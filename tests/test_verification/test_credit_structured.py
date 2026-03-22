"""WP4: Credit and structured products verification."""

import numpy as raw_np
import pytest

from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.copulas.gaussian import GaussianCopula
from trellis.models.copulas.factor import FactorCopula
from trellis.models.cashflow_engine.waterfall import Waterfall, Tranche
from trellis.models.cashflow_engine.prepayment import PSA, CPR
from trellis.models.cashflow_engine.amortization import level_pay


# ---------------------------------------------------------------------------
# Credit curve verification
# ---------------------------------------------------------------------------

class TestCreditCurveProperties:

    def test_survival_decreasing(self):
        cc = CreditCurve.flat(0.02)
        for t in [1, 5, 10, 20]:
            prev = float(cc.survival_probability(max(t - 1, 0.5)))
            curr = float(cc.survival_probability(t))
            assert curr <= prev

    def test_survival_at_zero_is_one(self):
        cc = CreditCurve.flat(0.02)
        assert float(cc.survival_probability(0.0)) == pytest.approx(1.0, rel=1e-10)

    def test_hazard_rate_positive(self):
        cc = CreditCurve([1, 5, 10], [0.01, 0.02, 0.03])
        for t in [0.5, 1, 3, 5, 7, 10]:
            assert float(cc.hazard_rate(t)) > 0

    def test_from_spreads_roundtrip(self):
        """CDS spreads → hazard rates → back to spreads."""
        spreads = {1.0: 0.005, 5.0: 0.01, 10.0: 0.015}
        R = 0.4
        cc = CreditCurve.from_spreads(spreads, recovery=R)
        for t, s in spreads.items():
            lam = float(cc.hazard_rate(t))
            recovered_spread = lam * (1 - R)
            assert recovered_spread == pytest.approx(s, rel=1e-6)


# ---------------------------------------------------------------------------
# Copula verification
# ---------------------------------------------------------------------------

class TestGaussianCopula:

    def test_independent_defaults(self):
        """With identity correlation, defaults are independent."""
        n = 5
        corr = raw_np.eye(n)
        copula = GaussianCopula(corr)
        rng = raw_np.random.default_rng(42)
        U = copula.sample_uniforms(100000, rng)
        # Each marginal should be approximately uniform
        for i in range(n):
            assert raw_np.mean(U[:, i]) == pytest.approx(0.5, abs=0.01)
        # Correlation between columns should be near zero
        for i in range(n):
            for j in range(i + 1, n):
                corr_ij = raw_np.corrcoef(U[:, i], U[:, j])[0, 1]
                assert abs(corr_ij) < 0.02

    def test_high_correlation_joint_defaults(self):
        """High correlation → more joint defaults."""
        n = 5
        lam = 0.02
        hazards = raw_np.full(n, lam)

        # Low correlation
        corr_low = raw_np.full((n, n), 0.1)
        raw_np.fill_diagonal(corr_low, 1.0)
        copula_low = GaussianCopula(corr_low)
        taus_low = copula_low.sample_default_times(hazards, 50000, raw_np.random.default_rng(42))
        joint_low = raw_np.mean(raw_np.all(taus_low < 5.0, axis=1))

        # High correlation
        corr_high = raw_np.full((n, n), 0.8)
        raw_np.fill_diagonal(corr_high, 1.0)
        copula_high = GaussianCopula(corr_high)
        taus_high = copula_high.sample_default_times(hazards, 50000, raw_np.random.default_rng(42))
        joint_high = raw_np.mean(raw_np.all(taus_high < 5.0, axis=1))

        assert joint_high > joint_low


class TestFactorCopula:

    def test_loss_distribution_sums_to_one(self):
        fc = FactorCopula(n_names=100, correlation=0.3)
        losses, probs = fc.loss_distribution(0.05)
        assert raw_np.sum(probs) == pytest.approx(1.0, rel=1e-6)

    def test_expected_loss_matches_marginal(self):
        """E[loss fraction] ≈ marginal default probability."""
        fc = FactorCopula(n_names=100, correlation=0.3)
        p = 0.05
        losses, probs = fc.loss_distribution(p)
        expected_loss = raw_np.sum(losses * probs) / 100
        assert expected_loss == pytest.approx(p, rel=0.05)

    def test_higher_correlation_fatter_tails(self):
        """Higher correlation → more probability in the tails."""
        p = 0.05
        fc_low = FactorCopula(n_names=100, correlation=0.1)
        fc_high = FactorCopula(n_names=100, correlation=0.5)
        _, probs_low = fc_low.loss_distribution(p)
        _, probs_high = fc_high.loss_distribution(p)
        # P(more than 20 defaults) should be higher with high correlation
        tail_low = raw_np.sum(probs_low[20:])
        tail_high = raw_np.sum(probs_high[20:])
        assert tail_high > tail_low

    def test_zero_correlation_binomial(self):
        """With zero correlation, loss distribution = binomial(n, p)."""
        from scipy.stats import binom
        n, p = 50, 0.03
        fc = FactorCopula(n_names=n, correlation=1e-6)
        losses, probs = fc.loss_distribution(p)
        binom_probs = binom.pmf(range(n + 1), n, p)
        # Should be very close to binomial
        assert raw_np.sum(raw_np.abs(probs - binom_probs)) < 0.01


# ---------------------------------------------------------------------------
# Waterfall verification
# ---------------------------------------------------------------------------

class TestWaterfall:

    def test_conservation_of_cash(self):
        """Total distributed ≤ total available."""
        tranches = [
            Tranche("A", 80e6, 0.04, 0),
            Tranche("B", 20e6, 0.06, 1),
        ]
        wf = Waterfall(tranches)
        cashflows = [(5e6, 3e6)] * 10  # 10 periods
        results = wf.run(cashflows, period=0.5)
        for dist in results:
            total_int = sum(d["interest"] for k, d in dist.items() if k != "_residual")
            total_prin = sum(d["principal"] for k, d in dist.items() if k != "_residual")
            residual = dist["_residual"]
            assert total_int + residual["interest"] == pytest.approx(5e6, rel=1e-6)
            assert total_prin + residual["principal"] == pytest.approx(3e6, rel=1e-6)

    def test_senior_paid_first(self):
        """Senior tranche gets interest before junior when cash is scarce."""
        tranches = [
            Tranche("Senior", 80e6, 0.04, 0),
            Tranche("Junior", 20e6, 0.08, 1),
        ]
        wf = Waterfall(tranches)
        # Only $1M interest available — not enough for both
        result = wf.distribute(1e6, 0, period=0.5)
        # Senior should get its full due or all available
        senior_due = 80e6 * 0.04 * 0.5  # $1.6M due
        assert result["Senior"]["interest"] == pytest.approx(1e6)  # gets all available
        assert result["Junior"]["interest"] == 0.0  # nothing left

    def test_balance_decreases(self):
        """Tranche balance decreases with principal payments."""
        t = Tranche("A", 100e6, 0.04, 0)
        wf = Waterfall([t])
        wf.distribute(2e6, 10e6, period=0.5)
        assert t.balance == pytest.approx(90e6)


# ---------------------------------------------------------------------------
# Prepayment model verification
# ---------------------------------------------------------------------------

class TestPrepaymentModels:

    def test_psa_ramp(self):
        """PSA: CPR ramps to 6% at month 30."""
        psa = PSA(speed=1.0)
        assert psa.cpr(1) == pytest.approx(0.002)
        assert psa.cpr(15) == pytest.approx(0.03)
        assert psa.cpr(30) == pytest.approx(0.06)
        assert psa.cpr(60) == pytest.approx(0.06)

    def test_psa_speed(self):
        """200% PSA = double the CPR."""
        psa = PSA(speed=2.0)
        assert psa.cpr(30) == pytest.approx(0.12)

    def test_smm_from_cpr(self):
        """SMM = 1 - (1 - CPR)^(1/12)."""
        cpr_val = 0.06
        expected_smm = 1 - (1 - cpr_val) ** (1 / 12)
        psa = PSA(speed=1.0)
        assert psa.smm(30) == pytest.approx(expected_smm, rel=1e-6)

    def test_cpr_constant(self):
        cpr = CPR(0.08)
        for m in [1, 12, 60]:
            assert cpr.cpr(m) == 0.08


# ---------------------------------------------------------------------------
# Amortization verification
# ---------------------------------------------------------------------------

class TestAmortization:

    def test_level_pay_total(self):
        """Total payments = principal + total interest."""
        balance = 1_000_000
        rate = 0.005  # monthly rate
        n = 360  # 30 years
        schedule = level_pay(balance, rate, n)
        total_paid = sum(i + p for i, p in schedule)
        total_principal = sum(p for _, p in schedule)
        assert total_principal == pytest.approx(balance, rel=1e-4)
        assert total_paid > balance  # total > principal (includes interest)

    def test_level_pay_constant_payment(self):
        """Each payment is the same amount."""
        schedule = level_pay(100000, 0.004, 120)
        payments = [i + p for i, p in schedule]
        for payment in payments:
            assert payment == pytest.approx(payments[0], rel=1e-6)

    def test_level_pay_final_balance_zero(self):
        """After all payments, balance = 0."""
        balance = 500000
        schedule = level_pay(balance, 0.005, 180)
        remaining = balance
        for _, principal in schedule:
            remaining -= principal
        assert abs(remaining) < 1.0  # within $1
