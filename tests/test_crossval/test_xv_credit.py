"""XV5: Credit cross-validation against external and independent internal evidence."""

from dataclasses import dataclass
from datetime import date

import numpy as raw_np
import pytest

# --- Trellis ---
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention

SETTLE = date(2024, 11, 15)


def _require_quantlib():
    pytest.importorskip("QuantLib")


def trellis_survival(hazard_rate, t):
    cc = CreditCurve.flat(hazard_rate)
    return float(cc.survival_probability(t))


def quantlib_survival(hazard_rate, t):
    _require_quantlib()
    import QuantLib as ql
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today

    quote = ql.QuoteHandle(ql.SimpleQuote(hazard_rate))
    flat_hazard = ql.FlatHazardRate(today, quote, ql.Actual365Fixed())
    target_date = today + ql.Period(int(t * 365), ql.Days)
    return flat_hazard.survivalProbability(target_date)


class TestCreditCrossValidation:

    def test_survival_prob_vs_quantlib(self):
        """Trellis survival probability matches QuantLib."""
        for lam in [0.01, 0.02, 0.05]:
            for t in [1, 5, 10]:
                trellis_sp = trellis_survival(lam, t)
                ql_sp = quantlib_survival(lam, t)
                assert trellis_sp == pytest.approx(ql_sp, rel=0.01), (
                    f"λ={lam}, t={t}: Trellis={trellis_sp:.6f}, QL={ql_sp:.6f}"
                )

    def test_hazard_rate_from_spreads(self):
        """CDS spread → hazard rate: λ ≈ spread / (1-R)."""
        spread = 0.01  # 100bp
        R = 0.4
        cc = CreditCurve.from_spreads({5.0: spread}, recovery=R)
        expected_lam = spread / (1 - R)
        assert float(cc.hazard_rate(5.0)) == pytest.approx(expected_lam, rel=1e-6)

    def test_survival_decreasing_all_libs(self):
        """Both agree: survival probability decreases with time."""
        for t1, t2 in [(1, 5), (5, 10)]:
            assert trellis_survival(0.02, t1) > trellis_survival(0.02, t2)
            assert quantlib_survival(0.02, t1) > quantlib_survival(0.02, t2)

    @pytest.mark.parametrize("rank", [1, 2])
    def test_homogeneous_rank_integration_matches_seeded_default_time_mc(self, rank):
        from trellis.core.differentiable import get_numpy
        from trellis.models.contingent_cashflows import (
            ProtectionPayment,
            nth_to_default_probability,
            protection_payment_pv,
            rank_trigger_probability,
        )
        from trellis.models.copulas.gaussian import GaussianCopula
        from trellis.models.credit_basket_copula import resolve_credit_basket_inputs

        @dataclass(frozen=True)
        class Spec:
            notional: float = 10_000_000.0
            n_names: int = 5
            n_th: int = rank
            end_date: date = date(2029, 11, 15)
            correlation: float = 0.3
            recovery: float = 0.4
            day_count: DayCountConvention = DayCountConvention.ACT_360

        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.04, max_tenor=10.0),
            credit_curve=CreditCurve.flat(0.03, max_tenor=10.0),
        )
        spec = Spec()
        resolved = resolve_credit_basket_inputs(market_state, spec)
        analytical_probability = nth_to_default_probability(
            resolved.n_names,
            spec.n_th,
            resolved.default_probability,
            resolved.correlation,
        )
        analytical_price = protection_payment_pv(
            ProtectionPayment(
                notional=resolved.notional,
                recovery=resolved.recovery,
                default_probability=analytical_probability,
                discount_factor=resolved.discount_factor,
            )
        )

        np = get_numpy()
        default_times = GaussianCopula(
            correlation=resolved.correlation,
            n_names=resolved.n_names,
        ).sample_default_times(
            np.full(resolved.n_names, resolved.hazard_rate),
            n_paths=250_000,
            rng=np.random.default_rng(42),
        )
        sampled_probability = rank_trigger_probability(
            default_times,
            rank=spec.n_th,
            horizon=resolved.horizon,
        )
        sampled_price = protection_payment_pv(
            ProtectionPayment(
                notional=resolved.notional,
                recovery=resolved.recovery,
                default_probability=sampled_probability,
                discount_factor=resolved.discount_factor,
            )
        )

        assert sampled_probability == pytest.approx(analytical_probability, rel=0.025)
        assert sampled_price == pytest.approx(analytical_price, rel=0.025)
