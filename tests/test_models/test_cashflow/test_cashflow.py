"""Tests for cashflow engine: amortization, prepayment, waterfall."""

import numpy as raw_np
import pytest

from trellis.models.cashflow_engine.amortization import level_pay
from trellis.models.cashflow_engine.prepayment import PSA, CPR, RateDependent
from trellis.models.cashflow_engine.waterfall import Tranche, Waterfall


# ---------------------------------------------------------------------------
# level_pay
# ---------------------------------------------------------------------------


class TestLevelPay:
    def test_total_payments_equal_principal_plus_interest(self):
        """Sum of all payments = principal + total interest."""
        balance = 100000.0
        annual_rate = 0.06
        monthly_rate = annual_rate / 12
        n_periods = 360  # 30 years

        schedule = level_pay(balance, monthly_rate, n_periods)
        total_interest = sum(interest for interest, _ in schedule)
        total_principal = sum(principal for _, principal in schedule)

        assert total_principal == pytest.approx(balance, rel=1e-4)
        # Total payments = total_interest + total_principal
        total_payments = total_interest + total_principal
        assert total_payments > balance  # must pay interest

    def test_final_balance_approx_zero(self):
        """After all payments, remaining balance should be approximately zero."""
        balance = 100000.0
        rate = 0.005  # monthly
        n = 360
        schedule = level_pay(balance, rate, n)
        remaining = balance
        for interest, principal in schedule:
            remaining -= principal
        assert remaining == pytest.approx(0.0, abs=0.01)

    def test_constant_total_payment(self):
        """Each period's total payment (interest + principal) should be constant."""
        balance = 50000.0
        rate = 0.004
        n = 120
        schedule = level_pay(balance, rate, n)
        payments = [i + p for i, p in schedule]
        for pmt in payments:
            assert pmt == pytest.approx(payments[0], rel=1e-6)


# ---------------------------------------------------------------------------
# PSA
# ---------------------------------------------------------------------------


class TestPSA:
    def test_cpr_ramps_up(self):
        """CPR at month 1 < CPR at month 30."""
        psa = PSA(speed=1.0)
        assert psa.cpr(1) < psa.cpr(30)

    def test_cpr_constant_after_30(self):
        """CPR is constant at 6% after month 30."""
        psa = PSA(speed=1.0)
        cpr_30 = psa.cpr(30)
        cpr_31 = psa.cpr(31)
        cpr_100 = psa.cpr(100)
        assert cpr_30 == pytest.approx(0.06, rel=1e-10)
        assert cpr_31 == pytest.approx(0.06, rel=1e-10)
        assert cpr_100 == pytest.approx(0.06, rel=1e-10)

    def test_speed_multiplier(self):
        """200% PSA doubles the CPR."""
        psa_100 = PSA(speed=1.0)
        psa_200 = PSA(speed=2.0)
        assert psa_200.cpr(15) == pytest.approx(2 * psa_100.cpr(15), rel=1e-10)

    def test_smm_positive(self):
        psa = PSA(speed=1.0)
        for m in [1, 15, 30, 60]:
            assert psa.smm(m) > 0


# ---------------------------------------------------------------------------
# CPR
# ---------------------------------------------------------------------------


class TestCPR:
    def test_constant_rate(self):
        """CPR is the same at all months."""
        cpr = CPR(rate=0.08)
        for m in [1, 10, 30, 100, 360]:
            assert cpr.cpr(m) == pytest.approx(0.08, rel=1e-12)

    def test_smm_consistent_with_cpr(self):
        """SMM = 1 - (1 - CPR)^(1/12)."""
        rate = 0.06
        cpr_model = CPR(rate)
        expected_smm = 1 - (1 - rate) ** (1 / 12)
        assert cpr_model.smm(5) == pytest.approx(expected_smm, rel=1e-10)


# ---------------------------------------------------------------------------
# RateDependent
# ---------------------------------------------------------------------------


class TestRateDependent:
    def test_higher_incentive_higher_cpr(self):
        """When current rate is much lower than coupon, CPR should be higher."""
        model = RateDependent(coupon=0.06, base_cpr=0.06,
                              incentive_mult=0.3, burnout=0.01)
        month = 12
        cpr_no_incentive = model.cpr(month, current_rate=0.06)
        cpr_with_incentive = model.cpr(month, current_rate=0.04)
        assert cpr_with_incentive > cpr_no_incentive

    def test_no_incentive_gives_base_cpr(self):
        """When current rate >= coupon, CPR = base_cpr."""
        model = RateDependent(coupon=0.06, base_cpr=0.06,
                              incentive_mult=0.3, burnout=0.01)
        assert model.cpr(1, current_rate=0.06) == pytest.approx(0.06, rel=1e-10)
        assert model.cpr(1, current_rate=0.07) == pytest.approx(0.06, rel=1e-10)


# ---------------------------------------------------------------------------
# Waterfall
# ---------------------------------------------------------------------------


class TestWaterfall:
    def test_senior_paid_before_junior(self):
        """Senior tranche receives interest/principal before junior."""
        senior = Tranche(name="A", notional=80.0, coupon=0.04, subordination=0.0)
        junior = Tranche(name="B", notional=20.0, coupon=0.06, subordination=1.0)
        wf = Waterfall([junior, senior])  # passed out of order on purpose

        # Limited cash: not enough for everyone
        result = wf.distribute(available_interest=2.0, available_principal=5.0, period=0.5)

        # Senior interest due: 80 * 0.04 * 0.5 = 1.6
        # Junior interest due: 20 * 0.06 * 0.5 = 0.6
        # Total interest available: 2.0, enough for both
        assert result["A"]["interest"] == pytest.approx(1.6, rel=1e-10)
        assert result["B"]["interest"] == pytest.approx(0.4, rel=1e-10)

        # Principal: senior gets first, 5.0 available, senior balance=80
        assert result["A"]["principal"] == pytest.approx(5.0, rel=1e-10)
        assert result["B"]["principal"] == pytest.approx(0.0, abs=1e-10)

    def test_total_distributed_leq_total_available(self):
        senior = Tranche(name="A", notional=100.0, coupon=0.05, subordination=0.0)
        junior = Tranche(name="B", notional=50.0, coupon=0.08, subordination=1.0)
        wf = Waterfall([senior, junior])

        avail_int, avail_prin = 10.0, 20.0
        result = wf.distribute(avail_int, avail_prin, period=1.0)

        total_int = sum(result[t.name]["interest"] for t in wf.tranches)
        total_prin = sum(result[t.name]["principal"] for t in wf.tranches)
        total_int += result["_residual"]["interest"]
        total_prin += result["_residual"]["principal"]

        assert total_int == pytest.approx(avail_int, rel=1e-10)
        assert total_prin == pytest.approx(avail_prin, rel=1e-10)

    def test_tranche_balance_decreases(self):
        """Tranche balance should decrease after principal distribution."""
        tranche = Tranche(name="A", notional=100.0, coupon=0.05, subordination=0.0)
        wf = Waterfall([tranche])

        initial_balance = tranche.balance
        wf.distribute(available_interest=5.0, available_principal=10.0, period=1.0)
        assert tranche.balance < initial_balance
        assert tranche.balance == pytest.approx(90.0, rel=1e-10)

    def test_waterfall_run_multiple_periods(self):
        """Run over multiple periods and verify balances decrease."""
        senior = Tranche(name="A", notional=100.0, coupon=0.05, subordination=0.0)
        wf = Waterfall([senior])
        cashflows = [(2.5, 10.0), (2.0, 10.0), (1.5, 10.0)]
        results = wf.run(cashflows, period=0.5)
        assert len(results) == 3
        assert senior.balance == pytest.approx(70.0, rel=1e-10)
