"""Tests for reference implementations: callable bond (tree), barrier (MC), nth-to-default (copula)."""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff, PresentValue
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _ms(rate=0.05, vol=0.20, credit_hz=None):
    kwargs = dict(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
    )
    if credit_hz is not None:
        kwargs["credit_curve"] = CreditCurve.flat(credit_hz)
    return MarketState(**kwargs)


def _straight_bond_pv(rate=0.05):
    bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                maturity=10, frequency=2)
    ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(rate))
    return price_payoff(DeterministicCashflowPayoff(bond), ms, day_count=bond.day_count)


# ---------------------------------------------------------------------------
# Callable Bond (tree-based)
# ---------------------------------------------------------------------------

class TestCallableBond:

    def test_returns_present_value(self):
        from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
        spec = CallableBondSpec(
            notional=100, coupon=0.05,
            start_date=SETTLE, end_date=date(2034, 11, 15),
            call_dates=[date(2027, 11, 15), date(2029, 11, 15), date(2031, 11, 15)],
        )
        payoff = CallableBondPayoff(spec)
        result = payoff.evaluate(_ms())
        assert isinstance(result, PresentValue)

    def test_positive_price(self):
        from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
        spec = CallableBondSpec(
            notional=100, coupon=0.05,
            start_date=SETTLE, end_date=date(2034, 11, 15),
            call_dates=[date(2027, 11, 15), date(2029, 11, 15)],
        )
        pv = price_payoff(CallableBondPayoff(spec), _ms())
        assert pv > 0

    def test_callable_leq_straight(self):
        """Callable bond ≤ straight bond at all rate levels."""
        from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
        spec = CallableBondSpec(
            notional=100, coupon=0.05,
            start_date=SETTLE, end_date=date(2034, 11, 15),
            call_dates=[date(2027, 11, 15), date(2029, 11, 15), date(2031, 11, 15)],
        )
        for rate in [0.03, 0.05, 0.07]:
            callable_pv = price_payoff(CallableBondPayoff(spec), _ms(rate=rate))
            straight_pv = _straight_bond_pv(rate)
            assert callable_pv <= straight_pv + 0.5, (
                f"Rate={rate:.0%}: callable ({callable_pv:.2f}) > straight ({straight_pv:.2f})"
            )

    def test_requirements(self):
        from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
        spec = CallableBondSpec(
            notional=100, coupon=0.05,
            start_date=SETTLE, end_date=date(2034, 11, 15),
            call_dates=[date(2029, 11, 15)],
        )
        assert CallableBondPayoff(spec).requirements == {"discount", "black_vol"}


# ---------------------------------------------------------------------------
# Barrier Option (MC-based)
# ---------------------------------------------------------------------------

class TestBarrierOption:

    def test_returns_present_value(self):
        from trellis.instruments.barrier_option import BarrierOptionPayoff, BarrierOptionSpec
        spec = BarrierOptionSpec(
            notional=100, spot=100, strike=100, barrier=80,
            expiry_date=date(2025, 11, 15), barrier_type="down_and_out",
        )
        result = BarrierOptionPayoff(spec).evaluate(_ms())
        assert isinstance(result, PresentValue)

    def test_positive_price(self):
        from trellis.instruments.barrier_option import BarrierOptionPayoff, BarrierOptionSpec
        spec = BarrierOptionSpec(
            notional=100, spot=100, strike=100, barrier=80,
            expiry_date=date(2025, 11, 15), barrier_type="down_and_out",
        )
        pv = price_payoff(BarrierOptionPayoff(spec), _ms())
        assert pv > 0

    def test_knock_out_leq_vanilla(self):
        """Knock-out option ≤ vanilla option."""
        from trellis.instruments.barrier_option import BarrierOptionPayoff, BarrierOptionSpec
        spec_ko = BarrierOptionSpec(
            notional=100, spot=100, strike=100, barrier=80,
            expiry_date=date(2025, 11, 15), barrier_type="down_and_out",
        )
        ko_pv = price_payoff(BarrierOptionPayoff(spec_ko), _ms())

        # Vanilla call reference (Black76 on forward)
        from scipy.stats import norm
        T, r, sigma = 1.0, 0.05, 0.20
        S, K = 100, 100
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        vanilla = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        assert ko_pv <= vanilla * 1.1  # within 10% tolerance for MC noise

    def test_requirements(self):
        from trellis.instruments.barrier_option import BarrierOptionPayoff, BarrierOptionSpec
        spec = BarrierOptionSpec(
            notional=100, spot=100, strike=100, barrier=80,
            expiry_date=date(2025, 11, 15),
        )
        assert BarrierOptionPayoff(spec).requirements == {"discount", "black_vol"}


# ---------------------------------------------------------------------------
# Nth-to-Default (copula-based)
# ---------------------------------------------------------------------------

class TestNthToDefault:

    def test_returns_present_value(self):
        from trellis.instruments.nth_to_default import NthToDefaultPayoff, NthToDefaultSpec
        spec = NthToDefaultSpec(
            notional=1_000_000, n_names=5, n_th=1,
            end_date=date(2029, 11, 15),
        )
        result = NthToDefaultPayoff(spec).evaluate(_ms(credit_hz=0.02))
        assert isinstance(result, PresentValue)

    def test_positive_protection_value(self):
        from trellis.instruments.nth_to_default import NthToDefaultPayoff, NthToDefaultSpec
        spec = NthToDefaultSpec(
            notional=1_000_000, n_names=5, n_th=1,
            end_date=date(2029, 11, 15),
        )
        pv = price_payoff(NthToDefaultPayoff(spec), _ms(credit_hz=0.02))
        assert pv > 0

    def test_first_to_default_geq_second(self):
        """First-to-default is more valuable than second-to-default."""
        from trellis.instruments.nth_to_default import NthToDefaultPayoff, NthToDefaultSpec
        ms = _ms(credit_hz=0.02)
        spec1 = NthToDefaultSpec(
            notional=1_000_000, n_names=5, n_th=1,
            end_date=date(2029, 11, 15),
        )
        spec2 = NthToDefaultSpec(
            notional=1_000_000, n_names=5, n_th=2,
            end_date=date(2029, 11, 15),
        )
        pv1 = price_payoff(NthToDefaultPayoff(spec1), ms)
        pv2 = price_payoff(NthToDefaultPayoff(spec2), ms)
        assert pv1 > pv2

    def test_requirements(self):
        from trellis.instruments.nth_to_default import NthToDefaultPayoff, NthToDefaultSpec
        spec = NthToDefaultSpec(
            notional=1_000_000, n_names=5, n_th=1,
            end_date=date(2029, 11, 15),
        )
        assert NthToDefaultPayoff(spec).requirements == {"discount", "credit"}
