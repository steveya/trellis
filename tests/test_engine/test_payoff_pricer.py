"""Tests for Payoff protocol, MarketState, and price_payoff."""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import DeterministicCashflowPayoff, Payoff
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond


SETTLE = date(2024, 11, 15)


def _curve():
    return YieldCurve.flat(0.045)


def _bond():
    return Bond(
        face=100, coupon=0.045, maturity_date=date(2034, 11, 15),
        maturity=10, frequency=2,
    )


def _market_state(curve=None):
    return MarketState(as_of=SETTLE, settlement=SETTLE, discount=curve or _curve())


# --- MarketState tests ---


class TestMarketState:

    def test_frozen(self):
        ms = _market_state()
        with pytest.raises(AttributeError):
            ms.as_of = date(2025, 1, 1)

    def test_available_capabilities_with_discount(self):
        ms = _market_state()
        assert {"discount_curve", "forward_curve"} <= ms.available_capabilities

    def test_available_capabilities_without_discount(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=None)
        assert ms.available_capabilities == set()

    def test_fields(self):
        ms = _market_state()
        assert ms.as_of == SETTLE
        assert ms.settlement == SETTLE
        assert ms.discount is not None


# --- MissingCapabilityError tests ---


class TestMissingCapabilityError:

    def test_message(self):
        err = MissingCapabilityError({"black_vol_surface"}, {"discount_curve"})
        assert "black_vol_surface" in str(err)
        assert "discount_curve" in str(err)

    def test_fields(self):
        err = MissingCapabilityError({"forward_curve"}, {"discount_curve"})
        assert err.missing == {"forward_curve"}
        assert err.available == {"discount_curve"}


# --- Payoff protocol tests ---


class TestPayoffProtocol:

    def test_deterministic_cashflow_is_payoff(self):
        adapter = DeterministicCashflowPayoff(_bond())
        assert isinstance(adapter, Payoff)

    def test_requirements(self):
        adapter = DeterministicCashflowPayoff(_bond())
        assert adapter.requirements == {"discount_curve"}

    def test_evaluate_returns_float(self):
        adapter = DeterministicCashflowPayoff(_bond())
        ms = _market_state()
        result = adapter.evaluate(ms)
        assert isinstance(result, float)
        assert result > 0

    def test_evaluate_matches_manual_discount(self):
        bond = _bond()
        adapter = DeterministicCashflowPayoff(bond, day_count=bond.day_count)
        ms = _market_state()
        pv = adapter.evaluate(ms)
        # Manual discount
        from trellis.core.date_utils import year_fraction
        schedule = bond.cashflows(SETTLE)
        manual_pv = sum(
            amt * float(ms.discount.discount(year_fraction(SETTLE, d, bond.day_count)))
            for d, amt in zip(schedule.dates, schedule.amounts)
        )
        assert pv == pytest.approx(manual_pv, rel=1e-10)

    def test_instrument_accessor(self):
        bond = _bond()
        adapter = DeterministicCashflowPayoff(bond)
        assert adapter.instrument is bond


# --- price_payoff tests ---


class TestPricePayoff:

    def test_basic_pricing(self):
        adapter = DeterministicCashflowPayoff(_bond())
        ms = _market_state()
        pv = price_payoff(DeterministicCashflowPayoff(_bond(), day_count=DayCountConvention.ACT_ACT), ms)
        assert 80 < pv < 120

    def test_missing_capability_raises(self):
        adapter = DeterministicCashflowPayoff(_bond())
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=None)
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(adapter, ms)
        assert "discount_curve" in exc_info.value.missing

    def test_zero_coupon_bond(self):
        bond = Bond(
            face=100, coupon=0.0,
            maturity_date=date(2034, 3, 19),
            maturity=10, frequency=2,
            issue_date=date(2024, 3, 19),
        )
        settle = date(2024, 3, 19)
        curve = YieldCurve.flat(0.05)
        ms = MarketState(as_of=settle, settlement=settle, discount=curve)
        adapter = DeterministicCashflowPayoff(bond)
        pv = price_payoff(adapter, ms)
        expected = 100 * np.exp(-0.05 * 10)
        assert pv == pytest.approx(expected, rel=0.01)

    def test_zero_pv_payoff(self):
        """A payoff returning 0.0 prices to zero."""

        class ZeroPayoff:
            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, market_state):
                return 0.0

        ms = _market_state()
        assert price_payoff(ZeroPayoff(), ms) == 0.0


# --- Round-trip tests: Bond via price_payoff must match price_instrument ---


class TestRoundTrip:

    def test_round_trip_flat_curve(self):
        bond = _bond()
        curve = _curve()
        result_a = price_instrument(bond, curve, SETTLE, greeks=None)

        adapter = DeterministicCashflowPayoff(bond)
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=curve)
        pv_b = price_payoff(DeterministicCashflowPayoff(bond, day_count=bond.day_count), ms)

        assert pv_b == pytest.approx(result_a.dirty_price, rel=1e-12)

    def test_round_trip_real_curve(self):
        from trellis.data.mock import MockDataProvider
        provider = MockDataProvider()
        yields = provider.fetch_yields(SETTLE)
        curve = YieldCurve.from_treasury_yields(yields)

        bond = _bond()
        result_a = price_instrument(bond, curve, SETTLE, greeks=None)

        adapter = DeterministicCashflowPayoff(bond)
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=curve)
        pv_b = price_payoff(DeterministicCashflowPayoff(bond, day_count=bond.day_count), ms)

        assert pv_b == pytest.approx(result_a.dirty_price, rel=1e-12)

    def test_round_trip_all_sample_bonds(self):
        from trellis.samples import (
            sample_bond_2y, sample_bond_5y,
            sample_bond_10y, sample_bond_30y, sample_curve,
        )
        curve = sample_curve()
        for bond_fn in [sample_bond_2y, sample_bond_5y, sample_bond_10y, sample_bond_30y]:
            bond = bond_fn()
            result_a = price_instrument(bond, curve, SETTLE, greeks=None)
            adapter = DeterministicCashflowPayoff(bond)
            ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=curve)
            pv_b = price_payoff(DeterministicCashflowPayoff(bond, day_count=bond.day_count), ms)
            assert pv_b == pytest.approx(result_a.dirty_price, rel=1e-12), (
                f"Round-trip failed for {bond_fn.__name__}"
            )


# --- Custom Payoff (verifies the protocol works beyond Bond adapter) ---


class TestCustomPayoff:

    def test_custom_payoff_returns_float(self):
        """A hand-rolled Payoff returning a float works with price_payoff."""
        from trellis.core.date_utils import year_fraction

        class SingleCashflowPayoff:
            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, market_state):
                t = year_fraction(market_state.settlement, date(2025, 11, 15))
                return 100.0 * market_state.discount.discount(t)

        payoff = SingleCashflowPayoff()
        assert isinstance(payoff, Payoff)
        ms = _market_state()
        pv = price_payoff(payoff, ms)
        assert pv > 0
        assert pv < 100  # discounted

    def test_payoff_with_extra_requirements_raises(self):
        """A payoff requiring something MarketState can't provide."""

        class VolPayoff:
            @property
            def requirements(self):
                return {"discount_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                return 0.0

        ms = _market_state()
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(VolPayoff(), ms)
        assert "black_vol_surface" in exc_info.value.missing


class TestEvaluateReturnsFloat:

    def test_float_passthrough(self):
        """evaluate() returns float → price_payoff returns it directly."""

        class DirectPayoff:
            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, market_state):
                return 42.57

        ms = _market_state()
        assert price_payoff(DirectPayoff(), ms) == 42.57

    def test_two_payoffs_same_result(self):
        """Two payoffs computing the same thing should agree."""
        from trellis.core.date_utils import year_fraction

        class PayoffA:
            @property
            def requirements(self):
                return {"discount_curve"}
            def evaluate(self, ms):
                t = year_fraction(ms.settlement, date(2025, 11, 15))
                return 100.0 * ms.discount.discount(t)

        class PayoffB:
            @property
            def requirements(self):
                return {"discount_curve"}
            def evaluate(self, ms):
                t = year_fraction(ms.settlement, date(2025, 11, 15))
                return 100.0 * ms.discount.discount(t)

        ms = _market_state()
        assert price_payoff(PayoffA(), ms) == price_payoff(PayoffB(), ms)
