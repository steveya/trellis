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
        assert ms.available_capabilities == {"discount", "forward_rate"}

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
        err = MissingCapabilityError({"black_vol"}, {"discount"})
        assert "black_vol" in str(err)
        assert "discount" in str(err)

    def test_fields(self):
        err = MissingCapabilityError({"forward_rate"}, {"discount"})
        assert err.missing == {"forward_rate"}
        assert err.available == {"discount"}


# --- Payoff protocol tests ---


class TestPayoffProtocol:

    def test_deterministic_cashflow_is_payoff(self):
        adapter = DeterministicCashflowPayoff(_bond())
        assert isinstance(adapter, Payoff)

    def test_requirements(self):
        adapter = DeterministicCashflowPayoff(_bond())
        assert adapter.requirements == {"discount"}

    def test_evaluate_returns_cashflows(self):
        from trellis.core.payoff import Cashflows
        adapter = DeterministicCashflowPayoff(_bond())
        ms = _market_state()
        result = adapter.evaluate(ms)
        assert isinstance(result, Cashflows)
        assert len(result.flows) > 0
        for cf_date, amount in result.flows:
            assert isinstance(cf_date, date)
            assert isinstance(amount, float)
            assert cf_date > SETTLE

    def test_evaluate_matches_bond_cashflows(self):
        bond = _bond()
        adapter = DeterministicCashflowPayoff(bond)
        ms = _market_state()
        result = adapter.evaluate(ms)
        bond_schedule = bond.cashflows(SETTLE)
        assert len(result.flows) == len(bond_schedule.dates)
        for (pf_date, pf_amt), b_date, b_amt in zip(
            result.flows, bond_schedule.dates, bond_schedule.amounts
        ):
            assert pf_date == b_date
            assert pf_amt == b_amt

    def test_instrument_accessor(self):
        bond = _bond()
        adapter = DeterministicCashflowPayoff(bond)
        assert adapter.instrument is bond


# --- price_payoff tests ---


class TestPricePayoff:

    def test_basic_pricing(self):
        adapter = DeterministicCashflowPayoff(_bond())
        ms = _market_state()
        pv = price_payoff(adapter, ms, day_count=DayCountConvention.ACT_ACT)
        assert 80 < pv < 120

    def test_missing_capability_raises(self):
        adapter = DeterministicCashflowPayoff(_bond())
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=None)
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(adapter, ms)
        assert "discount" in exc_info.value.missing

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

    def test_empty_cashflows_returns_zero(self):
        """A payoff with no cashflows prices to zero."""

        class EmptyPayoff:
            @property
            def requirements(self):
                return {"discount"}

            def evaluate(self, market_state):
                return []

        ms = _market_state()
        assert price_payoff(EmptyPayoff(), ms) == 0.0


# --- Round-trip tests: Bond via price_payoff must match price_instrument ---


class TestRoundTrip:

    def test_round_trip_flat_curve(self):
        bond = _bond()
        curve = _curve()
        result_a = price_instrument(bond, curve, SETTLE, greeks=None)

        adapter = DeterministicCashflowPayoff(bond)
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=curve)
        pv_b = price_payoff(adapter, ms, day_count=bond.day_count)

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
        pv_b = price_payoff(adapter, ms, day_count=bond.day_count)

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
            pv_b = price_payoff(adapter, ms, day_count=bond.day_count)
            assert pv_b == pytest.approx(result_a.dirty_price, rel=1e-12), (
                f"Round-trip failed for {bond_fn.__name__}"
            )


# --- Custom Payoff (verifies the protocol works beyond Bond adapter) ---


class TestCustomPayoff:

    def test_custom_payoff_implementation(self):
        """A hand-rolled Payoff should also work with price_payoff."""

        class SingleCashflow:
            @property
            def requirements(self):
                return {"discount"}

            def evaluate(self, market_state):
                return [(date(2025, 11, 15), 100.0)]

        payoff = SingleCashflow()
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
                return {"discount", "black_vol"}

            def evaluate(self, market_state):
                return []

        ms = _market_state()
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(VolPayoff(), ms)
        assert "black_vol" in exc_info.value.missing


# --- PresentValue return type ---


class TestPresentValueReturn:

    def test_present_value_passthrough(self):
        """A payoff returning PresentValue should not be discounted again."""
        from trellis.core.payoff import PresentValue

        class TreePayoff:
            @property
            def requirements(self):
                return {"discount"}

            def evaluate(self, market_state):
                return PresentValue(42.57)

        ms = _market_state()
        pv = price_payoff(TreePayoff(), ms)
        assert pv == 42.57

    def test_cashflows_return_matches_list(self):
        """Cashflows return type should produce same result as raw list."""
        from trellis.core.payoff import Cashflows

        class CfPayoff:
            @property
            def requirements(self):
                return {"discount"}

            def evaluate(self, market_state):
                return Cashflows([(date(2025, 11, 15), 100.0)])

        class ListPayoff:
            @property
            def requirements(self):
                return {"discount"}

            def evaluate(self, market_state):
                return [(date(2025, 11, 15), 100.0)]

        ms = _market_state()
        pv_cf = price_payoff(CfPayoff(), ms)
        pv_list = price_payoff(ListPayoff(), ms)
        assert pv_cf == pytest.approx(pv_list, rel=1e-12)

    def test_present_value_vs_manual_discount(self):
        """PresentValue(X) should equal Cashflows([(settlement, X)]) discounted by df(0)=1."""
        from trellis.core.payoff import Cashflows, PresentValue

        target_pv = 95.0

        class PVPayoff:
            @property
            def requirements(self):
                return {"discount"}
            def evaluate(self, ms):
                return PresentValue(target_pv)

        class CfPayoff:
            @property
            def requirements(self):
                return {"discount"}
            def evaluate(self, ms):
                # Cashflow at settlement → df(0) = 1.0 → no discounting
                return Cashflows([(ms.settlement, target_pv)])

        ms = _market_state()
        assert price_payoff(PVPayoff(), ms) == price_payoff(CfPayoff(), ms)
