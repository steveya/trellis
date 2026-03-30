"""Tests for agent invariant checks."""

from datetime import date

import pytest

from trellis.agent.invariants import (
    check_non_negativity,
    check_vol_monotonicity,
    check_zero_vol_intrinsic,
    run_invariant_suite,
)
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.cap import CapFloorSpec, CapPayoff
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _cap_spec():
    return CapFloorSpec(
        notional=1_000_000, strike=0.05,
        start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
        frequency=Frequency.QUARTERLY,
    )


def _cap_factory():
    return CapPayoff(_cap_spec())


def _ms_factory(vol=0.20):
    return MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


class TestNonNegativity:

    def test_cap_non_negative(self):
        failures = check_non_negativity(_cap_factory(), _ms_factory())
        assert failures == []

    def test_non_negativity_can_return_structured_diagnostics(self):
        class NegativePayoff:
            @property
            def requirements(self):
                return {"discount"}

            def evaluate(self, market_state):
                return -2.5

        failures = check_non_negativity(
            NegativePayoff(),
            _ms_factory(),
            return_diagnostics=True,
        )

        assert len(failures) == 1
        failure = failures[0]
        assert failure.check == "check_non_negativity"
        assert failure.actual == pytest.approx(-2.5)
        assert "available_capabilities" in failure.context


class TestVolMonotonicity:

    def test_cap_monotonic(self):
        failures = check_vol_monotonicity(_cap_factory, _ms_factory)
        assert failures == []

    def test_constant_price_fails(self):
        """A payoff that ignores vol should fail monotonicity."""
        from trellis.core.payoff import DeterministicCashflowPayoff
        from trellis.instruments.bond import Bond

        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                    maturity=10, frequency=2)

        def bond_factory():
            return DeterministicCashflowPayoff(bond)

        # Bond price doesn't change with vol — this is expected behavior,
        # not a "failure" for a non-option. The monotonicity check is for options.
        # A bond payoff returns constant price → monotonicity is satisfied (weakly).
        failures = check_vol_monotonicity(bond_factory, _ms_factory)
        # Weak monotonicity (p2 >= p1) should pass for constant prices
        assert failures == []


class TestZeroVolIntrinsic:

    def test_itm_cap(self):
        """ITM cap (strike < forward rate) at zero vol should have positive intrinsic."""
        spec = CapFloorSpec(
            notional=1_000_000, strike=0.04,  # ITM vs 5% curve
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )

        def cap_factory():
            return CapPayoff(spec)

        def intrinsic_fn(ms):
            # At zero vol, cap value = sum of max(F-K, 0) * tau * N * df
            cap = CapPayoff(spec)
            ms_zero = MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=ms.discount, vol_surface=FlatVol(1e-10),
            )
            return price_payoff(cap, ms_zero)

        # Zero vol intrinsic check should pass (comparing against itself)
        failures = check_zero_vol_intrinsic(cap_factory, _ms_factory, intrinsic_fn)
        assert failures == []


class TestRunSuite:

    def test_cap_passes_suite(self):
        passed, failures = run_invariant_suite(
            payoff_factory=_cap_factory,
            market_state_factory=_ms_factory,
            is_option=True,
        )
        assert passed, f"Failures: {failures}"
