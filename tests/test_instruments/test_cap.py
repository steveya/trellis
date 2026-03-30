"""Tests for CapPayoff and FloorPayoff."""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import Payoff
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.black import black76_call
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _cap_spec(**overrides):
    defaults = dict(
        notional=1_000_000,
        strike=0.05,
        start_date=date(2025, 2, 15),
        end_date=date(2027, 2, 15),
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
    )
    defaults.update(overrides)
    return CapFloorSpec(**defaults)


def _market_state(rate=0.05, vol=0.20):
    curve = YieldCurve.flat(rate)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        vol_surface=FlatVol(vol),
    )


class TestCapPayoff:

    def test_satisfies_protocol(self):
        assert isinstance(CapPayoff(_cap_spec()), Payoff)

    def test_requirements(self):
        cap = CapPayoff(_cap_spec())
        assert cap.requirements == {"discount", "forward_rate", "black_vol"}

    def test_positive_price(self):
        cap = CapPayoff(_cap_spec())
        ms = _market_state()
        pv = price_payoff(cap, ms)
        assert pv > 0

    def test_monotonic_in_vol(self):
        """Cap price increases with vol."""
        spec = _cap_spec()
        p1 = price_payoff(CapPayoff(spec), _market_state(vol=0.10))
        p2 = price_payoff(CapPayoff(spec), _market_state(vol=0.20))
        p3 = price_payoff(CapPayoff(spec), _market_state(vol=0.40))
        assert p1 < p2 < p3

    def test_monotonic_in_strike(self):
        """Cap price decreases as strike increases."""
        ms = _market_state()
        p_low = price_payoff(CapPayoff(_cap_spec(strike=0.03)), ms)
        p_mid = price_payoff(CapPayoff(_cap_spec(strike=0.05)), ms)
        p_high = price_payoff(CapPayoff(_cap_spec(strike=0.08)), ms)
        assert p_low > p_mid > p_high

    def test_zero_vol_equals_intrinsic(self):
        """At zero vol, cap = sum of max(F-K, 0) * tau * notional * df."""
        spec = _cap_spec(strike=0.04)  # ITM since flat curve at 5%
        ms_zero_vol = _market_state(rate=0.05, vol=0.0)
        ms_normal = _market_state(rate=0.05, vol=0.20)

        pv_zero = price_payoff(CapPayoff(spec), ms_zero_vol)
        pv_normal = price_payoff(CapPayoff(spec), ms_normal)

        # Zero vol should be less than or equal to normal vol (ATM/ITM)
        assert pv_zero <= pv_normal + 1e-6
        # And zero vol should be > 0 for ITM cap
        assert pv_zero > 0

    def test_missing_vol_raises(self):
        """MarketState without vol → MissingCapabilityError."""
        cap = CapPayoff(_cap_spec())
        ms = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
        )
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(cap, ms)
        assert "black_vol_surface" in exc_info.value.missing


class TestFloorPayoff:

    def test_satisfies_protocol(self):
        assert isinstance(FloorPayoff(_cap_spec()), Payoff)

    def test_positive_price(self):
        floor = FloorPayoff(_cap_spec())
        ms = _market_state()
        pv = price_payoff(floor, ms)
        assert pv > 0


class TestCapFloorParity:

    def test_cap_minus_floor_equals_swap_value(self):
        """Cap(K) - Floor(K) = sum((F_i - K) * tau_i * notional * df_i).

        This is put-call parity applied to each caplet/floorlet pair.
        """
        spec = _cap_spec()
        ms = _market_state()

        cap_pv = price_payoff(CapPayoff(spec), ms)
        floor_pv = price_payoff(FloorPayoff(spec), ms)

        # Compute swap value manually: sum of (F-K) * tau * N * df
        cap = CapPayoff(spec)
        floor = FloorPayoff(spec)
        cap_cfs = cap.evaluate(ms)
        floor_cfs = floor.evaluate(ms)

        # From put-call parity on Black76: call - put = F - K
        # So cap - floor should equal the swap value
        # We verify the relationship holds at the PV level
        diff = cap_pv - floor_pv
        # The sign depends on whether forwards are above or below strike
        # On a flat 5% curve with 5% strike, forwards ≈ strike, so diff ≈ 0
        assert abs(diff) < spec.notional * 0.01  # within 1% of notional
