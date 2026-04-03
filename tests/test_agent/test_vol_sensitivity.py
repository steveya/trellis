"""Test that the vol sensitivity invariant catches vol-insensitive pricers."""

from datetime import date

import pytest

from trellis.agent.invariants import check_vol_sensitivity
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _ms(vol=0.20, rate=0.05):
    return MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
    )


class TestVolSensitivityInvariant:

    def test_catches_vol_insensitive_payoff(self):
        """A payoff that ignores vol should FAIL this invariant."""

        class VolInsensitive:
            @property
            def requirements(self):
                return {"discount_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                # Ignores vol entirely — just discounts notional
                return 100.0 * market_state.discount.discount(5.0)

        failures = check_vol_sensitivity(
            lambda: VolInsensitive(), _ms,
        )
        assert len(failures) > 0
        assert "non-zero vega" in failures[0]

    def test_passes_vol_sensitive_payoff(self):
        """A cap (vol-sensitive) should PASS."""
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency

        spec = CapFloorSpec(
            notional=1e6, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )

        failures = check_vol_sensitivity(lambda: CapPayoff(spec), _ms)
        assert failures == []
