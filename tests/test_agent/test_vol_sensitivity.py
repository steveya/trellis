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

    def test_skips_flat_vol_bump_for_explicit_sabr_cap_strip(self):
        """A SABR-parameterized strip should not be forced through flat-surface vega."""
        from trellis.conventions.day_count import DayCountConvention
        from trellis.core.types import Frequency
        from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
        from trellis.instruments.cap import CapFloorSpec
        from trellis.models.rate_cap_floor import price_rate_cap_floor_strip_analytical

        class _SabrCapStripPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def spec(self):
                return self._spec

            @property
            def requirements(self):
                return {"discount_curve", "forward_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                return price_rate_cap_floor_strip_analytical(
                    market_state,
                    self._spec,
                    instrument_class="cap",
                )

        curve = DateAwareFlatYieldCurve(
            value_date=SETTLE,
            flat_rate=0.0425,
            curve_day_count=DayCountConvention.ACT_ACT_ISDA,
        )
        spec = CapFloorSpec(
            notional=1_000_000.0,
            strike=0.04,
            start_date=SETTLE,
            end_date=date(2029, 11, 15),
            frequency=Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            model="sabr",
            sabr={"alpha": 0.025, "beta": 0.5, "rho": -0.2, "nu": 0.35},
        )

        def _sabr_market_state(vol=0.20, rate=0.0425):
            del vol, rate
            return MarketState(
                as_of=SETTLE,
                settlement=SETTLE,
                discount=curve,
                forecast_curves={"USD-SOFR-3M": curve},
                vol_surface=FlatVol(0.20),
                model_parameters={
                    "sabr": {
                        "alpha": 0.025,
                        "beta": 0.5,
                        "rho": -0.2,
                        "nu": 0.35,
                    },
                },
            )

        failures = check_vol_sensitivity(
            lambda: _SabrCapStripPayoff(spec),
            _sabr_market_state,
        )
        assert failures == []
