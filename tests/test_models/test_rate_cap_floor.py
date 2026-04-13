from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.cap import CapFloorSpec
from trellis.models.rate_cap_floor import (
    price_rate_cap_floor_strip_analytical,
    price_rate_cap_floor_strip_monte_carlo,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state(rate=0.05, vol=0.20):
    curve = YieldCurve.flat(rate)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        vol_surface=FlatVol(vol),
    )


def _cap_spec(**overrides):
    defaults = dict(
        notional=1_000_000,
        strike=0.04,
        start_date=date(2025, 2, 15),
        end_date=date(2030, 2, 15),
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        rate_index=None,
    )
    defaults.update(overrides)
    return CapFloorSpec(**defaults)


def test_rate_cap_floor_strip_monte_carlo_tracks_analytical_reference():
    market_state = _market_state(rate=0.05, vol=0.20)
    spec = _cap_spec()

    analytical = price_rate_cap_floor_strip_analytical(
        market_state,
        spec,
        instrument_class="cap",
    )
    monte_carlo = price_rate_cap_floor_strip_monte_carlo(
        market_state,
        spec,
        instrument_class="cap",
        n_paths=20000,
        seed=7,
    )

    assert analytical > 0.0
    assert monte_carlo > 0.0
    assert monte_carlo == pytest.approx(analytical, rel=0.20)


def test_rate_cap_floor_strip_helpers_accept_keyword_contract_fields():
    market_state = _market_state(rate=0.05, vol=0.20)
    spec = _cap_spec(rate_index="USD-SOFR-3M")

    analytical_from_spec = price_rate_cap_floor_strip_analytical(
        market_state,
        spec,
        instrument_class="cap",
    )
    monte_carlo_from_spec = price_rate_cap_floor_strip_monte_carlo(
        market_state,
        spec,
        instrument_class="cap",
        n_paths=20000,
        seed=7,
    )

    analytical_from_kwargs = price_rate_cap_floor_strip_analytical(
        market_state=market_state,
        instrument_class="cap",
        notional=spec.notional,
        strike=spec.strike,
        start_date=spec.start_date,
        end_date=spec.end_date,
        frequency=spec.frequency,
        day_count=spec.day_count,
        rate_index=spec.rate_index,
    )
    monte_carlo_from_kwargs = price_rate_cap_floor_strip_monte_carlo(
        market_state=market_state,
        instrument_class="cap",
        notional=spec.notional,
        strike=spec.strike,
        start_date=spec.start_date,
        end_date=spec.end_date,
        frequency=spec.frequency,
        day_count=spec.day_count,
        rate_index=spec.rate_index,
        n_paths=20000,
        seed=7,
    )

    assert analytical_from_kwargs == pytest.approx(analytical_from_spec)
    assert monte_carlo_from_kwargs == pytest.approx(monte_carlo_from_spec)
