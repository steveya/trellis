from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.bermudanswaption import (
    BermudanSwaptionPayoff,
    BermudanSwaptionSpec,
)
from trellis.models.rate_style_swaption import (
    price_bermudan_swaption_black76_lower_bound,
)
from trellis.models.vol_surface import FlatVol


def _market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
    )


def test_admitted_bermudan_lower_bound_matches_retained_reference():
    market_state = _market_state()
    spec = BermudanSwaptionSpec(
        notional=1_000_000.0,
        strike=0.04,
        exercise_dates=(
            date(2027, 1, 1),
            date(2025, 1, 1),
            date(2026, 1, 1),
            date(2027, 1, 1),
            date(2031, 1, 1),
        ),
        swap_end=date(2030, 1, 1),
    )

    actual = BermudanSwaptionPayoff(spec).evaluate(market_state)
    expected = price_bermudan_swaption_black76_lower_bound(market_state, spec)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_admitted_bermudan_lower_bound_returns_zero_without_valid_exercise_date():
    market_state = _market_state()
    spec = BermudanSwaptionSpec(
        notional=1_000_000.0,
        strike=0.04,
        exercise_dates=(date(2025, 1, 1), date(2030, 1, 1)),
        swap_end=date(2030, 1, 1),
    )

    assert BermudanSwaptionPayoff(spec).evaluate(market_state) == 0.0
