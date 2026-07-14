from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pytest

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


def _market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        spot=100.0,
        underlier_spots={"SPX": 100.0},
        vol_surface=FlatVol(0.20),
    )


@dataclass(frozen=True)
class _ArithmeticAsianSpec:
    notional: float
    underlier: str
    strike: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str
    day_count: DayCountConvention
    dividend_yield: float = 0.0
    n_paths: int = 120_000
    seed: int | None = 42


def test_arithmetic_asian_analytical_tracks_bounded_monte_carlo_for_monthly_call():
    from trellis.models.asian_option import (
        price_arithmetic_asian_option_analytical,
        price_arithmetic_asian_option_monte_carlo,
    )

    spec = _ArithmeticAsianSpec(
        notional=100.0,
        underlier="SPX",
        strike=102.0,
        expiry_date=date(2025, 12, 31),
        observation_dates=(
            date(2025, 1, 31),
            date(2025, 2, 28),
            date(2025, 3, 31),
            date(2025, 4, 30),
            date(2025, 5, 31),
            date(2025, 6, 30),
            date(2025, 7, 31),
            date(2025, 8, 31),
            date(2025, 9, 30),
            date(2025, 10, 31),
            date(2025, 11, 30),
            date(2025, 12, 31),
        ),
        option_type="call",
        day_count=DayCountConvention.ACT_365,
    )

    analytical = price_arithmetic_asian_option_analytical(_market_state(), spec)
    monte_carlo = price_arithmetic_asian_option_monte_carlo(_market_state(), spec)

    assert analytical > 0.0
    assert monte_carlo > 0.0
    assert analytical == pytest.approx(monte_carlo, rel=0.08)


def test_arithmetic_asian_analytical_tracks_bounded_monte_carlo_for_weekly_put():
    from trellis.models.asian_option import (
        price_arithmetic_asian_option_analytical,
        price_arithmetic_asian_option_monte_carlo,
    )

    spec = _ArithmeticAsianSpec(
        notional=100.0,
        underlier="SPX",
        strike=104.0,
        expiry_date=date(2025, 1, 31),
        observation_dates=(
            date(2025, 1, 3),
            date(2025, 1, 10),
            date(2025, 1, 17),
            date(2025, 1, 24),
            date(2025, 1, 31),
        ),
        option_type="put",
        day_count=DayCountConvention.ACT_365,
        n_paths=150_000,
    )

    analytical = price_arithmetic_asian_option_analytical(_market_state(), spec)
    monte_carlo = price_arithmetic_asian_option_monte_carlo(_market_state(), spec)

    assert analytical >= 0.0
    assert monte_carlo >= 0.0
    assert analytical == pytest.approx(monte_carlo, rel=0.12, abs=0.5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_arithmetic_asian_analytical_handles_nonpositive_strike(option_type):
    from trellis.models.asian_option import (
        price_arithmetic_asian_option_analytical_result,
    )

    spec = _ArithmeticAsianSpec(
        notional=100.0,
        underlier="SPX",
        strike=-5.0,
        expiry_date=date(2025, 12, 31),
        observation_dates=(
            date(2025, 3, 31),
            date(2025, 6, 30),
            date(2025, 9, 30),
            date(2025, 12, 31),
        ),
        option_type=option_type,
        day_count=DayCountConvention.ACT_365,
    )

    result = price_arithmetic_asian_option_analytical_result(_market_state(), spec)
    maturity = year_fraction(
        date(2025, 1, 1),
        spec.expiry_date,
        spec.day_count,
    )
    expected = (
        0.0
        if option_type == "put"
        else spec.notional
        * math.exp(-0.03 * maturity)
        * (result.matched_mean - spec.strike)
    )

    assert result.price == pytest.approx(expected)
