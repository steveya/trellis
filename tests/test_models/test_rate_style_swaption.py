from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_style_swaption import (
    price_bermudan_swaption_black76_lower_bound,
    price_swaption_black76,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class _EuropeanSpec:
    notional = 100.0
    strike = 0.05
    expiry_date = date(2029, 11, 15)
    swap_start = expiry_date
    swap_end = date(2034, 11, 15)
    from trellis.core.types import DayCountConvention, Frequency
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.ACT_360
    rate_index = None
    is_payer = True


class _BermudanSpec:
    notional = 100.0
    strike = 0.05
    exercise_dates = "2025-11-15,2026-11-15,2027-11-15,2028-11-15,2029-11-15"
    swap_end = date(2030, 11, 15)
    from trellis.core.types import DayCountConvention, Frequency
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.ACT_360
    rate_index = None
    is_payer = True


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        vol_surface=FlatVol(vol),
    )


def test_price_swaption_black76_is_positive():
    price = price_swaption_black76(_market_state(), _EuropeanSpec())
    assert price > 0.0


def test_bermudan_lower_bound_uses_final_exercise_date():
    market_state = _market_state()
    lower_bound = price_bermudan_swaption_black76_lower_bound(market_state, _BermudanSpec())
    final_exercise = price_swaption_black76(
        market_state,
        _BermudanSpec(),
        expiry_date=date(2029, 11, 15),
    )

    assert lower_bound == final_exercise


def test_bermudan_lower_bound_increases_with_vol():
    low = price_bermudan_swaption_black76_lower_bound(_market_state(vol=0.10), _BermudanSpec())
    high = price_bermudan_swaption_black76_lower_bound(_market_state(vol=0.30), _BermudanSpec())
    assert high > low
