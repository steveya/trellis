from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.core.differentiable import gradient
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_style_swaption import (
    ResolvedSwaptionBlack76Inputs,
    price_bermudan_swaption_black76_lower_bound,
    price_swaption_black76_raw,
    price_swaption_black76,
    resolve_swaption_black76_inputs,
)
from trellis.models.black import black76_put
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
    exercise_dates = (
        date(2025, 11, 15),
        date(2026, 11, 15),
        date(2027, 11, 15),
        date(2028, 11, 15),
        date(2029, 11, 15),
    )
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


def _finite_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def test_price_swaption_black76_is_positive():
    price = price_swaption_black76(_market_state(), _EuropeanSpec())
    assert price > 0.0


def test_price_swaption_black76_raw_matches_public_wrapper():
    market_state = _market_state()
    resolved = resolve_swaption_black76_inputs(market_state, _EuropeanSpec())

    assert price_swaption_black76_raw(resolved) == pytest.approx(
        price_swaption_black76(market_state, _EuropeanSpec()),
        abs=1e-12,
    )


def test_price_swaption_black76_raw_receiver_branch_matches_black76_put():
    resolved = resolve_swaption_black76_inputs(_market_state(), _EuropeanSpec())
    receiver = replace(resolved, is_payer=False)

    expected = (
        receiver.notional
        * receiver.annuity
        * black76_put(
            receiver.forward_swap_rate,
            receiver.strike,
            receiver.vol,
            receiver.expiry_years,
        )
    )

    assert price_swaption_black76_raw(receiver) == pytest.approx(expected, abs=1e-12)


def test_price_swaption_black76_raw_vega_matches_finite_difference():
    resolved = resolve_swaption_black76_inputs(_market_state(), _EuropeanSpec())

    autodiff_vega = gradient(
        lambda vol: price_swaption_black76_raw(replace(resolved, vol=vol))
    )(resolved.vol)
    fd_vega = _finite_difference(
        lambda vol: price_swaption_black76_raw(replace(resolved, vol=vol)),
        resolved.vol,
    )

    assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)


def test_price_swaption_black76_raw_returns_zero_when_payment_count_is_zero():
    resolved = ResolvedSwaptionBlack76Inputs(
        expiry_date=_EuropeanSpec.expiry_date,
        expiry_years=5.0,
        annuity=4.2,
        forward_swap_rate=0.051,
        strike=0.05,
        vol=0.20,
        notional=100.0,
        is_payer=True,
        payment_count=0,
    )

    assert price_swaption_black76_raw(resolved) == 0.0


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
