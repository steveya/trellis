from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class _Spec:
    notional = 100.0
    spot = 100.0
    strike = 100.0
    expiry_date = date(2025, 11, 15)
    option_type = "call"


def _market_state(vol: float = 0.20, rate: float = 0.05) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=5.0),
        vol_surface=FlatVol(vol),
    )


def _black76_equity_price(spec: _Spec, *, rate: float = 0.05, vol: float = 0.20) -> float:
    maturity = 1.0
    df = float(YieldCurve.flat(rate, max_tenor=5.0).discount(maturity))
    forward = float(spec.spot) / max(df, 1e-12)
    if spec.option_type == "put":
        return float(spec.notional) * float(df) * float(
            black76_put(forward, float(spec.strike), vol, maturity)
        )
    return float(spec.notional) * float(df) * float(
        black76_call(forward, float(spec.strike), vol, maturity)
    )


def test_resolve_vanilla_equity_transform_inputs_reads_market_state_contract():
    from trellis.models.equity_option_transforms import (
        resolve_vanilla_equity_transform_inputs,
    )

    resolved = resolve_vanilla_equity_transform_inputs(
        _market_state(vol=0.30, rate=0.04),
        _Spec(),
        method="cos",
    )

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.04)
    assert resolved.sigma == pytest.approx(0.30)
    assert resolved.method == "cos"


@pytest.mark.parametrize("method", ["fft", "cos"])
def test_price_vanilla_equity_option_transform_matches_black76_for_calls(method: str):
    from trellis.models.equity_option_transforms import price_vanilla_equity_option_transform

    spec = _Spec()
    price = price_vanilla_equity_option_transform(
        _market_state(),
        spec,
        method=method,
    )

    reference = _black76_equity_price(spec)
    assert price == pytest.approx(reference, rel=0.03)


@pytest.mark.parametrize("method", ["fft", "cos"])
def test_price_vanilla_equity_option_transform_matches_black76_for_puts(method: str):
    from trellis.models.equity_option_transforms import price_vanilla_equity_option_transform

    spec = _Spec()
    spec.option_type = "put"
    price = price_vanilla_equity_option_transform(
        _market_state(),
        spec,
        method=method,
    )

    reference = _black76_equity_price(spec)
    assert price == pytest.approx(reference, rel=0.03)
