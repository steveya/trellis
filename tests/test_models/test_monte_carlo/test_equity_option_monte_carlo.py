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
    n_paths = 12_000
    n_steps = 64
    seed = 7


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


def test_resolve_vanilla_equity_monte_carlo_inputs_reads_market_state_contract():
    from trellis.models.equity_option_monte_carlo import resolve_vanilla_equity_monte_carlo_inputs

    resolved = resolve_vanilla_equity_monte_carlo_inputs(
        _market_state(vol=0.30, rate=0.04),
        _Spec(),
    )

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.04)
    assert resolved.sigma == pytest.approx(0.30)
    assert resolved.scheme == "exact"
    assert resolved.variance_reduction == "none"


@pytest.mark.parametrize("scheme", ["euler", "milstein", "exact", "log_euler"])
def test_price_vanilla_equity_option_monte_carlo_matches_black76_across_schemes(scheme: str):
    from trellis.models.equity_option_monte_carlo import (
        price_vanilla_equity_option_monte_carlo,
    )

    spec = _Spec()
    price = price_vanilla_equity_option_monte_carlo(
        _market_state(),
        spec,
        scheme=scheme,
        n_paths=20_000,
        n_steps=128,
        seed=11,
    )

    reference = _black76_equity_price(spec)
    assert price == pytest.approx(reference, rel=0.12)


def test_antithetic_and_control_variate_reduce_standard_error():
    from trellis.models.equity_option_monte_carlo import (
        price_vanilla_equity_option_monte_carlo_result,
    )

    market_state = _market_state()
    spec = _Spec()
    plain = price_vanilla_equity_option_monte_carlo_result(
        market_state,
        spec,
        scheme="exact",
        variance_reduction="none",
        n_paths=16_000,
        n_steps=96,
        seed=21,
    )
    antithetic = price_vanilla_equity_option_monte_carlo_result(
        market_state,
        spec,
        scheme="exact",
        variance_reduction="antithetic",
        n_paths=16_000,
        n_steps=96,
        seed=21,
    )
    control_variate = price_vanilla_equity_option_monte_carlo_result(
        market_state,
        spec,
        scheme="exact",
        variance_reduction="control_variate",
        n_paths=16_000,
        n_steps=96,
        seed=21,
    )

    assert antithetic.std_error < plain.std_error
    assert control_variate.std_error < plain.std_error
