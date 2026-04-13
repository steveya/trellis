from __future__ import annotations

from datetime import date
from dataclasses import replace

import pytest

from trellis.core.market_state import MarketState
from trellis.core.runtime_contract import wrap_market_state_with_contract
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        spot=100.0,
        underlier_spots={"SPX": 100.0, "NDX": 95.0},
        vol_surface=FlatVol(0.20),
        local_vol_surface=lambda spot, time: 0.01,
        local_vol_surfaces={"spx_local_vol": lambda spot, time: 0.01},
        model_parameters={"correlation_matrix": ((1.0, 0.35), (0.35, 1.0))},
    )


class _BestOfSpec:
    notional = 100.0
    underliers = "SPX,NDX"
    spots = "100.0,95.0"
    strike = 100.0
    expiry_date = date(2025, 11, 15)
    correlation = "1.0,0.35;0.35,1.0"
    weights = None
    vols = None
    dividend_yields = None
    basket_style = "best_of"
    option_type = "call"
    day_count = None
    n_paths = 40_000


class _SpreadSpec:
    notional = 100.0
    underliers = "SPX,NDX"
    spots = "100.0,95.0"
    strike = 5.0
    expiry_date = date(2025, 11, 15)
    correlation = "1.0,0.35;0.35,1.0"
    weights = "1.0,-1.0"
    vols = None
    dividend_yields = None
    basket_style = "spread"
    option_type = "call"
    day_count = None
    n_paths = 40_000


def test_resolve_basket_option_inputs_parses_underliers_and_explicit_correlation():
    from trellis.models.basket_option import resolve_basket_option_inputs

    resolved = resolve_basket_option_inputs(_market_state(), _SpreadSpec())

    assert resolved.semantics.constituent_names == ("SPX", "NDX")
    assert resolved.semantics.constituent_spots == pytest.approx((100.0, 95.0))
    assert resolved.correlation_matrix[0] == pytest.approx((1.0, 0.35))
    assert resolved.correlation_matrix[1] == pytest.approx((0.35, 1.0))
    assert resolved.weights == pytest.approx((1.0, -1.0))
    assert resolved.basket_style == "spread"


def test_resolve_basket_option_inputs_accepts_runtime_contract_proxy():
    from trellis.models.basket_option import resolve_basket_option_inputs

    proxied_market_state = wrap_market_state_with_contract(
        _market_state(),
        requirements={"discount_curve", "spot", "vol_surface"},
        context="BasketOptionPayoff",
    )

    resolved = resolve_basket_option_inputs(
        proxied_market_state,
        _SpreadSpec(),
        comparison_target="kirk_spread",
    )

    assert resolved.correlation_matrix[0] == pytest.approx((1.0, 0.35))
    assert resolved.correlation_matrix[1] == pytest.approx((0.35, 1.0))


def test_basket_option_helpers_keep_best_of_analytical_and_mc_within_tolerance():
    from trellis.models.basket_option import (
        price_basket_option_analytical,
        price_basket_option_monte_carlo,
    )

    market_state = _market_state()
    analytical = price_basket_option_analytical(
        market_state,
        _BestOfSpec(),
        comparison_target="stulz_rainbow",
    )
    monte_carlo = price_basket_option_monte_carlo(
        market_state,
        _BestOfSpec(),
        comparison_target="mc_rainbow",
        n_paths=20_000,
        seed=42,
    )

    assert analytical > 0.0
    assert monte_carlo > 0.0
    assert monte_carlo == pytest.approx(analytical, rel=0.12)


def test_spread_transform_proxy_matches_stabilized_analytical_kernel():
    from trellis.models.basket_option import (
        price_basket_option_analytical,
        price_basket_option_transform_proxy,
    )

    market_state = _market_state()
    analytical = price_basket_option_analytical(
        market_state,
        _SpreadSpec(),
        comparison_target="kirk_spread",
    )
    transform = price_basket_option_transform_proxy(
        market_state,
        _SpreadSpec(),
        comparison_target="fft_spread_2d",
    )

    assert transform == pytest.approx(analytical, rel=1e-12)


def test_basket_option_helpers_prefer_explicit_vol_surface_over_local_vol_overlay():
    from trellis.models.basket_option import price_basket_option_analytical

    market_state = _market_state()
    low_vol = price_basket_option_analytical(
        replace(market_state, vol_surface=FlatVol(0.05)),
        _SpreadSpec(),
        comparison_target="kirk_spread",
    )
    high_vol = price_basket_option_analytical(
        replace(market_state, vol_surface=FlatVol(0.40)),
        _SpreadSpec(),
        comparison_target="kirk_spread",
    )

    assert high_vol > low_vol
