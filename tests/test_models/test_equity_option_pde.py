"""Tests for the reusable vanilla-equity PDE helper surface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log, sqrt

import pytest
from scipy.stats import norm

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.black import black76_call
from trellis.models.equity_option_pde import (
    price_vanilla_equity_option_pde,
    resolve_vanilla_equity_pde_inputs,
    solve_vanilla_equity_option_pde_surface,
)
from trellis.models.vol_surface import FlatVol


def _bs_call(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2)


@dataclass(frozen=True)
class VanillaEquitySpec:
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"


def _market_state() -> MarketState:
    settle = date(2024, 11, 15)
    return MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )


def test_resolve_vanilla_equity_pde_inputs_uses_market_state_contract():
    settle = date(2024, 11, 15)
    spec = VanillaEquitySpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )

    resolved = resolve_vanilla_equity_pde_inputs(_market_state(), spec, theta=0.5)

    assert resolved.notional == pytest.approx(1.0)
    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity > 0.0
    assert resolved.rate == pytest.approx(0.05, rel=1e-3)
    assert resolved.sigma == pytest.approx(0.20, rel=1e-3)
    assert resolved.theta == pytest.approx(0.5)
    assert resolved.s_max >= 400.0
    assert resolved.n_x >= 5
    assert resolved.n_t >= 1
    assert settle == _market_state().settlement


def test_price_vanilla_equity_option_pde_matches_black_scholes_for_cn_call():
    spec = VanillaEquitySpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )

    price = price_vanilla_equity_option_pde(
        _market_state(),
        spec,
        theta=0.5,
        n_x=401,
        n_t=401,
    )

    assert price == pytest.approx(_bs_call(100.0, 100.0, 0.05, 0.20, 1.0), rel=0.02)


def test_theta_half_is_more_accurate_than_theta_one_for_same_grid():
    spec = VanillaEquitySpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )
    reference = _bs_call(100.0, 100.0, 0.05, 0.20, 1.0)

    cn_price = price_vanilla_equity_option_pde(
        _market_state(),
        spec,
        theta=0.5,
        n_x=301,
        n_t=301,
    )
    implicit_price = price_vanilla_equity_option_pde(
        _market_state(),
        spec,
        theta=1.0,
        n_x=301,
        n_t=301,
    )

    assert abs(cn_price - reference) <= abs(implicit_price - reference) + 1e-6


def test_price_vanilla_equity_option_pde_agrees_with_discounted_black76_call():
    spec = VanillaEquitySpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )

    price = price_vanilla_equity_option_pde(
        _market_state(),
        spec,
        theta=0.5,
        n_x=401,
        n_t=401,
    )

    maturity = 1.0
    rate = 0.05
    sigma = 0.20
    discount = exp(-rate * maturity)
    forward = spec.spot / discount
    analytical = discount * black76_call(forward, spec.strike, sigma, maturity)

    assert price == pytest.approx(analytical, rel=0.02)


def test_solve_vanilla_equity_option_pde_surface_returns_grid_aligned_values():
    spec = VanillaEquitySpec(
        notional=1.0,
        spot=100.0,
        strike=105.0,
        expiry_date=date(2025, 11, 15),
        option_type="put",
    )

    resolved, grid, values = solve_vanilla_equity_option_pde_surface(
        _market_state(),
        spec,
        theta=1.0,
        n_x=101,
        n_t=101,
    )

    assert resolved.option_type == "put"
    assert len(grid.x) == len(values)
    assert float(values[0]) >= 0.0


def test_price_vanilla_equity_option_pde_returns_intrinsic_after_expiry_without_market_data():
    expired_spec = VanillaEquitySpec(
        notional=2.0,
        spot=90.0,
        strike=100.0,
        expiry_date=date(2024, 11, 14),
        option_type="put",
    )
    expired_market_state = MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
    )

    price = price_vanilla_equity_option_pde(expired_market_state, expired_spec)

    assert price == pytest.approx(20.0)
