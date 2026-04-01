"""Tests for the reusable equity-option tree helper surface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log, sqrt

import pytest
from scipy.stats import norm

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.equity_option_tree import (
    build_vanilla_equity_lattice,
    price_vanilla_equity_option_on_lattice,
    price_vanilla_equity_option_tree,
    resolve_vanilla_equity_tree_inputs,
)
from trellis.models.vol_surface import FlatVol


def _bs_call(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2)


@dataclass(frozen=True)
class VanillaEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "put"
    exercise_style: str = "american"


def test_price_vanilla_equity_option_on_lattice_matches_bs_for_european_call():
    lattice = build_vanilla_equity_lattice(
        spot=100.0,
        rate=0.05,
        sigma=0.20,
        maturity=1.0,
        n_steps=200,
        model="crr",
    )
    price = price_vanilla_equity_option_on_lattice(
        lattice,
        strike=100.0,
        option_type="call",
        exercise_style="european",
    )
    assert price == pytest.approx(_bs_call(100.0, 100.0, 0.05, 0.20, 1.0), rel=0.02)


def test_price_vanilla_equity_option_tree_american_put_geq_european_value():
    settle = date(2024, 11, 15)
    expiry = date(2025, 11, 15)
    spec = VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=expiry,
        option_type="put",
        exercise_style="american",
    )
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )

    american = price_vanilla_equity_option_tree(market_state, spec, model="crr", n_steps=400)
    european = price_vanilla_equity_option_tree(
        market_state,
        VanillaEquitySpec(
            spot=100.0,
            strike=100.0,
            expiry_date=expiry,
            option_type="put",
            exercise_style="european",
        ),
        model="crr",
        n_steps=400,
    )

    assert american >= european - 1e-6


def test_price_vanilla_equity_option_tree_supports_jarrow_rudd():
    settle = date(2024, 11, 15)
    expiry = date(2025, 11, 15)
    spec = VanillaEquitySpec(
        spot=100.0,
        strike=105.0,
        expiry_date=expiry,
        option_type="call",
        exercise_style="european",
    )
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )

    price = price_vanilla_equity_option_tree(market_state, spec, model="jarrow_rudd", n_steps=300)

    assert price > 0.0


def test_resolve_vanilla_equity_tree_inputs_uses_market_state_contract():
    settle = date(2024, 11, 15)
    expiry = date(2025, 11, 15)
    spec = VanillaEquitySpec(
        spot=100.0,
        strike=105.0,
        expiry_date=expiry,
        option_type="call",
        exercise_style="bermudan",
    )
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )

    resolved = resolve_vanilla_equity_tree_inputs(market_state, spec)

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(105.0)
    assert resolved.maturity > 0.0
    assert resolved.rate == pytest.approx(0.05, rel=1e-3)
    assert resolved.sigma == pytest.approx(0.20, rel=1e-3)
    assert resolved.option_type == "call"
    assert resolved.exercise_style == "bermudan"
