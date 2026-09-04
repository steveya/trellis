"""Tests for checked fixed-strike lookback option helper surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.equity_exotics import price_equity_fixed_lookback_option_analytical
from trellis.models.vol_surface import FlatVol


@dataclass(frozen=True)
class LookbackSpec:
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    lookback_type: str = "fixed_strike"
    running_extreme: float | None = None
    n_paths: int = 80_000
    n_steps: int = 96
    seed: int | None = 42
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _market_state() -> MarketState:
    settle = date(2024, 11, 15)
    return MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )


def test_fixed_lookback_monte_carlo_tracks_analytical_call():
    from trellis.models.lookback_option import price_equity_fixed_lookback_option_monte_carlo

    spec = LookbackSpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )

    monte_carlo = price_equity_fixed_lookback_option_monte_carlo(_market_state(), spec)
    analytical = price_equity_fixed_lookback_option_analytical(_market_state(), spec)

    assert monte_carlo == pytest.approx(analytical, rel=0.06)


@pytest.mark.parametrize(
    ("option_type", "strike", "rate"),
    (("call", 101.0, 0.021), ("put", 99.0, -0.021)),
)
def test_fixed_lookback_low_volatility_large_carry_weight_tracks_monte_carlo(
    option_type,
    strike,
    rate,
):
    from trellis.models.lookback_option import (
        price_equity_fixed_lookback_option_monte_carlo_result,
    )

    market_state = MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(0.02),
    )
    spec = LookbackSpec(
        notional=1.0,
        spot=100.0,
        strike=strike,
        running_extreme=100.0,
        expiry_date=date(2025, 11, 15),
        option_type=option_type,
        n_paths=250_000,
        n_steps=96,
        seed=42,
    )

    monte_carlo = price_equity_fixed_lookback_option_monte_carlo_result(
        market_state,
        spec,
    )
    analytical = price_equity_fixed_lookback_option_analytical(
        market_state,
        spec,
    )

    assert analytical == pytest.approx(
        monte_carlo.price,
        abs=5.0 * monte_carlo.std_error,
    )


def test_fixed_lookback_monte_carlo_uses_running_extreme_for_call_intrinsic_floor():
    from trellis.models.lookback_option import price_equity_fixed_lookback_option_monte_carlo

    spec = LookbackSpec(
        notional=1.0,
        spot=100.0,
        strike=90.0,
        running_extreme=115.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
        n_paths=30_000,
    )

    price = price_equity_fixed_lookback_option_monte_carlo(_market_state(), spec)

    assert price > 115.0 - 90.0
