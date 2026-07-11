from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.equity_exotics import price_equity_variance_swap_analytical
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state(vol: float = 0.20, rate: float = 0.03) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=5.0),
        vol_surface=FlatVol(vol),
    )


def _spec(**overrides):
    defaults = {
        "notional": 10_000.0,
        "spot": 100.0,
        "strike_variance": 0.035,
        "expiry_date": date(2025, 11, 15),
        "realized_variance": 0.0,
        "day_count": DayCountConvention.ACT_365,
        "n_paths": 50_000,
        "n_steps": 96,
        "seed": 19,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_equity_variance_swap_monte_carlo_matches_flat_vol_log_contract():
    from trellis.models.variance_swap import (
        price_equity_variance_swap_monte_carlo_result,
    )

    market_state = _market_state(vol=0.20, rate=0.03)
    spec = _spec()

    result = price_equity_variance_swap_monte_carlo_result(market_state, spec)
    analytical = price_equity_variance_swap_analytical(market_state, spec)

    assert result.fair_strike_variance == pytest.approx(0.20**2, rel=0.04)
    assert result.price == pytest.approx(analytical, abs=3.0 * result.standard_error)


def test_equity_variance_swap_monte_carlo_scalar_returns_price():
    from trellis.models.variance_swap import price_equity_variance_swap_monte_carlo

    market_state = _market_state(vol=0.18, rate=0.02)
    spec = _spec(strike_variance=0.03, n_paths=20_000, seed=7)

    price = price_equity_variance_swap_monte_carlo(market_state, spec)

    assert isinstance(price, float)
    assert price > 0.0
