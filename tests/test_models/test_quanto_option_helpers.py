from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    domestic = YieldCurve.flat(0.05)
    foreign = YieldCurve.flat(0.03)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=domestic,
        forecast_curves={"EUR-DISC": foreign},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"quanto_correlation": 0.35},
    )


def test_quanto_option_helpers_match_adapter_prices():
    from trellis.instruments._agent.quantooptionanalytical import (
        QuantoOptionAnalyticalPayoff,
        QuantoOptionSpec as AnalyticalSpec,
    )
    from trellis.instruments._agent.quantooptionmontecarlo import (
        QuantoOptionMonteCarloPayoff,
        QuantoOptionSpec as MonteCarloSpec,
    )
    from trellis.models.quanto_option import (
        price_quanto_option_analytical_from_market_state,
        price_quanto_option_monte_carlo_from_market_state,
    )

    analytical_spec = AnalyticalSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
    )
    mc_spec = MonteCarloSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        n_paths=12_000,
        n_steps=64,
        seed=7,
    )
    market_state = _market_state()

    assert price_quanto_option_analytical_from_market_state(market_state, analytical_spec) == pytest.approx(
        QuantoOptionAnalyticalPayoff(analytical_spec).evaluate(market_state),
        rel=1e-12,
    )
    assert price_quanto_option_monte_carlo_from_market_state(market_state, mc_spec) == pytest.approx(
        QuantoOptionMonteCarloPayoff(mc_spec).evaluate(market_state),
        rel=1e-12,
    )
