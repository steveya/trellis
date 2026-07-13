from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state(*, correlation: float = 0.35) -> MarketState:
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
        model_parameters={"quanto_correlation": correlation},
    )


def test_quanto_compatibility_wrappers_match_primitive_composed_adapters():
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
        rel=0.04,
    )


def test_primitive_composed_quanto_adapters_use_intrinsic_value_at_expiry():
    from trellis.instruments._agent.quantooptionanalytical import (
        QuantoOptionAnalyticalPayoff,
        QuantoOptionSpec as AnalyticalSpec,
    )
    from trellis.instruments._agent.quantooptionmontecarlo import (
        QuantoOptionMonteCarloPayoff,
        QuantoOptionSpec as MonteCarloSpec,
    )

    terms = {
        "notional": 10_000.0,
        "strike": 95.0,
        "expiry_date": SETTLE,
        "fx_pair": "EURUSD",
    }
    market_state = _market_state()
    expected = 10_000.0 * (100.0 - 95.0)

    analytical = QuantoOptionAnalyticalPayoff(AnalyticalSpec(**terms))
    monte_carlo = QuantoOptionMonteCarloPayoff(MonteCarloSpec(**terms))

    assert analytical.evaluate(market_state) == pytest.approx(expected)
    assert monte_carlo.evaluate(market_state) == pytest.approx(expected)


def test_primitive_composed_quanto_lanes_agree_on_correlation_direction():
    from trellis.instruments._agent.quantooptionanalytical import (
        QuantoOptionAnalyticalPayoff,
        QuantoOptionSpec as AnalyticalSpec,
    )
    from trellis.instruments._agent.quantooptionmontecarlo import (
        QuantoOptionMonteCarloPayoff,
        QuantoOptionSpec as MonteCarloSpec,
    )

    terms = {
        "notional": 100_000.0,
        "strike": 100.0,
        "expiry_date": date(2025, 11, 15),
        "fx_pair": "EURUSD",
    }
    analytical = QuantoOptionAnalyticalPayoff(AnalyticalSpec(**terms))
    monte_carlo = QuantoOptionMonteCarloPayoff(
        MonteCarloSpec(**terms, n_paths=32_768, n_steps=64, seed=19)
    )
    negative_correlation = _market_state(correlation=-0.5)
    positive_correlation = _market_state(correlation=0.5)

    analytical_negative = analytical.evaluate(negative_correlation)
    analytical_positive = analytical.evaluate(positive_correlation)
    monte_carlo_negative = monte_carlo.evaluate(negative_correlation)
    monte_carlo_positive = monte_carlo.evaluate(positive_correlation)

    assert analytical_negative > analytical_positive
    assert monte_carlo_negative > monte_carlo_positive
    assert monte_carlo_negative == pytest.approx(analytical_negative, rel=0.04)
    assert monte_carlo_positive == pytest.approx(analytical_positive, rel=0.04)
