"""Tests for the reusable factor-state simulation substrate."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trellis.book import Book
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.swap import SwapPayoff, SwapSpec
from trellis.models.processes.heston import Heston
from trellis.models.vol_surface import FlatVol


def test_simulate_factor_state_observations_matches_vector_state_snapshots():
    from trellis.models.monte_carlo.event_state import event_step_indices
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.simulation_substrate import (
        simulate_factor_state_observations,
    )

    process = Heston(
        mu=0.03,
        kappa=1.8,
        theta=0.04,
        xi=0.35,
        rho=-0.55,
        v0=0.04,
    )
    initial_state = np.array([100.0, 0.04], dtype=float)
    observation_times = (0.0, 0.25, 0.75, 1.0)

    reduced_engine = MonteCarloEngine(
        process,
        n_paths=96,
        n_steps=16,
        seed=7,
        method="euler",
    )
    simulation = simulate_factor_state_observations(
        reduced_engine,
        initial_state,
        1.0,
        observation_times=observation_times,
        state_names=("spot", "variance"),
        process_family="heston",
    )

    full_engine = MonteCarloEngine(
        process,
        n_paths=96,
        n_steps=16,
        seed=7,
        method="euler",
    )
    full_paths = full_engine.simulate(initial_state, 1.0)
    expected_steps = event_step_indices(observation_times, 1.0, 16)

    assert simulation.process_family == "heston"
    assert simulation.measure == "risk_neutral"
    assert simulation.state_names == ("spot", "variance")
    assert simulation.observation_times == observation_times
    assert simulation.factor_paths.shape == (len(observation_times), 96, 2)

    for index, step in enumerate(expected_steps):
        np.testing.assert_allclose(
            simulation.factor_paths[index],
            full_paths[:, step, :],
            atol=0.0,
            rtol=0.0,
        )


def test_single_swap_future_value_cube_matches_today_pv_and_zeroes_after_maturity():
    from trellis.book import FutureValueCube
    from trellis.core.market_state import MarketState
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_future_value_cube,
    )

    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.042, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.046, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
        selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
    )
    spec = SwapSpec(
        notional=1_000_000.0,
        fixed_rate=0.045,
        start_date=settle,
        end_date=date(2027, 11, 15),
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
        is_payer=True,
    )

    cube = price_interest_rate_swap_future_value_cube(
        name="payer_swap",
        spec=spec,
        market_state=market_state,
        n_paths=512,
        n_steps=96,
        seed=19,
    )

    direct_pv = SwapPayoff(spec).evaluate(market_state)

    assert isinstance(cube, FutureValueCube)
    assert cube.position_names == ("payer_swap",)
    assert cube.value_semantics == "clean_future_value"
    assert cube.phase_semantics == "post_event"
    assert cube.measure == "risk_neutral"
    assert cube.compute_plan["engine_family"] == "simulation_substrate"
    assert cube.compute_plan["projection_family"] == "hull_white_1f_rate_projection"
    assert cube.observation_dates[0] == settle
    assert cube.observation_dates[-1] == spec.end_date
    np.testing.assert_allclose(
        cube.values_for_position("payer_swap")[0],
        np.full(cube.n_paths, direct_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("payer_swap")[-1],
        np.zeros(cube.n_paths, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    assert cube.expected_positive_exposure()[0] == pytest.approx(max(direct_pv, 0.0))


def test_swap_portfolio_future_value_cube_matches_position_pvs_on_shared_grid():
    from trellis.core.market_state import MarketState
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_portfolio_future_value_cube,
    )

    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.042, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.046, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
        selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
    )
    long_swap = SwapSpec(
        notional=1_000_000.0,
        fixed_rate=0.045,
        start_date=settle,
        end_date=date(2027, 11, 15),
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
        is_payer=True,
    )
    short_swap = SwapSpec(
        notional=2_000_000.0,
        fixed_rate=0.041,
        start_date=settle,
        end_date=date(2026, 11, 15),
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
        is_payer=False,
    )

    cube = price_interest_rate_swap_portfolio_future_value_cube(
        positions={"long_swap": long_swap, "short_swap": short_swap},
        market_state=market_state,
        n_paths=512,
        n_steps=96,
        seed=23,
    )

    long_pv = SwapPayoff(long_swap).evaluate(market_state)
    short_pv = SwapPayoff(short_swap).evaluate(market_state)

    assert cube.position_names == ("long_swap", "short_swap")
    assert cube.compute_plan["engine_family"] == "simulation_substrate"
    assert cube.compute_plan["observation_grid"] == "portfolio_float_boundary_union"
    assert cube.compute_plan["portfolio_size"] == 2
    assert cube.observation_dates[0] == settle
    assert cube.observation_dates[-1] == long_swap.end_date
    np.testing.assert_allclose(
        cube.values_for_position("long_swap")[0],
        np.full(cube.n_paths, long_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("short_swap")[0],
        np.full(cube.n_paths, short_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.portfolio_values()[0],
        np.full(cube.n_paths, long_pv + short_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("short_swap")[cube.date_index(short_swap.end_date)],
        np.zeros(cube.n_paths, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("short_swap")[-1],
        np.zeros(cube.n_paths, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("long_swap")[-1],
        np.zeros(cube.n_paths, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    assert cube.expected_positive_exposure()[0] == pytest.approx(max(long_pv + short_pv, 0.0))


def test_swap_portfolio_future_value_cube_accepts_book_inputs_and_scales_positions():
    from trellis.core.market_state import MarketState
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_portfolio_future_value_cube,
    )

    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.039, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.043, max_tenor=10.0)},
        vol_surface=FlatVol(0.18),
        selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
    )
    payer = SwapSpec(
        notional=1_000_000.0,
        fixed_rate=0.044,
        start_date=settle,
        end_date=date(2027, 11, 15),
        rate_index="USD-SOFR-3M",
        is_payer=True,
    )
    receiver = SwapSpec(
        notional=1_500_000.0,
        fixed_rate=0.040,
        start_date=settle,
        end_date=date(2026, 11, 15),
        rate_index="USD-SOFR-3M",
        is_payer=False,
    )
    book = Book(
        {
            "payer": SwapPayoff(payer),
            "receiver": SwapPayoff(receiver),
        },
        notionals={"payer": 1.0, "receiver": 0.5},
    )

    cube = price_interest_rate_swap_portfolio_future_value_cube(
        positions=book,
        market_state=market_state,
        n_paths=384,
        n_steps=80,
        seed=29,
    )

    payer_pv = SwapPayoff(payer).evaluate(market_state)
    receiver_pv = 0.5 * SwapPayoff(receiver).evaluate(market_state)

    np.testing.assert_allclose(
        cube.values_for_position("payer")[0],
        np.full(cube.n_paths, payer_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("receiver")[0],
        np.full(cube.n_paths, receiver_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    assert cube.position_provenance["receiver"]["book_notional_multiplier"] == pytest.approx(0.5)
