"""Tests for the reusable factor-state simulation substrate."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from trellis.book import Book
from trellis.core.date_utils import build_payment_timeline, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.swap import SwapPayoff, SwapSpec
from trellis.models.processes.heston import Heston
from trellis.models.vol_surface import FlatVol


def _flat_rates_market(
    settlement: date,
    *,
    discount_rate: float = 0.042,
    forward_rate: float = 0.046,
) -> MarketState:
    return MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(discount_rate, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(forward_rate, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        },
    )


def _vanilla_swap(
    *,
    settlement: date,
    notional: float,
    fixed_rate: float,
    end_date: date,
    is_payer: bool,
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
    float_frequency: Frequency = Frequency.QUARTERLY,
) -> SwapSpec:
    return SwapSpec(
        notional=notional,
        fixed_rate=fixed_rate,
        start_date=settlement,
        end_date=end_date,
        fixed_frequency=fixed_frequency,
        float_frequency=float_frequency,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
        is_payer=is_payer,
    )


def _reference_clean_value_on_reset_date(
    spec: SwapSpec,
    *,
    observation_date: date,
    discount_rate: float = 0.042,
    forward_rate: float = 0.046,
) -> float:
    """Independent deterministic reference for reset-date cube values.

    The simulation substrate values future cashflows against the original
    curve time origin, using initial-curve discount ratios from the observation
    time. This helper mirrors that mathematical contract without using the
    substrate projection or valuation model.
    """
    market = _flat_rates_market(
        spec.start_date,
        discount_rate=discount_rate,
        forward_rate=forward_rate,
    )
    observation_time = year_fraction(
        spec.start_date,
        observation_date,
        spec.float_day_count,
    )
    anchor_discount = max(float(market.discount.discount(observation_time)), 1e-12)
    forecast_curve = market.forecast_forward_curve(spec.rate_index)

    fixed_leg = 0.0
    for period in build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.fixed_frequency,
        day_count=spec.fixed_day_count,
        time_origin=spec.start_date,
    ):
        if period.end_date <= observation_date:
            continue
        discount_ratio = (
            float(market.discount.discount(float(period.t_payment))) / anchor_discount
        )
        fixed_leg += (
            float(spec.notional)
            * float(spec.fixed_rate)
            * float(period.accrual_fraction or 0.0)
            * discount_ratio
        )
        if period.start_date < observation_date < period.end_date:
            accrued = year_fraction(
                period.start_date,
                observation_date,
                spec.fixed_day_count,
            )
            fixed_leg -= float(spec.notional) * float(spec.fixed_rate) * float(accrued)

    float_leg = 0.0
    for period in build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.float_frequency,
        day_count=spec.float_day_count,
        time_origin=spec.start_date,
    ):
        if period.end_date <= observation_date or period.start_date < observation_date:
            continue
        forward = float(
            forecast_curve.forward_rate(float(period.t_start), float(period.t_end))
        )
        discount_ratio = (
            float(market.discount.discount(float(period.t_payment))) / anchor_discount
        )
        float_leg += (
            float(spec.notional)
            * forward
            * float(period.accrual_fraction or 0.0)
            * discount_ratio
        )

    sign = 1.0 if spec.is_payer else -1.0
    return sign * (float_leg - fixed_leg)


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


def test_simulation_substrate_exports_portfolio_future_value_helper():
    import trellis.models.monte_carlo.simulation_substrate as substrate

    assert "price_interest_rate_swap_portfolio_future_value_cube" in substrate.__all__


def test_swap_portfolio_future_value_cube_supports_degenerate_single_position_mapping():
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_portfolio_future_value_cube,
    )

    settlement = date(2024, 11, 15)
    market_state = _flat_rates_market(settlement)
    spec = _vanilla_swap(
        settlement=settlement,
        notional=1_000_000.0,
        fixed_rate=0.044,
        end_date=date(2027, 11, 15),
        is_payer=True,
    )

    cube = price_interest_rate_swap_portfolio_future_value_cube(
        positions={"single": spec},
        market_state=market_state,
        n_paths=72,
        n_steps=72,
        seed=43,
        mean_reversion=0.12,
        sigma=0.0,
    )

    direct_pv = SwapPayoff(spec).evaluate(market_state)

    assert cube.position_names == ("single",)
    assert cube.compute_plan["portfolio_size"] == 1
    assert cube.compute_plan["observation_grid"] == "float_boundary_dates"
    np.testing.assert_allclose(
        cube.values_for_position("single")[0],
        np.full(cube.n_paths, direct_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        cube.values_for_position("single")[-1],
        np.zeros(cube.n_paths, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )


def test_swap_portfolio_future_value_cube_invariants_cover_staggered_mixed_portfolio():
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_portfolio_future_value_cube,
    )

    settlement = date(2024, 11, 15)
    market_state = _flat_rates_market(settlement)
    payer = _vanilla_swap(
        settlement=settlement,
        notional=1_000_000.0,
        fixed_rate=0.045,
        end_date=date(2028, 11, 15),
        is_payer=True,
        float_frequency=Frequency.QUARTERLY,
    )
    receiver = _vanilla_swap(
        settlement=settlement,
        notional=750_000.0,
        fixed_rate=0.039,
        end_date=date(2026, 5, 15),
        is_payer=False,
        fixed_frequency=Frequency.ANNUAL,
        float_frequency=Frequency.MONTHLY,
    )

    cube = price_interest_rate_swap_portfolio_future_value_cube(
        positions={"payer": payer, "receiver": receiver},
        market_state=market_state,
        n_paths=96,
        n_steps=96,
        seed=31,
        mean_reversion=0.15,
        sigma=0.0,
    )

    payer_pv = SwapPayoff(payer).evaluate(market_state)
    receiver_pv = SwapPayoff(receiver).evaluate(market_state)
    payer_values = cube.values_for_position("payer")
    receiver_values = cube.values_for_position("receiver")

    assert cube.position_names == ("payer", "receiver")
    assert cube.compute_plan["observation_grid"] == "portfolio_float_boundary_union"
    assert cube.observation_dates[0] == settlement
    assert date(2024, 12, 15) in cube.observation_dates
    assert date(2025, 2, 15) in cube.observation_dates
    assert cube.observation_dates[-1] == payer.end_date

    np.testing.assert_allclose(payer_values[0], payer_pv, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(receiver_values[0], receiver_pv, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        cube.portfolio_values(),
        payer_values + receiver_values,
        atol=1e-10,
        rtol=1e-10,
    )
    assert cube.expected_positive_exposure()[0] == pytest.approx(
        max(payer_pv + receiver_pv, 0.0)
    )

    receiver_maturity_index = cube.date_index(receiver.end_date)
    payer_maturity_index = cube.date_index(payer.end_date)
    np.testing.assert_allclose(
        receiver_values[receiver_maturity_index:],
        np.zeros_like(receiver_values[receiver_maturity_index:]),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        payer_values[payer_maturity_index:],
        np.zeros_like(payer_values[payer_maturity_index:]),
        atol=1e-10,
        rtol=1e-10,
    )


def test_swap_future_value_cube_is_linear_in_contract_notional():
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_future_value_cube,
    )

    settlement = date(2024, 11, 15)
    market_state = _flat_rates_market(settlement)
    base = _vanilla_swap(
        settlement=settlement,
        notional=1_000_000.0,
        fixed_rate=0.043,
        end_date=date(2027, 11, 15),
        is_payer=True,
    )
    scaled = replace(base, notional=2_000_000.0)

    base_cube = price_interest_rate_swap_future_value_cube(
        name="base",
        spec=base,
        market_state=market_state,
        n_paths=80,
        n_steps=72,
        seed=37,
        mean_reversion=0.11,
        sigma=0.0,
    )
    scaled_cube = price_interest_rate_swap_future_value_cube(
        name="scaled",
        spec=scaled,
        market_state=market_state,
        n_paths=80,
        n_steps=72,
        seed=37,
        mean_reversion=0.11,
        sigma=0.0,
    )

    assert scaled_cube.observation_dates == base_cube.observation_dates
    np.testing.assert_allclose(
        scaled_cube.values_for_position("scaled"),
        2.0 * base_cube.values_for_position("base"),
        atol=1e-10,
        rtol=1e-10,
    )


def test_swap_future_value_cube_matches_independent_reset_date_reference():
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_future_value_cube,
    )

    settlement = date(2024, 11, 15)
    market_state = _flat_rates_market(settlement)
    spec = _vanilla_swap(
        settlement=settlement,
        notional=1_000_000.0,
        fixed_rate=0.044,
        end_date=date(2027, 11, 15),
        is_payer=True,
    )
    observation_date = date(2025, 11, 15)

    cube = price_interest_rate_swap_future_value_cube(
        name="payer",
        spec=spec,
        market_state=market_state,
        n_paths=96,
        n_steps=96,
        seed=41,
        mean_reversion=0.10,
        sigma=0.0,
    )

    expected = _reference_clean_value_on_reset_date(
        spec,
        observation_date=observation_date,
    )
    observed = cube.values_for_position("payer")[cube.date_index(observation_date)]

    np.testing.assert_allclose(
        observed,
        np.full(cube.n_paths, expected, dtype=float),
        atol=1e-8,
        rtol=1e-8,
    )
