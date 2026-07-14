from __future__ import annotations

import numpy as raw_np
import pytest


def test_path_reducer_finalizer_hides_internal_accumulator_for_streaming_and_replay():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.path_state import (
        MonteCarloPathRequirement,
        PathReducer,
    )
    from trellis.models.processes.gbm import GBM

    def initialize(initial, _n_steps):
        return raw_np.stack((initial, raw_np.ones_like(initial)), axis=1)

    def update(accumulator, values, _step):
        return raw_np.stack(
            (accumulator[:, 0] + values, accumulator[:, 1] + 1.0),
            axis=1,
        )

    reducer = PathReducer(
        name="path_mean",
        init_fn=initialize,
        update_fn=update,
        finalize_fn=lambda accumulator: accumulator[:, 0] / accumulator[:, 1],
    )
    reduced_requirement = MonteCarloPathRequirement(reducers=(reducer,))
    replay_requirement = MonteCarloPathRequirement(
        full_path=True,
        reducers=(reducer,),
    )
    process = GBM(mu=0.10, sigma=0.0)

    reduced_state = MonteCarloEngine(
        process,
        n_paths=8,
        n_steps=4,
        seed=7,
        method="exact",
    ).simulate_state(100.0, 1.0, reduced_requirement)
    replay_state = MonteCarloEngine(
        process,
        n_paths=8,
        n_steps=4,
        seed=7,
        method="exact",
    ).simulate_state(100.0, 1.0, replay_requirement)

    expected = raw_np.mean(replay_state.full_paths, axis=1)
    raw_np.testing.assert_allclose(
        reduced_state.reduced_value("path_mean"),
        expected,
    )
    raw_np.testing.assert_allclose(
        replay_state.reduced_value("path_mean"),
        expected,
    )
    assert reduced_state.reduced_value("path_mean").shape == (8,)
    assert reduced_state.full_paths is None

    invalid_reducer = PathReducer(
        name="invalid_finalizer",
        init_fn=lambda initial, _n_steps: initial,
        update_fn=lambda accumulator, _values, _step: accumulator,
        finalize_fn=lambda accumulator: raw_np.mean(accumulator),
    )
    with pytest.raises(ValueError, match="preserve.*path axis"):
        invalid_reducer.finalize(raw_np.ones(8))


def test_running_extremum_contract_and_full_path_statistic_are_explicit():
    from trellis.models.monte_carlo.path_statistics import (
        RunningExtremumContract,
        discrete_path_extremum,
    )

    paths = raw_np.array(
        [
            [100.0, 90.0, 110.0, 80.0, 120.0],
            [100.0, 105.0, 95.0, 100.0, 90.0],
        ]
    )
    maximum = RunningExtremumContract(
        n_steps=4,
        observation_steps=(0, 2, 4),
        direction="maximum",
    )
    minimum = RunningExtremumContract(
        n_steps=4,
        observation_steps=(0, 2, 4),
        direction="minimum",
    )
    prior_minimum = RunningExtremumContract(
        n_steps=4,
        observation_steps=(2, 4),
        direction="minimum",
        initial_extremum=85.0,
    )

    raw_np.testing.assert_allclose(
        discrete_path_extremum(paths, maximum),
        raw_np.array([120.0, 100.0]),
    )
    raw_np.testing.assert_allclose(
        discrete_path_extremum(paths, minimum),
        raw_np.array([100.0, 90.0]),
    )
    raw_np.testing.assert_allclose(
        discrete_path_extremum(paths, prior_minimum),
        raw_np.array([85.0, 85.0]),
    )


def test_squared_log_return_contract_defines_baseline_and_annualization():
    from trellis.models.monte_carlo.path_statistics import (
        SquaredLogReturnContract,
        annualized_squared_log_return_sum,
    )

    paths = raw_np.array(
        [
            [100.0, 90.0, 110.0, 80.0, 120.0],
            [100.0, 105.0, 95.0, 100.0, 90.0],
        ]
    )
    contract = SquaredLogReturnContract(
        n_steps=4,
        observation_steps=(0, 2, 4),
        annualization_factor=2.0,
    )
    delayed_baseline = SquaredLogReturnContract(
        n_steps=4,
        observation_steps=(1, 3),
        annualization_factor=1.0,
    )

    expected = 2.0 * raw_np.array(
        [
            raw_np.log(110.0 / 100.0) ** 2 + raw_np.log(120.0 / 110.0) ** 2,
            raw_np.log(95.0 / 100.0) ** 2 + raw_np.log(90.0 / 95.0) ** 2,
        ]
    )
    raw_np.testing.assert_allclose(
        annualized_squared_log_return_sum(paths, contract),
        expected,
    )
    raw_np.testing.assert_allclose(
        annualized_squared_log_return_sum(paths, delayed_baseline),
        raw_np.array(
            [raw_np.log(80.0 / 90.0) ** 2, raw_np.log(100.0 / 105.0) ** 2]
        ),
    )


def test_path_statistic_reducers_match_full_paths_with_bounded_storage():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement
    from trellis.models.monte_carlo.path_statistics import (
        RunningExtremumContract,
        SquaredLogReturnContract,
        annualized_squared_log_return_sum,
        build_running_extremum_reducer,
        build_squared_log_return_reducer,
        discrete_path_extremum,
    )
    from trellis.models.processes.gbm import GBM

    extrema_contract = RunningExtremumContract(
        n_steps=6,
        observation_steps=(0, 2, 4, 6),
        direction="maximum",
    )
    return_contract = SquaredLogReturnContract(
        n_steps=6,
        observation_steps=(0, 2, 4, 6),
        annualization_factor=1.0,
    )
    requirement = MonteCarloPathRequirement(
        reducers=(
            build_running_extremum_reducer(
                extrema_contract,
                name="scheduled_maximum",
            ),
            build_squared_log_return_reducer(
                return_contract,
                name="realized_variance",
            ),
        )
    )
    process = GBM(mu=0.04, sigma=0.20)
    reduced_state = MonteCarloEngine(
        process,
        n_paths=128,
        n_steps=6,
        seed=19,
        method="exact",
    ).simulate_state(100.0, 1.0, requirement)
    paths = MonteCarloEngine(
        process,
        n_paths=128,
        n_steps=6,
        seed=19,
        method="exact",
    ).simulate(100.0, 1.0)

    raw_np.testing.assert_allclose(
        reduced_state.reduced_value("scheduled_maximum"),
        discrete_path_extremum(paths, extrema_contract),
        rtol=0.0,
        atol=0.0,
    )
    raw_np.testing.assert_allclose(
        reduced_state.reduced_value("realized_variance"),
        annualized_squared_log_return_sum(paths, return_contract),
        rtol=1e-14,
        atol=1e-14,
    )
    assert reduced_state.full_paths is None
    assert reduced_state.reduced_value("scheduled_maximum").shape == (128,)
    assert reduced_state.reduced_value("realized_variance").shape == (128,)


def test_path_statistics_preserve_smooth_autograd_operations():
    from trellis.core.differentiable import get_numpy, gradient
    from trellis.models.monte_carlo.path_statistics import (
        RunningExtremumContract,
        SquaredLogReturnContract,
        annualized_squared_log_return_sum,
        discrete_path_extremum,
    )

    np = get_numpy()
    maximum = RunningExtremumContract(
        n_steps=1,
        observation_steps=(0, 1),
        direction="maximum",
    )
    realized = SquaredLogReturnContract(
        n_steps=1,
        observation_steps=(0, 1),
        annualization_factor=2.0,
    )

    maximum_sensitivity = gradient(
        lambda terminal: discrete_path_extremum(
            np.array([[100.0, terminal]]),
            maximum,
        )[0]
    )(110.0)
    variance_sensitivity = gradient(
        lambda terminal: annualized_squared_log_return_sum(
            np.array([[100.0, terminal]]),
            realized,
        )[0]
    )(110.0)

    assert maximum_sensitivity == pytest.approx(1.0)
    assert variance_sensitivity == pytest.approx(
        4.0 * raw_np.log(110.0 / 100.0) / 110.0
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=0,
                observation_steps=(0,),
            ),
            "n_steps must be a positive integer",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=4.5,
                observation_steps=(0, 4),
            ),
            "n_steps must be a positive integer",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=4,
                observation_steps=(),
            ),
            "at least one",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=4,
                observation_steps=(0, 2, 2),
            ),
            "strictly increasing",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=4,
                observation_steps=(0, 5),
            ),
            "cannot exceed n_steps",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=4,
                observation_steps=(0, 4),
                direction="up",
            ),
            "direction",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["RunningExtremumContract"],
            ).RunningExtremumContract(
                n_steps=4,
                observation_steps=(0, 4),
                initial_extremum=float("inf"),
            ),
            "initial_extremum must be finite and positive",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["SquaredLogReturnContract"],
            ).SquaredLogReturnContract(
                n_steps=4,
                observation_steps=(0,),
            ),
            "at least two",
        ),
        (
            lambda: __import__(
                "trellis.models.monte_carlo.path_statistics",
                fromlist=["SquaredLogReturnContract"],
            ).SquaredLogReturnContract(
                n_steps=4,
                observation_steps=(0, 4),
                annualization_factor=0.0,
            ),
            "annualization_factor must be finite and positive",
        ),
    ],
)
def test_path_statistic_contracts_fail_closed(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_path_statistics_reject_grid_shape_and_level_contract_violations():
    from trellis.models.monte_carlo.path_statistics import (
        RunningExtremumContract,
        SquaredLogReturnContract,
        annualized_squared_log_return_sum,
        build_running_extremum_reducer,
        build_squared_log_return_reducer,
        discrete_path_extremum,
    )

    extrema = RunningExtremumContract(
        n_steps=4,
        observation_steps=(0, 2, 4),
    )
    squared_returns = SquaredLogReturnContract(
        n_steps=4,
        observation_steps=(0, 2, 4),
    )

    with pytest.raises(ValueError, match="execution grid has 5 steps; expected 4"):
        discrete_path_extremum(raw_np.ones((2, 6)) * 100.0, extrema)
    with pytest.raises(ValueError, match="scalar state per path"):
        annualized_squared_log_return_sum(
            raw_np.ones((2, 5, 2)) * 100.0,
            squared_returns,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        discrete_path_extremum(
            raw_np.array([[100.0, 90.0, 0.0, 95.0, 105.0]]),
            extrema,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        annualized_squared_log_return_sum(
            raw_np.array([[100.0, 90.0, float("nan"), 95.0, 105.0]]),
            squared_returns,
        )

    extrema_reducer = build_running_extremum_reducer(extrema)
    with pytest.raises(ValueError, match="execution grid has 5 steps; expected 4"):
        extrema_reducer.init(raw_np.array([100.0]), 5)
    with pytest.raises(ValueError, match="scalar state per path"):
        extrema_reducer.init(raw_np.ones((2, 2)) * 100.0, 4)
    extrema_accumulator = extrema_reducer.init(raw_np.array([100.0]), 4)
    with pytest.raises(ValueError, match="finite and positive"):
        extrema_reducer.update(extrema_accumulator, raw_np.array([0.0]), 2)

    return_reducer = build_squared_log_return_reducer(squared_returns)
    return_accumulator = return_reducer.init(raw_np.array([100.0]), 4)
    with pytest.raises(ValueError, match="finite and positive"):
        return_reducer.update(return_accumulator, raw_np.array([-1.0]), 2)
