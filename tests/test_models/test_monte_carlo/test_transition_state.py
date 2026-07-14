from __future__ import annotations

import math

import numpy as raw_np
import pytest


def _extremum_requirement(
    *,
    n_steps: int,
    full_path: bool = False,
    direction: str = "maximum",
):
    from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement
    from trellis.models.monte_carlo.transition_state import (
        ConditionalBridgeExtremumContract,
        build_conditional_bridge_extremum_reducer,
    )

    reducer = build_conditional_bridge_extremum_reducer(
        ConditionalBridgeExtremumContract(
            n_steps=n_steps,
            transition_steps=tuple(range(1, n_steps + 1)),
            direction=direction,
        ),
        name=f"continuous_{direction}",
    )
    return MonteCarloPathRequirement(
        full_path=full_path,
        transition_reducers=(reducer,),
    )


def test_conditional_log_bridge_extremum_matches_inverse_cdf_identity():
    from trellis.models.monte_carlo.transition_state import (
        ScalarTransitionObservation,
        conditional_log_bridge_extremum,
    )

    previous_log = raw_np.array([0.10, -0.05, 0.20])
    current_log = raw_np.array([-0.05, 0.15, 0.20])
    uniforms = raw_np.array([0.25, 0.50, 0.90])
    variance = 0.09
    observation = ScalarTransitionObservation(
        previous_values=raw_np.exp(previous_log),
        current_values=raw_np.exp(current_log),
        step=3,
        start_time=0.50,
        end_time=0.75,
        bridge_coordinate="log",
        bridge_variance=variance,
        bridge_uniforms=uniforms,
    )
    radicand = (
        (previous_log - current_log) ** 2
        - 2.0 * variance * raw_np.log1p(-uniforms)
    )
    expected_maximum = raw_np.exp(
        0.5 * (previous_log + current_log + raw_np.sqrt(radicand))
    )
    expected_minimum = raw_np.exp(
        0.5 * (previous_log + current_log - raw_np.sqrt(radicand))
    )

    maximum = conditional_log_bridge_extremum(observation, direction="maximum")
    minimum = conditional_log_bridge_extremum(observation, direction="minimum")

    raw_np.testing.assert_allclose(maximum, expected_maximum, rtol=1e-14)
    raw_np.testing.assert_allclose(minimum, expected_minimum, rtol=1e-14)
    assert raw_np.all(maximum >= raw_np.maximum(observation.previous_values, observation.current_values))
    assert raw_np.all(minimum <= raw_np.minimum(observation.previous_values, observation.current_values))


def test_zero_variance_bridge_extremum_is_the_endpoint_extremum():
    from trellis.models.monte_carlo.transition_state import (
        ScalarTransitionObservation,
        conditional_log_bridge_extremum,
    )

    observation = ScalarTransitionObservation(
        previous_values=raw_np.array([90.0, 110.0]),
        current_values=raw_np.array([100.0, 95.0]),
        step=1,
        start_time=0.0,
        end_time=0.5,
        bridge_coordinate="log",
        bridge_variance=0.0,
        bridge_uniforms=raw_np.array([0.25, 0.75]),
    )

    raw_np.testing.assert_array_equal(
        conditional_log_bridge_extremum(observation, direction="maximum"),
        raw_np.array([100.0, 110.0]),
    )
    raw_np.testing.assert_array_equal(
        conditional_log_bridge_extremum(observation, direction="minimum"),
        raw_np.array([90.0, 95.0]),
    )


def test_conditional_bridge_extremum_distribution_matches_fine_grid_reference():
    from scipy.integrate import quad

    from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
    from trellis.models.monte_carlo.transition_state import (
        ScalarTransitionObservation,
        conditional_log_bridge_extremum,
    )

    n_paths = 4_096
    n_fine_steps = 512
    variance = 0.09
    rng = raw_np.random.default_rng(2718)
    bridge_paths = brownian_bridge(
        variance,
        n_fine_steps,
        n_paths,
        end_values=raw_np.zeros(n_paths),
        bridge_shocks=rng.standard_normal((n_paths, n_fine_steps)),
    )
    coarse_levels = raw_np.exp(raw_np.max(bridge_paths[:, ::32], axis=1))
    fine_levels = raw_np.exp(raw_np.max(bridge_paths, axis=1))

    expected_level = quad(
        lambda level: math.exp(level)
        * (4.0 * level / variance)
        * math.exp(-2.0 * level * level / variance),
        0.0,
        raw_np.inf,
    )[0]
    observation = ScalarTransitionObservation(
        previous_values=raw_np.ones(100_000),
        current_values=raw_np.ones(100_000),
        step=1,
        start_time=0.0,
        end_time=1.0,
        bridge_coordinate="log",
        bridge_variance=variance,
        bridge_uniforms=raw_np.random.default_rng(31415).random(100_000),
    )
    exact_levels = conditional_log_bridge_extremum(
        observation,
        direction="maximum",
    )

    coarse_error = abs(float(raw_np.mean(coarse_levels)) - expected_level)
    fine_error = abs(float(raw_np.mean(fine_levels)) - expected_level)
    exact_error = abs(float(raw_np.mean(exact_levels)) - expected_level)
    assert raw_np.all(fine_levels >= coarse_levels)
    assert fine_error < coarse_error
    assert exact_error < 0.004


def test_transition_reducers_match_streaming_full_path_and_explicit_replay():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.transition_state import (
        MonteCarloRandomInputs,
        replay_scalar_transition_reducers,
    )
    from trellis.models.monte_carlo.variance_reduction import (
        sobol_transition_inputs,
    )
    from trellis.models.processes.gbm import GBM

    n_paths = 256
    n_steps = 8
    reduced_requirement = _extremum_requirement(n_steps=n_steps)
    full_requirement = _extremum_requirement(n_steps=n_steps, full_path=True)
    random_inputs = sobol_transition_inputs(
        n_paths,
        n_steps,
        n_factors=1,
        seed=17,
    )
    assert isinstance(random_inputs, MonteCarloRandomInputs)
    process = GBM(mu=0.04, sigma=0.25)

    reduced = MonteCarloEngine(
        process,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=91,
        method="exact",
    ).simulate_state(
        100.0,
        1.0,
        reduced_requirement,
        random_inputs=random_inputs,
    )
    full = MonteCarloEngine(
        process,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=19,
        method="exact",
    ).simulate_state(
        100.0,
        1.0,
        full_requirement,
        random_inputs=random_inputs,
    )
    replayed = replay_scalar_transition_reducers(
        full.full_paths,
        process=process,
        maturity=1.0,
        reducers=full_requirement.transition_reducers,
        transition_uniforms=random_inputs.transition_uniforms,
    )

    assert reduced.full_paths is None
    raw_np.testing.assert_array_equal(reduced.terminal_values, full.terminal_values)
    for name in ("continuous_maximum",):
        raw_np.testing.assert_array_equal(
            reduced.reduced_value(name),
            full.reduced_value(name),
        )
        raw_np.testing.assert_array_equal(
            reduced.reduced_value(name),
            replayed[name],
        )


def test_state_aware_price_accepts_typed_transition_random_inputs():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.path_state import StateAwarePayoff
    from trellis.models.monte_carlo.variance_reduction import (
        sobol_transition_inputs,
    )
    from trellis.models.processes.gbm import GBM

    n_paths = 256
    n_steps = 8
    requirement = _extremum_requirement(n_steps=n_steps)
    payoff = StateAwarePayoff(
        path_requirement=requirement,
        evaluate_paths_fn=lambda _paths: (_ for _ in ()).throw(
            AssertionError("explicit transition pricing must use reduced state")
        ),
        evaluate_state_fn=lambda state: raw_np.maximum(
            state.reduced_value("continuous_maximum") - 100.0,
            0.0,
        ),
        name="transition_state_proof",
    )
    random_inputs = sobol_transition_inputs(
        n_paths,
        n_steps,
        seed=43,
    )
    engine = MonteCarloEngine(
        GBM(mu=0.04, sigma=0.20),
        n_paths=n_paths,
        n_steps=n_steps,
        method="exact",
    )

    result = engine.price(
        100.0,
        1.0,
        payoff,
        discount_rate=0.04,
        random_inputs=random_inputs,
        return_paths=False,
    )

    assert result["price"] > 0.0
    assert result["paths"] is None
    assert result["path_state"] is not None
    assert result["path_state"].full_paths is None


def test_seeded_pseudo_bridge_randomness_is_reproducible_for_full_and_reduced_state():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.processes.gbm import GBM

    process = GBM(mu=0.03, sigma=0.20)
    reduced_requirement = _extremum_requirement(n_steps=6)
    full_requirement = _extremum_requirement(n_steps=6, full_path=True)

    reduced = MonteCarloEngine(
        process,
        n_paths=128,
        n_steps=6,
        seed=23,
        method="exact",
    ).simulate_state(100.0, 1.0, reduced_requirement)
    full = MonteCarloEngine(
        process,
        n_paths=128,
        n_steps=6,
        seed=23,
        method="exact",
    ).simulate_state(100.0, 1.0, full_requirement)

    raw_np.testing.assert_array_equal(reduced.terminal_values, full.terminal_values)
    raw_np.testing.assert_array_equal(
        reduced.reduced_value("continuous_maximum"),
        full.reduced_value("continuous_maximum"),
    )


def test_transition_requirement_does_not_perturb_seeded_process_paths():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.processes.gbm import GBM

    process = GBM(mu=0.03, sigma=0.20)
    ordinary_paths = MonteCarloEngine(
        process,
        n_paths=128,
        n_steps=6,
        seed=29,
        method="exact",
    ).simulate(100.0, 1.0)
    transition_state = MonteCarloEngine(
        process,
        n_paths=128,
        n_steps=6,
        seed=29,
        method="exact",
    ).simulate_state(
        100.0,
        1.0,
        _extremum_requirement(n_steps=6, full_path=True),
    )

    raw_np.testing.assert_array_equal(ordinary_paths, transition_state.full_paths)


def test_explicit_process_shocks_require_separate_transition_uniforms():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.processes.gbm import GBM

    engine = MonteCarloEngine(
        GBM(mu=0.03, sigma=0.20),
        n_paths=16,
        n_steps=4,
        method="exact",
    )
    shocks = raw_np.zeros((16, 4))

    with pytest.raises(ValueError, match="random_inputs.*transition_uniforms"):
        engine.simulate_state(
            100.0,
            1.0,
            _extremum_requirement(n_steps=4),
            shocks=shocks,
        )


def test_differentiable_state_price_rejects_unused_transition_uniforms():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.path_state import terminal_value_payoff
    from trellis.models.monte_carlo.transition_state import MonteCarloRandomInputs
    from trellis.models.processes.gbm import GBM

    engine = MonteCarloEngine(
        GBM(mu=0.03, sigma=0.20),
        n_paths=16,
        n_steps=4,
        method="exact",
    )
    random_inputs = MonteCarloRandomInputs(
        process_shocks=raw_np.zeros((16, 4)),
        transition_uniforms=raw_np.full((16, 4), 0.5),
    )

    with pytest.raises(ValueError, match="no transition reducer"):
        engine.price(
            100.0,
            1.0,
            terminal_value_payoff(lambda terminal: terminal),
            discount_rate=0.03,
            random_inputs=random_inputs,
            differentiable=True,
            return_paths=False,
        )


def test_multiple_stochastic_transition_reducers_fail_closed_without_joint_law():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement
    from trellis.models.processes.gbm import GBM

    maximum = _extremum_requirement(
        n_steps=4,
        direction="maximum",
    ).transition_reducers[0]
    minimum = _extremum_requirement(
        n_steps=4,
        direction="minimum",
    ).transition_reducers[0]
    requirement = MonteCarloPathRequirement(
        transition_reducers=(maximum, minimum),
    )

    with pytest.raises(NotImplementedError, match="one stochastic transition reducer"):
        MonteCarloEngine(
            GBM(mu=0.03, sigma=0.20),
            n_paths=16,
            n_steps=4,
            seed=13,
            method="exact",
        ).simulate_state(100.0, 1.0, requirement)


@pytest.mark.parametrize(
    ("process", "method", "message"),
    [
        pytest.param(
            None,
            "exact",
            "conditional scalar bridge capability",
            id="unsupported-process",
        ),
        pytest.param(
            "gbm",
            "euler",
            "exact scalar transitions",
            id="unsupported-euler-scheme",
        ),
        pytest.param(
            "correlated_gbm",
            "exact",
            "scalar state",
            id="unsupported-vector-state",
        ),
    ],
)
def test_transition_state_fails_closed_for_unsupported_process_or_scheme(
    process,
    method,
    message,
):
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.processes.correlated_gbm import CorrelatedGBM
    from trellis.models.processes.gbm import GBM
    from trellis.models.processes.vasicek import Vasicek

    if process is None:
        process = Vasicek(a=0.10, b=0.03, sigma=0.01)
    elif process == "gbm":
        process = GBM(mu=0.03, sigma=0.20)
    else:
        process = CorrelatedGBM(
            mu=[0.03, 0.04],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.2], [0.2, 1.0]],
        )
    x0 = raw_np.array([100.0, 95.0]) if process.state_dim == 2 else 100.0
    engine = MonteCarloEngine(
        process,
        n_paths=16,
        n_steps=4,
        seed=13,
        method=method,
    )

    with pytest.raises((ValueError, NotImplementedError), match=message):
        engine.simulate_state(
            x0,
            1.0,
            _extremum_requirement(n_steps=4),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bridge_coordinate": ""}, "non-empty"),
        ({"bridge_variance": -0.01}, "non-negative"),
        ({"bridge_uniforms": raw_np.array([0.0, 0.5])}, "strictly between"),
        ({"current_values": raw_np.array([100.0])}, "same shape"),
        ({"previous_values": raw_np.array([100.0, raw_np.nan])}, "finite"),
    ],
)
def test_scalar_transition_observation_rejects_invalid_bridge_inputs(kwargs, message):
    from trellis.models.monte_carlo.transition_state import ScalarTransitionObservation

    values = {
        "previous_values": raw_np.array([100.0, 105.0]),
        "current_values": raw_np.array([101.0, 103.0]),
        "step": 1,
        "start_time": 0.0,
        "end_time": 0.25,
        "bridge_coordinate": "log",
        "bridge_variance": 0.01,
        "bridge_uniforms": raw_np.array([0.25, 0.75]),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ScalarTransitionObservation(**values)


@pytest.mark.parametrize(
    ("coordinate", "previous_values", "message"),
    [
        ("state", raw_np.array([100.0, 105.0]), "log bridge coordinate"),
        ("log", raw_np.array([100.0, -1.0]), "finite and positive"),
    ],
)
def test_conditional_log_bridge_kernel_owns_its_coordinate_and_level_contract(
    coordinate,
    previous_values,
    message,
):
    from trellis.models.monte_carlo.transition_state import (
        ScalarTransitionObservation,
        conditional_log_bridge_extremum,
    )

    observation = ScalarTransitionObservation(
        previous_values=previous_values,
        current_values=raw_np.array([101.0, 103.0]),
        step=1,
        start_time=0.0,
        end_time=0.25,
        bridge_coordinate=coordinate,
        bridge_variance=0.01,
        bridge_uniforms=raw_np.array([0.25, 0.75]),
    )

    with pytest.raises((ValueError, NotImplementedError), match=message):
        conditional_log_bridge_extremum(observation, direction="maximum")


def test_sobol_transition_inputs_use_distinct_reproducible_coordinates():
    from trellis.models.monte_carlo.variance_reduction import sobol_transition_inputs

    first = sobol_transition_inputs(
        256,
        5,
        n_factors=2,
        seed=31,
    )
    second = sobol_transition_inputs(
        256,
        5,
        n_factors=2,
        seed=31,
    )

    assert first.process_shocks.shape == (256, 5, 2)
    assert first.transition_uniforms.shape == (256, 5)
    raw_np.testing.assert_array_equal(first.process_shocks, second.process_shocks)
    raw_np.testing.assert_array_equal(
        first.transition_uniforms,
        second.transition_uniforms,
    )
    assert raw_np.all(
        (first.transition_uniforms > 0.0)
        & (first.transition_uniforms < 1.0)
    )


def test_only_constant_gbm_exposes_the_admitted_log_bridge_capability():
    from trellis.models.monte_carlo.transition_state import (
        ScalarConditionalBridgeProcess,
    )
    from trellis.models.processes.gbm import GBM, PiecewiseConstantGBM

    constant = GBM(mu=0.03, sigma=0.20)
    piecewise = PiecewiseConstantGBM(
        interval_ends=(0.50, 1.00),
        mus=(0.03, 0.04),
        sigmas=(0.20, 0.30),
    )

    assert constant.conditional_bridge_coordinate == "log"
    assert constant.conditional_bridge_variance(0.25, 0.75) == pytest.approx(0.02)
    assert isinstance(constant, ScalarConditionalBridgeProcess)
    assert not isinstance(piecewise, ScalarConditionalBridgeProcess)
