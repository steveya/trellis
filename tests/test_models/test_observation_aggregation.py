from __future__ import annotations

import numpy as raw_np
import pytest


def test_weighted_observation_contract_validates_schedule_weights_and_grid():
    from trellis.models.observation_aggregation import WeightedObservationContract

    contract = WeightedObservationContract(
        observation_times=(0.0, 0.5, 1.0),
        weights=(0.2, 0.3, 0.5),
    )

    assert contract.observation_times == (0.0, 0.5, 1.0)
    assert contract.weights == (0.2, 0.3, 0.5)
    assert contract.observation_steps(maturity=1.0, n_steps=4) == (0, 2, 4)

    with pytest.raises(ValueError, match="at least one"):
        WeightedObservationContract(observation_times=(), weights=())
    with pytest.raises(ValueError, match="strictly increasing"):
        WeightedObservationContract(
            observation_times=(0.5, 0.5),
            weights=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="non-negative"):
        WeightedObservationContract(
            observation_times=(-0.1, 1.0),
            weights=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="finite"):
        WeightedObservationContract(
            observation_times=(0.5, float("nan")),
            weights=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="same length"):
        WeightedObservationContract(
            observation_times=(0.5, 1.0),
            weights=(1.0,),
        )
    with pytest.raises(ValueError, match="finite"):
        WeightedObservationContract(
            observation_times=(0.5, 1.0),
            weights=(0.5, float("inf")),
        )
    with pytest.raises(ValueError, match="cannot exceed maturity"):
        contract.observation_steps(maturity=0.75, n_steps=4)
    with pytest.raises(ValueError, match="represented exactly"):
        WeightedObservationContract(
            observation_times=(0.3, 1.0),
            weights=(0.5, 0.5),
        ).observation_steps(maturity=1.0, n_steps=4)
    with pytest.raises(ValueError, match="distinct simulation steps"):
        WeightedObservationContract(
            observation_times=(0.2, 0.24),
            weights=(0.5, 0.5),
        ).observation_steps(maturity=1.0, n_steps=2)


def test_weighted_observation_sum_uses_explicit_weights_and_preserves_autograd():
    from trellis.core.differentiable import get_numpy, gradient
    from trellis.models.observation_aggregation import (
        WeightedObservationContract,
        weighted_observation_sum,
    )

    contract = WeightedObservationContract(
        observation_times=(0.0, 0.5, 1.0),
        weights=(0.2, 0.3, 0.5),
    )
    observations = raw_np.array(
        [
            [100.0, 110.0, 120.0],
            [80.0, 90.0, 70.0],
        ]
    )

    raw_np.testing.assert_allclose(
        weighted_observation_sum(observations, contract),
        raw_np.array([113.0, 78.0]),
    )

    np = get_numpy()
    sensitivity = gradient(
        lambda final_level: weighted_observation_sum(
            np.array([100.0, 110.0, final_level]),
            contract,
        )
    )(120.0)
    assert sensitivity == pytest.approx(0.5)

    spread_contract = WeightedObservationContract(
        observation_times=(0.5, 1.0),
        weights=(1.0, -1.0),
    )
    raw_np.testing.assert_allclose(
        weighted_observation_sum(
            raw_np.array([[110.0, 100.0], [90.0, 95.0]]),
            spread_contract,
        ),
        raw_np.array([10.0, -5.0]),
    )

    with pytest.raises(ValueError, match="trailing value per observation time"):
        weighted_observation_sum(raw_np.ones((2, 2)), contract)


def test_weighted_observation_payoff_matches_full_paths_and_reduced_state():
    from trellis.core.differentiable import get_numpy
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.observation_aggregation import (
        WeightedObservationContract,
        weighted_observation_payoff,
    )
    from trellis.models.processes.gbm import GBM

    np = get_numpy()
    contract = WeightedObservationContract(
        observation_times=(0.0, 0.5, 1.0),
        weights=(1.0 / 3.0,) * 3,
    )
    payoff = weighted_observation_payoff(
        contract,
        maturity=1.0,
        n_steps=4,
        settlement_fn=lambda average: np.maximum(average - 105.0, 0.0),
        reducer_name="weighted_levels",
    )
    paths = raw_np.array(
        [
            [100.0, 103.0, 110.0, 115.0, 120.0],
            [100.0, 98.0, 95.0, 92.0, 90.0],
        ]
    )

    raw_np.testing.assert_allclose(
        payoff(paths),
        raw_np.array([5.0, 0.0]),
    )

    from trellis.models.monte_carlo.path_state import MonteCarloPathState

    reduced_state = MonteCarloPathState(
        initial_value=100.0,
        n_steps=4,
        terminal_values=paths[:, -1],
        reducer_values={"weighted_levels": raw_np.array([110.0, 95.0])},
    )
    raw_np.testing.assert_allclose(payoff.evaluate_state(reduced_state), payoff(paths))

    engine = MonteCarloEngine(
        GBM(mu=0.10, sigma=0.0),
        n_paths=16,
        n_steps=4,
        seed=7,
        method="exact",
    )
    result = engine.price(
        100.0,
        1.0,
        payoff,
        return_paths=False,
    )

    expected_average = 100.0 * (
        1.0 + raw_np.exp(0.05) + raw_np.exp(0.10)
    ) / 3.0
    assert result["price"] == pytest.approx(max(expected_average - 105.0, 0.0))
    assert result["paths"] is None
    assert result["path_state"] is not None
    assert result["path_state"].full_paths is None
    assert tuple(result["path_state"].reducer_values) == ("weighted_levels",)


def test_weighted_observation_payoff_rejects_vector_state_and_grid_mismatch():
    from trellis.models.monte_carlo.path_state import MonteCarloPathState
    from trellis.models.observation_aggregation import (
        WeightedObservationContract,
        build_weighted_observation_reducer,
        weighted_observation_payoff,
    )

    contract = WeightedObservationContract(
        observation_times=(0.5, 1.0),
        weights=(0.5, 0.5),
    )
    payoff = weighted_observation_payoff(
        contract,
        maturity=1.0,
        n_steps=4,
        settlement_fn=lambda aggregate: aggregate,
    )
    with pytest.raises(ValueError, match="scalar state per path"):
        payoff(raw_np.ones((2, 5, 2)))
    with pytest.raises(ValueError, match="execution grid has 8 steps; expected 4"):
        payoff(raw_np.ones((2, 9)))

    scalar_settlement = weighted_observation_payoff(
        contract,
        maturity=1.0,
        n_steps=4,
        settlement_fn=lambda aggregate: 1.0,
    )
    with pytest.raises(ValueError, match="one value per path"):
        scalar_settlement(raw_np.ones((2, 5)))

    reducer = build_weighted_observation_reducer(
        contract,
        maturity=1.0,
        n_steps=4,
    )
    with pytest.raises(ValueError, match="scalar state per path"):
        reducer.init(raw_np.ones((2, 2)), 4)
    with pytest.raises(ValueError, match="execution grid has 8 steps; expected 4"):
        reducer.init(raw_np.ones(2), 8)

    state = MonteCarloPathState(
        initial_value=100.0,
        n_steps=8,
        terminal_values=raw_np.array([100.0]),
        reducer_values={"weighted_observations": raw_np.array([100.0])},
    )
    with pytest.raises(ValueError, match="execution grid has 8 steps; expected 4"):
        payoff.evaluate_state(state)
