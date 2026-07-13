from __future__ import annotations

import numpy as raw_np
import pytest


def test_observation_return_contract_validates_schedule_direction_and_bounds():
    from trellis.models.observation_returns import ObservationReturnContract

    contract = ObservationReturnContract(
        observation_times=(0.25, 0.5, 1.0),
        direction="down",
        local_floor=0.0,
        local_cap=0.08,
        global_floor=0.0,
        global_cap=0.15,
        payoff_scale=100.0,
    )

    assert contract.observation_times == (0.25, 0.5, 1.0)
    assert contract.direction == "down"
    assert contract.observation_steps(maturity=1.0, n_steps=4) == (1, 2, 4)

    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationReturnContract(observation_times=(0.5, 0.5, 1.0))
    with pytest.raises(ValueError, match="positive"):
        ObservationReturnContract(observation_times=(0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        ObservationReturnContract(observation_times=(0.5, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        ObservationReturnContract(observation_times=(0.5, float("inf")))
    with pytest.raises(ValueError, match="direction"):
        ObservationReturnContract(observation_times=(1.0,), direction="call")
    with pytest.raises(ValueError, match="local_cap"):
        ObservationReturnContract(
            observation_times=(1.0,),
            local_floor=0.10,
            local_cap=0.05,
        )
    with pytest.raises(ValueError, match="represented exactly"):
        ObservationReturnContract(
            observation_times=(0.3, 1.0),
        ).observation_steps(maturity=1.0, n_steps=4)


def test_bounded_observation_return_sum_applies_local_and_global_bounds():
    from trellis.models.observation_returns import (
        ObservationReturnContract,
        bounded_observation_return_sum,
    )

    contract = ObservationReturnContract(
        observation_times=(0.25, 0.5, 1.0),
        local_floor=0.0,
        local_cap=0.08,
        global_floor=0.0,
        global_cap=0.15,
        payoff_scale=100.0,
    )
    gross_returns = raw_np.array(
        [
            [1.10, 0.95, 1.30],
            [0.90, 1.02, 1.04],
        ]
    )

    values = bounded_observation_return_sum(gross_returns, contract)

    raw_np.testing.assert_allclose(values, raw_np.array([15.0, 6.0]))

    down_contract = ObservationReturnContract(
        observation_times=(0.5, 1.0),
        direction="down",
        local_floor=0.0,
        local_cap=0.08,
        global_floor=0.0,
        global_cap=1.0,
    )
    raw_np.testing.assert_allclose(
        bounded_observation_return_sum(
            raw_np.array([[0.90, 1.20]]),
            down_contract,
        ),
        raw_np.array([0.08]),
    )

    for invalid_return in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            bounded_observation_return_sum(
                raw_np.array([[invalid_return, 1.0]]),
                down_contract,
            )


def test_bounded_observation_return_sum_preserves_autograd_inside_bounds():
    from trellis.core.differentiable import get_numpy, gradient
    from trellis.models.observation_returns import (
        ObservationReturnContract,
        bounded_observation_return_sum,
    )

    np = get_numpy()
    contract = ObservationReturnContract(
        observation_times=(1.0,),
        local_floor=0.0,
        local_cap=0.20,
        global_floor=0.0,
        global_cap=0.50,
        payoff_scale=100.0,
    )

    sensitivity = gradient(
        lambda gross_return: bounded_observation_return_sum(
            np.array([gross_return]),
            contract,
        )
    )(1.05)

    assert sensitivity == pytest.approx(100.0)


def test_observation_return_payoff_matches_full_paths_and_reduced_state():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.observation_returns import (
        ObservationReturnContract,
        observation_return_payoff,
    )
    from trellis.models.processes.gbm import GBM

    contract = ObservationReturnContract(
        observation_times=(0.5, 1.0),
        local_floor=0.0,
        local_cap=0.08,
        global_floor=0.0,
        global_cap=0.15,
        payoff_scale=100.0,
    )
    payoff = observation_return_payoff(
        contract,
        maturity=1.0,
        n_steps=4,
        reducer_name="reset_returns",
    )
    paths = raw_np.array(
        [
            [100.0, 103.0, 110.0, 105.0, 121.0],
            [100.0, 98.0, 90.0, 95.0, 91.8],
        ]
    )

    raw_np.testing.assert_allclose(payoff(paths), raw_np.array([15.0, 2.0]))

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

    expected_period_return = raw_np.exp(0.05) - 1.0
    assert result["price"] == pytest.approx(100.0 * 2.0 * expected_period_return)
    assert result["paths"] is None
    assert result["path_state"] is not None
    assert result["path_state"].full_paths is None
    assert tuple(result["path_state"].reducer_values) == ("reset_returns",)


def test_observation_return_payoff_rejects_nonpositive_observed_levels():
    from trellis.models.observation_returns import (
        ObservationReturnContract,
        build_observation_return_reducer,
        observation_return_payoff,
    )

    contract = ObservationReturnContract(observation_times=(1.0,))
    payoff = observation_return_payoff(
        contract,
        maturity=1.0,
        n_steps=1,
    )
    with pytest.raises(ValueError, match="observed levels must be finite and positive"):
        payoff(raw_np.array([[-100.0, -90.0]]))

    reducer = build_observation_return_reducer(
        contract,
        maturity=1.0,
        n_steps=1,
    )
    with pytest.raises(ValueError, match="observed levels must be finite and positive"):
        reducer.init(raw_np.array([-100.0]), 1)

    accumulator = reducer.init(raw_np.array([100.0]), 1)
    with pytest.raises(ValueError, match="observed levels must be finite and positive"):
        reducer.update(accumulator, raw_np.array([-90.0]), 1)
