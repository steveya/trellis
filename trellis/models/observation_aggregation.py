"""Product-neutral weighted aggregation on scheduled Monte Carlo observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as raw_np

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.event_state import event_step_indices
from trellis.models.monte_carlo.path_state import (
    MonteCarloPathRequirement,
    MonteCarloPathState,
    PathReducer,
    StateAwarePayoff,
)

np = get_numpy()


def _to_backend_array(values):
    if isinstance(values, raw_np.ndarray) or hasattr(values, "_value"):
        return values
    return np.asarray(values)


def _validation_view(values) -> raw_np.ndarray:
    return raw_np.asarray(getattr(values, "_value", values))


def _validate_execution_grid(actual_steps: int, expected_steps: int) -> None:
    actual = int(actual_steps)
    expected = int(expected_steps)
    if actual != expected:
        raise ValueError(f"execution grid has {actual} steps; expected {expected}")


def _single_state_cross_section(values):
    array = _to_backend_array(values)
    view = _validation_view(array)
    if view.ndim == 1:
        return array
    if view.ndim == 2 and view.shape[1] == 1:
        return array[:, 0]
    raise ValueError("weighted observation aggregation requires scalar state per path")


def _scalar_path_matrix(paths):
    values = _to_backend_array(paths)
    view = _validation_view(values)
    if view.ndim == 3 and view.shape[2] == 1:
        values = values[:, :, 0]
        view = _validation_view(values)
    if view.ndim != 2:
        raise ValueError("weighted observation aggregation requires scalar state per path")
    return values


def _validate_finite_values(values, *, name: str) -> None:
    if raw_np.any(~raw_np.isfinite(_validation_view(values))):
        raise ValueError(f"{name} must contain finite values")


@dataclass(frozen=True)
class WeightedObservationContract:
    """Explicit times and weights for a scalar scheduled path aggregate."""

    observation_times: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        times = tuple(float(time) for time in self.observation_times)
        weights = tuple(float(weight) for weight in self.weights)
        if not times:
            raise ValueError("observation_times must contain at least one value")
        if len(weights) != len(times):
            raise ValueError("weights and observation_times must have the same length")
        if any(not isfinite(time) for time in times):
            raise ValueError("observation_times must contain finite values")
        if any(time < 0.0 for time in times):
            raise ValueError("observation_times must be non-negative")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("observation_times must be strictly increasing")
        if any(not isfinite(weight) for weight in weights):
            raise ValueError("weights must contain finite values")
        object.__setattr__(self, "observation_times", times)
        object.__setattr__(self, "weights", weights)

    def observation_steps(self, *, maturity: float, n_steps: int) -> tuple[int, ...]:
        """Map every observation to one distinct, exact simulation step."""
        horizon = float(maturity)
        step_count = int(n_steps)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if step_count <= 0:
            raise ValueError("n_steps must be positive")
        if self.observation_times[-1] > horizon + 1e-12:
            raise ValueError("observation_times cannot exceed maturity")

        steps = event_step_indices(self.observation_times, horizon, step_count)
        if len(steps) != len(self.observation_times):
            raise ValueError("observation_times must map to distinct simulation steps")
        grid_times = tuple(horizon * step / step_count for step in steps)
        if any(
            abs(grid_time - observation_time) > 1e-12
            for grid_time, observation_time in zip(
                grid_times,
                self.observation_times,
                strict=True,
            )
        ):
            raise ValueError(
                "observation_times must be represented exactly on the simulation grid"
            )
        return steps

    def resolve_uniform_grid_steps(
        self,
        *,
        maturity: float,
        n_steps: int | None = None,
        min_steps: int = 1,
        max_steps: int = 4096,
    ) -> int:
        """Validate an explicit grid or find the smallest exact bounded grid."""
        lower = int(min_steps)
        upper = int(max_steps)
        if lower <= 0:
            raise ValueError("min_steps must be positive")
        if upper < lower:
            raise ValueError("max_steps must be at least min_steps")

        if n_steps is not None:
            step_count = int(n_steps)
            if step_count < lower:
                raise ValueError("n_steps must be at least min_steps")
            if step_count > upper:
                raise ValueError("n_steps must not exceed max_steps")
            self.observation_steps(maturity=maturity, n_steps=step_count)
            return step_count

        horizon = float(maturity)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if self.observation_times[-1] > horizon + 1e-12:
            raise ValueError("observation_times cannot exceed maturity")

        for candidate in range(lower, upper + 1):
            try:
                self.observation_steps(maturity=horizon, n_steps=candidate)
            except ValueError:
                continue
            return candidate
        raise ValueError(
            "no exact uniform simulation grid exists between "
            f"{lower} and {upper} steps"
        )


def weighted_observation_sum(observations, contract: WeightedObservationContract):
    """Return the explicitly weighted sum along the observation axis."""
    values = _to_backend_array(observations)
    view = _validation_view(values)
    if view.ndim == 0 or view.shape[-1] != len(contract.observation_times):
        raise ValueError(
            "observations must have one trailing value per observation time"
        )
    _validate_finite_values(values, name="observations")
    weights = np.asarray(contract.weights)
    aggregate = np.sum(values * weights, axis=-1)
    _validate_finite_values(aggregate, name="weighted observation aggregate")
    return aggregate


def build_weighted_observation_reducer(
    contract: WeightedObservationContract,
    *,
    maturity: float,
    n_steps: int,
    name: str = "weighted_observations",
) -> PathReducer:
    """Build reduced state for an explicitly weighted scalar observation sum."""
    reducer_name = str(name).strip()
    if not reducer_name:
        raise ValueError("weighted observation reducer name must be non-empty")
    steps = contract.observation_steps(maturity=maturity, n_steps=n_steps)
    weights_by_step = dict(zip(steps, contract.weights, strict=True))

    def _init(initial_values, total_steps):
        _validate_execution_grid(total_steps, n_steps)
        initial = _single_state_cross_section(initial_values)
        _validate_finite_values(initial, name="observed levels")
        if 0 in weights_by_step:
            return float(weights_by_step[0]) * initial
        return np.zeros_like(initial)

    def _update(accumulator, values, step):
        current_accumulator = _to_backend_array(accumulator)
        weight = weights_by_step.get(int(step))
        if weight is None:
            return current_accumulator
        current = _single_state_cross_section(values)
        _validate_finite_values(current, name="observed levels")
        return current_accumulator + float(weight) * current

    return PathReducer(
        name=reducer_name,
        init_fn=_init,
        update_fn=_update,
    )


def _settled_path_values(settlement_fn: Callable, aggregate, *, n_paths: int):
    settled = _to_backend_array(settlement_fn(aggregate))
    view = _validation_view(settled)
    if view.shape != (int(n_paths),):
        raise ValueError("settlement_fn must return one value per path")
    _validate_finite_values(settled, name="settled path values")
    return settled


def weighted_observation_payoff(
    contract: WeightedObservationContract,
    *,
    maturity: float,
    n_steps: int,
    settlement_fn: Callable,
    reducer_name: str = "weighted_observations",
    name: str = "weighted_observation_payoff",
    derivative_metadata: Mapping[str, object] | None = None,
) -> StateAwarePayoff:
    """Compose weighted path state with caller-supplied settlement semantics."""
    if not callable(settlement_fn):
        raise TypeError("settlement_fn must be callable")
    steps = contract.observation_steps(maturity=maturity, n_steps=n_steps)
    reducer = build_weighted_observation_reducer(
        contract,
        maturity=maturity,
        n_steps=n_steps,
        name=reducer_name,
    )

    def _evaluate_paths(paths):
        values = _scalar_path_matrix(paths)
        view = _validation_view(values)
        _validate_execution_grid(view.shape[1] - 1, n_steps)
        aggregate = weighted_observation_sum(values[:, steps], contract)
        return _settled_path_values(
            settlement_fn,
            aggregate,
            n_paths=view.shape[0],
        )

    def _evaluate_state(state: MonteCarloPathState):
        _validate_execution_grid(state.n_steps, n_steps)
        aggregate = _to_backend_array(state.reduced_value(reducer_name))
        _validate_finite_values(aggregate, name="weighted observation aggregate")
        return _settled_path_values(
            settlement_fn,
            aggregate,
            n_paths=state.n_paths,
        )

    return StateAwarePayoff(
        path_requirement=MonteCarloPathRequirement(reducers=(reducer,)),
        evaluate_paths_fn=_evaluate_paths,
        evaluate_state_fn=_evaluate_state,
        name=str(name or "weighted_observation_payoff"),
        derivative_metadata=dict(derivative_metadata or {}),
    )


__all__ = [
    "WeightedObservationContract",
    "build_weighted_observation_reducer",
    "weighted_observation_payoff",
    "weighted_observation_sum",
]
