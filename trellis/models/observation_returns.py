"""Product-neutral payoff algebra for scheduled observation returns."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, isnan

import numpy as raw_np

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.event_state import event_step_indices
from trellis.models.monte_carlo.path_state import (
    MonteCarloPathRequirement,
    PathReducer,
    StateAwarePayoff,
)

np = get_numpy()


def _to_backend_array(values):
    if isinstance(values, raw_np.ndarray) or hasattr(values, "_value"):
        return values
    return np.asarray(values)


def _validation_view(values):
    return raw_np.asarray(getattr(values, "_value", values))


def _normalized_direction(value: str) -> str:
    direction = str(value or "up").strip().lower().replace("-", "_")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    return direction


def _validated_bound(value: float, *, name: str) -> float:
    bound = float(value)
    if isnan(bound):
        raise ValueError(f"{name} cannot be NaN")
    return bound


@dataclass(frozen=True)
class ObservationReturnContract:
    """Typed terms for a bounded sum of consecutive simple returns."""

    observation_times: tuple[float, ...]
    direction: str = "up"
    local_floor: float = -inf
    local_cap: float = inf
    global_floor: float = -inf
    global_cap: float = inf
    payoff_scale: float = 1.0

    def __post_init__(self) -> None:
        times = tuple(float(time) for time in self.observation_times)
        if any(not isfinite(time) for time in times):
            raise ValueError("observation_times must contain finite values")
        if not times or any(time <= 0.0 for time in times):
            raise ValueError("observation_times must contain positive values")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("observation_times must be strictly increasing")

        local_floor = _validated_bound(self.local_floor, name="local_floor")
        local_cap = _validated_bound(self.local_cap, name="local_cap")
        global_floor = _validated_bound(self.global_floor, name="global_floor")
        global_cap = _validated_bound(self.global_cap, name="global_cap")
        if local_cap < local_floor:
            raise ValueError("local_cap must be greater than or equal to local_floor")
        if global_cap < global_floor:
            raise ValueError("global_cap must be greater than or equal to global_floor")
        payoff_scale = float(self.payoff_scale)
        if not isfinite(payoff_scale):
            raise ValueError("payoff_scale must be finite")

        object.__setattr__(self, "observation_times", times)
        object.__setattr__(self, "direction", _normalized_direction(self.direction))
        object.__setattr__(self, "local_floor", local_floor)
        object.__setattr__(self, "local_cap", local_cap)
        object.__setattr__(self, "global_floor", global_floor)
        object.__setattr__(self, "global_cap", global_cap)
        object.__setattr__(self, "payoff_scale", payoff_scale)

    def observation_steps(self, *, maturity: float, n_steps: int) -> tuple[int, ...]:
        """Map observation times onto distinct simulation steps or fail closed."""
        horizon = float(maturity)
        step_count = int(n_steps)
        if horizon <= 0.0 or step_count <= 0:
            raise ValueError("maturity and n_steps must be positive")
        if self.observation_times[-1] > horizon + 1e-12:
            raise ValueError("observation_times cannot exceed maturity")
        steps = event_step_indices(self.observation_times, horizon, step_count)
        if len(steps) != len(self.observation_times) or steps[0] <= 0:
            raise ValueError(
                "observation_times must map to distinct positive simulation steps"
            )
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


def simple_observation_returns(gross_returns, *, direction: str = "up"):
    """Convert positive gross returns into signed simple returns."""
    values = _to_backend_array(gross_returns)
    view = _validation_view(values)
    if raw_np.any(~raw_np.isfinite(view)) or raw_np.any(view <= 0.0):
        raise ValueError("gross_returns must be finite and strictly positive")
    simple = values - 1.0
    return simple if _normalized_direction(direction) == "up" else -simple


def bounded_observation_return_sum(
    gross_returns,
    contract: ObservationReturnContract,
):
    """Apply local bounds, sum interval returns, then apply global bounds."""
    values = _to_backend_array(gross_returns)
    view = _validation_view(values)
    if view.ndim == 0 or view.shape[-1] != len(contract.observation_times):
        raise ValueError(
            "gross_returns must have one trailing value per observation time"
        )
    returns = simple_observation_returns(values, direction=contract.direction)
    locally_bounded = np.clip(returns, contract.local_floor, contract.local_cap)
    accumulated = np.sum(locally_bounded, axis=-1)
    settled = np.clip(accumulated, contract.global_floor, contract.global_cap)
    return contract.payoff_scale * settled


def _single_state_cross_section(values):
    array = _to_backend_array(values)
    view = _validation_view(array)
    if view.ndim == 1:
        return array
    if view.ndim == 2 and view.shape[1] == 1:
        return array[:, 0]
    raise ValueError(
        "observation-return reducers require a scalar state per path"
    )


def build_observation_return_reducer(
    contract: ObservationReturnContract,
    *,
    maturity: float,
    n_steps: int,
    name: str = "observation_returns",
) -> PathReducer:
    """Build reduced state for consecutive bounded observation returns."""
    reducer_name = str(name).strip()
    if not reducer_name:
        raise ValueError("observation-return reducer name must be non-empty")
    observation_steps = frozenset(
        contract.observation_steps(maturity=maturity, n_steps=n_steps)
    )

    def _init(initial_values, total_steps):
        del total_steps
        previous = _single_state_cross_section(initial_values)
        return np.stack((previous, np.zeros_like(previous)), axis=1)

    def _update(accumulator, values, step):
        current_accumulator = _to_backend_array(accumulator)
        if int(step) not in observation_steps:
            return current_accumulator
        previous = current_accumulator[:, 0]
        accumulated = current_accumulator[:, 1]
        current = _single_state_cross_section(values)
        gross_return = current / previous
        interval_return = simple_observation_returns(
            gross_return,
            direction=contract.direction,
        )
        locally_bounded = np.clip(
            interval_return,
            contract.local_floor,
            contract.local_cap,
        )
        return np.stack((current, accumulated + locally_bounded), axis=1)

    return PathReducer(name=reducer_name, init_fn=_init, update_fn=_update)


def observation_return_payoff(
    contract: ObservationReturnContract,
    *,
    maturity: float,
    n_steps: int,
    reducer_name: str = "observation_returns",
) -> StateAwarePayoff:
    """Build a full-path/reduced-state payoff for bounded observation returns."""
    steps = contract.observation_steps(maturity=maturity, n_steps=n_steps)
    reducer = build_observation_return_reducer(
        contract,
        maturity=maturity,
        n_steps=n_steps,
        name=reducer_name,
    )
    requirement = MonteCarloPathRequirement(reducers=(reducer,))

    def _evaluate_paths(paths):
        values = _to_backend_array(paths)
        view = _validation_view(values)
        if view.ndim == 3 and view.shape[2] == 1:
            values = values[:, :, 0]
            view = _validation_view(values)
        if view.ndim != 2 or view.shape[1] <= max(steps):
            raise ValueError("paths do not contain every required observation step")
        level_steps = (0, *steps)
        levels = values[:, level_steps]
        gross_returns = levels[:, 1:] / levels[:, :-1]
        return bounded_observation_return_sum(gross_returns, contract)

    def _evaluate_state(state):
        reduced = _to_backend_array(state.reduced_value(reducer_name))
        accumulated = reduced[:, 1]
        settled = np.clip(
            accumulated,
            contract.global_floor,
            contract.global_cap,
        )
        return contract.payoff_scale * settled

    return StateAwarePayoff(
        path_requirement=requirement,
        evaluate_paths_fn=_evaluate_paths,
        evaluate_state_fn=_evaluate_state,
        name=reducer_name,
    )


__all__ = [
    "ObservationReturnContract",
    "bounded_observation_return_sum",
    "build_observation_return_reducer",
    "observation_return_payoff",
    "simple_observation_returns",
]
