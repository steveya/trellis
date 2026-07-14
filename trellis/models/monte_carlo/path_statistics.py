"""Product-neutral discrete statistics for scalar Monte Carlo paths."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as raw_np

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.path_state import PathReducer

np = get_numpy()


def _to_backend_array(values):
    if isinstance(values, raw_np.ndarray) or hasattr(values, "_value"):
        return values
    return np.asarray(values)


def _validation_view(values) -> raw_np.ndarray:
    return raw_np.asarray(getattr(values, "_value", values))


def _normalize_observation_steps(
    values: tuple[int, ...],
    *,
    n_steps: int,
    minimum_count: int,
) -> tuple[int, ...]:
    steps: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("observation_steps must contain integer steps")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_steps must contain integer steps") from exc
        if not isfinite(numeric) or not numeric.is_integer():
            raise ValueError("observation_steps must contain integer steps")
        steps.append(int(numeric))
    if len(steps) < minimum_count:
        qualifier = "one" if minimum_count == 1 else "two"
        raise ValueError(
            f"observation_steps must contain at least {qualifier} value"
            + ("s" if minimum_count != 1 else "")
        )
    if any(step < 0 for step in steps):
        raise ValueError("observation_steps must be non-negative")
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError("observation_steps must be strictly increasing")
    if steps[-1] > n_steps:
        raise ValueError("observation_steps cannot exceed n_steps")
    return tuple(steps)


def _validated_n_steps(value) -> int:
    if isinstance(value, bool):
        raise ValueError("n_steps must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_steps must be a positive integer") from exc
    if not isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError("n_steps must be a positive integer")
    return int(numeric)


def _validate_grid(actual_steps: int, expected_steps: int) -> None:
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
    raise ValueError("path statistics require a scalar state per path")


def _scalar_path_matrix(paths, *, n_steps: int):
    values = _to_backend_array(paths)
    view = _validation_view(values)
    if view.ndim == 3 and view.shape[2] == 1:
        values = values[:, :, 0]
        view = _validation_view(values)
    if view.ndim != 2:
        raise ValueError("path statistics require a scalar state per path")
    _validate_grid(view.shape[1] - 1, n_steps)
    return values


def _validate_positive_levels(values) -> None:
    view = _validation_view(values)
    if raw_np.any(~raw_np.isfinite(view)) or raw_np.any(view <= 0.0):
        raise ValueError("observed path levels must be finite and positive")


def _validated_reducer_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("path-statistic reducer name must be non-empty")
    return normalized


@dataclass(frozen=True)
class RunningExtremumContract:
    """Explicit discrete observations for one positive-path extremum."""

    n_steps: int
    observation_steps: tuple[int, ...]
    direction: str = "maximum"
    initial_extremum: float | None = None

    def __post_init__(self) -> None:
        n_steps = _validated_n_steps(self.n_steps)
        steps = _normalize_observation_steps(
            self.observation_steps,
            n_steps=n_steps,
            minimum_count=1,
        )
        direction = str(self.direction or "").strip().lower()
        if direction not in {"minimum", "maximum"}:
            raise ValueError("direction must be 'minimum' or 'maximum'")
        initial_extremum = self.initial_extremum
        if initial_extremum is not None:
            initial_extremum = float(initial_extremum)
            if not isfinite(initial_extremum) or initial_extremum <= 0.0:
                raise ValueError(
                    "initial_extremum must be finite and positive"
                )
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "observation_steps", steps)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "initial_extremum", initial_extremum)


@dataclass(frozen=True)
class SquaredLogReturnContract:
    """Explicit discrete observations and scaling for squared log returns.

    The first selected observation is the return baseline and contributes no
    squared return. ``annualization_factor`` multiplies the complete sum; the
    caller owns the convention used to derive that factor.
    """

    n_steps: int
    observation_steps: tuple[int, ...]
    annualization_factor: float = 1.0

    def __post_init__(self) -> None:
        n_steps = _validated_n_steps(self.n_steps)
        steps = _normalize_observation_steps(
            self.observation_steps,
            n_steps=n_steps,
            minimum_count=2,
        )
        factor = float(self.annualization_factor)
        if not isfinite(factor) or factor <= 0.0:
            raise ValueError(
                "annualization_factor must be finite and positive"
            )
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "observation_steps", steps)
        object.__setattr__(self, "annualization_factor", factor)


def discrete_path_extremum(paths, contract: RunningExtremumContract):
    """Return one minimum or maximum over the declared discrete steps."""
    values = _scalar_path_matrix(paths, n_steps=contract.n_steps)
    observed = values[:, contract.observation_steps]
    _validate_positive_levels(observed)
    if contract.direction == "minimum":
        result = np.min(observed, axis=1)
        if contract.initial_extremum is not None:
            result = np.minimum(result, contract.initial_extremum)
        return result
    result = np.max(observed, axis=1)
    if contract.initial_extremum is not None:
        result = np.maximum(result, contract.initial_extremum)
    return result


def annualized_squared_log_return_sum(
    paths,
    contract: SquaredLogReturnContract,
):
    """Return the scaled sum of consecutive selected squared log returns."""
    values = _scalar_path_matrix(paths, n_steps=contract.n_steps)
    observed = values[:, contract.observation_steps]
    _validate_positive_levels(observed)
    log_returns = np.log(observed[:, 1:] / observed[:, :-1])
    return contract.annualization_factor * np.sum(
        log_returns * log_returns,
        axis=1,
    )


def build_running_extremum_reducer(
    contract: RunningExtremumContract,
    *,
    name: str = "running_extremum",
) -> PathReducer:
    """Build bounded state for the declared discrete path extremum."""
    reducer_name = _validated_reducer_name(name)
    observed_steps = frozenset(contract.observation_steps)

    def initialize(initial_values, total_steps):
        _validate_grid(total_steps, contract.n_steps)
        initial = _single_state_cross_section(initial_values)
        if contract.initial_extremum is not None:
            accumulator = np.ones_like(initial) * contract.initial_extremum
            if 0 not in observed_steps:
                return accumulator
        elif 0 not in observed_steps:
            sentinel = raw_np.inf if contract.direction == "minimum" else -raw_np.inf
            return np.ones_like(initial) * sentinel
        else:
            accumulator = initial
        _validate_positive_levels(initial)
        if contract.direction == "minimum":
            return np.minimum(accumulator, initial)
        return np.maximum(accumulator, initial)

    def update(accumulator, values, step):
        current_accumulator = _to_backend_array(accumulator)
        if int(step) not in observed_steps:
            return current_accumulator
        current = _single_state_cross_section(values)
        _validate_positive_levels(current)
        if contract.direction == "minimum":
            return np.minimum(current_accumulator, current)
        return np.maximum(current_accumulator, current)

    return PathReducer(
        name=reducer_name,
        init_fn=initialize,
        update_fn=update,
    )


def build_squared_log_return_reducer(
    contract: SquaredLogReturnContract,
    *,
    name: str = "squared_log_returns",
) -> PathReducer:
    """Build bounded state for the declared squared-log-return sum."""
    reducer_name = _validated_reducer_name(name)
    observed_steps = frozenset(contract.observation_steps)
    baseline_step = contract.observation_steps[0]

    def initialize(initial_values, total_steps):
        _validate_grid(total_steps, contract.n_steps)
        initial = _single_state_cross_section(initial_values)
        if baseline_step == 0:
            _validate_positive_levels(initial)
        return np.stack((initial, np.zeros_like(initial)), axis=1)

    def update(accumulator, values, step):
        current_accumulator = _to_backend_array(accumulator)
        current_step = int(step)
        if current_step not in observed_steps:
            return current_accumulator
        current = _single_state_cross_section(values)
        _validate_positive_levels(current)
        accumulated = current_accumulator[:, 1]
        if current_step == baseline_step and baseline_step != 0:
            return np.stack((current, accumulated), axis=1)
        previous = current_accumulator[:, 0]
        _validate_positive_levels(previous)
        log_return = np.log(current / previous)
        return np.stack(
            (
                current,
                accumulated
                + contract.annualization_factor * log_return * log_return,
            ),
            axis=1,
        )

    return PathReducer(
        name=reducer_name,
        init_fn=initialize,
        update_fn=update,
        finalize_fn=lambda accumulator: accumulator[:, 1],
    )


__all__ = [
    "RunningExtremumContract",
    "SquaredLogReturnContract",
    "annualized_squared_log_return_sum",
    "build_running_extremum_reducer",
    "build_squared_log_return_reducer",
    "discrete_path_extremum",
]
