"""Product-neutral state for statistics defined between path endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Protocol, runtime_checkable

import numpy as raw_np


@runtime_checkable
class ScalarConditionalBridgeProcess(Protocol):
    """Process capability for an exact conditional scalar bridge."""

    @property
    def conditional_bridge_coordinate(self) -> str:
        """Return the Gaussian bridge coordinate used between endpoints."""
        ...

    def conditional_bridge_variance(
        self,
        start_time: float,
        end_time: float,
    ) -> float:
        """Return integrated variance in the bridge coordinate."""
        ...


@dataclass(frozen=True)
class MonteCarloRandomInputs:
    """Explicit process normals and an optional auxiliary transition channel."""

    process_shocks: raw_np.ndarray
    transition_uniforms: raw_np.ndarray | None = None


def _path_cross_section(values, *, name: str) -> raw_np.ndarray:
    array = raw_np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must contain one scalar state per path")
    return array


def _positive_path_cross_section(values, *, name: str) -> raw_np.ndarray:
    array = _path_cross_section(values, name=name)
    if raw_np.any(~raw_np.isfinite(array)) or raw_np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite and positive path levels")
    return array


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


def _validated_transition_steps(
    values: tuple[int, ...],
    *,
    n_steps: int,
) -> tuple[int, ...]:
    steps: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("transition_steps must contain integer ending steps")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "transition_steps must contain integer ending steps"
            ) from exc
        if not isfinite(numeric) or not numeric.is_integer():
            raise ValueError("transition_steps must contain integer ending steps")
        steps.append(int(numeric))
    if not steps:
        raise ValueError("transition_steps must contain at least one ending step")
    if any(step < 1 for step in steps):
        raise ValueError("transition_steps must be between 1 and n_steps")
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError("transition_steps must be strictly increasing")
    if steps[-1] > n_steps:
        raise ValueError("transition_steps cannot exceed n_steps")
    return tuple(steps)


def _validated_reducer_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("transition reducer name must be non-empty")
    return normalized


@dataclass(frozen=True)
class ScalarTransitionObservation:
    """One exact scalar transition plus independent bridge randomness."""

    previous_values: raw_np.ndarray
    current_values: raw_np.ndarray
    step: int
    start_time: float
    end_time: float
    bridge_coordinate: str
    bridge_variance: float
    bridge_uniforms: raw_np.ndarray

    def __post_init__(self) -> None:
        previous = _path_cross_section(
            self.previous_values,
            name="previous_values",
        )
        current = _path_cross_section(
            self.current_values,
            name="current_values",
        )
        if raw_np.any(~raw_np.isfinite(previous)) or raw_np.any(
            ~raw_np.isfinite(current)
        ):
            raise ValueError(
                "transition endpoint values must be finite"
            )
        if current.shape != previous.shape:
            raise ValueError(
                "previous_values and current_values must have the same shape"
            )
        uniforms = _path_cross_section(
            self.bridge_uniforms,
            name="bridge_uniforms",
        )
        if uniforms.shape != previous.shape:
            raise ValueError(
                "bridge_uniforms and transition endpoint values must have the same shape"
            )
        if raw_np.any(~raw_np.isfinite(uniforms)) or raw_np.any(
            (uniforms <= 0.0) | (uniforms >= 1.0)
        ):
            raise ValueError("bridge_uniforms must be strictly between zero and one")
        step = int(self.step)
        if step < 1:
            raise ValueError("transition step must be positive")
        start_time = float(self.start_time)
        end_time = float(self.end_time)
        if (
            not isfinite(start_time)
            or not isfinite(end_time)
            or start_time < 0.0
            or end_time <= start_time
        ):
            raise ValueError(
                "transition times must be finite, non-negative, and strictly increasing"
            )
        coordinate = str(self.bridge_coordinate or "").strip().lower()
        if not coordinate:
            raise ValueError("bridge_coordinate must be non-empty")
        variance = float(self.bridge_variance)
        if not isfinite(variance) or variance < 0.0:
            raise ValueError("bridge_variance must be finite and non-negative")
        object.__setattr__(self, "previous_values", previous)
        object.__setattr__(self, "current_values", current)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)
        object.__setattr__(self, "bridge_coordinate", coordinate)
        object.__setattr__(self, "bridge_variance", variance)
        object.__setattr__(self, "bridge_uniforms", uniforms)


@dataclass(frozen=True)
class ScalarTransitionReducer:
    """Incremental statistic whose update consumes one scalar transition."""

    name: str
    init_fn: Callable[[raw_np.ndarray, int], raw_np.ndarray]
    update_fn: Callable[
        [raw_np.ndarray, ScalarTransitionObservation],
        raw_np.ndarray,
    ]
    finalize_fn: Callable[[raw_np.ndarray], raw_np.ndarray] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validated_reducer_name(self.name))

    def init(self, initial_values: raw_np.ndarray, n_steps: int) -> raw_np.ndarray:
        """Initialize pathwise transition state."""
        return raw_np.asarray(self.init_fn(initial_values, n_steps), dtype=float)

    def update(
        self,
        accumulator: raw_np.ndarray,
        observation: ScalarTransitionObservation,
    ) -> raw_np.ndarray:
        """Advance the statistic with one transition observation."""
        return raw_np.asarray(
            self.update_fn(accumulator, observation),
            dtype=float,
        )

    def finalize(self, accumulator: raw_np.ndarray) -> raw_np.ndarray:
        """Project private accumulator state into its published statistic."""
        current = raw_np.asarray(accumulator, dtype=float)
        result = (
            current
            if self.finalize_fn is None
            else raw_np.asarray(self.finalize_fn(current), dtype=float)
        )
        if current.ndim == 0 or result.ndim == 0 or result.shape[0] != current.shape[0]:
            raise ValueError(
                "finalize_fn must preserve the transition accumulator path axis"
            )
        return result


@dataclass(frozen=True)
class ConditionalBridgeExtremumContract:
    """Selected exact scalar transitions for one conditional extremum."""

    n_steps: int
    transition_steps: tuple[int, ...]
    direction: str = "maximum"
    initial_extremum: float | None = None

    def __post_init__(self) -> None:
        n_steps = _validated_n_steps(self.n_steps)
        transition_steps = _validated_transition_steps(
            self.transition_steps,
            n_steps=n_steps,
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
        object.__setattr__(self, "transition_steps", transition_steps)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "initial_extremum", initial_extremum)


def resolve_scalar_bridge_parameters(
    process: ScalarConditionalBridgeProcess,
    start_time: float,
    end_time: float,
) -> tuple[str, float]:
    """Resolve and validate one process-supplied bridge capability."""
    if not isinstance(process, ScalarConditionalBridgeProcess):
        raise NotImplementedError(
            "transition reducers require a conditional scalar bridge capability"
        )
    coordinate = str(process.conditional_bridge_coordinate or "").strip().lower()
    if not coordinate:
        raise ValueError("process conditional bridge coordinate must be non-empty")
    variance = float(
        process.conditional_bridge_variance(start_time, end_time)
    )
    if not isfinite(variance) or variance < 0.0:
        raise ValueError(
            "process conditional bridge variance must be finite and non-negative"
        )
    return coordinate, variance


def conditional_log_bridge_extremum(
    observation: ScalarTransitionObservation,
    *,
    direction: str,
) -> raw_np.ndarray:
    """Sample a conditional minimum or maximum in the exact log bridge."""
    normalized_direction = str(direction or "").strip().lower()
    if normalized_direction not in {"minimum", "maximum"}:
        raise ValueError("direction must be 'minimum' or 'maximum'")
    if observation.bridge_coordinate != "log":
        raise NotImplementedError(
            "conditional extrema currently require a log bridge coordinate"
        )
    previous_values = _positive_path_cross_section(
        observation.previous_values,
        name="previous_values",
    )
    current_values = _positive_path_cross_section(
        observation.current_values,
        name="current_values",
    )
    if observation.bridge_variance == 0.0:
        if normalized_direction == "minimum":
            return raw_np.minimum(
                previous_values,
                current_values,
            )
        return raw_np.maximum(
            previous_values,
            current_values,
        )
    previous_log = raw_np.log(previous_values)
    current_log = raw_np.log(current_values)
    radicand = (
        (previous_log - current_log) ** 2
        - 2.0
        * observation.bridge_variance
        * raw_np.log1p(-observation.bridge_uniforms)
    )
    root = raw_np.sqrt(raw_np.maximum(radicand, 0.0))
    sign = 1.0 if normalized_direction == "maximum" else -1.0
    return raw_np.exp(0.5 * (previous_log + current_log + sign * root))


def build_conditional_bridge_extremum_reducer(
    contract: ConditionalBridgeExtremumContract,
    *,
    name: str = "conditional_bridge_extremum",
) -> ScalarTransitionReducer:
    """Build bounded transition state for a conditional scalar extremum."""
    reducer_name = _validated_reducer_name(name)
    selected_steps = frozenset(contract.transition_steps)

    def initialize(initial_values, total_steps):
        if int(total_steps) != contract.n_steps:
            raise ValueError(
                f"execution grid has {int(total_steps)} steps; expected {contract.n_steps}"
            )
        initial = _positive_path_cross_section(
            initial_values,
            name="initial_values",
        )
        if contract.initial_extremum is None:
            return initial.copy()
        prior = raw_np.ones_like(initial) * contract.initial_extremum
        if contract.direction == "minimum":
            return raw_np.minimum(prior, initial)
        return raw_np.maximum(prior, initial)

    def update(accumulator, observation):
        current = _path_cross_section(accumulator, name="accumulator")
        if observation.step not in selected_steps:
            return current
        transition_extremum = conditional_log_bridge_extremum(
            observation,
            direction=contract.direction,
        )
        if contract.direction == "minimum":
            return raw_np.minimum(current, transition_extremum)
        return raw_np.maximum(current, transition_extremum)

    return ScalarTransitionReducer(
        name=reducer_name,
        init_fn=initialize,
        update_fn=update,
    )


def _coerce_transition_uniforms(
    reducers: tuple[ScalarTransitionReducer, ...],
    transition_uniforms: raw_np.ndarray,
    *,
    n_paths: int,
    n_steps: int,
) -> raw_np.ndarray:
    if len(reducers) != 1:
        raise NotImplementedError(
            "transition state currently supports one stochastic transition reducer; "
            "a joint auxiliary-randomness law is required for multiple reducers"
        )
    values = raw_np.asarray(transition_uniforms, dtype=float)
    if values.shape != (n_paths, n_steps):
        raise ValueError(
            f"transition_uniforms must have shape ({n_paths}, {n_steps})"
        )
    if raw_np.any(~raw_np.isfinite(values)) or raw_np.any(
        (values <= 0.0) | (values >= 1.0)
    ):
        raise ValueError(
            "transition_uniforms must be strictly between zero and one"
        )
    return values


def replay_scalar_transition_reducers(
    paths: raw_np.ndarray,
    *,
    process: ScalarConditionalBridgeProcess,
    maturity: float,
    reducers: tuple[ScalarTransitionReducer, ...],
    transition_uniforms: raw_np.ndarray,
) -> dict[str, raw_np.ndarray]:
    """Replay transition reducers against a fully materialized scalar path."""
    path_values = raw_np.asarray(paths, dtype=float)
    if path_values.ndim == 3 and path_values.shape[2] == 1:
        path_values = path_values[:, :, 0]
    if path_values.ndim != 2:
        raise ValueError("transition replay requires scalar paths")
    n_paths, columns = path_values.shape
    n_steps = columns - 1
    if n_steps <= 0:
        raise ValueError("transition replay requires at least one path step")
    maturity_value = float(maturity)
    if not isfinite(maturity_value) or maturity_value <= 0.0:
        raise ValueError("maturity must be finite and positive")
    reducer_names = [reducer.name for reducer in reducers]
    if len(reducer_names) != len(set(reducer_names)):
        raise ValueError("transition reducer names must be unique")
    uniforms = _coerce_transition_uniforms(
        reducers,
        transition_uniforms,
        n_paths=n_paths,
        n_steps=n_steps,
    )
    values = {
        reducer.name: reducer.init(path_values[:, 0], n_steps)
        for reducer in reducers
    }
    dt = maturity_value / n_steps
    for step in range(1, n_steps + 1):
        start_time = (step - 1) * dt
        end_time = step * dt
        coordinate, variance = resolve_scalar_bridge_parameters(
            process,
            start_time,
            end_time,
        )
        for reducer in reducers:
            observation = ScalarTransitionObservation(
                previous_values=path_values[:, step - 1],
                current_values=path_values[:, step],
                step=step,
                start_time=start_time,
                end_time=end_time,
                bridge_coordinate=coordinate,
                bridge_variance=variance,
                bridge_uniforms=uniforms[:, step - 1],
            )
            values[reducer.name] = reducer.update(
                values[reducer.name],
                observation,
            )
    return {
        reducer.name: reducer.finalize(values[reducer.name])
        for reducer in reducers
    }


__all__ = [
    "ConditionalBridgeExtremumContract",
    "MonteCarloRandomInputs",
    "ScalarConditionalBridgeProcess",
    "ScalarTransitionObservation",
    "ScalarTransitionReducer",
    "build_conditional_bridge_extremum_reducer",
    "conditional_log_bridge_extremum",
    "replay_scalar_transition_reducers",
    "resolve_scalar_bridge_parameters",
]
