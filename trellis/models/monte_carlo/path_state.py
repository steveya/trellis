"""Data structures for tracking what the Monte Carlo engine records per path.

Instead of storing every time step of every path (expensive), payoffs
can declare exactly what they need: full paths, snapshots at specific
steps, barrier-crossing flags, or custom running statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as raw_np

from trellis.core.differentiable import get_numpy

np = get_numpy()


def _to_backend_array(values):
    if isinstance(values, raw_np.ndarray):
        return values
    if hasattr(values, "_value"):
        return values
    return raw_np.asarray(values, dtype=float)


def _validation_view(values):
    return getattr(values, "_value", values)


def _normalize_steps(steps: tuple[int, ...]) -> tuple[int, ...]:
    """Return sorted, unique, non-negative observation steps."""
    return tuple(sorted({int(step) for step in steps if int(step) >= 0}))


def _materialize_initial_cross_section(
    initial_value,
    n_paths: int,
    dtype,
):
    """Expand the initial state into one pathwise cross-section."""
    del dtype
    initial_array = _to_backend_array(initial_value)
    initial_view = _validation_view(initial_array)
    if initial_view.ndim == 0:
        return np.ones(n_paths) * initial_array
    if initial_view.ndim == 1:
        return np.ones((n_paths, 1)) * initial_array[np.newaxis, :]
    raise ValueError("initial_value must be scalar or one-dimensional")


@dataclass(frozen=True)
class BarrierMonitor:
    """Specification for tracking whether simulated paths cross a price level.

    Attributes:
        name: Identifier for this monitor (e.g. "knock_out").
        level: The barrier price level.
        direction: "up" if the barrier is above spot, "down" if below.
        observation_steps: Which time steps to check (empty = check all steps).
    """

    name: str
    level: float
    direction: str
    observation_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        normalized_steps = _normalize_steps(self.observation_steps)
        if normalized_steps != self.observation_steps:
            object.__setattr__(self, "observation_steps", normalized_steps)


@dataclass(frozen=True)
class PathReducer:
    """A running statistic computed incrementally as paths are simulated.

    Instead of storing the full path, a PathReducer maintains a summary
    (e.g. running max, running average) that is updated at each time step
    via init_fn (called once at step 0) and update_fn (called at each step).
    """

    name: str
    init_fn: Callable[[raw_np.ndarray, int], raw_np.ndarray]
    update_fn: Callable[[raw_np.ndarray, raw_np.ndarray, int], raw_np.ndarray]

    def init(self, initial_values: raw_np.ndarray, n_steps: int) -> raw_np.ndarray:
        """Initialize the reducer accumulator from the starting cross-section."""
        return _to_backend_array(self.init_fn(initial_values, n_steps))

    def update(self, accumulator: raw_np.ndarray, values: raw_np.ndarray, step: int) -> raw_np.ndarray:
        """Advance the accumulator with one new cross-section."""
        return _to_backend_array(self.update_fn(accumulator, values, step))


@dataclass(frozen=True)
class MonteCarloPathRequirement:
    """Declare the minimal path state a payoff needs from the engine."""

    full_path: bool = False
    snapshot_steps: tuple[int, ...] = ()
    barrier_monitors: tuple[BarrierMonitor, ...] = ()
    reducers: tuple[PathReducer, ...] = ()

    def __post_init__(self) -> None:
        normalized_steps = _normalize_steps(self.snapshot_steps)
        if normalized_steps != self.snapshot_steps:
            object.__setattr__(self, "snapshot_steps", normalized_steps)

        monitor_names = [monitor.name for monitor in self.barrier_monitors]
        if len(set(monitor_names)) != len(monitor_names):
            raise ValueError("barrier monitor names must be unique")

        reducer_names = [reducer.name for reducer in self.reducers]
        if len(set(reducer_names)) != len(reducer_names):
            raise ValueError("reducer names must be unique")

    @classmethod
    def full_paths(cls) -> "MonteCarloPathRequirement":
        """Request the full simulated path matrix."""
        return cls(full_path=True)

    @classmethod
    def terminal_only(cls) -> "MonteCarloPathRequirement":
        """Request only the terminal state."""
        return cls()

    @classmethod
    def snapshots(cls, steps: list[int] | tuple[int, ...]) -> "MonteCarloPathRequirement":
        """Request snapshots at selected step indices."""
        return cls(snapshot_steps=tuple(steps))

    @property
    def uses_reduced_storage(self) -> bool:
        """Return whether the requirement can avoid storing the full path matrix."""
        return not self.full_path


@dataclass(frozen=True)
class MonteCarloPathState:
    """Compact state returned by reduced-storage Monte Carlo simulation."""

    initial_value: float | raw_np.ndarray
    n_steps: int
    terminal_values: raw_np.ndarray
    full_paths: raw_np.ndarray | None = None
    snapshots: Mapping[int, raw_np.ndarray] = field(default_factory=dict)
    barrier_hits: Mapping[str, raw_np.ndarray] = field(default_factory=dict)
    reducer_values: Mapping[str, raw_np.ndarray] = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        """Return the number of simulated paths."""
        return int(self.terminal_values.shape[0])

    def snapshot(self, step: int) -> raw_np.ndarray:
        """Return the simulated state at the requested step."""
        step = int(step)
        if step == 0:
            return _materialize_initial_cross_section(
                self.initial_value,
                self.n_paths,
                self.terminal_values.dtype,
            )
        if step == self.n_steps:
            return self.terminal_values
        if self.full_paths is not None:
            return self.full_paths[:, step]
        if step in self.snapshots:
            return self.snapshots[step]
        raise KeyError(f"snapshot for step {step} was not stored")

    def barrier_hit(self, name: str) -> raw_np.ndarray:
        """Return the stored barrier-hit indicator for the named monitor."""
        try:
            return self.barrier_hits[name]
        except KeyError as exc:
            raise KeyError(f"barrier monitor '{name}' was not stored") from exc

    def reduced_value(self, name: str) -> raw_np.ndarray:
        """Return the stored reduced statistic for the named reducer."""
        try:
            return self.reducer_values[name]
        except KeyError as exc:
            raise KeyError(f"reducer '{name}' was not stored") from exc

    def materialize_full_paths(self) -> raw_np.ndarray:
        """Return the full path matrix when it was explicitly stored."""
        if self.full_paths is None:
            raise ValueError("full paths were not stored for this simulation")
        return self.full_paths


@dataclass(frozen=True)
class StateAwarePayoff:
    """Payoff adapter with an explicit path-state contract."""

    path_requirement: MonteCarloPathRequirement
    evaluate_paths_fn: Callable[[raw_np.ndarray], raw_np.ndarray]
    evaluate_state_fn: Callable[[MonteCarloPathState], raw_np.ndarray]
    name: str | None = None
    derivative_metadata: Mapping[str, object] = field(default_factory=dict)

    def __call__(self, paths: raw_np.ndarray) -> raw_np.ndarray:
        """Evaluate the payoff from a full path matrix."""
        return _to_backend_array(self.evaluate_paths_fn(paths))

    def evaluate_state(self, state: MonteCarloPathState) -> raw_np.ndarray:
        """Evaluate the payoff from a reduced path state."""
        return _to_backend_array(self.evaluate_state_fn(state))


def terminal_value_payoff(
    payoff_fn: Callable[[raw_np.ndarray], raw_np.ndarray],
    *,
    name: str | None = None,
) -> StateAwarePayoff:
    """Wrap a terminal-only payoff so the engine can skip full path storage."""

    def evaluate_paths(paths: raw_np.ndarray) -> raw_np.ndarray:
        return _to_backend_array(payoff_fn(paths[:, -1]))

    def evaluate_state(state: MonteCarloPathState) -> raw_np.ndarray:
        return _to_backend_array(payoff_fn(state.terminal_values))

    return StateAwarePayoff(
        path_requirement=MonteCarloPathRequirement.terminal_only(),
        evaluate_paths_fn=evaluate_paths,
        evaluate_state_fn=evaluate_state,
        name=name or getattr(payoff_fn, "__name__", "terminal_value_payoff"),
    )


def barrier_payoff(
    *,
    barrier: float,
    direction: str,
    knock: str,
    terminal_payoff_fn: Callable[[raw_np.ndarray], raw_np.ndarray],
    scale: float = 1.0,
    name: str | None = None,
) -> StateAwarePayoff:
    """Wrap a barrier payoff with a stored barrier-hit statistic."""

    monitor = BarrierMonitor(name="barrier", level=barrier, direction=direction)

    if knock not in {"in", "out"}:
        raise ValueError("knock must be 'in' or 'out'")

    def _barrier_hit_from_paths(paths: raw_np.ndarray) -> raw_np.ndarray:
        axes = tuple(range(1, paths.ndim))
        if direction == "down":
            return raw_np.any(paths <= barrier, axis=axes)
        return raw_np.any(paths >= barrier, axis=axes)

    def _apply_barrier(vanilla: raw_np.ndarray, breached: raw_np.ndarray) -> raw_np.ndarray:
        if knock == "out":
            return raw_np.where(breached, 0.0, vanilla)
        return raw_np.where(breached, vanilla, 0.0)

    def evaluate_paths(paths: raw_np.ndarray) -> raw_np.ndarray:
        vanilla = raw_np.asarray(terminal_payoff_fn(paths[:, -1]), dtype=float)
        return scale * _apply_barrier(vanilla, _barrier_hit_from_paths(paths))

    def evaluate_state(state: MonteCarloPathState) -> raw_np.ndarray:
        vanilla = raw_np.asarray(terminal_payoff_fn(state.terminal_values), dtype=float)
        return scale * _apply_barrier(vanilla, state.barrier_hit(monitor.name))

    return StateAwarePayoff(
        path_requirement=MonteCarloPathRequirement(barrier_monitors=(monitor,)),
        evaluate_paths_fn=evaluate_paths,
        evaluate_state_fn=evaluate_state,
        name=name or "barrier_payoff",
        derivative_metadata={
            "discontinuous_features": ("barrier_monitor",),
            "unsupported_reason": "barrier_monitor_discontinuity",
        },
    )
