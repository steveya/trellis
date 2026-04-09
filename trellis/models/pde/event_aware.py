"""Generic event-aware 1D PDE rollback assembly.

This module provides a bounded runtime substrate for one-dimensional PDE
problems with deterministic event dates and single-controller projections.
It is intentionally lower-level than any product adapter: callers assemble a
problem from typed operator, grid, boundary, terminal, and event specs, then
run a generic backward rollback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as raw_np

from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator, PDEOperator
from trellis.models.pde.rate_operator import HullWhitePDEOperator
from trellis.models.pde.theta_method import _theta_step

EventTransformKind = Literal["add_cashflow", "project_max", "project_min", "state_remap"]
BoundaryPolicy = Literal["none", "linear_extrapolation"]
OperatorFamily = Literal["black_scholes_1d", "local_vol_1d", "hull_white_1f"]


@dataclass(frozen=True)
class EventAwarePDEGridSpec:
    """Typed grid inputs for event-aware PDE assembly."""

    x_min: float
    x_max: float
    n_x: int
    maturity: float
    n_t: int
    log_spacing: bool = False


@dataclass(frozen=True)
class EventAwarePDEOperatorSpec:
    """Typed one-dimensional operator selection."""

    family: str
    sigma: float | None = None
    sigma_fn: Callable | None = None
    r: float | None = None
    r_fn: Callable[[float], float] | None = None
    mean_reversion: float | None = None
    theta_fn: Callable[[float], float] | None = None
    r0: float | None = None


@dataclass(frozen=True)
class EventAwarePDEBoundarySpec:
    """Boundary callables and bounded post-step policies."""

    lower: float | Callable = 0.0
    upper: float | Callable = 0.0
    post_step_policy: BoundaryPolicy = "none"


@dataclass(frozen=True)
class EventAwarePDETransform:
    """Deterministic transform applied at an event time."""

    kind: EventTransformKind
    payload: object = 0.0
    label: str = ""


@dataclass(frozen=True)
class EventAwarePDEEventBucket:
    """Event-time transform bucket in continuous time."""

    time: float
    transforms: tuple[EventAwarePDETransform, ...]
    label: str = ""


@dataclass(frozen=True)
class CompiledEventAwarePDEEventBucket:
    """Step-indexed event bucket used during rollback."""

    step_index: int
    time: float
    transforms: tuple[EventAwarePDETransform, ...]
    label: str = ""


@dataclass(frozen=True)
class EventAwarePDEProblemSpec:
    """High-level problem spec before numerical assembly."""

    grid_spec: EventAwarePDEGridSpec
    operator_spec: EventAwarePDEOperatorSpec
    terminal_condition: raw_np.ndarray | Callable[[raw_np.ndarray], raw_np.ndarray]
    boundary_spec: EventAwarePDEBoundarySpec = EventAwarePDEBoundarySpec()
    event_buckets: tuple[EventAwarePDEEventBucket, ...] = ()
    theta: float = 0.5
    rannacher_timesteps: int = 0


@dataclass(frozen=True)
class EventAwarePDEProblem:
    """Compiled event-aware PDE rollback problem."""

    grid: Grid
    operator: PDEOperator
    terminal_condition: raw_np.ndarray
    boundary_spec: EventAwarePDEBoundarySpec
    event_buckets: tuple[CompiledEventAwarePDEEventBucket, ...]
    theta: float = 0.5
    rannacher_timesteps: int = 0


def build_event_aware_pde_problem(spec: EventAwarePDEProblemSpec) -> EventAwarePDEProblem:
    """Compile a typed event-aware PDE problem into numerical objects."""
    if spec.boundary_spec.post_step_policy not in {"none", "linear_extrapolation"}:
        raise ValueError(
            f"Unsupported boundary post_step_policy {spec.boundary_spec.post_step_policy!r}"
        )
    grid = Grid(
        x_min=float(spec.grid_spec.x_min),
        x_max=float(spec.grid_spec.x_max),
        n_x=max(int(spec.grid_spec.n_x), 5),
        T=max(float(spec.grid_spec.maturity), 0.0),
        n_t=max(int(spec.grid_spec.n_t), 1),
        log_spacing=bool(spec.grid_spec.log_spacing),
    )
    operator = build_event_aware_pde_operator(spec.operator_spec)
    terminal = _coerce_vector(
        _evaluate_callable(spec.terminal_condition, ((grid.x,),), ("terminal_condition",)),
        grid.x,
        name="terminal_condition",
    )
    compiled_buckets = _compile_event_buckets(spec.event_buckets, grid)
    return EventAwarePDEProblem(
        grid=grid,
        operator=operator,
        terminal_condition=terminal.astype(float),
        boundary_spec=spec.boundary_spec,
        event_buckets=compiled_buckets,
        theta=float(spec.theta),
        rannacher_timesteps=max(int(spec.rannacher_timesteps), 0),
    )


def build_event_aware_pde_operator(spec: EventAwarePDEOperatorSpec) -> PDEOperator:
    """Resolve the bounded operator-family plugin surface."""
    family = str(spec.family)
    if family == "black_scholes_1d":
        sigma_fn = _resolve_sigma_fn(spec)
        r_fn = _resolve_rate_fn(spec)
        return BlackScholesOperator(sigma_fn=sigma_fn, r_fn=r_fn)
    if family == "local_vol_1d":
        sigma_fn = _resolve_sigma_fn(spec)
        r_fn = _resolve_rate_fn(spec)
        return BlackScholesOperator(sigma_fn=sigma_fn, r_fn=r_fn)
    if family == "hull_white_1f":
        if spec.sigma is None:
            raise ValueError("hull_white_1f operator requires sigma")
        if spec.mean_reversion is None:
            raise ValueError("hull_white_1f operator requires mean_reversion")
        return HullWhitePDEOperator(
            sigma=float(spec.sigma),
            a=float(spec.mean_reversion),
            theta_fn=spec.theta_fn,
            r0=float(spec.r0) if spec.r0 is not None else 0.05,
        )
    raise ValueError(f"Unsupported PDE operator family {family!r}")


def solve_event_aware_pde(problem: EventAwarePDEProblem) -> raw_np.ndarray:
    """Run the generic one-dimensional event-aware rollback."""
    grid = problem.grid
    values = problem.terminal_condition.copy().astype(float)
    n_int = grid.n_x - 2
    event_buckets_by_step: dict[int, list[CompiledEventAwarePDEEventBucket]] = {}
    for bucket in problem.event_buckets:
        event_buckets_by_step.setdefault(bucket.step_index, []).append(bucket)

    for bucket in event_buckets_by_step.get(grid.n_t, ()):
        values = apply_event_bucket(values, grid.x, grid.T, bucket)

    for step in range(grid.n_t - 1, -1, -1):
        t = step * grid.dt
        steps_from_maturity = grid.n_t - step
        step_theta = (
            1.0
            if 0 < problem.rannacher_timesteps and steps_from_maturity <= problem.rannacher_timesteps
            else problem.theta
        )
        a_coeff, b_coeff, c_coeff = problem.operator.coefficients(grid.x, t, grid.dt)
        lower = evaluate_boundary_value(problem.boundary_spec.lower, t, values, grid)
        upper = evaluate_boundary_value(problem.boundary_spec.upper, t, values, grid)
        values = _theta_step(values, a_coeff, b_coeff, c_coeff, step_theta, lower, upper, n_int)
        if problem.boundary_spec.post_step_policy == "linear_extrapolation":
            values = _apply_linear_extrapolation(values)
        for bucket in event_buckets_by_step.get(step, ()):
            values = apply_event_bucket(values, grid.x, t, bucket)
    return values


def interpolate_pde_values(values: raw_np.ndarray, x_grid: raw_np.ndarray, x0: float) -> float:
    """Linearly interpolate the rollback solution at a point."""
    idx = raw_np.searchsorted(x_grid, x0)
    idx = max(1, min(int(idx), len(x_grid) - 1))
    weight = float((x0 - x_grid[idx - 1]) / (x_grid[idx] - x_grid[idx - 1]))
    return float(values[idx - 1] * (1.0 - weight) + values[idx] * weight)


def apply_event_bucket(
    values: raw_np.ndarray,
    x_grid: raw_np.ndarray,
    t: float,
    bucket: EventAwarePDEEventBucket | CompiledEventAwarePDEEventBucket,
) -> raw_np.ndarray:
    """Apply a transform bucket in the declared order."""
    updated = raw_np.asarray(values, dtype=float).copy()
    for transform in bucket.transforms:
        updated = apply_event_transform(updated, x_grid, t, transform)
    return updated


def apply_event_transform(
    values: raw_np.ndarray,
    x_grid: raw_np.ndarray,
    t: float,
    transform: EventAwarePDETransform,
) -> raw_np.ndarray:
    """Apply a single event transform."""
    current = raw_np.asarray(values, dtype=float)
    if transform.kind == "add_cashflow":
        return current + _evaluate_transform_payload(transform.payload, x_grid, t, current, "add_cashflow")
    if transform.kind == "project_max":
        return raw_np.maximum(current, _evaluate_transform_payload(transform.payload, x_grid, t, current, "project_max"))
    if transform.kind == "project_min":
        return raw_np.minimum(current, _evaluate_transform_payload(transform.payload, x_grid, t, current, "project_min"))
    if transform.kind == "state_remap":
        remapped = _evaluate_callable(
            transform.payload,
            (
                (x_grid, current, t),
                (x_grid, t, current),
                (x_grid, current),
                (x_grid,),
            ),
            ("state_remap",),
        )
        return _coerce_vector(remapped, x_grid, name="state_remap")
    raise ValueError(f"Unsupported event transform kind {transform.kind!r}")


def evaluate_boundary_value(boundary: float | Callable, t: float, values: raw_np.ndarray, grid: Grid) -> float:
    """Evaluate a bounded boundary callable or scalar."""
    if callable(boundary):
        result = _evaluate_callable(
            boundary,
            (
                (t, values, grid),
                (t, values),
                (t, grid),
                (t,),
                (),
            ),
            ("boundary",),
        )
        return float(result)
    return float(boundary)


def _apply_linear_extrapolation(values: raw_np.ndarray) -> raw_np.ndarray:
    updated = raw_np.asarray(values, dtype=float).copy()
    if len(updated) >= 3:
        updated[0] = 2.0 * updated[1] - updated[2]
        updated[-1] = 2.0 * updated[-2] - updated[-3]
    return updated


def _compile_event_buckets(
    buckets: tuple[EventAwarePDEEventBucket, ...],
    grid: Grid,
) -> tuple[CompiledEventAwarePDEEventBucket, ...]:
    compiled: list[CompiledEventAwarePDEEventBucket] = []
    supported_kinds = {"add_cashflow", "project_max", "project_min", "state_remap"}
    for bucket in buckets:
        time = float(bucket.time)
        if time < 0.0 or time > grid.T + 1e-12:
            raise ValueError(f"Event time {time} lies outside PDE horizon [0, {grid.T}]")
        step_index = int(round(time / grid.dt))
        if step_index < 0 or step_index > grid.n_t:
            raise ValueError(f"Event time {time} resolves to invalid step {step_index}")
        for transform in bucket.transforms:
            if transform.kind not in supported_kinds:
                raise ValueError(f"Unsupported event transform kind {transform.kind!r}")
        compiled.append(
            CompiledEventAwarePDEEventBucket(
                step_index=step_index,
                time=time,
                transforms=tuple(bucket.transforms),
                label=bucket.label,
            )
        )
    compiled.sort(key=lambda bucket: (bucket.step_index, bucket.time))
    return tuple(compiled)


def _resolve_rate_fn(spec: EventAwarePDEOperatorSpec) -> Callable[[float], float]:
    if spec.r_fn is not None:
        return spec.r_fn
    if spec.r is None:
        raise ValueError(f"{spec.family} operator requires r or r_fn")
    rate = float(spec.r)
    return lambda _t: rate


def _resolve_sigma_fn(spec: EventAwarePDEOperatorSpec) -> Callable:
    if spec.sigma_fn is not None:
        return spec.sigma_fn
    if spec.sigma is None:
        raise ValueError(f"{spec.family} operator requires sigma or sigma_fn")
    sigma = float(spec.sigma)
    return lambda _x, _t: sigma


def _evaluate_transform_payload(
    payload: object,
    x_grid: raw_np.ndarray,
    t: float,
    values: raw_np.ndarray,
    name: str,
) -> raw_np.ndarray:
    if callable(payload):
        result = _evaluate_callable(
            payload,
            (
                (x_grid, values, t),
                (x_grid, t, values),
                (x_grid, t),
                (x_grid,),
                (t,),
                (),
            ),
            (name,),
        )
    else:
        result = payload
    return _coerce_vector(result, x_grid, name=name)


def _evaluate_callable(fn: object, arg_sets, names: tuple[str, ...]):
    if not callable(fn):
        return fn
    last_error: TypeError | None = None
    for args in arg_sets:
        try:
            return fn(*args)
        except TypeError as exc:
            last_error = exc
            continue
    readable = "/".join(names)
    raise TypeError(f"Callable payload for {readable} does not match any supported signature") from last_error


def _coerce_vector(value: object, x_grid: raw_np.ndarray, *, name: str) -> raw_np.ndarray:
    array = raw_np.asarray(value, dtype=float)
    if array.ndim == 0:
        return raw_np.full(x_grid.shape, float(array), dtype=float)
    if array.shape != x_grid.shape:
        raise ValueError(f"{name} payload must have shape {x_grid.shape}, got {array.shape}")
    return array.astype(float)
