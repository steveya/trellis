"""Reusable path-event state helpers for path-dependent Monte Carlo routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement

np = get_numpy()


def _to_backend_array(values):
    if hasattr(values, "_value"):
        return values
    return np.asarray(values)


def _validation_view(values):
    return getattr(values, "_value", values)


def _maybe_copy_array(values):
    copier = getattr(values, "copy", None)
    return copier() if callable(copier) else values


def event_step_indices(
    event_times: tuple[float, ...],
    T: float,
    n_steps: int,
) -> tuple[int, ...]:
    """Map event times onto deterministic Monte Carlo step indices."""
    if T <= 0.0 or n_steps <= 0:
        return (0,)
    if not event_times:
        return (n_steps,)

    scaled = [
        int(np.clip(np.rint((float(time) / float(T)) * n_steps), 0, n_steps))
        for time in event_times
    ]
    return tuple(dict.fromkeys(sorted(scaled)))


def build_event_path_requirement(
    event_times: tuple[float, ...],
    T: float,
    n_steps: int,
) -> MonteCarloPathRequirement:
    """Request the snapshot steps needed by a path-event replay."""
    return MonteCarloPathRequirement(
        snapshot_steps=event_step_indices(event_times, T, n_steps),
    )


def _normalize_rule(value: str | None, default: str) -> str:
    rule = (value or default).strip().lower().replace("-", "_").replace(" ", "_")
    return rule or default


def _normalize_kind(value: str) -> str:
    kind = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not kind:
        raise ValueError("event kind must be non-empty")
    return kind


def _normalize_payload(payload: Mapping[str, object] | None) -> Mapping[str, object]:
    if payload is None:
        payload = {}
    normalized = {str(key): value for key, value in dict(payload).items()}
    return MappingProxyType(dict(sorted(normalized.items(), key=lambda item: item[0])))


def _event_sort_key(event: "PathEventSpec") -> tuple[int, int, str, str]:
    return (int(event.step), int(event.priority), event.kind, event.name)


def _coerce_observation_matrix(cross_section) -> np.ndarray:
    values = _to_backend_array(cross_section)
    values_view = _validation_view(values)
    if values_view.ndim == 1:
        values = values[:, np.newaxis]
        values_view = _validation_view(values)
    if values_view.ndim != 2:
        raise ValueError(
            f"Observation cross-sections must have rank 1 or 2; received shape {values_view.shape}."
        )
    return values


def _coerce_event_vector(cross_section) -> np.ndarray:
    values = _to_backend_array(cross_section)
    values_view = _validation_view(values)
    if values_view.ndim == 1:
        return values
    if values_view.ndim == 2 and values_view.shape[1] == 1:
        return values[:, 0]
    raise ValueError(
        f"Event cross-sections must be scalar-valued or have one column; received shape {values_view.shape}."
    )


def _coerce_reference_vector(initial_values, n_assets: int) -> np.ndarray:
    reference = _to_backend_array(initial_values)
    reference_view = _validation_view(reference)
    if reference_view.ndim == 0:
        if n_assets != 1:
            raise ValueError(
                f"Scalar reference values can only be used for one-dimensional events; got {n_assets} assets."
            )
        return np.ones(1) * reference
    if reference_view.shape != (n_assets,):
        raise ValueError(
            f"Reference values must have shape ({n_assets},); got {reference_view.shape}."
        )
    return reference


def _select_remaining_constituent(
    step_returns: np.ndarray,
    remaining_mask: np.ndarray,
    *,
    worst_rule: bool,
):
    """Return the selected constituent index and simple return for each path."""
    masked = np.where(remaining_mask, step_returns, np.inf if worst_rule else -np.inf)
    selected_indices = np.argmin(masked, axis=1) if worst_rule else np.argmax(masked, axis=1)
    selected_returns = step_returns[np.arange(step_returns.shape[0]), selected_indices]
    return selected_indices, selected_returns


def _aggregate_locked_returns(
    accumulated: np.ndarray,
    selected_counts: np.ndarray,
    aggregation_rule: str,
) -> np.ndarray:
    """Aggregate locked simple returns into the maturity settlement payoff."""
    normalized_rule = _normalize_rule(aggregation_rule, "average_locked_returns")
    if normalized_rule in {"average_locked_returns", "average_selected_returns"}:
        denominator = np.where(selected_counts > 0, selected_counts, 1.0)
        return accumulated / denominator
    if normalized_rule == "sum_locked_returns":
        return accumulated
    raise ValueError(
        f"Unsupported aggregation_rule {aggregation_rule!r}; expected average_locked_returns or sum_locked_returns"
    )


def _validate_selection_rule(selection_rule: str) -> bool:
    normalized = _normalize_rule(selection_rule, "best_of_remaining")
    if normalized in {"best_of_remaining", "best_of", "best", "max_remaining", "max"}:
        return False
    if normalized in {"worst_of_remaining", "worst_of", "worst", "min_remaining", "min"}:
        return True
    raise ValueError(
        f"Unsupported selection_rule {selection_rule!r}; expected best_of_remaining or worst_of_remaining"
    )


def _is_worst_rule(selection_rule: str) -> bool:
    normalized = _normalize_rule(selection_rule, "best_of_remaining")
    return normalized in {"worst_of_remaining", "worst_of", "worst", "min_remaining", "min"}


def _validate_direction(direction: str | None, default: str) -> str:
    normalized = _normalize_rule(direction, default)
    if normalized not in {"up", "down"}:
        raise ValueError(
            f"Unsupported direction {direction!r}; expected up or down"
        )
    return normalized


@dataclass(frozen=True)
class PathEventSpec:
    """A deterministic path-dependent event to replay against Monte Carlo state."""

    name: str
    kind: str
    step: int
    priority: int = 0
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "priority", int(self.priority))
        if self.step < 0:
            raise ValueError("event step must be non-negative")
        object.__setattr__(self, "payload", _normalize_payload(self.payload))

    def get(self, key: str, default=None):
        """Return a payload value with a default."""
        return self.payload.get(key, default)


@dataclass(frozen=True)
class PathEventRecord:
    """Frozen replay record produced by the path-event substrate."""

    name: str
    kind: str
    step: int
    priority: int = 0
    payload: Mapping[str, object] = field(default_factory=dict)
    selected_indices: np.ndarray | None = None
    selected_values: np.ndarray | None = None
    barrier_hit: np.ndarray | None = None
    exercise_triggered: np.ndarray | None = None
    exercise_value: np.ndarray | None = None
    coupon_cashflow: np.ndarray | None = None
    settlement_value: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "priority", int(self.priority))
        if self.step < 0:
            raise ValueError("event step must be non-negative")
        object.__setattr__(self, "payload", _normalize_payload(self.payload))


@dataclass(frozen=True)
class PathEventTimeline:
    """Ordered path-event sequence for deterministic replay."""

    events: tuple[PathEventSpec, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.events, key=_event_sort_key))
        names = [event.name for event in ordered]
        if len(set(names)) != len(names):
            raise ValueError("path-event names must be unique")
        object.__setattr__(self, "events", ordered)

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def steps(self) -> tuple[int, ...]:
        """Return the ordered replay steps for the timeline."""
        return tuple(event.step for event in self.events)


@dataclass(frozen=True)
class PathEventState:
    """Deterministic replay state for path-dependent derivative events."""

    initial_values: np.ndarray | float
    n_paths: int
    n_steps: int
    records: tuple[PathEventRecord, ...] = ()
    remaining_mask: np.ndarray | None = None
    locked_returns: np.ndarray | None = None
    selected_counts: np.ndarray | None = None
    selected_indices: Mapping[str, np.ndarray] = field(default_factory=dict)
    selected_values: Mapping[str, np.ndarray] = field(default_factory=dict)
    barrier_hits: Mapping[str, np.ndarray] = field(default_factory=dict)
    exercise_flags: Mapping[str, np.ndarray] = field(default_factory=dict)
    exercise_values: Mapping[str, np.ndarray] = field(default_factory=dict)
    coupon_cashflows: Mapping[str, np.ndarray] = field(default_factory=dict)
    settlement_values: Mapping[str, np.ndarray] = field(default_factory=dict)
    reducer_values: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def event_count(self) -> int:
        """Return the number of replayed events."""
        return len(self.records)

    def barrier_hit(self, name: str) -> np.ndarray:
        """Return the stored barrier indicator for the named event."""
        try:
            return self.barrier_hits[name]
        except KeyError as exc:
            raise KeyError(f"barrier event '{name}' was not stored") from exc

    def exercise_triggered(self, name: str) -> np.ndarray:
        """Return the stored exercise trigger indicator for the named event."""
        try:
            return self.exercise_flags[name]
        except KeyError as exc:
            raise KeyError(f"exercise event '{name}' was not stored") from exc

    def settlement_value(self, name: str) -> np.ndarray:
        """Return the stored settlement value for the named event."""
        try:
            return self.settlement_values[name]
        except KeyError as exc:
            raise KeyError(f"settlement event '{name}' was not stored") from exc

    def coupon_cashflow(self, name: str) -> np.ndarray:
        """Return the stored coupon cashflow for the named event."""
        try:
            return self.coupon_cashflows[name]
        except KeyError as exc:
            raise KeyError(f"coupon event '{name}' was not stored") from exc

    def reduced_value(self, name: str) -> np.ndarray:
        """Return the stored reduced path statistic for the named reducer."""
        try:
            return self.reducer_values[name]
        except KeyError as exc:
            raise KeyError(f"reducer '{name}' was not stored") from exc


def replay_path_event_timeline(
    cross_sections: Sequence[object],
    initial_values,
    event_timeline: PathEventTimeline | Sequence[PathEventSpec],
    reducer_values: Mapping[str, object] | None = None,
) -> PathEventState:
    """Replay a path-event timeline against a sequence of deterministic cross-sections."""
    specs = tuple(event_timeline.events) if isinstance(event_timeline, PathEventTimeline) else tuple(event_timeline)
    if not specs:
        raise ValueError("event_timeline must contain at least one event")
    if len(cross_sections) != len(specs):
        raise ValueError("cross_sections must align one-to-one with event_timeline")

    ordered_pairs = sorted(zip(specs, cross_sections), key=lambda item: _event_sort_key(item[0]))
    first_section = _to_backend_array(ordered_pairs[0][1])
    n_paths = int(_validation_view(first_section).shape[0])
    n_steps = max(event.step for event in specs)
    state = PathEventState(
        initial_values=_maybe_copy_array(_to_backend_array(initial_values)),
        n_paths=n_paths,
        n_steps=n_steps,
        reducer_values={
            str(name): _maybe_copy_array(_to_backend_array(values))
            for name, values in dict(reducer_values or {}).items()
        },
    )

    for spec, cross_section in ordered_pairs:
        state = apply_path_event_spec(state, spec, cross_section)

    return state


def apply_path_event_spec(
    state: PathEventState,
    spec: PathEventSpec,
    cross_section,
) -> PathEventState:
    """Advance a path-event state by one event specification."""
    kind = _normalize_kind(spec.kind)
    if kind == "observation":
        return _apply_observation_event(state, spec, cross_section)
    if kind == "barrier":
        return _apply_barrier_event(state, spec, cross_section)
    if kind == "coupon":
        return _apply_coupon_event(state, spec, cross_section)
    if kind == "exercise":
        return _apply_exercise_event(state, spec, cross_section)
    if kind == "settlement":
        return _apply_settlement_event(state, spec, cross_section)
    raise ValueError(
        f"Unsupported path event kind {spec.kind!r}; expected observation, barrier, coupon, exercise, or settlement"
    )


def _apply_observation_event(
    state: PathEventState,
    spec: PathEventSpec,
    cross_section,
) -> PathEventState:
    step_values = _coerce_observation_matrix(cross_section)
    n_paths, n_assets = step_values.shape
    reference = _coerce_reference_vector(state.initial_values, n_assets)

    remaining_mask = (
        np.ones((n_paths, n_assets), dtype=bool)
        if state.remaining_mask is None
        else np.asarray(state.remaining_mask, dtype=bool).copy()
    )
    if remaining_mask.shape != (n_paths, n_assets):
        raise ValueError(
            f"Observation state must have remaining_mask shape {(n_paths, n_assets)}; got {remaining_mask.shape}."
        )

    locked_returns = (
        np.zeros(n_paths, dtype=float)
        if state.locked_returns is None
        else np.asarray(state.locked_returns, dtype=float).copy()
    )
    selected_counts = (
        np.zeros(n_paths, dtype=float)
        if state.selected_counts is None
        else np.asarray(state.selected_counts, dtype=float).copy()
    )

    step_returns = step_values / reference[None, :] - 1.0
    selection_rule = _normalize_rule(spec.payload.get("selection_rule"), "best_of_remaining")
    lock_rule = _normalize_rule(spec.payload.get("lock_rule"), "remove_selected")
    if lock_rule != "remove_selected":
        raise ValueError(
            f"Unsupported lock_rule {lock_rule!r}; expected remove_selected"
        )
    selection_count = max(int(spec.payload.get("selection_count", 1)), 1)
    worst_rule = _validate_selection_rule(selection_rule)

    selected_indices = None
    selected_values = None
    for _ in range(selection_count):
        active = remaining_mask.any(axis=1)
        if not active.any():
            break
        selected_indices, selected_values = _select_remaining_constituent(
            step_returns,
            remaining_mask,
            worst_rule=worst_rule,
        )
        selected_values = np.where(active, selected_values, 0.0)
        locked_returns += selected_values
        selected_counts += active.astype(float)
        rows = np.where(active)[0]
        remaining_mask[rows, selected_indices[active]] = False

    records = state.records + (
        PathEventRecord(
            name=spec.name,
            kind=spec.kind,
            step=spec.step,
            priority=spec.priority,
            payload=spec.payload,
            selected_indices=None if selected_indices is None else np.asarray(selected_indices, dtype=int).copy(),
            selected_values=None if selected_values is None else _maybe_copy_array(selected_values),
        ),
    )
    selected_indices_map = dict(state.selected_indices)
    selected_values_map = dict(state.selected_values)
    if selected_indices is not None and selected_values is not None:
        selected_indices_map[spec.name] = np.asarray(selected_indices, dtype=int).copy()
        selected_values_map[spec.name] = _maybe_copy_array(selected_values)

    return PathEventState(
        initial_values=state.initial_values,
        n_paths=state.n_paths,
        n_steps=max(state.n_steps, spec.step),
        records=records,
        remaining_mask=remaining_mask,
        locked_returns=locked_returns,
        selected_counts=selected_counts,
        selected_indices=selected_indices_map,
        selected_values=selected_values_map,
        barrier_hits=dict(state.barrier_hits),
        exercise_flags=dict(state.exercise_flags),
        exercise_values=dict(state.exercise_values),
        coupon_cashflows=dict(state.coupon_cashflows),
        settlement_values=dict(state.settlement_values),
        reducer_values=dict(state.reducer_values),
    )


def _apply_barrier_event(
    state: PathEventState,
    spec: PathEventSpec,
    cross_section,
) -> PathEventState:
    values = _coerce_event_vector(cross_section)
    direction = _validate_direction(spec.payload.get("direction"), "down")
    level = float(spec.payload["level"])
    hit = values <= level if direction == "down" else values >= level
    hit = np.asarray(hit, dtype=bool)

    barrier_hits = dict(state.barrier_hits)
    existing = barrier_hits.get(spec.name)
    barrier_hits[spec.name] = hit if existing is None else (np.asarray(existing, dtype=bool) | hit)

    records = state.records + (
        PathEventRecord(
            name=spec.name,
            kind=spec.kind,
            step=spec.step,
            priority=spec.priority,
            payload=spec.payload,
            barrier_hit=_maybe_copy_array(hit),
        ),
    )

    return PathEventState(
        initial_values=state.initial_values,
        n_paths=state.n_paths,
        n_steps=max(state.n_steps, spec.step),
        records=records,
        remaining_mask=state.remaining_mask,
        locked_returns=state.locked_returns,
        selected_counts=state.selected_counts,
        selected_indices=dict(state.selected_indices),
        selected_values=dict(state.selected_values),
        barrier_hits=barrier_hits,
        exercise_flags=dict(state.exercise_flags),
        exercise_values=dict(state.exercise_values),
        coupon_cashflows=dict(state.coupon_cashflows),
        settlement_values=dict(state.settlement_values),
        reducer_values=dict(state.reducer_values),
    )


def _apply_coupon_event(
    state: PathEventState,
    spec: PathEventSpec,
    cross_section,
) -> PathEventState:
    _ = _coerce_event_vector(cross_section)
    amount = spec.payload.get("coupon_amount", spec.payload.get("amount"))
    if amount is None:
        coupon_rate = float(spec.payload.get("coupon_rate", 0.0))
        notional = float(spec.payload.get("notional", 1.0))
        amount_value = coupon_rate * notional
    else:
        amount_value = float(amount)
    cashflow = np.full(state.n_paths, amount_value, dtype=float)

    coupon_cashflows = dict(state.coupon_cashflows)
    existing = coupon_cashflows.get(spec.name)
    coupon_cashflows[spec.name] = cashflow if existing is None else (np.asarray(existing, dtype=float) + cashflow)

    records = state.records + (
        PathEventRecord(
            name=spec.name,
            kind=spec.kind,
            step=spec.step,
            priority=spec.priority,
            payload=spec.payload,
            coupon_cashflow=cashflow.copy(),
        ),
    )

    return PathEventState(
        initial_values=state.initial_values,
        n_paths=state.n_paths,
        n_steps=max(state.n_steps, spec.step),
        records=records,
        remaining_mask=state.remaining_mask,
        locked_returns=state.locked_returns,
        selected_counts=state.selected_counts,
        selected_indices=dict(state.selected_indices),
        selected_values=dict(state.selected_values),
        barrier_hits=dict(state.barrier_hits),
        exercise_flags=dict(state.exercise_flags),
        exercise_values=dict(state.exercise_values),
        coupon_cashflows=coupon_cashflows,
        settlement_values=dict(state.settlement_values),
        reducer_values=dict(state.reducer_values),
    )


def _apply_exercise_event(
    state: PathEventState,
    spec: PathEventSpec,
    cross_section,
) -> PathEventState:
    values = _coerce_event_vector(cross_section)
    rule = _normalize_rule(spec.payload.get("exercise_rule", spec.payload.get("rule")), "holder_put")
    direction = _validate_direction(
        spec.payload.get("direction"),
        "up" if rule in {"issuer_call", "call", "callable"} else "down",
    )
    threshold = spec.payload.get("threshold", spec.payload.get("trigger_level"))
    if threshold is None:
        raise ValueError("exercise events require a threshold or trigger_level")
    threshold_value = float(threshold)
    exercise_value = float(spec.payload.get("exercise_value", threshold_value))
    triggered = values >= threshold_value if direction == "up" else values <= threshold_value
    triggered = np.asarray(triggered, dtype=bool)
    exercise_values = np.full(state.n_paths, exercise_value, dtype=float)

    exercise_flags = dict(state.exercise_flags)
    existing_flags = exercise_flags.get(spec.name)
    exercise_flags[spec.name] = triggered if existing_flags is None else (np.asarray(existing_flags, dtype=bool) | triggered)

    exercise_values_map = dict(state.exercise_values)
    exercise_values_map[spec.name] = exercise_values

    records = state.records + (
        PathEventRecord(
            name=spec.name,
            kind=spec.kind,
            step=spec.step,
            priority=spec.priority,
            payload=spec.payload,
            exercise_triggered=_maybe_copy_array(triggered),
            exercise_value=_maybe_copy_array(exercise_values),
        ),
    )

    return PathEventState(
        initial_values=state.initial_values,
        n_paths=state.n_paths,
        n_steps=max(state.n_steps, spec.step),
        records=records,
        remaining_mask=state.remaining_mask,
        locked_returns=state.locked_returns,
        selected_counts=state.selected_counts,
        selected_indices=dict(state.selected_indices),
        selected_values=dict(state.selected_values),
        barrier_hits=dict(state.barrier_hits),
        exercise_flags=exercise_flags,
        exercise_values=exercise_values_map,
        coupon_cashflows=dict(state.coupon_cashflows),
        settlement_values=dict(state.settlement_values),
        reducer_values=dict(state.reducer_values),
    )


def _apply_settlement_event(
    state: PathEventState,
    spec: PathEventSpec,
    cross_section,
) -> PathEventState:
    rule = _normalize_rule(spec.payload.get("rule"), "terminal_value")
    terminal_values = None
    if rule not in {"average_locked_returns", "average_selected_returns", "sum_locked_returns"}:
        terminal_values = _coerce_event_vector(cross_section)

    if rule in {"average_locked_returns", "average_selected_returns"}:
        locked_returns = (
            np.zeros(state.n_paths, dtype=float)
            if state.locked_returns is None
            else _to_backend_array(state.locked_returns)
        )
        selected_counts = (
            np.zeros(state.n_paths, dtype=float)
            if state.selected_counts is None
            else _to_backend_array(state.selected_counts)
        )
        settlement = _aggregate_locked_returns(locked_returns, selected_counts, rule)
    elif rule == "sum_locked_returns":
        settlement = (
            np.zeros(state.n_paths, dtype=float)
            if state.locked_returns is None
            else np.asarray(state.locked_returns, dtype=float)
        )
    elif rule == "knock_out_terminal":
        barrier_event = str(spec.payload.get("barrier_event", spec.name))
        hit = state.barrier_hits.get(barrier_event)
        if hit is None:
            raise KeyError(f"settlement event {spec.name!r} requires barrier_event {barrier_event!r}")
        settlement = np.where(np.asarray(hit, dtype=bool), 0.0, terminal_values)
    elif rule == "exercise_or_terminal":
        exercise_event = str(spec.payload.get("exercise_event", spec.name))
        triggered = state.exercise_flags.get(exercise_event)
        exercise_values = state.exercise_values.get(exercise_event)
        if triggered is None or exercise_values is None:
            raise KeyError(f"settlement event {spec.name!r} requires exercise_event {exercise_event!r}")
        settlement = np.where(
            np.asarray(triggered, dtype=bool),
            _to_backend_array(exercise_values),
            terminal_values,
        )
    elif rule == "discounted_swap_pv":
        settlement = _discounted_swap_pv_settlement(
            state,
            spec,
            current_short_rate=_to_backend_array(terminal_values),
        )
    elif rule == "terminal_value":
        settlement = terminal_values
    else:
        raise ValueError(
            f"Unsupported settlement rule {rule!r}; expected average_locked_returns, sum_locked_returns, terminal_value, knock_out_terminal, exercise_or_terminal, or discounted_swap_pv"
        )

    coupon_events = spec.payload.get("coupon_events")
    if coupon_events is None:
        coupon_total = np.zeros(state.n_paths, dtype=float)
        for value in state.coupon_cashflows.values():
            coupon_total = coupon_total + _to_backend_array(value)
    else:
        if isinstance(coupon_events, str):
            coupon_event_names = (coupon_events,)
        else:
            coupon_event_names = tuple(coupon_events)
        coupon_total = np.zeros(state.n_paths, dtype=float)
        for event_name in coupon_event_names:
            if event_name not in state.coupon_cashflows:
                raise KeyError(f"settlement event {spec.name!r} requires coupon_event {event_name!r}")
            coupon_total = coupon_total + _to_backend_array(state.coupon_cashflows[event_name])

    settlement = _to_backend_array(settlement) + coupon_total

    settlement_values = dict(state.settlement_values)
    settlement_values[spec.name] = settlement
    records = state.records + (
        PathEventRecord(
            name=spec.name,
            kind=spec.kind,
            step=spec.step,
            priority=spec.priority,
            payload=spec.payload,
            settlement_value=_maybe_copy_array(settlement),
        ),
    )

    return PathEventState(
        initial_values=state.initial_values,
        n_paths=state.n_paths,
        n_steps=max(state.n_steps, spec.step),
        records=records,
        remaining_mask=state.remaining_mask,
        locked_returns=state.locked_returns,
        selected_counts=state.selected_counts,
        selected_indices=dict(state.selected_indices),
        selected_values=dict(state.selected_values),
        barrier_hits=dict(state.barrier_hits),
        exercise_flags=dict(state.exercise_flags),
        exercise_values=dict(state.exercise_values),
        coupon_cashflows=dict(state.coupon_cashflows),
        settlement_values=settlement_values,
        reducer_values=dict(state.reducer_values),
    )


def _discounted_swap_pv_settlement(
    state: PathEventState,
    spec: PathEventSpec,
    *,
    current_short_rate: np.ndarray,
) -> np.ndarray:
    """Return one discounted European swap payoff from a short-rate cross-section."""
    payment_times = np.asarray(spec.payload.get("payment_times", ()), dtype=float)
    accrual_fractions = np.asarray(spec.payload.get("accrual_fractions", ()), dtype=float)
    anchor_discount_factors = np.asarray(spec.payload.get("anchor_discount_factors", ()), dtype=float)
    if payment_times.size == 0 or accrual_fractions.size == 0 or anchor_discount_factors.size == 0:
        raise ValueError(
            f"settlement event {spec.name!r} with rule discounted_swap_pv requires payment_times, accrual_fractions, and anchor_discount_factors"
        )
    if not (
        payment_times.shape == accrual_fractions.shape
        and payment_times.shape == anchor_discount_factors.shape
    ):
        raise ValueError(
            f"settlement event {spec.name!r} requires payment_times, accrual_fractions, and anchor_discount_factors with matching shapes"
        )

    reducer_name = str(spec.payload.get("discount_reducer_name", "")).strip()
    if not reducer_name:
        raise ValueError(
            f"settlement event {spec.name!r} with rule discounted_swap_pv requires discount_reducer_name"
        )
    if reducer_name not in state.reducer_values:
        raise KeyError(
            f"settlement event {spec.name!r} requires reducer {reducer_name!r}"
        )

    exercise_time = max(float(spec.payload.get("exercise_time", 0.0)), 1e-12)
    anchor_discount_to_exercise = max(
        float(spec.payload.get("anchor_discount_to_exercise", 1.0)),
        1e-12,
    )
    mean_reversion = float(spec.payload.get("mean_reversion", 0.1))
    anchor_short_rate = float(
        spec.payload.get(
            "anchor_short_rate",
            -np.log(anchor_discount_to_exercise) / exercise_time,
        )
    )
    curve_basis_spread = float(spec.payload.get("curve_basis_spread", 0.0))
    strike = float(spec.payload.get("strike", 0.0))
    notional = float(spec.payload.get("notional", 1.0))
    payer_sign = 1.0 if bool(spec.payload.get("is_payer", True)) else -1.0
    discount_to_exercise = _to_backend_array(state.reducer_values[reducer_name])

    tau = np.maximum(payment_times - exercise_time, 0.0)
    if abs(mean_reversion) < 1e-12:
        B = tau
    else:
        B = (1.0 - np.exp(-mean_reversion * tau)) / mean_reversion

    anchor_ratio = anchor_discount_factors / anchor_discount_to_exercise
    short_rate = _to_backend_array(current_short_rate)[:, np.newaxis]
    bond_prices = anchor_ratio[np.newaxis, :] * np.exp(
        -B[np.newaxis, :] * (short_rate - anchor_short_rate)
    )
    annuity = np.sum(accrual_fractions[np.newaxis, :] * bond_prices, axis=1)
    forward_swap_rate = np.where(
        annuity > 1e-12,
        (1.0 - bond_prices[:, -1]) / annuity,
        0.0,
    )
    adjusted_forward = forward_swap_rate + curve_basis_spread
    intrinsic = np.maximum(payer_sign * (adjusted_forward - strike), 0.0)
    return discount_to_exercise * notional * annuity * intrinsic


__all__ = [
    "PathEventRecord",
    "PathEventSpec",
    "PathEventState",
    "PathEventTimeline",
    "apply_path_event_spec",
    "build_event_path_requirement",
    "event_step_indices",
    "replay_path_event_timeline",
]
