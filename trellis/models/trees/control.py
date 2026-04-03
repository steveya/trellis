"""Stable lattice exercise/control contracts.

This module separates product exercise semantics from the low-level
``lattice_backward_induction(...)`` string arguments. Products and generated
routes should build a checked-in ``LatticeExercisePolicy`` first, then pass it
to the lattice engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from numbers import Real
from typing import Iterable, Sequence

from trellis.core.date_utils import (
    build_exercise_timeline_from_dates as _build_exercise_timeline_from_dates,
    build_payment_timeline as _build_payment_timeline,
)
from trellis.core.types import ContractTimeline, EventSchedule, SchedulePeriod


class ExerciseObjective(str, Enum):
    """Economic objective used when combining continuation and exercise."""

    HOLDER_MAXIMIZE = "max"
    ISSUER_MINIMIZE = "min"


@dataclass(frozen=True)
class LatticeExercisePolicy:
    """Normalized lattice exercise policy.

    Parameters
    ----------
    exercise_style:
        Product-level semantic style, e.g. ``"issuer_call"`` or ``"bermudan"``.
    exercise_type:
        Lattice rollback type, e.g. ``"european"``, ``"american"``, or ``"bermudan"``.
    exercise_steps:
        Sorted lattice steps where exercise is allowed for schedule-dependent
        policies.
    objective:
        Economic objective for combining continuation and exercise values.
    exercise_fn:
        Concrete Python callable consumed by ``lattice_backward_induction``.
    """

    exercise_style: str
    exercise_type: str
    exercise_steps: tuple[int, ...] = ()
    objective: ExerciseObjective = ExerciseObjective.HOLDER_MAXIMIZE
    exercise_fn: object | None = max

    @property
    def objective_name(self) -> str:
        """Return the canonical objective label."""
        return self.objective.value


def build_payment_timeline(
    start: date,
    end: date,
    frequency,
    **kwargs,
) -> ContractTimeline:
    """Re-export the canonical payment-timeline builder from this lattice facade."""
    return _build_payment_timeline(start, end, frequency, **kwargs)


def build_exercise_timeline_from_dates(
    dates: Iterable[date | str],
    **kwargs,
) -> ContractTimeline:
    """Re-export the canonical exercise-timeline builder from this lattice facade."""
    return _build_exercise_timeline_from_dates(dates, **kwargs)


def resolve_lattice_exercise_policy(
    exercise_style: str,
    *,
    exercise_steps: Sequence[int] | None = None,
) -> LatticeExercisePolicy:
    """Map product exercise semantics onto a normalized lattice policy."""
    style = (exercise_style or "none").strip().lower()
    steps = _normalize_exercise_steps(exercise_steps)

    if style in {"none", "european"}:
        return LatticeExercisePolicy(
            exercise_style=style,
            exercise_type="european",
            exercise_steps=(),
            objective=ExerciseObjective.HOLDER_MAXIMIZE,
            exercise_fn=max,
        )

    if style == "american":
        return LatticeExercisePolicy(
            exercise_style=style,
            exercise_type="american",
            exercise_steps=(),
            objective=ExerciseObjective.HOLDER_MAXIMIZE,
            exercise_fn=max,
        )

    if style in {"bermudan", "holder_put"}:
        return LatticeExercisePolicy(
            exercise_style=style,
            exercise_type="bermudan",
            exercise_steps=steps,
            objective=ExerciseObjective.HOLDER_MAXIMIZE,
            exercise_fn=max,
        )

    if style == "issuer_call":
        return LatticeExercisePolicy(
            exercise_style=style,
            exercise_type="bermudan",
            exercise_steps=steps,
            objective=ExerciseObjective.ISSUER_MINIMIZE,
            exercise_fn=min,
        )

    raise ValueError(
        "Unsupported lattice exercise style "
        f"{exercise_style!r}. Expected one of "
        "'none', 'european', 'american', 'bermudan', 'holder_put', 'issuer_call'."
    )


def resolve_lattice_exercise_policy_from_control_style(
    control_style: str,
    *,
    exercise_steps: Sequence[int] | None = None,
    exercise_style: str | None = None,
) -> LatticeExercisePolicy:
    """Map tranche-1 controller-style labels onto a normalized lattice policy."""
    normalized_control = (control_style or "identity").strip().lower()
    if normalized_control == "issuer_min":
        target_style = exercise_style or "issuer_call"
    elif normalized_control == "holder_max":
        target_style = exercise_style or "bermudan"
    elif normalized_control == "identity":
        target_style = exercise_style or "european"
    else:
        raise ValueError(
            "Unsupported lattice controller style "
            f"{control_style!r}. Expected one of 'identity', 'holder_max', or 'issuer_min'."
        )
    return resolve_lattice_exercise_policy(target_style, exercise_steps=exercise_steps)


def lattice_steps_from_timeline(
    timeline: ContractTimeline | EventSchedule | Sequence[SchedulePeriod | date | Real],
    dt: float | None = None,
    n_steps: int | None = None,
    allow_step_zero: bool = False,
    allow_terminal_step: bool = False,
) -> tuple[int, ...] | dict[date | float, int]:
    """Map a timeline onto lattice steps.

    Supported modes
    ---------------
    - periodized timeline + ``dt`` + ``n_steps`` -> tuple of step indices
    - numeric event times + ``dt`` + ``n_steps`` -> tuple of step indices
    - explicit event dates or times without lattice spacing -> mapping from the
      original event to a monotone 1-based step index
    """
    if dt is None or n_steps is None:
        return _ordinal_step_map(timeline)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    events = [t_event for t_event in _event_times(timeline) if t_event is not None]
    steps: list[int] = []
    singleton_terminal_compat = len(events) == 1
    for t_event in events:
        step = lattice_step_from_time(
            t_event,
            dt=dt,
            n_steps=n_steps,
            allow_step_zero=allow_step_zero,
            # Compatibility bridge: many generated routes map maturity cashflows
            # as singleton event lists and then index `[0]`. Preserve the
            # historical exclusion for multi-event exercise timelines, but keep
            # singleton terminal events addressable.
            allow_terminal_step=allow_terminal_step or singleton_terminal_compat,
        )
        if step is None:
            continue
        steps.append(step)
    return _normalize_exercise_steps(steps)


def lattice_step_from_time(
    t_event: SchedulePeriod | float | Real,
    *,
    dt: float,
    n_steps: int,
    allow_step_zero: bool = False,
    allow_terminal_step: bool = True,
) -> int | None:
    """Map one event time to one lattice step.

    Use this for payment or maturity cashflows where callers need one concrete
    step index instead of a filtered timeline tuple.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    t_value = _single_event_time(t_event)
    if t_value is None:
        return None
    step = int(round(float(t_value) / dt))
    if step < 0:
        return None
    if not allow_step_zero and step == 0:
        return None
    if step > n_steps:
        return None
    if not allow_terminal_step and step >= n_steps:
        return None
    return step


def merge_lattice_exercise_policy(
    *,
    exercise_policy: LatticeExercisePolicy | None,
    exercise_type: str,
    exercise_steps: Sequence[int] | None,
    exercise_fn,
) -> tuple[str, tuple[int, ...], object]:
    """Merge optional legacy keyword arguments with a normalized policy object."""
    if exercise_policy is None:
        policy = resolve_lattice_exercise_policy(
            exercise_type,
            exercise_steps=exercise_steps,
        )
        effective_fn = exercise_fn if exercise_fn is not None else policy.exercise_fn
        return policy.exercise_type, _normalize_exercise_steps(exercise_steps), effective_fn

    effective_type = exercise_policy.exercise_type
    effective_steps = exercise_policy.exercise_steps
    effective_fn = exercise_policy.exercise_fn

    legacy_steps = _normalize_exercise_steps(exercise_steps)
    normalized_type = (exercise_type or "european").strip().lower()
    if normalized_type != "european" and normalized_type != effective_type:
        raise ValueError(
            "exercise_policy conflicts with explicit exercise_type: "
            f"{effective_type!r} vs {normalized_type!r}"
        )
    if legacy_steps and legacy_steps != effective_steps:
        raise ValueError(
            "exercise_policy conflicts with explicit exercise_steps: "
            f"{effective_steps!r} vs {legacy_steps!r}"
        )
    if exercise_fn is not None and exercise_fn is not effective_fn:
        raise ValueError("exercise_policy conflicts with explicit exercise_fn")
    return effective_type, effective_steps, effective_fn


def _normalize_exercise_steps(exercise_steps: Sequence[int] | None) -> tuple[int, ...]:
    """Return sorted, unique positive lattice exercise steps."""
    if not exercise_steps:
        return ()
    return tuple(sorted({int(step) for step in exercise_steps if int(step) >= 0}))


def _iter_periods(
    timeline: ContractTimeline | EventSchedule | Sequence[SchedulePeriod | date | Real],
) -> Iterable[SchedulePeriod]:
    """Return a sequence of schedule periods from supported timeline containers."""
    if isinstance(timeline, ContractTimeline | EventSchedule):
        return timeline
    return timeline


def _event_times(
    timeline: ContractTimeline | EventSchedule | Sequence[SchedulePeriod | date | Real],
) -> Iterable[float | None]:
    """Yield event times from supported timeline containers."""
    if isinstance(timeline, ContractTimeline | EventSchedule):
        for period in timeline:
            yield period.t_payment if period.t_payment is not None else period.t_end
        return

    for item in timeline:
        if isinstance(item, SchedulePeriod):
            yield item.t_payment if item.t_payment is not None else item.t_end
        elif isinstance(item, Real):
            yield float(item)
        else:
            yield None


def _single_event_time(item: SchedulePeriod | float | Real) -> float | None:
    """Extract one event time from a scalar or period object."""
    if isinstance(item, SchedulePeriod):
        if item.t_payment is not None:
            return float(item.t_payment)
        if item.t_end is not None:
            return float(item.t_end)
        return None
    if isinstance(item, Real):
        return float(item)
    return None


def _ordinal_step_map(
    timeline: ContractTimeline | EventSchedule | Sequence[SchedulePeriod | date | Real],
) -> dict[date | float, int]:
    """Return a monotone 1-based event-to-step map when lattice spacing is absent."""
    items: list[date | float] = []
    if isinstance(timeline, ContractTimeline | EventSchedule):
        for period in timeline:
            items.append(period.payment_date)
    else:
        for item in timeline:
            if isinstance(item, SchedulePeriod):
                items.append(item.payment_date)
            elif isinstance(item, date):
                items.append(item)
            elif isinstance(item, Real):
                items.append(float(item))

    if not items:
        return {}
    ordered = list(dict.fromkeys(items))
    return {event: index for index, event in enumerate(ordered, start=1)}
