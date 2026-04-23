"""Generic event-aware Monte Carlo problem assembly.

This module provides a bounded runtime substrate for schedule-driven,
single-state Monte Carlo problems. It compiles typed process, event, and
reduced-state contracts into reusable runtime objects built on top of the
existing Monte Carlo engine, path-state, and path-event replay helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import numpy as raw_np

from trellis.core.date_utils import build_payment_timeline, year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import SchedulePeriod
from trellis.models.hull_white_parameters import resolve_hull_white_parameters
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.event_state import (
    PathEventSpec,
    PathEventState,
    PathEventTimeline,
    event_step_indices,
    replay_path_event_timeline,
)
from trellis.models.monte_carlo.path_state import (
    MonteCarloPathRequirement,
    PathReducer,
    StateAwarePayoff,
    terminal_value_payoff,
)
from trellis.models.processes.gbm import GBM
from trellis.models.processes.hull_white import HullWhite
from trellis.models.processes.local_vol import LocalVol

np = get_numpy()


def _to_backend_array(values):
    if isinstance(values, raw_np.ndarray):
        return values
    if hasattr(values, "_value"):
        return values
    return raw_np.asarray(values, dtype=float)


def _normalize_payload(payload: Mapping[str, object] | None) -> Mapping[str, object]:
    if payload is None:
        payload = {}
    normalized = {str(key): value for key, value in dict(payload).items()}
    return MappingProxyType(dict(sorted(normalized.items(), key=lambda item: item[0])))


def _coerce_time(value: float) -> float:
    time = float(value)
    if time < 0.0:
        raise ValueError("event times must be non-negative")
    return time


def _coerce_priority(value: int) -> int:
    priority = int(value)
    if priority < 0:
        raise ValueError("event priorities must be non-negative")
    return priority


@dataclass(frozen=True)
class EventAwareMonteCarloProcessSpec:
    """Typed single-state process selection for event-aware Monte Carlo."""

    family: str
    risk_free_rate: float | None = None
    dividend_yield: float = 0.0
    sigma: float | None = None
    local_vol_surface: Callable | None = None
    mean_reversion: float | None = None
    theta: float = 0.0
    theta_fn: Callable[[float], float] | None = None
    simulation_method: str = ""


@dataclass(frozen=True)
class EventAwareMonteCarloEvent:
    """One deterministic event in continuous time."""

    time: float
    name: str
    kind: str
    priority: int = 0
    schedule_role: str = ""
    phase: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _coerce_time(self.time))
        object.__setattr__(self, "name", str(self.name))
        kind = str(self.kind).strip().lower().replace("-", "_").replace(" ", "_")
        if not kind:
            raise ValueError("event kind must be non-empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "priority", _coerce_priority(self.priority))
        object.__setattr__(self, "schedule_role", str(self.schedule_role))
        object.__setattr__(self, "phase", str(self.phase))
        object.__setattr__(self, "payload", _normalize_payload(self.payload))


@dataclass(frozen=True)
class EventAwareMonteCarloProblemSpec:
    """High-level event-aware Monte Carlo problem before numerical assembly."""

    process_spec: EventAwareMonteCarloProcessSpec
    initial_state: float | raw_np.ndarray
    maturity: float
    n_steps: int = 100
    discount_rate: float = 0.0
    path_requirement_kind: str = "terminal_only"
    reducer_kind: str = "terminal_payoff"
    terminal_payoff: Callable[[raw_np.ndarray], raw_np.ndarray] | None = None
    state_payoff: Callable[[PathEventState], raw_np.ndarray] | None = None
    path_reducers: tuple[PathReducer, ...] = ()
    event_specs: tuple[EventAwareMonteCarloEvent, ...] = ()
    settlement_event: str = ""


@dataclass(frozen=True)
class EventAwareMonteCarloProblem:
    """Compiled event-aware Monte Carlo problem."""

    process: object
    initial_state: float | raw_np.ndarray
    maturity: float
    n_steps: int
    discount_rate: float
    simulation_method: str
    path_requirement: MonteCarloPathRequirement
    payoff: StateAwarePayoff
    event_timeline: PathEventTimeline | None = None


def build_event_aware_monte_carlo_process(spec: EventAwareMonteCarloProcessSpec):
    """Resolve the bounded process-family plugin surface."""
    family = str(spec.family).strip()
    if family == "gbm_1d":
        if spec.risk_free_rate is None or spec.sigma is None:
            raise ValueError("gbm_1d process requires risk_free_rate and sigma")
        return GBM(
            mu=float(spec.risk_free_rate) - float(spec.dividend_yield),
            sigma=float(spec.sigma),
        )
    if family == "local_vol_1d":
        if spec.risk_free_rate is None or spec.local_vol_surface is None:
            raise ValueError("local_vol_1d process requires risk_free_rate and local_vol_surface")
        return LocalVol(
            mu=float(spec.risk_free_rate) - float(spec.dividend_yield),
            vol_fn=spec.local_vol_surface,
        )
    if family == "hull_white_1f":
        if spec.mean_reversion is None or spec.sigma is None:
            raise ValueError("hull_white_1f process requires mean_reversion and sigma")
        return HullWhite(
            a=float(spec.mean_reversion),
            sigma=float(spec.sigma),
            theta_fn=spec.theta_fn,
            theta=float(spec.theta),
        )
    raise ValueError(f"Unsupported Monte Carlo process family {family!r}")


def build_event_aware_monte_carlo_problem(
    spec: EventAwareMonteCarloProblemSpec,
) -> EventAwareMonteCarloProblem:
    """Compile a typed event-aware Monte Carlo problem into runtime objects."""
    maturity = max(float(spec.maturity), 0.0)
    n_steps = max(int(spec.n_steps), 1)
    process = build_event_aware_monte_carlo_process(spec.process_spec)
    simulation_method = _resolve_simulation_method(spec.process_spec)
    event_timeline = _compile_event_timeline(spec.event_specs, maturity=maturity, n_steps=n_steps)
    path_requirement = _build_path_requirement(
        spec.path_requirement_kind,
        event_timeline=event_timeline,
        path_reducers=spec.path_reducers,
    )
    payoff = _build_payoff(
        spec,
        path_requirement=path_requirement,
        event_timeline=event_timeline,
    )
    return EventAwareMonteCarloProblem(
        process=process,
        initial_state=spec.initial_state,
        maturity=maturity,
        n_steps=n_steps,
        discount_rate=float(spec.discount_rate),
        simulation_method=simulation_method,
        path_requirement=path_requirement,
        payoff=payoff,
        event_timeline=event_timeline,
    )


def build_event_aware_monte_carlo_problem_from_family_ir(
    family_ir,
    *,
    process_spec,
    initial_state,
    maturity: float,
    event_time_map: Mapping[str, float] | None = None,
    event_payloads: Mapping[str, Mapping[str, object]] | None = None,
    discount_rate: float = 0.0,
    n_steps: int = 100,
    terminal_payoff=None,
    state_payoff=None,
    path_reducers: tuple[PathReducer, ...] = (),
    settlement_event: str | None = None,
) -> EventAwareMonteCarloProblemSpec:
    """Bridge one typed family-IR packet onto a runtime Monte Carlo problem spec.

    This helper stays on the public runtime surface so generated adapters do
    not need to import `trellis.agent.*` modules during validation or replay.
    """
    event_time_map = dict(event_time_map or {})
    event_payloads = dict(event_payloads or {})
    compiled_events: list[EventAwareMonteCarloEvent] = []
    for bucket in getattr(family_ir, "event_timeline", ()) or ():
        event_date = str(getattr(bucket, "event_date", "") or "").strip()
        if event_date not in event_time_map:
            raise KeyError(f"Missing event time for MC event date {event_date!r}")
        event_time = float(event_time_map[event_date])
        for priority, event in enumerate(getattr(bucket, "events", ()) or ()):
            event_name = str(getattr(event, "event_name", "") or "").strip() or f"event_{priority}"
            compiled_events.append(
                EventAwareMonteCarloEvent(
                    time=event_time,
                    name=event_name,
                    kind=str(getattr(event, "event_kind", "") or "observation"),
                    priority=priority,
                    schedule_role=str(getattr(event, "schedule_role", "") or ""),
                    phase=str(getattr(event, "phase", "") or ""),
                    payload=event_payloads.get(event_name, {}),
                )
            )

    resolved_settlement_event = str(settlement_event or "").strip()
    if not resolved_settlement_event:
        for event in reversed(compiled_events):
            if event.kind == "settlement":
                resolved_settlement_event = event.name
                break

    return EventAwareMonteCarloProblemSpec(
        process_spec=process_spec,
        initial_state=initial_state,
        maturity=float(maturity),
        n_steps=max(int(n_steps), 1),
        discount_rate=float(discount_rate),
        path_requirement_kind=str(
            getattr(getattr(family_ir, "path_requirement_spec", None), "requirement_kind", "")
            or "terminal_only"
        ),
        reducer_kind=str(
            getattr(getattr(family_ir, "payoff_reducer_spec", None), "reducer_kind", "")
            or "terminal_payoff"
        ),
        terminal_payoff=terminal_payoff,
        state_payoff=state_payoff,
        path_reducers=tuple(path_reducers),
        event_specs=tuple(compiled_events),
        settlement_event=resolved_settlement_event,
    )


def build_event_time_map_from_family_ir(
    family_ir,
    *,
    time_origin,
    day_count,
) -> Mapping[str, float]:
    """Return event-date to event-time bindings for one timed family IR."""
    origin = time_origin if isinstance(time_origin, date) else date.fromisoformat(str(time_origin))
    mapping: dict[str, float] = {}
    for event_date in getattr(family_ir, "event_dates", ()) or ():
        text = str(event_date or "").strip()
        if not text:
            continue
        mapping[text] = float(year_fraction(origin, date.fromisoformat(text), day_count))
    return MappingProxyType(dict(sorted(mapping.items(), key=lambda item: item[0])))


def build_timed_event_aware_monte_carlo_problem_from_family_ir(
    family_ir,
    *,
    process_spec,
    initial_state,
    maturity: float,
    time_origin,
    day_count,
    event_payloads: Mapping[str, Mapping[str, object]] | None = None,
    discount_rate: float = 0.0,
    n_steps: int = 100,
    terminal_payoff=None,
    state_payoff=None,
    path_reducers: tuple[PathReducer, ...] = (),
    settlement_event: str | None = None,
) -> EventAwareMonteCarloProblemSpec:
    """Bridge one typed timed family IR onto a runtime Monte Carlo problem spec.

    This wrapper keeps generated adapters off manual event-date arrays and
    onto the family IR's typed schedule authority.
    """
    event_time_map = build_event_time_map_from_family_ir(
        family_ir,
        time_origin=time_origin,
        day_count=day_count,
    )
    return build_event_aware_monte_carlo_problem_from_family_ir(
        family_ir,
        process_spec=process_spec,
        initial_state=initial_state,
        maturity=maturity,
        event_time_map=event_time_map,
        event_payloads=event_payloads,
        discount_rate=discount_rate,
        n_steps=n_steps,
        terminal_payoff=terminal_payoff,
        state_payoff=state_payoff,
        path_reducers=path_reducers,
        settlement_event=settlement_event,
    )


def build_discounted_swap_pv_payload(
    *,
    payment_timeline: Iterable[SchedulePeriod],
    discount_curve,
    forward_curve=None,
    exercise_time: float,
    discount_reducer_name: str,
    mean_reversion: float,
    strike: float,
    notional: float = 1.0,
    is_payer: bool = True,
    anchor_short_rate: float | None = None,
) -> Mapping[str, object]:
    """Build one shared settlement payload for the `discounted_swap_pv` rule."""
    periods = tuple(payment_timeline or ())
    if not periods:
        raise ValueError("discounted_swap_pv payload requires at least one payment period")

    reducer_name = str(discount_reducer_name).strip()
    if not reducer_name:
        raise ValueError("discounted_swap_pv payload requires discount_reducer_name")

    exercise_time = max(float(exercise_time), 1e-12)
    anchor_discount_to_exercise = max(float(discount_curve.discount(exercise_time)), 1e-12)
    payment_times: list[float] = []
    accrual_fractions: list[float] = []
    anchor_discount_factors: list[float] = []
    annuity = 0.0
    forecast_float_leg = 0.0

    for period in periods:
        t_payment = (
            float(period.t_payment)
            if period.t_payment is not None
            else float(period.t_end if period.t_end is not None else 0.0)
        )
        if t_payment <= exercise_time:
            continue
        t_start = float(period.t_start if period.t_start is not None else exercise_time)
        t_end = float(period.t_end if period.t_end is not None else t_payment)
        accrual = float(
            period.accrual_fraction
            if period.accrual_fraction is not None
            else max(t_end - t_start, 0.0)
        )
        anchor_df = max(float(discount_curve.discount(t_payment)), 1e-12)
        ratio = anchor_df / anchor_discount_to_exercise

        payment_times.append(t_payment)
        accrual_fractions.append(accrual)
        anchor_discount_factors.append(anchor_df)
        annuity += accrual * ratio
        if forward_curve is not None:
            if hasattr(forward_curve, "forward_rate"):
                forward_rate = float(forward_curve.forward_rate(t_start, t_end))
            else:
                start_df = max(float(forward_curve.discount(t_start)), 1e-12)
                end_df = max(float(forward_curve.discount(t_end)), 1e-12)
                forward_rate = (start_df / end_df - 1.0) / max(t_end - t_start, 1e-12)
            forecast_float_leg += forward_rate * accrual * ratio

    if not payment_times:
        raise ValueError("discounted_swap_pv payload requires payment periods after exercise_time")

    discount_par_rate = 0.0
    if annuity > 1e-12:
        discount_par_rate = (
            1.0 - (anchor_discount_factors[-1] / anchor_discount_to_exercise)
        ) / annuity
    forecast_par_rate = (
        forecast_float_leg / annuity
        if annuity > 1e-12 and forward_curve is not None
        else discount_par_rate
    )
    if anchor_short_rate is None:
        anchor_short_rate = -raw_np.log(anchor_discount_to_exercise) / exercise_time

    return MappingProxyType(
        {
            "rule": "discounted_swap_pv",
            "payment_times": tuple(payment_times),
            "accrual_fractions": tuple(accrual_fractions),
            "anchor_discount_to_exercise": float(anchor_discount_to_exercise),
            "anchor_discount_factors": tuple(anchor_discount_factors),
            "discount_reducer_name": reducer_name,
            "mean_reversion": float(mean_reversion),
            "anchor_short_rate": float(anchor_short_rate),
            "curve_basis_spread": float(forecast_par_rate - discount_par_rate),
            "strike": float(strike),
            "notional": float(notional),
            "is_payer": bool(is_payer),
            "exercise_time": float(exercise_time),
        }
    )


def resolve_hull_white_monte_carlo_process_inputs(
    market_state,
    *,
    option_horizon: float,
    strike: float,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    default_mean_reversion: float = 0.1,
) -> tuple[EventAwareMonteCarloProcessSpec, float]:
    """Resolve a reusable Hull-White MC process spec from market inputs.

    The helper keeps adapters on the bounded model/calibration path by reading
    the short-rate sigma from calibrated model parameters when present and
    otherwise deriving the same Black-vol-based default used by the checked-in
    rate-tree helpers.
    """
    if getattr(market_state, "discount", None) is None:
        raise ValueError("Hull-White MC process inputs require market_state.discount")

    horizon = max(float(option_horizon), 1e-6)
    r0 = float(market_state.discount.zero_rate(max(horizon / 2.0, 1e-6)))
    default_sigma = None
    if getattr(market_state, "vol_surface", None) is not None:
        black_vol = float(
            market_state.vol_surface.black_vol(
                max(horizon, 1e-6),
                max(abs(float(strike)), 1e-6),
            )
        )
        default_sigma = black_vol * max(abs(r0), 1e-6)

    resolved_mean_reversion, resolved_sigma = resolve_hull_white_parameters(
        market_state,
        mean_reversion=mean_reversion,
        sigma=sigma,
        default_mean_reversion=default_mean_reversion,
        default_sigma=default_sigma,
    )
    return (
        EventAwareMonteCarloProcessSpec(
            family="hull_white_1f",
            mean_reversion=float(resolved_mean_reversion),
            sigma=float(resolved_sigma),
            theta=float(resolved_mean_reversion) * float(r0),
        ),
        float(r0),
    )


def price_event_aware_monte_carlo(
    problem: EventAwareMonteCarloProblem | EventAwareMonteCarloProblemSpec | None = None,
    *,
    problem_spec: EventAwareMonteCarloProblemSpec | None = None,
    process=None,
    initial_state=None,
    maturity: float | None = None,
    n_steps: int | None = None,
    discount_rate: float = 0.0,
    simulation_method: str = "",
    storage_policy: str | MonteCarloPathRequirement = "auto",
    payoff=None,
    payoff_fn=None,
    terminal_payoff=None,
    n_paths: int = 10_000,
    seed: int | None = None,
    return_paths: bool = False,
    shocks=None,
    differentiable: bool = False,
) -> dict:
    """Price an event-aware Monte Carlo workload from a runtime or assembly surface.

    Accepted call shapes:

    - ``price_event_aware_monte_carlo(problem, ...)`` with a compiled
      :class:`EventAwareMonteCarloProblem`
    - ``price_event_aware_monte_carlo(problem_spec=..., ...)`` or
      ``price_event_aware_monte_carlo(problem_spec_obj, ...)`` with an
      :class:`EventAwareMonteCarloProblemSpec`
    - ``price_event_aware_monte_carlo(process=..., initial_state=..., maturity=..., ...)``
      for thin adapters that already resolved a runtime process and a simple
      path payoff but should not need to instantiate ``MonteCarloEngine``
      directly
    """
    compiled_problem: EventAwareMonteCarloProblem | None = None
    if isinstance(problem, EventAwareMonteCarloProblem):
        if problem_spec is not None or process is not None:
            raise ValueError("Provide either a compiled problem or assembly inputs, not both")
        compiled_problem = problem
    else:
        if isinstance(problem, EventAwareMonteCarloProblemSpec):
            if problem_spec is not None:
                raise ValueError("Provide either problem or problem_spec, not both")
            problem_spec = problem
        if problem_spec is not None:
            if process is not None:
                raise ValueError("Provide either problem_spec or a runtime process bundle, not both")
            compiled_problem = build_event_aware_monte_carlo_problem(problem_spec)

    if compiled_problem is not None:
        engine = MonteCarloEngine(
            compiled_problem.process,
            n_paths=n_paths,
            n_steps=compiled_problem.n_steps,
            seed=seed,
            method=compiled_problem.simulation_method,
        )
        return engine.price(
            compiled_problem.initial_state,
            compiled_problem.maturity,
            compiled_problem.payoff,
            discount_rate=compiled_problem.discount_rate,
            storage_policy=compiled_problem.path_requirement,
            return_paths=return_paths,
            shocks=shocks,
            differentiable=differentiable,
        )

    if process is None:
        raise ValueError("Provide a compiled problem, problem_spec, or runtime process bundle")
    if initial_state is None or maturity is None:
        raise ValueError("Runtime process pricing requires initial_state and maturity")

    resolved_payoff = payoff if payoff is not None else payoff_fn
    if resolved_payoff is None and terminal_payoff is not None:
        def _terminal_only_path_payoff(paths):
            terminal = _to_backend_array(paths[:, -1])
            return _to_backend_array(terminal_payoff(terminal))

        resolved_payoff = _terminal_only_path_payoff
    if resolved_payoff is None:
        raise ValueError("Runtime process pricing requires payoff, payoff_fn, or terminal_payoff")

    engine = MonteCarloEngine(
        process,
        n_paths=n_paths,
        n_steps=max(int(n_steps or 100), 1),
        seed=seed,
        method=_resolve_simulation_method_from_runtime_process(process, simulation_method),
    )
    return engine.price(
        initial_state,
        float(maturity),
        resolved_payoff,
        discount_rate=float(discount_rate),
        storage_policy=storage_policy,
        return_paths=return_paths,
        shocks=shocks,
        differentiable=differentiable,
    )


def _resolve_simulation_method(spec: EventAwareMonteCarloProcessSpec) -> str:
    method = str(spec.simulation_method or "").strip().lower()
    if method:
        return method
    family = str(spec.family).strip()
    if family in {"gbm_1d", "hull_white_1f"}:
        return "exact"
    if family == "local_vol_1d":
        return "euler"
    return "euler"


def _resolve_simulation_method_from_runtime_process(process, method: str = "") -> str:
    normalized = str(method or "").strip().lower()
    if normalized:
        return normalized
    if isinstance(process, (GBM, HullWhite)):
        return "exact"
    if isinstance(process, LocalVol):
        return "euler"
    return "euler"


def _compile_event_timeline(
    event_specs: tuple[EventAwareMonteCarloEvent, ...],
    *,
    maturity: float,
    n_steps: int,
) -> PathEventTimeline | None:
    if not event_specs:
        return None

    unique_times = tuple(dict.fromkeys(sorted(float(event.time) for event in event_specs)))
    time_to_step = dict(
        zip(
            unique_times,
            event_step_indices(unique_times, maturity, n_steps),
            strict=True,
        )
    )
    compiled = tuple(
        PathEventSpec(
            name=event.name,
            kind=event.kind,
            step=int(time_to_step[float(event.time)]),
            priority=int(event.priority),
            payload=event.payload,
        )
        for event in event_specs
    )
    return PathEventTimeline(compiled)


def _build_path_requirement(
    requirement_kind: str,
    *,
    event_timeline: PathEventTimeline | None,
    path_reducers: tuple[PathReducer, ...],
) -> MonteCarloPathRequirement:
    normalized = str(requirement_kind or "terminal_only").strip().lower()
    if normalized == "terminal_only":
        return MonteCarloPathRequirement(reducers=tuple(path_reducers))
    if normalized == "event_replay":
        if event_timeline is None:
            raise ValueError("event_replay path requirements require an event timeline")
        return MonteCarloPathRequirement(
            snapshot_steps=tuple(sorted(set(event_timeline.steps))),
            reducers=tuple(path_reducers),
        )
    raise ValueError(f"Unsupported Monte Carlo path requirement kind {requirement_kind!r}")


def _build_payoff(
    spec: EventAwareMonteCarloProblemSpec,
    *,
    path_requirement: MonteCarloPathRequirement,
    event_timeline: PathEventTimeline | None,
) -> StateAwarePayoff:
    reducer_kind = str(spec.reducer_kind or "terminal_payoff").strip().lower()
    if reducer_kind == "terminal_payoff":
        if spec.terminal_payoff is None:
            raise ValueError("terminal_payoff reducer requires terminal_payoff")
        return terminal_value_payoff(spec.terminal_payoff, name="event_aware_terminal_payoff")

    if event_timeline is None:
        raise ValueError(f"{reducer_kind} reducer requires an event timeline")

    evaluate_state = _resolve_event_state_payoff(
        reducer_kind,
        settlement_event=spec.settlement_event,
        state_payoff=spec.state_payoff,
        event_timeline=event_timeline,
    )

    def evaluate_paths(paths: raw_np.ndarray) -> raw_np.ndarray:
        replayed = _replay_from_paths(
            paths,
            spec.initial_state,
            event_timeline,
            path_requirement=path_requirement,
        )
        return _to_backend_array(evaluate_state(replayed))

    def evaluate_state_from_reduced(state) -> raw_np.ndarray:
        replayed = _replay_from_state(state, spec.initial_state, event_timeline)
        return _to_backend_array(evaluate_state(replayed))

    return StateAwarePayoff(
        path_requirement=path_requirement,
        evaluate_paths_fn=evaluate_paths,
        evaluate_state_fn=evaluate_state_from_reduced,
        name=f"event_aware_{reducer_kind}",
        derivative_metadata=_event_timeline_derivative_metadata(event_timeline),
    )


def _event_timeline_derivative_metadata(event_timeline: PathEventTimeline | None) -> dict[str, object]:
    if event_timeline is None:
        return {}
    features = []
    for event in event_timeline:
        if event.kind == "barrier":
            features.append("barrier_event")
        elif event.kind == "exercise":
            features.append("exercise_event")
    features = list(dict.fromkeys(features))
    if not features:
        return {}
    return {
        "discontinuous_features": tuple(features),
        "unsupported_reason": f"{features[0]}_discontinuity",
    }


def _resolve_event_state_payoff(
    reducer_kind: str,
    *,
    settlement_event: str,
    state_payoff,
    event_timeline: PathEventTimeline,
):
    if state_payoff is not None:
        return state_payoff

    default_settlement = str(settlement_event or "").strip() or _default_settlement_event(event_timeline)
    if reducer_kind == "compiled_schedule_payoff" and default_settlement:
        return lambda state: _to_backend_array(state.settlement_value(default_settlement))

    raise ValueError(
        f"{reducer_kind} reducer requires either state_payoff or a settlement event"
    )


def _default_settlement_event(event_timeline: PathEventTimeline) -> str:
    for event in reversed(tuple(event_timeline.events)):
        if event.kind == "settlement":
            return event.name
    return ""


def _replay_from_paths(
    paths: raw_np.ndarray,
    initial_state,
    event_timeline: PathEventTimeline,
    *,
    path_requirement: MonteCarloPathRequirement,
) -> PathEventState:
    cross_sections = tuple(_to_backend_array(paths[:, event.step]) for event in event_timeline)
    return replay_path_event_timeline(
        cross_sections,
        initial_values=initial_state,
        event_timeline=event_timeline,
        reducer_values=_replay_reducer_values(paths, path_requirement),
    )


def _replay_from_state(
    state,
    initial_state,
    event_timeline: PathEventTimeline,
) -> PathEventState:
    cross_sections = tuple(_to_backend_array(state.snapshot(event.step)) for event in event_timeline)
    return replay_path_event_timeline(
        cross_sections,
        initial_values=initial_state,
        event_timeline=event_timeline,
        reducer_values=getattr(state, "reducer_values", None),
    )


def _replay_reducer_values(
    paths: raw_np.ndarray,
    path_requirement: MonteCarloPathRequirement,
) -> dict[str, raw_np.ndarray]:
    if not path_requirement.reducers:
        return {}
    initial = _to_backend_array(paths[:, 0])
    total_steps = max(paths.shape[1] - 1, 1)
    reduced = {
        reducer.name: reducer.init(initial, total_steps)
        for reducer in path_requirement.reducers
    }
    for step in range(1, paths.shape[1]):
        cross_section = _to_backend_array(paths[:, step])
        for reducer in path_requirement.reducers:
            reduced[reducer.name] = reducer.update(
                reduced[reducer.name],
                cross_section,
                step,
            )
    return reduced


def build_short_rate_discount_reducer(
    *,
    name: str,
    maturity: float,
) -> PathReducer:
    """Return a reduced-state discount-factor accumulator for short-rate paths."""
    horizon = max(float(maturity), 1e-12)
    total_steps_holder = {"n_steps": 1}

    def _init(initial_values: raw_np.ndarray, n_steps: int) -> raw_np.ndarray:
        total_steps_holder["n_steps"] = max(int(n_steps), 1)
        return np.ones(initial_values.shape[0])

    def _update(accumulator: raw_np.ndarray, values: raw_np.ndarray, step: int) -> raw_np.ndarray:
        del step
        dt = horizon / max(int(total_steps_holder["n_steps"]), 1)
        return _to_backend_array(accumulator) * np.exp(-_to_backend_array(values) * dt)

    return PathReducer(
        name=str(name),
        init_fn=_init,
        update_fn=_update,
    )


__all__ = [
    "EventAwareMonteCarloEvent",
    "EventAwareMonteCarloProblem",
    "EventAwareMonteCarloProblemSpec",
    "EventAwareMonteCarloProcessSpec",
    "build_event_time_map_from_family_ir",
    "build_payment_timeline",
    "build_discounted_swap_pv_payload",
    "build_event_aware_monte_carlo_problem",
    "build_event_aware_monte_carlo_problem_from_family_ir",
    "build_event_aware_monte_carlo_process",
    "build_timed_event_aware_monte_carlo_problem_from_family_ir",
    "build_short_rate_discount_reducer",
    "price_event_aware_monte_carlo",
    "resolve_hull_white_monte_carlo_process_inputs",
]
