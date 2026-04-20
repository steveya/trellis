"""Algebraic Contract IR for structural payoff matching.

This module is intentionally additive to the existing ``ProductIR`` surface.
It introduces a richer tree representation for contract payoff structure
without changing any existing route selection paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from types import MappingProxyType
from typing import Any, Mapping


class ContractIRWellFormednessError(ValueError):
    """Raised when a Contract IR node violates a local or global invariant."""


def _freeze_mapping(mapping: Mapping[tuple[object, ...], float] | None) -> Mapping[tuple[object, ...], float]:
    return MappingProxyType(dict(mapping or {}))


def _require_non_empty_text(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractIRWellFormednessError(f"{label} must be a non-empty string")
    return text


def _as_float(value: float | int, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive only
        raise ContractIRWellFormednessError(f"{label} must be numeric") from exc


@dataclass(frozen=True)
class Singleton:
    t: date

    def __post_init__(self):
        if not isinstance(self.t, date):
            raise ContractIRWellFormednessError("Singleton schedule requires a date")

    def key(self) -> tuple[date, ...]:
        return (self.t,)


@dataclass(frozen=True)
class FiniteSchedule:
    dates: tuple[date, ...]

    def __post_init__(self):
        if not isinstance(self.dates, tuple):
            object.__setattr__(self, "dates", tuple(self.dates))
        if not self.dates:
            raise ContractIRWellFormednessError("FiniteSchedule must be non-empty")
        for item in self.dates:
            if not isinstance(item, date):
                raise ContractIRWellFormednessError("FiniteSchedule entries must be dates")
        if any(left >= right for left, right in zip(self.dates, self.dates[1:])):
            raise ContractIRWellFormednessError(
                "FiniteSchedule dates must be strictly increasing"
            )

    def key(self) -> tuple[date, ...]:
        return self.dates


@dataclass(frozen=True)
class ContinuousInterval:
    t_start: date
    t_end: date

    def __post_init__(self):
        if not isinstance(self.t_start, date) or not isinstance(self.t_end, date):
            raise ContractIRWellFormednessError(
                "ContinuousInterval requires date endpoints"
            )
        if self.t_start > self.t_end:
            raise ContractIRWellFormednessError(
                "ContinuousInterval requires t_start <= t_end"
            )

    def key(self) -> tuple[date, date]:
        return (self.t_start, self.t_end)


Schedule = Singleton | FiniteSchedule | ContinuousInterval


@dataclass(frozen=True)
class EquitySpot:
    name: str
    dynamics: str

    def __post_init__(self):
        object.__setattr__(self, "name", _require_non_empty_text(self.name, label="EquitySpot.name"))
        object.__setattr__(
            self,
            "dynamics",
            _require_non_empty_text(self.dynamics, label="EquitySpot.dynamics"),
        )


@dataclass(frozen=True)
class RateCurve:
    name: str
    dynamics: str

    def __post_init__(self):
        object.__setattr__(self, "name", _require_non_empty_text(self.name, label="RateCurve.name"))
        object.__setattr__(
            self,
            "dynamics",
            _require_non_empty_text(self.dynamics, label="RateCurve.dynamics"),
        )


@dataclass(frozen=True)
class ForwardRate:
    name: str
    dynamics: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "name",
            _require_non_empty_text(self.name, label="ForwardRate.name"),
        )
        object.__setattr__(
            self,
            "dynamics",
            _require_non_empty_text(self.dynamics, label="ForwardRate.dynamics"),
        )


@dataclass(frozen=True)
class QuoteCurve:
    name: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "name",
            _require_non_empty_text(self.name, label="QuoteCurve.name"),
        )


@dataclass(frozen=True)
class QuoteSurface:
    name: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "name",
            _require_non_empty_text(self.name, label="QuoteSurface.name"),
        )


UnderlyingSpecLeaf = EquitySpot | RateCurve | ForwardRate | QuoteCurve | QuoteSurface


@dataclass(frozen=True)
class CompositeUnderlying:
    parts: tuple[UnderlyingSpecLeaf, ...]

    def __post_init__(self):
        if not isinstance(self.parts, tuple):
            object.__setattr__(self, "parts", tuple(self.parts))
        if not self.parts:
            raise ContractIRWellFormednessError("CompositeUnderlying must be non-empty")
        for part in self.parts:
            if not isinstance(part, (EquitySpot, RateCurve, ForwardRate, QuoteCurve, QuoteSurface)):
                raise ContractIRWellFormednessError(
                    "CompositeUnderlying.parts must contain only UnderlyingSpecLeaf values"
                )
        names = [part.name for part in self.parts]
        if len(set(names)) != len(names):
            raise ContractIRWellFormednessError(
                "CompositeUnderlying leaf names must be unique"
            )


UnderlyingSpec = EquitySpot | RateCurve | ForwardRate | QuoteCurve | QuoteSurface | CompositeUnderlying


@dataclass(frozen=True)
class Underlying:
    spec: UnderlyingSpec

    def __post_init__(self):
        if not isinstance(
            self.spec,
            (EquitySpot, RateCurve, ForwardRate, QuoteCurve, QuoteSurface, CompositeUnderlying),
        ):
            raise ContractIRWellFormednessError(
                "Underlying.spec must be an UnderlyingSpec"
            )


@dataclass(frozen=True)
class Exercise:
    style: str
    schedule: Schedule

    def __post_init__(self):
        normalized_style = _require_non_empty_text(self.style, label="Exercise.style").lower()
        object.__setattr__(self, "style", normalized_style)
        if normalized_style == "european" and not isinstance(self.schedule, Singleton):
            raise ContractIRWellFormednessError(
                "Exercise(style='european') requires a Singleton schedule"
            )
        if normalized_style == "bermudan" and not isinstance(self.schedule, FiniteSchedule):
            raise ContractIRWellFormednessError(
                "Exercise(style='bermudan') requires a FiniteSchedule"
            )
        if normalized_style == "american" and not isinstance(self.schedule, ContinuousInterval):
            raise ContractIRWellFormednessError(
                "Exercise(style='american') requires a ContinuousInterval"
            )
        if normalized_style not in {"european", "bermudan", "american"}:
            raise ContractIRWellFormednessError(
                f"unsupported Exercise.style {normalized_style!r}"
            )


@dataclass(frozen=True)
class Observation:
    kind: str
    schedule: Schedule

    def __post_init__(self):
        normalized_kind = _require_non_empty_text(self.kind, label="Observation.kind").lower()
        object.__setattr__(self, "kind", normalized_kind)
        if normalized_kind == "terminal" and not isinstance(self.schedule, Singleton):
            raise ContractIRWellFormednessError(
                "Observation(kind='terminal') requires a Singleton schedule"
            )
        if normalized_kind == "schedule" and not isinstance(self.schedule, FiniteSchedule):
            raise ContractIRWellFormednessError(
                "Observation(kind='schedule') requires a FiniteSchedule"
            )
        if normalized_kind == "path_dependent" and not isinstance(
            self.schedule,
            (FiniteSchedule, ContinuousInterval),
        ):
            raise ContractIRWellFormednessError(
                "Observation(kind='path_dependent') requires a FiniteSchedule or ContinuousInterval"
            )
        if normalized_kind not in {"terminal", "schedule", "path_dependent"}:
            raise ContractIRWellFormednessError(
                f"unsupported Observation.kind {normalized_kind!r}"
            )


@dataclass(frozen=True)
class Constant:
    value: float

    def __post_init__(self):
        object.__setattr__(self, "value", _as_float(self.value, label="Constant.value"))


@dataclass(frozen=True)
class Spot:
    underlier_id: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "underlier_id",
            _require_non_empty_text(self.underlier_id, label="Spot.underlier_id"),
        )


@dataclass(frozen=True)
class Forward:
    underlier_id: str
    schedule: Schedule

    def __post_init__(self):
        object.__setattr__(
            self,
            "underlier_id",
            _require_non_empty_text(self.underlier_id, label="Forward.underlier_id"),
        )
        if not isinstance(self.schedule, (Singleton, FiniteSchedule, ContinuousInterval)):
            raise ContractIRWellFormednessError("Forward schedule must be a Schedule")


@dataclass(frozen=True)
class SwapRate:
    underlier_id: str
    schedule: FiniteSchedule

    def __post_init__(self):
        object.__setattr__(
            self,
            "underlier_id",
            _require_non_empty_text(self.underlier_id, label="SwapRate.underlier_id"),
        )
        if not isinstance(self.schedule, FiniteSchedule):
            raise ContractIRWellFormednessError(
                "SwapRate schedule must be a FiniteSchedule"
            )


@dataclass(frozen=True)
class Annuity:
    underlier_id: str
    schedule: FiniteSchedule

    def __post_init__(self):
        object.__setattr__(
            self,
            "underlier_id",
            _require_non_empty_text(self.underlier_id, label="Annuity.underlier_id"),
        )
        if not isinstance(self.schedule, FiniteSchedule):
            raise ContractIRWellFormednessError(
                "Annuity schedule must be a FiniteSchedule"
            )


@dataclass(frozen=True)
class Strike:
    value: float

    def __post_init__(self):
        object.__setattr__(self, "value", _as_float(self.value, label="Strike.value"))


@dataclass(frozen=True)
class ParRateTenor:
    tenor: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "tenor",
            _require_non_empty_text(self.tenor, label="ParRateTenor.tenor"),
        )


@dataclass(frozen=True)
class ZeroRateTenor:
    tenor: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "tenor",
            _require_non_empty_text(self.tenor, label="ZeroRateTenor.tenor"),
        )


@dataclass(frozen=True)
class ForwardRateInterval:
    start_tenor: str
    end_tenor: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "start_tenor",
            _require_non_empty_text(self.start_tenor, label="ForwardRateInterval.start_tenor"),
        )
        object.__setattr__(
            self,
            "end_tenor",
            _require_non_empty_text(self.end_tenor, label="ForwardRateInterval.end_tenor"),
        )


CurveCoordinate = ParRateTenor | ZeroRateTenor | ForwardRateInterval


@dataclass(frozen=True)
class VolPoint:
    option_tenor: str
    strike: float
    strike_style: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "option_tenor",
            _require_non_empty_text(self.option_tenor, label="VolPoint.option_tenor"),
        )
        object.__setattr__(self, "strike", _as_float(self.strike, label="VolPoint.strike"))
        object.__setattr__(
            self,
            "strike_style",
            _require_non_empty_text(self.strike_style, label="VolPoint.strike_style"),
        )


@dataclass(frozen=True)
class VolDeltaPoint:
    option_tenor: str
    delta: float
    delta_style: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "option_tenor",
            _require_non_empty_text(self.option_tenor, label="VolDeltaPoint.option_tenor"),
        )
        object.__setattr__(self, "delta", _as_float(self.delta, label="VolDeltaPoint.delta"))
        object.__setattr__(
            self,
            "delta_style",
            _require_non_empty_text(self.delta_style, label="VolDeltaPoint.delta_style"),
        )


SurfaceCoordinate = VolPoint | VolDeltaPoint


@dataclass(frozen=True)
class CurveQuote:
    curve_id: str
    coordinate: CurveCoordinate
    convention: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "curve_id",
            _require_non_empty_text(self.curve_id, label="CurveQuote.curve_id"),
        )
        if not isinstance(self.coordinate, (ParRateTenor, ZeroRateTenor, ForwardRateInterval)):
            raise ContractIRWellFormednessError(
                "CurveQuote coordinate must be a CurveCoordinate"
            )
        object.__setattr__(
            self,
            "convention",
            _require_non_empty_text(self.convention, label="CurveQuote.convention"),
        )


@dataclass(frozen=True)
class SurfaceQuote:
    surface_id: str
    coordinate: SurfaceCoordinate
    convention: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "surface_id",
            _require_non_empty_text(self.surface_id, label="SurfaceQuote.surface_id"),
        )
        if not isinstance(self.coordinate, (VolPoint, VolDeltaPoint)):
            raise ContractIRWellFormednessError(
                "SurfaceQuote coordinate must be a SurfaceCoordinate"
            )
        object.__setattr__(
            self,
            "convention",
            _require_non_empty_text(self.convention, label="SurfaceQuote.convention"),
        )


PayoffExpr = Any
Predicate = Any


@dataclass(frozen=True)
class LinearBasket:
    terms: tuple[tuple[float, PayoffExpr], ...]

    def __post_init__(self):
        if not isinstance(self.terms, tuple):
            object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise ContractIRWellFormednessError("LinearBasket must be non-empty")
        normalized: list[tuple[float, PayoffExpr]] = []
        for weight, expr in self.terms:
            normalized.append((_as_float(weight, label="LinearBasket weight"), expr))
        object.__setattr__(self, "terms", tuple(normalized))


@dataclass(frozen=True)
class ArithmeticMean:
    expr: PayoffExpr
    schedule: FiniteSchedule

    def __post_init__(self):
        if not isinstance(self.schedule, FiniteSchedule):
            raise ContractIRWellFormednessError(
                "ArithmeticMean schedule must be a FiniteSchedule"
            )


@dataclass(frozen=True)
class VarianceObservable:
    underlier_id: str
    interval: ContinuousInterval

    def __post_init__(self):
        object.__setattr__(
            self,
            "underlier_id",
            _require_non_empty_text(
                self.underlier_id,
                label="VarianceObservable.underlier_id",
            ),
        )
        if not isinstance(self.interval, ContinuousInterval):
            raise ContractIRWellFormednessError(
                "VarianceObservable interval must be a ContinuousInterval"
            )


@dataclass(frozen=True)
class Max:
    args: tuple[PayoffExpr, ...]

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if len(self.args) < 1:
            raise ContractIRWellFormednessError("Max requires at least one argument")


@dataclass(frozen=True)
class Min:
    args: tuple[PayoffExpr, ...]

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if len(self.args) < 1:
            raise ContractIRWellFormednessError("Min requires at least one argument")


@dataclass(frozen=True)
class Add:
    args: tuple[PayoffExpr, ...]

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if len(self.args) < 2:
            raise ContractIRWellFormednessError("Add requires at least two arguments")


@dataclass(frozen=True)
class Sub:
    lhs: PayoffExpr
    rhs: PayoffExpr


@dataclass(frozen=True)
class Mul:
    args: tuple[PayoffExpr, ...]

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if len(self.args) < 2:
            raise ContractIRWellFormednessError("Mul requires at least two arguments")


@dataclass(frozen=True)
class Scaled:
    scalar: PayoffExpr
    body: PayoffExpr


@dataclass(frozen=True)
class Gt:
    lhs: PayoffExpr
    rhs: PayoffExpr


@dataclass(frozen=True)
class Ge:
    lhs: PayoffExpr
    rhs: PayoffExpr


@dataclass(frozen=True)
class Lt:
    lhs: PayoffExpr
    rhs: PayoffExpr


@dataclass(frozen=True)
class Le:
    lhs: PayoffExpr
    rhs: PayoffExpr


@dataclass(frozen=True)
class And:
    args: tuple[Predicate, ...]

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if not self.args:
            raise ContractIRWellFormednessError("And requires at least one predicate")


@dataclass(frozen=True)
class Or:
    args: tuple[Predicate, ...]

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if not self.args:
            raise ContractIRWellFormednessError("Or requires at least one predicate")


@dataclass(frozen=True)
class Not:
    arg: Predicate


@dataclass(frozen=True)
class Indicator:
    predicate: Predicate


PayoffExpr = (
    Constant
    | Spot
    | Forward
    | SwapRate
    | Annuity
    | Strike
    | CurveQuote
    | SurfaceQuote
    | LinearBasket
    | ArithmeticMean
    | VarianceObservable
    | Max
    | Min
    | Add
    | Sub
    | Mul
    | Scaled
    | Indicator
)

Predicate = Gt | Ge | Lt | Le | And | Or | Not


@dataclass(frozen=True)
class ContractIR:
    payoff: PayoffExpr
    exercise: Exercise
    observation: Observation
    underlying: Underlying

    def __post_init__(self):
        namespace = _underlier_namespace(self.underlying.spec)
        _validate_payoff_expr(self.payoff, namespace=namespace)
        _validate_schedule_alignment(self)


def _underlier_namespace(spec: UnderlyingSpec) -> tuple[str, ...]:
    if isinstance(spec, CompositeUnderlying):
        return tuple(part.name for part in spec.parts)
    return (spec.name,)


def _validate_underlier_reference(underlier_id: str, *, namespace: tuple[str, ...]) -> None:
    if underlier_id not in namespace:
        raise ContractIRWellFormednessError(
            f"payoff underlier {underlier_id!r} is not present in the contract underlier namespace"
        )


def _validate_predicate(predicate: Predicate, *, namespace: tuple[str, ...]) -> None:
    if isinstance(predicate, (Gt, Ge, Lt, Le)):
        _validate_payoff_expr(predicate.lhs, namespace=namespace)
        _validate_payoff_expr(predicate.rhs, namespace=namespace)
        return
    if isinstance(predicate, (And, Or)):
        for child in predicate.args:
            _validate_predicate(child, namespace=namespace)
        return
    if isinstance(predicate, Not):
        _validate_predicate(predicate.arg, namespace=namespace)
        return
    raise ContractIRWellFormednessError(
        f"unsupported predicate node {type(predicate).__name__}"
    )


def _validate_payoff_expr(expr: PayoffExpr, *, namespace: tuple[str, ...]) -> None:
    if isinstance(expr, (Constant, Strike)):
        return
    if isinstance(expr, Spot):
        _validate_underlier_reference(expr.underlier_id, namespace=namespace)
        return
    if isinstance(expr, Forward):
        _validate_underlier_reference(expr.underlier_id, namespace=namespace)
        return
    if isinstance(expr, (SwapRate, Annuity)):
        _validate_underlier_reference(expr.underlier_id, namespace=namespace)
        return
    if isinstance(expr, CurveQuote):
        _validate_underlier_reference(expr.curve_id, namespace=namespace)
        return
    if isinstance(expr, SurfaceQuote):
        _validate_underlier_reference(expr.surface_id, namespace=namespace)
        return
    if isinstance(expr, VarianceObservable):
        _validate_underlier_reference(expr.underlier_id, namespace=namespace)
        return
    if isinstance(expr, ArithmeticMean):
        _validate_payoff_expr(expr.expr, namespace=namespace)
        return
    if isinstance(expr, LinearBasket):
        for _, child in expr.terms:
            _validate_payoff_expr(child, namespace=namespace)
        return
    if isinstance(expr, (Max, Min, Add, Mul)):
        for child in expr.args:
            _validate_payoff_expr(child, namespace=namespace)
        return
    if isinstance(expr, Sub):
        _validate_payoff_expr(expr.lhs, namespace=namespace)
        _validate_payoff_expr(expr.rhs, namespace=namespace)
        return
    if isinstance(expr, Scaled):
        _validate_payoff_expr(expr.scalar, namespace=namespace)
        _validate_payoff_expr(expr.body, namespace=namespace)
        return
    if isinstance(expr, Indicator):
        _validate_predicate(expr.predicate, namespace=namespace)
        return
    raise ContractIRWellFormednessError(
        f"unsupported payoff node {type(expr).__name__}"
    )


def _validate_schedule_alignment(contract: ContractIR) -> None:
    if contract.observation.kind == "terminal" and contract.exercise.style == "european":
        return
    # Phase 2 does not require exercise/observation schedule equality globally.


@dataclass(frozen=True)
class PayoffEvalEnv:
    """Sparse numeric environment for payoff-level semantic checks."""

    values: Mapping[tuple[object, ...], float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "values", _freeze_mapping(self.values))


def _lookup_value(
    env: PayoffEvalEnv,
    *,
    key: tuple[object, ...],
    fallback: tuple[object, ...] | None = None,
) -> float:
    values = env.values
    if key in values:
        return float(values[key])
    if fallback is not None and fallback in values:
        return float(values[fallback])
    raise KeyError(f"missing payoff-evaluation input for key {key!r}")


def evaluate_payoff_expr(
    expr: PayoffExpr,
    env: PayoffEvalEnv,
    *,
    as_of: date | None = None,
) -> float:
    if isinstance(expr, Constant):
        return expr.value
    if isinstance(expr, Strike):
        return expr.value
    if isinstance(expr, Spot):
        if as_of is not None:
            try:
                return _lookup_value(
                    env,
                    key=("spot", expr.underlier_id, as_of),
                    fallback=("spot", expr.underlier_id),
                )
            except KeyError:
                return _lookup_value(env, key=("spot", expr.underlier_id))
        return _lookup_value(env, key=("spot", expr.underlier_id))
    if isinstance(expr, Forward):
        schedule_key = expr.schedule.key()
        if as_of is not None:
            return _lookup_value(
                env,
                key=("forward", expr.underlier_id, schedule_key, as_of),
                fallback=("forward", expr.underlier_id, schedule_key),
            )
        return _lookup_value(env, key=("forward", expr.underlier_id, schedule_key))
    if isinstance(expr, SwapRate):
        schedule_key = expr.schedule.key()
        if as_of is not None:
            return _lookup_value(
                env,
                key=("swap_rate", expr.underlier_id, schedule_key, as_of),
                fallback=("swap_rate", expr.underlier_id, schedule_key),
            )
        return _lookup_value(env, key=("swap_rate", expr.underlier_id, schedule_key))
    if isinstance(expr, Annuity):
        schedule_key = expr.schedule.key()
        if as_of is not None:
            return _lookup_value(
                env,
                key=("annuity", expr.underlier_id, schedule_key, as_of),
                fallback=("annuity", expr.underlier_id, schedule_key),
            )
        return _lookup_value(env, key=("annuity", expr.underlier_id, schedule_key))
    if isinstance(expr, CurveQuote):
        return _lookup_value(
            env,
            key=("curve_quote", expr.curve_id, expr.coordinate, expr.convention),
        )
    if isinstance(expr, SurfaceQuote):
        return _lookup_value(
            env,
            key=("surface_quote", expr.surface_id, expr.coordinate, expr.convention),
        )
    if isinstance(expr, VarianceObservable):
        return _lookup_value(
            env,
            key=(
                "variance_observable",
                expr.underlier_id,
                expr.interval.t_start,
                expr.interval.t_end,
            ),
        )
    if isinstance(expr, LinearBasket):
        return math.fsum(
            weight * evaluate_payoff_expr(child, env, as_of=as_of)
            for weight, child in expr.terms
        )
    if isinstance(expr, ArithmeticMean):
        return math.fsum(
            evaluate_payoff_expr(expr.expr, env, as_of=scheduled_date)
            for scheduled_date in expr.schedule.dates
        ) / len(expr.schedule.dates)
    if isinstance(expr, Max):
        return max(evaluate_payoff_expr(child, env, as_of=as_of) for child in expr.args)
    if isinstance(expr, Min):
        return min(evaluate_payoff_expr(child, env, as_of=as_of) for child in expr.args)
    if isinstance(expr, Add):
        return math.fsum(
            evaluate_payoff_expr(child, env, as_of=as_of) for child in expr.args
        )
    if isinstance(expr, Sub):
        return evaluate_payoff_expr(expr.lhs, env, as_of=as_of) - evaluate_payoff_expr(
            expr.rhs,
            env,
            as_of=as_of,
        )
    if isinstance(expr, Mul):
        result = 1.0
        for child in expr.args:
            result *= evaluate_payoff_expr(child, env, as_of=as_of)
        return result
    if isinstance(expr, Scaled):
        return evaluate_payoff_expr(expr.scalar, env, as_of=as_of) * evaluate_payoff_expr(
            expr.body,
            env,
            as_of=as_of,
        )
    if isinstance(expr, Indicator):
        return 1.0 if evaluate_predicate(expr.predicate, env, as_of=as_of) else 0.0
    raise TypeError(f"unsupported payoff node {type(expr).__name__}")


def evaluate_predicate(
    predicate: Predicate,
    env: PayoffEvalEnv,
    *,
    as_of: date | None = None,
) -> bool:
    if isinstance(predicate, Gt):
        return evaluate_payoff_expr(predicate.lhs, env, as_of=as_of) > evaluate_payoff_expr(
            predicate.rhs,
            env,
            as_of=as_of,
        )
    if isinstance(predicate, Ge):
        return evaluate_payoff_expr(predicate.lhs, env, as_of=as_of) >= evaluate_payoff_expr(
            predicate.rhs,
            env,
            as_of=as_of,
        )
    if isinstance(predicate, Lt):
        return evaluate_payoff_expr(predicate.lhs, env, as_of=as_of) < evaluate_payoff_expr(
            predicate.rhs,
            env,
            as_of=as_of,
        )
    if isinstance(predicate, Le):
        return evaluate_payoff_expr(predicate.lhs, env, as_of=as_of) <= evaluate_payoff_expr(
            predicate.rhs,
            env,
            as_of=as_of,
        )
    if isinstance(predicate, And):
        return all(evaluate_predicate(child, env, as_of=as_of) for child in predicate.args)
    if isinstance(predicate, Or):
        return any(evaluate_predicate(child, env, as_of=as_of) for child in predicate.args)
    if isinstance(predicate, Not):
        return not evaluate_predicate(predicate.arg, env, as_of=as_of)
    raise TypeError(f"unsupported predicate node {type(predicate).__name__}")


def canonicalize(expr: PayoffExpr) -> PayoffExpr:
    if isinstance(
        expr,
        (
            Constant,
            Spot,
            Forward,
            SwapRate,
            Annuity,
            Strike,
            CurveQuote,
            SurfaceQuote,
            VarianceObservable,
        ),
    ):
        return expr
    if isinstance(expr, ArithmeticMean):
        return ArithmeticMean(canonicalize(expr.expr), expr.schedule)
    if isinstance(expr, LinearBasket):
        normalized_terms: list[tuple[float, PayoffExpr]] = []
        for weight, child in expr.terms:
            if float(weight) == 0.0:
                continue
            normalized_terms.append((float(weight), canonicalize(child)))
        if not normalized_terms:
            return Constant(0.0)
        if len(normalized_terms) == 1:
            weight, child = normalized_terms[0]
            return canonicalize(Scaled(Constant(weight), child))
        return LinearBasket(tuple(normalized_terms))
    if isinstance(expr, Sub):
        lhs = canonicalize(expr.lhs)
        rhs = canonicalize(expr.rhs)
        if isinstance(lhs, Constant) and isinstance(rhs, Constant):
            return Constant(lhs.value - rhs.value)
        if isinstance(rhs, Constant) and rhs.value == 0.0:
            return lhs
        if lhs == rhs:
            return Constant(0.0)
        return Sub(lhs, rhs)
    if isinstance(expr, Scaled):
        scalar = canonicalize(expr.scalar)
        body = canonicalize(expr.body)
        if isinstance(body, Constant) and body.value == 0.0:
            return Constant(0.0)
        if isinstance(scalar, Constant):
            if scalar.value == 0.0:
                return Constant(0.0)
            if scalar.value == 1.0:
                return body
            if isinstance(body, Constant):
                return Constant(scalar.value * body.value)
            if isinstance(body, Scaled) and isinstance(body.scalar, Constant):
                return canonicalize(
                    Scaled(Constant(scalar.value * body.scalar.value), body.body)
                )
        return Scaled(scalar, body)
    if isinstance(expr, Add):
        normalized = _canonicalize_variadic_children(expr.args, Add)
        constant_sum = 0.0
        out: list[PayoffExpr] = []
        for child in normalized:
            if isinstance(child, Constant):
                constant_sum += child.value
            else:
                out.append(child)
        if constant_sum != 0.0:
            out.append(Constant(constant_sum))
        out.sort(key=_expr_sort_key)
        if not out:
            return Constant(0.0)
        if len(out) == 1:
            return out[0]
        return Add(tuple(out))
    if isinstance(expr, Mul):
        normalized = _canonicalize_variadic_children(expr.args, Mul)
        constant_product = 1.0
        out: list[PayoffExpr] = []
        for child in normalized:
            if isinstance(child, Constant):
                if child.value == 0.0:
                    return Constant(0.0)
                constant_product *= child.value
            else:
                out.append(child)
        if constant_product != 1.0 or not out:
            out.append(Constant(constant_product))
        out = [child for child in out if not (isinstance(child, Constant) and child.value == 1.0 and len(out) > 1)]
        out.sort(key=_expr_sort_key)
        if not out:
            return Constant(1.0)
        if len(out) == 1:
            return out[0]
        return Mul(tuple(out))
    if isinstance(expr, Max):
        return _canonicalize_extrema(Max, expr.args)
    if isinstance(expr, Min):
        return _canonicalize_extrema(Min, expr.args)
    if isinstance(expr, Indicator):
        return Indicator(_canonicalize_predicate(expr.predicate))
    raise TypeError(f"unsupported payoff node {type(expr).__name__}")


def _canonicalize_extrema(cls, args: tuple[PayoffExpr, ...]) -> PayoffExpr:
    normalized = _canonicalize_variadic_children(args, cls)
    if all(isinstance(child, Constant) for child in normalized):
        values = [child.value for child in normalized]
        return Constant(max(values) if cls is Max else min(values))

    factored = _factor_common_positive_scalar(cls, tuple(normalized))
    if factored is not None:
        return canonicalize(factored)

    unique: list[PayoffExpr] = []
    seen: set[tuple[object, ...]] = set()
    for child in sorted(normalized, key=_expr_sort_key):
        key = _expr_sort_key(child)
        if key in seen:
            continue
        seen.add(key)
        unique.append(child)
    if len(unique) == 1:
        return unique[0]
    return cls(tuple(unique))


def _factor_common_positive_scalar(cls, args: tuple[PayoffExpr, ...]) -> PayoffExpr | None:
    scaled_args = [child for child in args if isinstance(child, Scaled)]
    if not scaled_args:
        return None
    common_scalar = scaled_args[0].scalar
    if not _is_locally_nonnegative(common_scalar):
        return None
    if any(child.scalar != common_scalar for child in scaled_args[1:]):
        return None
    inner_args: list[PayoffExpr] = []
    for child in args:
        if isinstance(child, Scaled) and child.scalar == common_scalar:
            inner_args.append(child.body)
            continue
        if isinstance(child, Constant) and child.value == 0.0:
            inner_args.append(Constant(0.0))
            continue
        return None
    return Scaled(common_scalar, cls(tuple(inner_args)))


def _flatten_variadic(args: tuple[PayoffExpr, ...], cls) -> list[PayoffExpr]:
    out: list[PayoffExpr] = []
    for child in args:
        if isinstance(child, cls):
            out.extend(_flatten_variadic(child.args, cls))
        else:
            out.append(child)
    return out


def _canonicalize_variadic_children(args: tuple[PayoffExpr, ...], cls) -> list[PayoffExpr]:
    out: list[PayoffExpr] = []
    for child in args:
        normalized = canonicalize(child)
        if isinstance(normalized, cls):
            out.extend(_flatten_variadic(normalized.args, cls))
        else:
            out.append(normalized)
    return out


def _canonicalize_predicate(predicate: Predicate) -> Predicate:
    if isinstance(predicate, Gt):
        return Gt(canonicalize(predicate.lhs), canonicalize(predicate.rhs))
    if isinstance(predicate, Ge):
        return Ge(canonicalize(predicate.lhs), canonicalize(predicate.rhs))
    if isinstance(predicate, Lt):
        return Lt(canonicalize(predicate.lhs), canonicalize(predicate.rhs))
    if isinstance(predicate, Le):
        return Le(canonicalize(predicate.lhs), canonicalize(predicate.rhs))
    if isinstance(predicate, And):
        children = _flatten_predicates(predicate.args, And)
        normalized = [_canonicalize_predicate(child) for child in children]
        deduped = _dedupe_predicates(normalized)
        if len(deduped) == 1:
            return deduped[0]
        return And(tuple(sorted(deduped, key=_predicate_sort_key)))
    if isinstance(predicate, Or):
        children = _flatten_predicates(predicate.args, Or)
        normalized = [_canonicalize_predicate(child) for child in children]
        deduped = _dedupe_predicates(normalized)
        if len(deduped) == 1:
            return deduped[0]
        return Or(tuple(sorted(deduped, key=_predicate_sort_key)))
    if isinstance(predicate, Not):
        return Not(_canonicalize_predicate(predicate.arg))
    raise TypeError(f"unsupported predicate node {type(predicate).__name__}")


def _flatten_predicates(args: tuple[Predicate, ...], cls) -> list[Predicate]:
    out: list[Predicate] = []
    for child in args:
        if isinstance(child, cls):
            out.extend(_flatten_predicates(child.args, cls))
        else:
            out.append(child)
    return out


def _dedupe_predicates(args: list[Predicate]) -> list[Predicate]:
    out: list[Predicate] = []
    seen: set[tuple[object, ...]] = set()
    for child in args:
        key = _predicate_sort_key(child)
        if key in seen:
            continue
        seen.add(key)
        out.append(child)
    return out


def _is_locally_nonnegative(expr: PayoffExpr) -> bool:
    if isinstance(expr, Constant):
        return expr.value >= 0.0
    if isinstance(expr, Annuity):
        return True
    return False


def _schedule_key(schedule: Schedule) -> tuple[object, ...]:
    if isinstance(schedule, Singleton):
        return ("singleton", schedule.t)
    if isinstance(schedule, FiniteSchedule):
        return ("finite", *schedule.dates)
    if isinstance(schedule, ContinuousInterval):
        return ("interval", schedule.t_start, schedule.t_end)
    raise TypeError(f"unsupported schedule {type(schedule).__name__}")


def _expr_sort_key(expr: PayoffExpr) -> tuple[object, ...]:
    if isinstance(expr, Sub):
        return (0, _expr_sort_key(expr.lhs), _expr_sort_key(expr.rhs))
    if isinstance(expr, Spot):
        return (1, expr.underlier_id)
    if isinstance(expr, Forward):
        return (2, expr.underlier_id, _schedule_key(expr.schedule))
    if isinstance(expr, SwapRate):
        return (3, expr.underlier_id, _schedule_key(expr.schedule))
    if isinstance(expr, Annuity):
        return (4, expr.underlier_id, _schedule_key(expr.schedule))
    if isinstance(expr, CurveQuote):
        return (5, expr.curve_id, _coordinate_sort_key(expr.coordinate), expr.convention)
    if isinstance(expr, SurfaceQuote):
        return (6, expr.surface_id, _coordinate_sort_key(expr.coordinate), expr.convention)
    if isinstance(expr, LinearBasket):
        return (
            7,
            tuple((weight, _expr_sort_key(child)) for weight, child in expr.terms),
        )
    if isinstance(expr, ArithmeticMean):
        return (8, _expr_sort_key(expr.expr), _schedule_key(expr.schedule))
    if isinstance(expr, VarianceObservable):
        return (9, expr.underlier_id, expr.interval.t_start, expr.interval.t_end)
    if isinstance(expr, Constant):
        return (10, expr.value)
    if isinstance(expr, Indicator):
        return (11, _predicate_sort_key(expr.predicate))
    if isinstance(expr, Scaled):
        return (12, _expr_sort_key(expr.scalar), _expr_sort_key(expr.body))
    if isinstance(expr, Add):
        return (13, tuple(_expr_sort_key(child) for child in expr.args))
    if isinstance(expr, Mul):
        return (14, tuple(_expr_sort_key(child) for child in expr.args))
    if isinstance(expr, Max):
        return (15, tuple(_expr_sort_key(child) for child in expr.args))
    if isinstance(expr, Min):
        return (16, tuple(_expr_sort_key(child) for child in expr.args))
    if isinstance(expr, Strike):
        return (17, expr.value)
    raise TypeError(f"unsupported payoff node {type(expr).__name__}")


def _coordinate_sort_key(coordinate: CurveCoordinate | SurfaceCoordinate) -> tuple[object, ...]:
    if isinstance(coordinate, ParRateTenor):
        return (0, coordinate.tenor)
    if isinstance(coordinate, ZeroRateTenor):
        return (1, coordinate.tenor)
    if isinstance(coordinate, ForwardRateInterval):
        return (2, coordinate.start_tenor, coordinate.end_tenor)
    if isinstance(coordinate, VolPoint):
        return (3, coordinate.option_tenor, coordinate.strike, coordinate.strike_style)
    if isinstance(coordinate, VolDeltaPoint):
        return (4, coordinate.option_tenor, coordinate.delta, coordinate.delta_style)
    raise TypeError(f"unsupported coordinate node {type(coordinate).__name__}")


def _predicate_sort_key(predicate: Predicate) -> tuple[object, ...]:
    if isinstance(predicate, Gt):
        return (0, _expr_sort_key(predicate.lhs), _expr_sort_key(predicate.rhs))
    if isinstance(predicate, Ge):
        return (1, _expr_sort_key(predicate.lhs), _expr_sort_key(predicate.rhs))
    if isinstance(predicate, Lt):
        return (2, _expr_sort_key(predicate.lhs), _expr_sort_key(predicate.rhs))
    if isinstance(predicate, Le):
        return (3, _expr_sort_key(predicate.lhs), _expr_sort_key(predicate.rhs))
    if isinstance(predicate, And):
        return (4, tuple(_predicate_sort_key(child) for child in predicate.args))
    if isinstance(predicate, Or):
        return (5, tuple(_predicate_sort_key(child) for child in predicate.args))
    if isinstance(predicate, Not):
        return (6, _predicate_sort_key(predicate.arg))
    raise TypeError(f"unsupported predicate node {type(predicate).__name__}")


__all__ = [
    "Add",
    "And",
    "Annuity",
    "ArithmeticMean",
    "CompositeUnderlying",
    "Constant",
    "ContinuousInterval",
    "ContractIR",
    "ContractIRWellFormednessError",
    "CurveQuote",
    "EquitySpot",
    "Exercise",
    "FiniteSchedule",
    "Forward",
    "ForwardRate",
    "ForwardRateInterval",
    "Ge",
    "Gt",
    "Indicator",
    "Le",
    "LinearBasket",
    "Lt",
    "Max",
    "Min",
    "Mul",
    "Not",
    "Observation",
    "Or",
    "PayoffEvalEnv",
    "ParRateTenor",
    "QuoteCurve",
    "QuoteSurface",
    "RateCurve",
    "Scaled",
    "Singleton",
    "Spot",
    "Strike",
    "Sub",
    "SurfaceQuote",
    "SwapRate",
    "Underlying",
    "VarianceObservable",
    "VolDeltaPoint",
    "VolPoint",
    "ZeroRateTenor",
    "canonicalize",
    "evaluate_payoff_expr",
    "evaluate_predicate",
]
