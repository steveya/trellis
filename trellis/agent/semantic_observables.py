"""Semantic observable and predicate grammar for conditional coupon contracts.

The first executable conditional-accrual wave admits only single rate-index
observables. CMS, spread, spot, basket, and transform nodes are intentionally
representable but blocked until a later route/lowering ticket admits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PredicateGrammarValidationError(ValueError):
    """Raised when a semantic observable or predicate expression is malformed."""


FIRST_WAVE_CONDITIONAL_ACCRUAL_OBSERVABLES = frozenset({"rate_index"})
SUPPORTED_MISSING_FIXING_POLICIES = frozenset(
    {
        "fail_fast",
        "project_forward_for_future_only",
        "require_history_for_observed",
    }
)


def _text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PredicateGrammarValidationError(f"{label} must be a non-empty string")
    return text


def _optional_text(value: object) -> str:
    return str(value or "").strip()


def _lower_text(value: object, *, label: str) -> str:
    return _text(value, label=label).lower()


def _float(value: float | int, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive only
        raise PredicateGrammarValidationError(f"{label} must be numeric") from exc


def _tuple_text(values: object, *, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    if not result:
        raise PredicateGrammarValidationError(f"{label} must be non-empty")
    return tuple(result)


@dataclass(frozen=True)
class ObservationMetadata:
    """Observation schedule and fixing policy attached to one observable."""

    schedule_role: str = "observation_dates"
    fixing_date_role: str = "fixing_dates"
    missing_fixing_policy: str = "fail_fast"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule_role",
            _text(self.schedule_role, label="ObservationMetadata.schedule_role"),
        )
        object.__setattr__(
            self,
            "fixing_date_role",
            _text(self.fixing_date_role, label="ObservationMetadata.fixing_date_role"),
        )
        policy = _lower_text(
            self.missing_fixing_policy,
            label="ObservationMetadata.missing_fixing_policy",
        )
        if policy not in SUPPORTED_MISSING_FIXING_POLICIES:
            raise PredicateGrammarValidationError(
                "ObservationMetadata.missing_fixing_policy must be one of "
                f"{sorted(SUPPORTED_MISSING_FIXING_POLICIES)}"
            )
        object.__setattr__(self, "missing_fixing_policy", policy)


@dataclass(frozen=True)
class ObservableSupportBlocker:
    """Structured blocker for an observable family not admitted in this slice."""

    observable_family: str
    blocker_id: str
    reason: str
    required_ticket: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observable_family",
            _lower_text(
                self.observable_family,
                label="ObservableSupportBlocker.observable_family",
            ),
        )
        object.__setattr__(
            self,
            "blocker_id",
            _text(self.blocker_id, label="ObservableSupportBlocker.blocker_id"),
        )
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, label="ObservableSupportBlocker.reason"),
        )
        object.__setattr__(self, "required_ticket", _optional_text(self.required_ticket))


@dataclass(frozen=True)
class RateIndexObservable:
    """Single rate-index observable admitted for first-wave range accruals."""

    observable_id: str
    index_name: str
    tenor: str = ""
    observation: ObservationMetadata = field(default_factory=ObservationMetadata)
    observable_family: str = field(default="rate_index", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observable_id",
            _text(self.observable_id, label="RateIndexObservable.observable_id"),
        )
        object.__setattr__(
            self,
            "index_name",
            _text(self.index_name, label="RateIndexObservable.index_name").upper(),
        )
        object.__setattr__(self, "tenor", _optional_text(self.tenor).upper())
        if not isinstance(self.observation, ObservationMetadata):
            raise PredicateGrammarValidationError(
                "RateIndexObservable.observation must be ObservationMetadata"
            )


@dataclass(frozen=True)
class CmsRateObservable:
    """CMS rate placeholder; representable but not admitted for this wave."""

    observable_id: str
    curve_id: str
    tenor: str
    observation: ObservationMetadata = field(default_factory=ObservationMetadata)
    observable_family: str = field(default="cms_rate", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observable_id",
            _text(self.observable_id, label="CmsRateObservable.observable_id"),
        )
        object.__setattr__(
            self,
            "curve_id",
            _text(self.curve_id, label="CmsRateObservable.curve_id").upper(),
        )
        object.__setattr__(
            self,
            "tenor",
            _text(self.tenor, label="CmsRateObservable.tenor").upper(),
        )
        if not isinstance(self.observation, ObservationMetadata):
            raise PredicateGrammarValidationError(
                "CmsRateObservable.observation must be ObservationMetadata"
            )


@dataclass(frozen=True)
class SpotObservable:
    """Spot observable placeholder for later conditional equity/FX coupons."""

    observable_id: str
    asset_id: str
    observation: ObservationMetadata = field(default_factory=ObservationMetadata)
    observable_family: str = field(default="spot", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observable_id",
            _text(self.observable_id, label="SpotObservable.observable_id"),
        )
        object.__setattr__(
            self,
            "asset_id",
            _text(self.asset_id, label="SpotObservable.asset_id").upper(),
        )
        if not isinstance(self.observation, ObservationMetadata):
            raise PredicateGrammarValidationError(
                "SpotObservable.observation must be ObservationMetadata"
            )


@dataclass(frozen=True)
class BasketObservable:
    """Basket observable placeholder for later conditional basket coupons."""

    observable_id: str
    constituents: tuple[str, ...]
    observation: ObservationMetadata = field(default_factory=ObservationMetadata)
    observable_family: str = field(default="basket", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observable_id",
            _text(self.observable_id, label="BasketObservable.observable_id"),
        )
        object.__setattr__(
            self,
            "constituents",
            _tuple_text(self.constituents, label="BasketObservable.constituents"),
        )
        if not isinstance(self.observation, ObservationMetadata):
            raise PredicateGrammarValidationError(
                "BasketObservable.observation must be ObservationMetadata"
            )


@dataclass(frozen=True)
class TransformObservable:
    """Transform placeholder over one or more observable inputs."""

    observable_id: str
    transform_id: str
    inputs: tuple["ObservableExpr", ...]
    observation: ObservationMetadata = field(default_factory=ObservationMetadata)
    observable_family: str = field(default="transform", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observable_id",
            _text(self.observable_id, label="TransformObservable.observable_id"),
        )
        object.__setattr__(
            self,
            "transform_id",
            _text(self.transform_id, label="TransformObservable.transform_id"),
        )
        if not isinstance(self.inputs, tuple):
            object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.inputs:
            raise PredicateGrammarValidationError("TransformObservable.inputs must be non-empty")
        for item in self.inputs:
            _require_observable_expr(item, label="TransformObservable.inputs")
        if not isinstance(self.observation, ObservationMetadata):
            raise PredicateGrammarValidationError(
                "TransformObservable.observation must be ObservationMetadata"
            )


@dataclass(frozen=True)
class SpreadObservable:
    """Spread placeholder over two observable expressions."""

    left: "ObservableExpr"
    right: "ObservableExpr"
    spread_id: str = ""
    observation: ObservationMetadata = field(default_factory=ObservationMetadata)
    observable_family: str = field(default="spread", init=False)

    def __post_init__(self) -> None:
        _require_observable_expr(self.left, label="SpreadObservable.left")
        _require_observable_expr(self.right, label="SpreadObservable.right")
        object.__setattr__(self, "spread_id", _optional_text(self.spread_id))
        if not isinstance(self.observation, ObservationMetadata):
            raise PredicateGrammarValidationError(
                "SpreadObservable.observation must be ObservationMetadata"
            )


ObservableExpr = (
    RateIndexObservable
    | CmsRateObservable
    | SpotObservable
    | BasketObservable
    | TransformObservable
    | SpreadObservable
)


@dataclass(frozen=True)
class BetweenPredicate:
    observable: ObservableExpr
    lower_bound: float
    upper_bound: float
    inclusive_lower: bool = True
    inclusive_upper: bool = True

    def __post_init__(self) -> None:
        _require_observable_expr(self.observable, label="BetweenPredicate.observable")
        lower = _float(self.lower_bound, label="BetweenPredicate.lower_bound")
        upper = _float(self.upper_bound, label="BetweenPredicate.upper_bound")
        if lower > upper:
            raise PredicateGrammarValidationError(
                "BetweenPredicate requires lower_bound <= upper_bound"
            )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "inclusive_lower", bool(self.inclusive_lower))
        object.__setattr__(self, "inclusive_upper", bool(self.inclusive_upper))


@dataclass(frozen=True)
class GreaterThanPredicate:
    observable: ObservableExpr
    threshold: float
    inclusive: bool = False

    def __post_init__(self) -> None:
        _require_observable_expr(self.observable, label="GreaterThanPredicate.observable")
        object.__setattr__(
            self,
            "threshold",
            _float(self.threshold, label="GreaterThanPredicate.threshold"),
        )
        object.__setattr__(self, "inclusive", bool(self.inclusive))


@dataclass(frozen=True)
class LessThanPredicate:
    observable: ObservableExpr
    threshold: float
    inclusive: bool = False

    def __post_init__(self) -> None:
        _require_observable_expr(self.observable, label="LessThanPredicate.observable")
        object.__setattr__(
            self,
            "threshold",
            _float(self.threshold, label="LessThanPredicate.threshold"),
        )
        object.__setattr__(self, "inclusive", bool(self.inclusive))


@dataclass(frozen=True)
class AndPredicate:
    predicates: tuple["PredicateExpr", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", _predicate_tuple(self.predicates, label="AndPredicate.predicates"))


@dataclass(frozen=True)
class OrPredicate:
    predicates: tuple["PredicateExpr", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", _predicate_tuple(self.predicates, label="OrPredicate.predicates"))


@dataclass(frozen=True)
class NotPredicate:
    predicate: "PredicateExpr"

    def __post_init__(self) -> None:
        _require_predicate_expr(self.predicate, label="NotPredicate.predicate")


PredicateExpr = (
    BetweenPredicate
    | GreaterThanPredicate
    | LessThanPredicate
    | AndPredicate
    | OrPredicate
    | NotPredicate
)


def _require_observable_expr(value: object, *, label: str) -> None:
    if not isinstance(
        value,
        (
            RateIndexObservable,
            CmsRateObservable,
            SpotObservable,
            BasketObservable,
            TransformObservable,
            SpreadObservable,
        ),
    ):
        raise PredicateGrammarValidationError(f"{label} must be an ObservableExpr")


def _require_predicate_expr(value: object, *, label: str) -> None:
    if not isinstance(
        value,
        (
            BetweenPredicate,
            GreaterThanPredicate,
            LessThanPredicate,
            AndPredicate,
            OrPredicate,
            NotPredicate,
        ),
    ):
        raise PredicateGrammarValidationError(f"{label} must be a PredicateExpr")


def _predicate_tuple(values: object, *, label: str) -> tuple[PredicateExpr, ...]:
    if values is None:
        raise PredicateGrammarValidationError(f"{label} must be non-empty")
    if not isinstance(values, tuple):
        values = tuple(values)
    if not values:
        raise PredicateGrammarValidationError(f"{label} must be non-empty")
    for value in values:
        _require_predicate_expr(value, label=label)
    return values


def _dedupe_rate_index_keys(
    keys: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return tuple(result)


def _rate_index_keys_from_observable(
    observable: ObservableExpr,
) -> tuple[tuple[str, str], ...]:
    if isinstance(observable, RateIndexObservable):
        return ((observable.index_name, observable.tenor),)
    if isinstance(observable, SpreadObservable):
        return _dedupe_rate_index_keys(
            (
                *_rate_index_keys_from_observable(observable.left),
                *_rate_index_keys_from_observable(observable.right),
            )
        )
    if isinstance(observable, TransformObservable):
        return _dedupe_rate_index_keys(
            tuple(
                key
                for item in observable.inputs
                for key in _rate_index_keys_from_observable(item)
            )
        )
    return ()


def _rate_index_keys_from_predicate(
    predicate: PredicateExpr,
) -> tuple[tuple[str, str], ...]:
    if isinstance(predicate, (BetweenPredicate, GreaterThanPredicate, LessThanPredicate)):
        return _rate_index_keys_from_observable(predicate.observable)
    if isinstance(predicate, (AndPredicate, OrPredicate)):
        return _dedupe_rate_index_keys(
            tuple(
                key
                for item in predicate.predicates
                for key in _rate_index_keys_from_predicate(item)
            )
        )
    if isinstance(predicate, NotPredicate):
        return _rate_index_keys_from_predicate(predicate.predicate)
    raise PredicateGrammarValidationError("predicate must be a PredicateExpr")


def observable_support_blockers(observable: ObservableExpr) -> tuple[ObservableSupportBlocker, ...]:
    """Return blockers that prevent first-wave conditional-accrual admission."""
    _require_observable_expr(observable, label="observable")
    if isinstance(observable, RateIndexObservable):
        return ()
    if isinstance(observable, SpreadObservable):
        return (
            _blocker(
                "spread",
                "conditional_accrual_spread_observable_pending",
                "Spread observables are representable but not admitted for first-wave conditional accrual lowering.",
                required_ticket="QUA-1118",
            ),
            *observable_support_blockers(observable.left),
            *observable_support_blockers(observable.right),
        )
    if isinstance(observable, TransformObservable):
        child_blockers = tuple(
            blocker
            for item in observable.inputs
            for blocker in observable_support_blockers(item)
        )
        return (
            _blocker(
                "transform",
                "conditional_accrual_transform_observable_pending",
                "Transform observables require route-specific lowering support.",
                required_ticket="follow_on",
            ),
            *child_blockers,
        )
    family = observable.observable_family
    return (
        _blocker(
            family,
            f"conditional_accrual_{family}_observable_pending",
            f"{family} observables are representable but not admitted for first-wave conditional accrual lowering.",
            required_ticket="QUA-1118" if family == "cms_rate" else "follow_on",
        ),
    )


def predicate_support_blockers(predicate: PredicateExpr) -> tuple[ObservableSupportBlocker, ...]:
    """Return all observable-support blockers reachable from a predicate."""
    _require_predicate_expr(predicate, label="predicate")
    if isinstance(predicate, (BetweenPredicate, GreaterThanPredicate, LessThanPredicate)):
        blockers = observable_support_blockers(predicate.observable)
    elif isinstance(predicate, (AndPredicate, OrPredicate)):
        blockers = tuple(
            blocker
            for item in predicate.predicates
            for blocker in predicate_support_blockers(item)
        )
    elif isinstance(predicate, NotPredicate):
        blockers = predicate_support_blockers(predicate.predicate)
    else:
        raise PredicateGrammarValidationError("predicate must be a PredicateExpr")
    rate_index_keys = _rate_index_keys_from_predicate(predicate)
    if len(rate_index_keys) > 1:
        blockers = (
            *blockers,
            _blocker(
                "multi_index",
                "conditional_accrual_multi_index_predicate_pending",
                "First-wave conditional accrual lowering admits exactly one rate-index observable identity.",
                required_ticket="QUA-1118",
            ),
        )
    return _dedupe_blockers(blockers)


def validate_conditional_accrual_predicate(predicate: PredicateExpr) -> tuple[ObservableSupportBlocker, ...]:
    """Validate that a predicate is admitted for first-wave conditional accrual."""
    blockers = predicate_support_blockers(predicate)
    if blockers:
        families = ", ".join(blocker.observable_family for blocker in blockers)
        raise PredicateGrammarValidationError(
            f"unsupported observable family for conditional accrual predicate: {families}"
        )
    return ()


def _blocker(
    observable_family: str,
    blocker_id: str,
    reason: str,
    *,
    required_ticket: str,
) -> ObservableSupportBlocker:
    return ObservableSupportBlocker(
        observable_family=observable_family,
        blocker_id=blocker_id,
        reason=reason,
        required_ticket=required_ticket,
    )


def _dedupe_blockers(blockers) -> tuple[ObservableSupportBlocker, ...]:
    result: list[ObservableSupportBlocker] = []
    seen: set[tuple[str, str]] = set()
    for blocker in blockers:
        key = (blocker.observable_family, blocker.blocker_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(blocker)
    return tuple(result)


__all__ = [
    "AndPredicate",
    "BasketObservable",
    "BetweenPredicate",
    "CmsRateObservable",
    "FIRST_WAVE_CONDITIONAL_ACCRUAL_OBSERVABLES",
    "GreaterThanPredicate",
    "LessThanPredicate",
    "NotPredicate",
    "ObservableExpr",
    "ObservableSupportBlocker",
    "ObservationMetadata",
    "OrPredicate",
    "PredicateExpr",
    "PredicateGrammarValidationError",
    "RateIndexObservable",
    "SpotObservable",
    "SpreadObservable",
    "SUPPORTED_MISSING_FIXING_POLICIES",
    "TransformObservable",
    "observable_support_blockers",
    "predicate_support_blockers",
    "validate_conditional_accrual_predicate",
]
