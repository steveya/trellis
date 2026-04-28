"""Execution-layer IR value objects.

The XIR.0 execution seam is intentionally model-free and route-free.  It
records contractual execution structure only; downstream pricing routes,
discounting policy, measure choice, and model selection remain outside this
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, TypeAlias, runtime_checkable


class ExecutionIRWellFormednessError(ValueError):
    """Raised when an execution IR node violates a local invariant."""


MetadataItems: TypeAlias = tuple[tuple[str, object], ...]


def _text(value: object) -> str:
    return str(value or "").strip()


def _lower_text(value: object) -> str:
    return _text(value).lower()


def _upper_text(value: object) -> str:
    return _text(value).upper()


def _tuple_text(values: object) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(values, str):
        values = (values,)
    for value in values or ():
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _frozen_text_set(values: object) -> frozenset[str]:
    return frozenset(_tuple_text(values))


def _metadata_items(metadata: Mapping[str, object] | MetadataItems | None) -> MetadataItems:
    if metadata is None:
        return ()
    if isinstance(metadata, Mapping):
        items = metadata.items()
    else:
        items = metadata
    normalized: list[tuple[str, object]] = []
    seen: set[str] = set()
    for key, value in items:
        text_key = _text(key)
        if not text_key or text_key in seen:
            continue
        seen.add(text_key)
        normalized.append((text_key, value))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@runtime_checkable
class SupportsSourceTrack(Protocol):
    """Structural source shape accepted by execution compiler entrypoints."""

    semantic_id: str


@dataclass(frozen=True)
class SourceTrack:
    """Metadata describing the upstream semantic authority for this execution IR."""

    source_kind: str
    semantic_id: str = ""
    product_family: str = ""
    instrument_class: str = ""
    source_ref: str = ""
    source_metadata: Mapping[str, object] | MetadataItems | None = None

    def __post_init__(self) -> None:
        source_kind = _lower_text(self.source_kind)
        if not source_kind:
            raise ExecutionIRWellFormednessError(
                "SourceTrack.source_kind must be a non-empty string"
            )
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "semantic_id", _text(self.semantic_id))
        object.__setattr__(self, "product_family", _lower_text(self.product_family))
        object.__setattr__(self, "instrument_class", _lower_text(self.instrument_class))
        object.__setattr__(self, "source_ref", _text(self.source_ref))
        object.__setattr__(
            self,
            "source_metadata",
            _metadata_items(self.source_metadata),
        )


@dataclass(frozen=True)
class RequirementHints:
    """Route-free requirements implied by execution structure."""

    market_inputs: frozenset[str] = frozenset()
    state_variables: frozenset[str] = frozenset()
    timeline_roles: frozenset[str] = frozenset()
    unsupported_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_inputs", _frozen_text_set(self.market_inputs))
        object.__setattr__(
            self,
            "state_variables",
            _frozen_text_set(self.state_variables),
        )
        object.__setattr__(self, "timeline_roles", _frozen_text_set(self.timeline_roles))
        object.__setattr__(
            self,
            "unsupported_reasons",
            _tuple_text(self.unsupported_reasons),
        )


@dataclass(frozen=True)
class ExecutionMetadata:
    """Operator-facing metadata for diagnostics and summaries."""

    schema_version: str = "xir.0"
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    unsupported_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, object] | MetadataItems | None = None

    def __post_init__(self) -> None:
        schema_version = _lower_text(self.schema_version) or "xir.0"
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "tags", _tuple_text(self.tags))
        object.__setattr__(self, "notes", _tuple_text(self.notes))
        object.__setattr__(
            self,
            "unsupported_reasons",
            _tuple_text(self.unsupported_reasons),
        )
        object.__setattr__(self, "metadata", _metadata_items(self.metadata))


@dataclass(frozen=True)
class KnownCashflowObligation:
    """Known undiscounted cashflow obligation."""

    obligation_id: str
    payment_date: object
    currency: str
    amount: float
    payer: str = ""
    receiver: str = ""
    obligation_kind: str = field(default="known_cashflow", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id))
        object.__setattr__(self, "currency", _upper_text(self.currency))
        object.__setattr__(self, "amount", float(self.amount))
        object.__setattr__(self, "payer", _lower_text(self.payer))
        object.__setattr__(self, "receiver", _lower_text(self.receiver))


@dataclass(frozen=True)
class CouponLegExecution:
    """Placeholder for a future lowered coupon-leg execution program."""

    obligation_id: str
    leg_id: str = ""
    currency: str = ""
    schedule_role: str = ""
    formula_ref: str = ""
    obligation_kind: str = field(default="coupon_leg", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id))
        object.__setattr__(self, "leg_id", _text(self.leg_id))
        object.__setattr__(self, "currency", _upper_text(self.currency))
        object.__setattr__(self, "schedule_role", _lower_text(self.schedule_role))
        object.__setattr__(self, "formula_ref", _text(self.formula_ref))


@dataclass(frozen=True)
class PeriodRateOptionStripExecution:
    """Placeholder for future cap/floor-style period-rate option strips."""

    obligation_id: str
    strip_id: str = ""
    currency: str = ""
    schedule_role: str = ""
    option_style: str = ""
    obligation_kind: str = field(default="period_rate_option_strip", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id))
        object.__setattr__(self, "strip_id", _text(self.strip_id))
        object.__setattr__(self, "currency", _upper_text(self.currency))
        object.__setattr__(self, "schedule_role", _lower_text(self.schedule_role))
        object.__setattr__(self, "option_style", _lower_text(self.option_style))


@dataclass(frozen=True)
class ContingentSettlement:
    """Placeholder for settlement conditional on an event or state expression."""

    obligation_id: str
    condition_ref: str = ""
    settlement_ref: str = ""
    currency: str = ""
    obligation_kind: str = field(default="contingent_settlement", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id))
        object.__setattr__(self, "condition_ref", _text(self.condition_ref))
        object.__setattr__(self, "settlement_ref", _text(self.settlement_ref))
        object.__setattr__(self, "currency", _upper_text(self.currency))


@dataclass(frozen=True)
class PrincipalExchange:
    """Placeholder for initial, final, or intermediate principal exchanges."""

    obligation_id: str
    exchange_date: object
    currency: str
    amount: float
    direction: str = ""
    obligation_kind: str = field(default="principal_exchange", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id))
        object.__setattr__(self, "currency", _upper_text(self.currency))
        object.__setattr__(self, "amount", float(self.amount))
        object.__setattr__(self, "direction", _lower_text(self.direction))


ExecutionObligation: TypeAlias = (
    KnownCashflowObligation
    | CouponLegExecution
    | PeriodRateOptionStripExecution
    | ContingentSettlement
    | PrincipalExchange
)


@dataclass(frozen=True)
class ObservableBinding:
    """Route-free observable reference used by execution obligations/events."""

    observable_id: str
    observable_kind: str
    source_ref: str = ""
    currency: str = ""
    tenor: str = ""
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] | MetadataItems | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observable_id", _text(self.observable_id))
        object.__setattr__(self, "observable_kind", _lower_text(self.observable_kind))
        object.__setattr__(self, "source_ref", _text(self.source_ref))
        object.__setattr__(self, "currency", _upper_text(self.currency))
        object.__setattr__(self, "tenor", _text(self.tenor))
        object.__setattr__(self, "tags", _tuple_text(self.tags))
        object.__setattr__(self, "metadata", _metadata_items(self.metadata))


@dataclass(frozen=True)
class SpotObservableRef(ObservableBinding):
    """Spot observable reference."""

    observable_kind: str = "spot"


@dataclass(frozen=True)
class ForwardRateObservableRef(ObservableBinding):
    """Forward-rate observable reference."""

    observable_kind: str = "forward_rate"


@dataclass(frozen=True)
class SwapRateObservableRef(ObservableBinding):
    """Swap-rate observable reference."""

    observable_kind: str = "swap_rate"


@dataclass(frozen=True)
class CurveQuoteObservableRef(ObservableBinding):
    """Curve-quote observable reference."""

    observable_kind: str = "curve_quote"


@dataclass(frozen=True)
class SurfaceQuoteObservableRef(ObservableBinding):
    """Surface-quote observable reference."""

    observable_kind: str = "surface_quote"


@dataclass(frozen=True)
class ExecutionEvent:
    """Placeholder event in the execution timeline."""

    event_id: str
    event_kind: str
    schedule_role: str = ""
    phase: str = ""
    event_date: object | None = None
    metadata: Mapping[str, object] | MetadataItems | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id))
        object.__setattr__(self, "event_kind", _lower_text(self.event_kind))
        object.__setattr__(self, "schedule_role", _lower_text(self.schedule_role))
        object.__setattr__(self, "phase", _lower_text(self.phase))
        object.__setattr__(self, "metadata", _metadata_items(self.metadata))


@dataclass(frozen=True)
class ExecutionEventPlan:
    """Ordered event-plan placeholder."""

    events: tuple[ExecutionEvent, ...] = ()
    phase_order: tuple[str, ...] = (
        "fixing",
        "observation",
        "accrual_boundary",
        "coupon",
        "payment",
        "decision",
        "termination",
        "state_reset",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events or ()))
        object.__setattr__(self, "phase_order", _tuple_text(self.phase_order))
        for event in self.events:
            if not isinstance(event, ExecutionEvent):
                raise ExecutionIRWellFormednessError(
                    "ExecutionEventPlan.events must contain ExecutionEvent values"
                )


@dataclass(frozen=True)
class ExecutionStateField:
    """Placeholder state-field declaration."""

    name: str
    domain: str = ""
    initial_value: object | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name))
        object.__setattr__(self, "domain", _lower_text(self.domain))
        object.__setattr__(self, "tags", _tuple_text(self.tags))


@dataclass(frozen=True)
class ExecutionStateSchema:
    """State schema placeholder for dynamic execution programs."""

    fields: tuple[ExecutionStateField, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields or ()))
        seen: set[str] = set()
        for state_field in self.fields:
            if not isinstance(state_field, ExecutionStateField):
                raise ExecutionIRWellFormednessError(
                    "ExecutionStateSchema.fields must contain ExecutionStateField values"
                )
            if state_field.name and state_field.name in seen:
                raise ExecutionIRWellFormednessError(
                    f"duplicate execution state field {state_field.name!r}"
                )
            seen.add(state_field.name)


@dataclass(frozen=True)
class DecisionAction:
    """Placeholder decision action."""

    action_id: str
    action_type: str = ""
    controller_role: str = ""
    schedule_role: str = ""
    state_updates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id))
        object.__setattr__(self, "action_type", _lower_text(self.action_type))
        object.__setattr__(self, "controller_role", _lower_text(self.controller_role))
        object.__setattr__(self, "schedule_role", _lower_text(self.schedule_role))
        object.__setattr__(self, "state_updates", _tuple_text(self.state_updates))


@dataclass(frozen=True)
class DecisionProgram:
    """Decision-program placeholder."""

    actions: tuple[DecisionAction, ...] = ()
    controller_role: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions or ()))
        object.__setattr__(self, "controller_role", _lower_text(self.controller_role))
        for action in self.actions:
            if not isinstance(action, DecisionAction):
                raise ExecutionIRWellFormednessError(
                    "DecisionProgram.actions must contain DecisionAction values"
                )


@dataclass(frozen=True)
class SettlementStep:
    """Placeholder settlement program step."""

    step_id: str
    settlement_kind: str = ""
    expression: str = ""
    currency: str = ""
    payment_date_role: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _text(self.step_id))
        object.__setattr__(self, "settlement_kind", _lower_text(self.settlement_kind))
        object.__setattr__(self, "expression", _text(self.expression))
        object.__setattr__(self, "currency", _upper_text(self.currency))
        object.__setattr__(self, "payment_date_role", _lower_text(self.payment_date_role))


@dataclass(frozen=True)
class SettlementProgram:
    """Settlement-program placeholder."""

    steps: tuple[SettlementStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps or ()))
        for step in self.steps:
            if not isinstance(step, SettlementStep):
                raise ExecutionIRWellFormednessError(
                    "SettlementProgram.steps must contain SettlementStep values"
                )


@dataclass(frozen=True)
class ContractExecutionIR:
    """Route-free, model-free execution seam for upstream contract semantics."""

    source_track: SourceTrack
    obligations: tuple[ExecutionObligation, ...] = ()
    observables: tuple[ObservableBinding, ...] = ()
    event_plan: ExecutionEventPlan = field(default_factory=ExecutionEventPlan)
    state_schema: ExecutionStateSchema = field(default_factory=ExecutionStateSchema)
    decision_program: DecisionProgram = field(default_factory=DecisionProgram)
    settlement_program: SettlementProgram = field(default_factory=SettlementProgram)
    requirement_hints: RequirementHints = field(default_factory=RequirementHints)
    execution_metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.source_track, SourceTrack):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.source_track must be a SourceTrack"
            )
        object.__setattr__(self, "obligations", tuple(self.obligations or ()))
        object.__setattr__(self, "observables", tuple(self.observables or ()))
        for obligation in self.obligations:
            if not isinstance(
                obligation,
                (
                    KnownCashflowObligation,
                    CouponLegExecution,
                    PeriodRateOptionStripExecution,
                    ContingentSettlement,
                    PrincipalExchange,
                ),
            ):
                raise ExecutionIRWellFormednessError(
                    "ContractExecutionIR.obligations contains an unsupported value"
                )
        for observable in self.observables:
            if not isinstance(observable, ObservableBinding):
                raise ExecutionIRWellFormednessError(
                    "ContractExecutionIR.observables must contain ObservableBinding values"
                )
        if not isinstance(self.event_plan, ExecutionEventPlan):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.event_plan must be an ExecutionEventPlan"
            )
        if not isinstance(self.state_schema, ExecutionStateSchema):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.state_schema must be an ExecutionStateSchema"
            )
        if not isinstance(self.decision_program, DecisionProgram):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.decision_program must be a DecisionProgram"
            )
        if not isinstance(self.settlement_program, SettlementProgram):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.settlement_program must be a SettlementProgram"
            )
        if not isinstance(self.requirement_hints, RequirementHints):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.requirement_hints must be RequirementHints"
            )
        if not isinstance(self.execution_metadata, ExecutionMetadata):
            raise ExecutionIRWellFormednessError(
                "ContractExecutionIR.execution_metadata must be ExecutionMetadata"
            )

    @classmethod
    def empty(
        cls,
        *,
        source_track: SourceTrack,
        unsupported_reasons: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
    ) -> "ContractExecutionIR":
        """Build an explicit empty seam artifact for unsupported lowerings."""
        reasons = _tuple_text(unsupported_reasons)
        return cls(
            source_track=source_track,
            requirement_hints=RequirementHints(unsupported_reasons=reasons),
            execution_metadata=ExecutionMetadata(
                tags=tags,
                unsupported_reasons=reasons,
            ),
        )
