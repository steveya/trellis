"""Dynamic event/state/control semantics over static contract bases."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from trellis.agent.contract_ir import ContractIR
from trellis.agent.static_leg_contract import SettlementRule, StaticLegContractIR


class DynamicContractIRWellFormednessError(ValueError):
    """Raised when a dynamic semantic contract violates a local invariant."""


def _require_text(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DynamicContractIRWellFormednessError(f"{label} must be a non-empty string")
    return text


DEFAULT_EVENT_ORDERING = (
    "observation",
    "coupon",
    "payment",
    "decision",
    "termination",
    "state_update",
)


@dataclass(frozen=True)
class StateFieldSpec:
    name: str
    domain: str
    initial_value: object
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, label="StateFieldSpec.name"))
        object.__setattr__(self, "domain", _require_text(self.domain, label="StateFieldSpec.domain"))
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class StateSchema:
    fields: tuple[StateFieldSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            object.__setattr__(self, "fields", tuple(self.fields))
        seen: set[str] = set()
        for field_spec in self.fields:
            if not isinstance(field_spec, StateFieldSpec):
                raise DynamicContractIRWellFormednessError(
                    "StateSchema.fields must contain StateFieldSpec values"
                )
            if field_spec.name in seen:
                raise DynamicContractIRWellFormednessError(
                    f"duplicate state field {field_spec.name!r}"
                )
            seen.add(field_spec.name)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field_spec.name for field_spec in self.fields)


@dataclass(frozen=True)
class StateUpdateSpec:
    field_name: str
    update_expression: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_name",
            _require_text(self.field_name, label="StateUpdateSpec.field_name"),
        )
        object.__setattr__(
            self,
            "update_expression",
            _require_text(
                self.update_expression,
                label="StateUpdateSpec.update_expression",
            ),
        )


@dataclass(frozen=True)
class ActionSpec:
    action_name: str
    action_type: str
    description: str = ""
    action_domain: str = "discrete"
    quantity_source: str = ""
    bounds_expression: str = ""
    state_updates: tuple[StateUpdateSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_name",
            _require_text(self.action_name, label="ActionSpec.action_name"),
        )
        object.__setattr__(
            self,
            "action_type",
            _require_text(self.action_type, label="ActionSpec.action_type").lower(),
        )
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(
            self,
            "action_domain",
            _require_text(self.action_domain, label="ActionSpec.action_domain").lower(),
        )
        object.__setattr__(self, "quantity_source", str(self.quantity_source or "").strip())
        object.__setattr__(self, "bounds_expression", str(self.bounds_expression or "").strip())
        if not isinstance(self.state_updates, tuple):
            object.__setattr__(self, "state_updates", tuple(self.state_updates))
        if self.action_domain not in {"discrete", "continuous", "singular"}:
            raise DynamicContractIRWellFormednessError(
                "ActionSpec.action_domain must be one of 'discrete', 'continuous', or 'singular'"
            )
        if self.action_domain != "discrete" and not self.quantity_source:
            raise DynamicContractIRWellFormednessError(
                "continuous or singular actions must declare ActionSpec.quantity_source"
            )
        for update in self.state_updates:
            if not isinstance(update, StateUpdateSpec):
                raise DynamicContractIRWellFormednessError(
                    "ActionSpec.state_updates must contain StateUpdateSpec values"
                )


@dataclass(frozen=True)
class ObservationEvent:
    label: str
    schedule_role: str
    observed_terms: tuple[str, ...]
    phase: str = "observation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="ObservationEvent.label"))
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="ObservationEvent.schedule_role"),
        )
        if not isinstance(self.observed_terms, tuple):
            object.__setattr__(self, "observed_terms", tuple(self.observed_terms))
        object.__setattr__(self, "phase", _require_text(self.phase, label="ObservationEvent.phase").lower())


@dataclass(frozen=True)
class CouponEvent:
    label: str
    schedule_role: str
    coupon_formula: str
    state_updates: tuple[StateUpdateSpec, ...] = ()
    phase: str = "coupon"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="CouponEvent.label"))
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="CouponEvent.schedule_role"),
        )
        object.__setattr__(
            self,
            "coupon_formula",
            _require_text(self.coupon_formula, label="CouponEvent.coupon_formula"),
        )
        if not isinstance(self.state_updates, tuple):
            object.__setattr__(self, "state_updates", tuple(self.state_updates))
        object.__setattr__(self, "phase", _require_text(self.phase, label="CouponEvent.phase").lower())


@dataclass(frozen=True)
class PaymentEvent:
    label: str
    schedule_role: str
    cashflow_formula: str
    phase: str = "payment"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="PaymentEvent.label"))
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="PaymentEvent.schedule_role"),
        )
        object.__setattr__(
            self,
            "cashflow_formula",
            _require_text(self.cashflow_formula, label="PaymentEvent.cashflow_formula"),
        )
        object.__setattr__(self, "phase", _require_text(self.phase, label="PaymentEvent.phase").lower())


@dataclass(frozen=True)
class DecisionEvent:
    label: str
    schedule_role: str
    action_set: tuple[ActionSpec, ...]
    controller_role: str
    phase: str = "decision"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="DecisionEvent.label"))
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="DecisionEvent.schedule_role"),
        )
        if not isinstance(self.action_set, tuple):
            object.__setattr__(self, "action_set", tuple(self.action_set))
        if not self.action_set:
            raise DynamicContractIRWellFormednessError(
                "DecisionEvent.action_set must be non-empty"
            )
        for action in self.action_set:
            if not isinstance(action, ActionSpec):
                raise DynamicContractIRWellFormednessError(
                    "DecisionEvent.action_set must contain ActionSpec values"
                )
        object.__setattr__(
            self,
            "controller_role",
            _require_text(self.controller_role, label="DecisionEvent.controller_role").lower(),
        )
        object.__setattr__(self, "phase", _require_text(self.phase, label="DecisionEvent.phase").lower())


@dataclass(frozen=True)
class AutomaticTerminationEvent:
    label: str
    trigger: str
    settlement_expression: str = ""
    phase: str = "termination"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label",
            _require_text(self.label, label="AutomaticTerminationEvent.label"),
        )
        object.__setattr__(
            self,
            "trigger",
            _require_text(self.trigger, label="AutomaticTerminationEvent.trigger"),
        )
        object.__setattr__(self, "settlement_expression", str(self.settlement_expression or "").strip())
        object.__setattr__(
            self,
            "phase",
            _require_text(self.phase, label="AutomaticTerminationEvent.phase").lower(),
        )


@dataclass(frozen=True)
class StateResetEvent:
    label: str
    schedule_role: str
    state_updates: tuple[StateUpdateSpec, ...]
    phase: str = "state_update"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="StateResetEvent.label"))
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="StateResetEvent.schedule_role"),
        )
        if not isinstance(self.state_updates, tuple):
            object.__setattr__(self, "state_updates", tuple(self.state_updates))
        if not self.state_updates:
            raise DynamicContractIRWellFormednessError(
                "StateResetEvent.state_updates must be non-empty"
            )
        object.__setattr__(
            self,
            "phase",
            _require_text(self.phase, label="StateResetEvent.phase").lower(),
        )


ContractEvent = (
    ObservationEvent
    | CouponEvent
    | PaymentEvent
    | DecisionEvent
    | AutomaticTerminationEvent
    | StateResetEvent
)


@dataclass(frozen=True)
class TerminationRule:
    label: str
    trigger: str
    settlement_expression: str = ""
    event_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="TerminationRule.label"))
        object.__setattr__(self, "trigger", _require_text(self.trigger, label="TerminationRule.trigger"))
        object.__setattr__(self, "settlement_expression", str(self.settlement_expression or "").strip())
        object.__setattr__(self, "event_label", str(self.event_label or "").strip())


@dataclass(frozen=True)
class EventTimeBucket:
    event_date: date
    phase_sequence: tuple[str, ...]
    events: tuple[ContractEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_date, date):
            raise DynamicContractIRWellFormednessError(
                "EventTimeBucket.event_date must be a date"
            )
        if not isinstance(self.phase_sequence, tuple):
            object.__setattr__(self, "phase_sequence", tuple(self.phase_sequence))
        if not self.phase_sequence:
            raise DynamicContractIRWellFormednessError(
                "EventTimeBucket.phase_sequence must be non-empty"
            )
        if not isinstance(self.events, tuple):
            object.__setattr__(self, "events", tuple(self.events))
        if not self.events:
            raise DynamicContractIRWellFormednessError(
                "EventTimeBucket.events must be non-empty"
            )
        for event in self.events:
            if not isinstance(
                event,
                (
                    ObservationEvent,
                    CouponEvent,
                    PaymentEvent,
                    DecisionEvent,
                    AutomaticTerminationEvent,
                    StateResetEvent,
                ),
            ):
                raise DynamicContractIRWellFormednessError(
                    "EventTimeBucket.events must contain ContractEvent values"
                )


@dataclass(frozen=True)
class EventProgram:
    ordering: tuple[str, ...] = DEFAULT_EVENT_ORDERING
    buckets: tuple[EventTimeBucket, ...] = ()
    termination_rules: tuple[TerminationRule, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ordering, tuple):
            object.__setattr__(self, "ordering", tuple(self.ordering))
        if len(set(self.ordering)) != len(self.ordering):
            raise DynamicContractIRWellFormednessError(
                "EventProgram.ordering must not repeat phases"
            )
        if not isinstance(self.buckets, tuple):
            object.__setattr__(self, "buckets", tuple(self.buckets))
        if not isinstance(self.termination_rules, tuple):
            object.__setattr__(self, "termination_rules", tuple(self.termination_rules))
        previous_date: date | None = None
        seen_event_labels: set[str] = set()
        for bucket in self.buckets:
            if not isinstance(bucket, EventTimeBucket):
                raise DynamicContractIRWellFormednessError(
                    "EventProgram.buckets must contain EventTimeBucket values"
                )
            if previous_date is not None and bucket.event_date < previous_date:
                raise DynamicContractIRWellFormednessError(
                    "EventProgram buckets must be ordered by event_date"
                )
            previous_date = bucket.event_date
            if any(phase not in self.ordering for phase in bucket.phase_sequence):
                raise DynamicContractIRWellFormednessError(
                    "EventTimeBucket.phase_sequence must be drawn from EventProgram.ordering"
                )
            for event in bucket.events:
                if event.phase not in self.ordering:
                    raise DynamicContractIRWellFormednessError(
                        f"event phase {event.phase!r} is not present in EventProgram.ordering"
                    )
                if event.label in seen_event_labels:
                    raise DynamicContractIRWellFormednessError(
                        f"duplicate event label {event.label!r}"
                    )
                seen_event_labels.add(event.label)
        for rule in self.termination_rules:
            if not isinstance(rule, TerminationRule):
                raise DynamicContractIRWellFormednessError(
                    "EventProgram.termination_rules must contain TerminationRule values"
                )
            if rule.event_label and rule.event_label not in seen_event_labels:
                raise DynamicContractIRWellFormednessError(
                    f"TerminationRule.event_label {rule.event_label!r} does not reference an event label"
                )


@dataclass(frozen=True)
class ControlProgram:
    controller_role: str
    decision_style: str
    decision_event_labels: tuple[str, ...]
    admissible_actions: tuple[ActionSpec, ...]
    inventory_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_role",
            _require_text(self.controller_role, label="ControlProgram.controller_role").lower(),
        )
        object.__setattr__(
            self,
            "decision_style",
            _require_text(self.decision_style, label="ControlProgram.decision_style").lower(),
        )
        if not isinstance(self.decision_event_labels, tuple):
            object.__setattr__(self, "decision_event_labels", tuple(self.decision_event_labels))
        if not self.decision_event_labels:
            raise DynamicContractIRWellFormednessError(
                "ControlProgram.decision_event_labels must be non-empty"
            )
        if not isinstance(self.admissible_actions, tuple):
            object.__setattr__(self, "admissible_actions", tuple(self.admissible_actions))
        if not self.admissible_actions:
            raise DynamicContractIRWellFormednessError(
                "ControlProgram.admissible_actions must be non-empty"
            )
        for action in self.admissible_actions:
            if not isinstance(action, ActionSpec):
                raise DynamicContractIRWellFormednessError(
                    "ControlProgram.admissible_actions must contain ActionSpec values"
                )
        if not isinstance(self.inventory_fields, tuple):
            object.__setattr__(self, "inventory_fields", tuple(self.inventory_fields))
        for field_name in self.inventory_fields:
            _require_text(field_name, label="ControlProgram.inventory_fields")


BaseContract = ContractIR | StaticLegContractIR


@dataclass(frozen=True)
class DynamicContractIR:
    base_contract: BaseContract | None
    semantic_family: str = ""
    base_track: str = ""
    state_schema: StateSchema = field(default_factory=StateSchema)
    event_program: EventProgram = field(default_factory=EventProgram)
    control_program: ControlProgram | None = None
    settlement: SettlementRule = field(default_factory=SettlementRule)

    def __post_init__(self) -> None:
        if self.base_contract is not None and not isinstance(
            self.base_contract,
            (ContractIR, StaticLegContractIR),
        ):
            raise DynamicContractIRWellFormednessError(
                "DynamicContractIR.base_contract must be a ContractIR, StaticLegContractIR, or None"
            )
        if not isinstance(self.state_schema, StateSchema):
            raise DynamicContractIRWellFormednessError(
                "DynamicContractIR.state_schema must be a StateSchema"
            )
        if not isinstance(self.event_program, EventProgram):
            raise DynamicContractIRWellFormednessError(
                "DynamicContractIR.event_program must be an EventProgram"
            )
        if not isinstance(self.settlement, SettlementRule):
            raise DynamicContractIRWellFormednessError(
                "DynamicContractIR.settlement must be a SettlementRule"
            )
        object.__setattr__(self, "semantic_family", str(self.semantic_family or "").strip())
        object.__setattr__(self, "base_track", str(self.base_track or "").strip())
        if self.base_track and self.base_track not in {
            "payoff_expression",
            "quoted_observable",
            "static_leg",
        }:
            raise DynamicContractIRWellFormednessError(
                "DynamicContractIR.base_track must be one of 'payoff_expression', "
                "'quoted_observable', or 'static_leg'"
            )
        field_names = set(self.state_schema.field_names)
        decision_events: list[DecisionEvent] = []
        event_labels: set[str] = set()
        for bucket in self.event_program.buckets:
            for event in bucket.events:
                event_labels.add(event.label)
                if isinstance(event, DecisionEvent):
                    decision_events.append(event)
                updates = ()
                if isinstance(event, (CouponEvent, StateResetEvent)):
                    updates = event.state_updates
                for update in updates:
                    if update.field_name not in field_names:
                        raise DynamicContractIRWellFormednessError(
                            f"state update references unknown field {update.field_name!r}"
                        )
        if decision_events and self.control_program is None:
            raise DynamicContractIRWellFormednessError(
                "DynamicContractIR requires a ControlProgram when DecisionEvent is present"
            )
        if self.control_program is not None:
            if set(self.control_program.inventory_fields) - field_names:
                raise DynamicContractIRWellFormednessError(
                    "ControlProgram.inventory_fields must reference declared state fields"
                )
            decision_labels = {event.label for event in decision_events}
            if set(self.control_program.decision_event_labels) - decision_labels:
                raise DynamicContractIRWellFormednessError(
                    "ControlProgram.decision_event_labels must reference DecisionEvent labels"
                )
            decision_roles = {event.controller_role for event in decision_events}
            if decision_roles and decision_roles != {self.control_program.controller_role}:
                raise DynamicContractIRWellFormednessError(
                    "DecisionEvent controller_role values must agree with ControlProgram.controller_role"
                )
            action_names = {action.action_name for action in self.control_program.admissible_actions}
            for event in decision_events:
                event_action_names = {action.action_name for action in event.action_set}
                if not event_action_names.issubset(action_names):
                    raise DynamicContractIRWellFormednessError(
                        "DecisionEvent actions must be declared in ControlProgram.admissible_actions"
                    )
                for action in event.action_set:
                    for update in action.state_updates:
                        if update.field_name not in field_names:
                            raise DynamicContractIRWellFormednessError(
                                f"action update references unknown field {update.field_name!r}"
                            )


__all__ = [
    "ActionSpec",
    "AutomaticTerminationEvent",
    "ControlProgram",
    "CouponEvent",
    "DEFAULT_EVENT_ORDERING",
    "DecisionEvent",
    "DynamicContractIR",
    "DynamicContractIRWellFormednessError",
    "EventProgram",
    "EventTimeBucket",
    "ObservationEvent",
    "PaymentEvent",
    "StateFieldSpec",
    "StateResetEvent",
    "StateSchema",
    "StateUpdateSpec",
    "TerminationRule",
]
