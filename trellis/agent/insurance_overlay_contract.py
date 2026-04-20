"""Bounded semantic wrapper for insurance-style overlays above financial control."""

from __future__ import annotations

from dataclasses import dataclass, field

from trellis.agent.dynamic_contract_ir import (
    DynamicContractIR,
    StateFieldSpec,
    StateUpdateSpec,
)


class InsuranceOverlayContractWellFormednessError(ValueError):
    """Raised when an insurance overlay contract violates a local invariant."""


def _require_text(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise InsuranceOverlayContractWellFormednessError(f"{label} must be a non-empty string")
    return text


@dataclass(frozen=True)
class PolicyStateSchema:
    fields: tuple[StateFieldSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple):
            object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields:
            raise InsuranceOverlayContractWellFormednessError(
                "PolicyStateSchema.fields must be non-empty"
            )
        seen: set[str] = set()
        for field_spec in self.fields:
            if not isinstance(field_spec, StateFieldSpec):
                raise InsuranceOverlayContractWellFormednessError(
                    "PolicyStateSchema.fields must contain StateFieldSpec values"
                )
            if "policy_state" not in field_spec.tags:
                raise InsuranceOverlayContractWellFormednessError(
                    "PolicyStateSchema fields must carry the 'policy_state' tag"
                )
            if field_spec.name in seen:
                raise InsuranceOverlayContractWellFormednessError(
                    f"duplicate policy-state field {field_spec.name!r}"
                )
            seen.add(field_spec.name)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field_spec.name for field_spec in self.fields)


@dataclass(frozen=True)
class OverlayParameterSpec:
    name: str
    parameter_kind: str
    value: object
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, label="OverlayParameterSpec.name"))
        object.__setattr__(
            self,
            "parameter_kind",
            _require_text(self.parameter_kind, label="OverlayParameterSpec.parameter_kind"),
        )
        if not isinstance(self.notes, tuple):
            object.__setattr__(self, "notes", tuple(self.notes))


@dataclass(frozen=True)
class OverlayParameterSet:
    parameters: tuple[OverlayParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple):
            object.__setattr__(self, "parameters", tuple(self.parameters))
        seen: set[str] = set()
        for parameter in self.parameters:
            if not isinstance(parameter, OverlayParameterSpec):
                raise InsuranceOverlayContractWellFormednessError(
                    "OverlayParameterSet.parameters must contain OverlayParameterSpec values"
                )
            if parameter.name in seen:
                raise InsuranceOverlayContractWellFormednessError(
                    f"duplicate overlay parameter {parameter.name!r}"
                )
            seen.add(parameter.name)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)


@dataclass(frozen=True)
class OverlayTransitionEvent:
    label: str
    schedule_role: str
    trigger_expression: str
    state_updates: tuple[StateUpdateSpec, ...] = ()
    cashflow_adjustment: str = ""
    phase: str = "overlay_transition"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label",
            _require_text(self.label, label="OverlayTransitionEvent.label"),
        )
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="OverlayTransitionEvent.schedule_role"),
        )
        object.__setattr__(
            self,
            "trigger_expression",
            _require_text(
                self.trigger_expression,
                label="OverlayTransitionEvent.trigger_expression",
            ),
        )
        if not isinstance(self.state_updates, tuple):
            object.__setattr__(self, "state_updates", tuple(self.state_updates))
        object.__setattr__(self, "cashflow_adjustment", str(self.cashflow_adjustment or "").strip())
        object.__setattr__(
            self,
            "phase",
            _require_text(self.phase, label="OverlayTransitionEvent.phase").lower(),
        )
        if not self.state_updates:
            raise InsuranceOverlayContractWellFormednessError(
                "OverlayTransitionEvent.state_updates must be non-empty"
            )
        for update in self.state_updates:
            if not isinstance(update, StateUpdateSpec):
                raise InsuranceOverlayContractWellFormednessError(
                    "OverlayTransitionEvent.state_updates must contain StateUpdateSpec values"
                )


@dataclass(frozen=True)
class OverlayFeeEvent:
    label: str
    schedule_role: str
    fee_formula: str
    state_updates: tuple[StateUpdateSpec, ...] = ()
    phase: str = "overlay_fee"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _require_text(self.label, label="OverlayFeeEvent.label"))
        object.__setattr__(
            self,
            "schedule_role",
            _require_text(self.schedule_role, label="OverlayFeeEvent.schedule_role"),
        )
        object.__setattr__(
            self,
            "fee_formula",
            _require_text(self.fee_formula, label="OverlayFeeEvent.fee_formula"),
        )
        if not isinstance(self.state_updates, tuple):
            object.__setattr__(self, "state_updates", tuple(self.state_updates))
        object.__setattr__(self, "phase", _require_text(self.phase, label="OverlayFeeEvent.phase").lower())
        for update in self.state_updates:
            if not isinstance(update, StateUpdateSpec):
                raise InsuranceOverlayContractWellFormednessError(
                    "OverlayFeeEvent.state_updates must contain StateUpdateSpec values"
                )


OverlayEvent = OverlayTransitionEvent | OverlayFeeEvent


@dataclass(frozen=True)
class OverlayCompositionRule:
    composition_style: str
    policy_state_field: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "composition_style",
            _require_text(
                self.composition_style,
                label="OverlayCompositionRule.composition_style",
            ).lower(),
        )
        object.__setattr__(self, "policy_state_field", str(self.policy_state_field or "").strip())
        if not isinstance(self.notes, tuple):
            object.__setattr__(self, "notes", tuple(self.notes))


@dataclass(frozen=True)
class InsuranceOverlayContractIR:
    core_contract: DynamicContractIR
    policy_state_schema: PolicyStateSchema
    overlay_events: tuple[OverlayEvent, ...]
    composition_rule: OverlayCompositionRule
    semantic_family: str = ""
    overlay_parameters: OverlayParameterSet = field(default_factory=OverlayParameterSet)

    def __post_init__(self) -> None:
        if not isinstance(self.core_contract, DynamicContractIR):
            raise InsuranceOverlayContractWellFormednessError(
                "InsuranceOverlayContractIR.core_contract must be a DynamicContractIR"
            )
        if not isinstance(self.policy_state_schema, PolicyStateSchema):
            raise InsuranceOverlayContractWellFormednessError(
                "InsuranceOverlayContractIR.policy_state_schema must be a PolicyStateSchema"
            )
        if not isinstance(self.overlay_events, tuple):
            object.__setattr__(self, "overlay_events", tuple(self.overlay_events))
        if not self.overlay_events:
            raise InsuranceOverlayContractWellFormednessError(
                "InsuranceOverlayContractIR.overlay_events must be non-empty"
            )
        if not isinstance(self.composition_rule, OverlayCompositionRule):
            raise InsuranceOverlayContractWellFormednessError(
                "InsuranceOverlayContractIR.composition_rule must be an OverlayCompositionRule"
            )
        if not isinstance(self.overlay_parameters, OverlayParameterSet):
            raise InsuranceOverlayContractWellFormednessError(
                "InsuranceOverlayContractIR.overlay_parameters must be an OverlayParameterSet"
            )

        object.__setattr__(
            self,
            "semantic_family",
            str(self.semantic_family or self.core_contract.semantic_family or "").strip(),
        )

        core_field_names = set(self.core_contract.state_schema.field_names)
        for field_spec in self.core_contract.state_schema.fields:
            if {"policy_state", "insurance_overlay"} & set(field_spec.tags):
                raise InsuranceOverlayContractWellFormednessError(
                    "InsuranceOverlayContractIR requires an overlay-free core contract"
                )

        policy_field_names = set(self.policy_state_schema.field_names)
        if core_field_names & policy_field_names:
            raise InsuranceOverlayContractWellFormednessError(
                "policy-state field names must not collide with core state fields"
            )

        if (
            self.composition_rule.policy_state_field
            and self.composition_rule.policy_state_field not in policy_field_names
        ):
            raise InsuranceOverlayContractWellFormednessError(
                "OverlayCompositionRule.policy_state_field must reference a declared policy-state field"
            )

        seen_labels: set[str] = set()
        for event in self.overlay_events:
            if not isinstance(event, (OverlayTransitionEvent, OverlayFeeEvent)):
                raise InsuranceOverlayContractWellFormednessError(
                    "InsuranceOverlayContractIR.overlay_events must contain overlay event values"
                )
            if event.label in seen_labels:
                raise InsuranceOverlayContractWellFormednessError(
                    f"duplicate overlay event label {event.label!r}"
                )
            seen_labels.add(event.label)
            for update in getattr(event, "state_updates", ()):
                if update.field_name not in policy_field_names:
                    raise InsuranceOverlayContractWellFormednessError(
                        f"overlay event references unknown policy-state field {update.field_name!r}"
                    )


__all__ = [
    "InsuranceOverlayContractIR",
    "InsuranceOverlayContractWellFormednessError",
    "OverlayCompositionRule",
    "OverlayEvent",
    "OverlayFeeEvent",
    "OverlayParameterSet",
    "OverlayParameterSpec",
    "OverlayTransitionEvent",
    "PolicyStateSchema",
]
