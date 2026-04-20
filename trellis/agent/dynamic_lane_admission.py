"""Bounded post-Phase-4 admission for dynamic semantic lowering lanes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from trellis.agent.dynamic_contract_ir import (
    CouponEvent,
    DecisionEvent,
    DynamicContractIR,
    StateResetEvent,
)


class DynamicLaneAdmissionError(ValueError):
    """Raised when a dynamic contract falls outside the admitted lane cohorts."""


def _default_benchmark_plan() -> "DynamicBenchmarkPlan":
    return DynamicBenchmarkPlan(
        cohort_id="unclassified_dynamic_lane",
        proving_family="dynamic_lane",
        validation_mode="benchmark_plan",
    )


@dataclass(frozen=True)
class DynamicBenchmarkPlan:
    """Minimal parity / benchmark ledger attached to one admitted lane."""

    cohort_id: str
    proving_family: str
    validation_mode: str
    reference_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutomaticEventStateLaneAdmission:
    """Typed admission contract for automatic stateful products."""

    semantic_family: str
    base_track: str
    event_ordering: tuple[str, ...]
    schedule_roles: tuple[str, ...]
    state_fields: tuple[str, ...]
    state_update_fields: tuple[str, ...]
    termination_rule_labels: tuple[str, ...]
    candidate_numerical_lanes: tuple[str, ...]
    benchmark_plan: DynamicBenchmarkPlan = field(default_factory=_default_benchmark_plan)
    lane: str = "automatic_event_state"


@dataclass(frozen=True)
class DiscreteControlLaneAdmission:
    """Typed admission contract for discrete holder / issuer control."""

    semantic_family: str
    base_track: str
    controller_role: str
    decision_style: str
    decision_event_labels: tuple[str, ...]
    action_names: tuple[str, ...]
    inventory_fields: tuple[str, ...]
    candidate_numerical_lanes: tuple[str, ...]
    benchmark_plan: DynamicBenchmarkPlan = field(default_factory=_default_benchmark_plan)
    lane: str = "discrete_control"


@dataclass(frozen=True)
class ContinuousControlLaneAdmission:
    """Typed admission contract for continuous or singular control."""

    semantic_family: str
    base_track: str
    controller_role: str
    decision_style: str
    controlled_state_fields: tuple[str, ...]
    magnitude_action_names: tuple[str, ...]
    action_domains: tuple[str, ...]
    candidate_numerical_lanes: tuple[str, ...]
    approximation_policy_required: bool = True
    benchmark_plan: DynamicBenchmarkPlan = field(default_factory=_default_benchmark_plan)
    lane: str = "continuous_control"


DynamicLaneAdmission = (
    AutomaticEventStateLaneAdmission
    | DiscreteControlLaneAdmission
    | ContinuousControlLaneAdmission
)


def compile_dynamic_lane_admission(contract: DynamicContractIR | None) -> DynamicLaneAdmission:
    """Compile one bounded dynamic semantic contract onto an admitted lane."""

    if contract is None:
        raise DynamicLaneAdmissionError("dynamic lane admission requires a DynamicContractIR")
    if not isinstance(contract, DynamicContractIR):
        raise TypeError("contract must be a DynamicContractIR")
    if str(contract.base_track or "").strip().lower() == "quoted_observable":
        raise DynamicLaneAdmissionError(
            "quoted-observable dynamic hybrids remain deferred from the admitted dynamic lanes"
        )

    if _is_continuous_control(contract):
        return _compile_continuous_control_lane(contract)
    if _is_discrete_control(contract):
        return _compile_discrete_control_lane(contract)
    if _is_automatic_event_state(contract):
        return _compile_automatic_event_state_lane(contract)

    raise DynamicLaneAdmissionError(
        "contract does not match an admitted automatic, discrete-control, or continuous-control cohort"
    )


def _is_continuous_control(contract: DynamicContractIR) -> bool:
    control_program = contract.control_program
    if control_program is None:
        return False
    return any(
        action.action_domain in {"continuous", "singular"}
        for action in control_program.admissible_actions
    )


def _is_discrete_control(contract: DynamicContractIR) -> bool:
    control_program = contract.control_program
    if control_program is None:
        return False
    return all(
        action.action_domain == "discrete"
        for action in control_program.admissible_actions
    )


def _is_automatic_event_state(contract: DynamicContractIR) -> bool:
    return contract.control_program is None and bool(contract.event_program.termination_rules)


def _compile_automatic_event_state_lane(
    contract: DynamicContractIR,
) -> AutomaticEventStateLaneAdmission:
    semantic_family = _automatic_family(contract)
    benchmark_plan = _automatic_benchmark_plan(semantic_family)
    return AutomaticEventStateLaneAdmission(
        semantic_family=semantic_family,
        base_track=_normalized_base_track(contract),
        event_ordering=tuple(contract.event_program.ordering),
        schedule_roles=_schedule_roles(contract),
        state_fields=tuple(contract.state_schema.field_names),
        state_update_fields=_state_update_fields(contract),
        termination_rule_labels=tuple(rule.label for rule in contract.event_program.termination_rules),
        candidate_numerical_lanes=("event_aware_monte_carlo",),
        benchmark_plan=benchmark_plan,
    )


def _compile_discrete_control_lane(
    contract: DynamicContractIR,
) -> DiscreteControlLaneAdmission:
    control_program = contract.control_program
    if control_program is None:
        raise DynamicLaneAdmissionError(
            "_compile_discrete_control_lane requires a ControlProgram"
        )
    semantic_family = _discrete_family(contract)
    candidate_lanes = _discrete_candidate_lanes(semantic_family)
    benchmark_plan = _discrete_benchmark_plan(semantic_family)
    return DiscreteControlLaneAdmission(
        semantic_family=semantic_family,
        base_track=_normalized_base_track(contract),
        controller_role=control_program.controller_role,
        decision_style=control_program.decision_style,
        decision_event_labels=tuple(control_program.decision_event_labels),
        action_names=tuple(action.action_name for action in control_program.admissible_actions),
        inventory_fields=tuple(control_program.inventory_fields),
        candidate_numerical_lanes=candidate_lanes,
        benchmark_plan=benchmark_plan,
    )


def _compile_continuous_control_lane(
    contract: DynamicContractIR,
) -> ContinuousControlLaneAdmission:
    control_program = contract.control_program
    if control_program is None:
        raise DynamicLaneAdmissionError(
            "_compile_continuous_control_lane requires a ControlProgram"
        )
    semantic_family = _continuous_family(contract)
    magnitude_actions = tuple(
        action.action_name
        for action in control_program.admissible_actions
        if action.action_domain in {"continuous", "singular"}
    )
    action_domains = _unique(
        action.action_domain for action in control_program.admissible_actions
    )
    return ContinuousControlLaneAdmission(
        semantic_family=semantic_family,
        base_track=_normalized_base_track(contract),
        controller_role=control_program.controller_role,
        decision_style=control_program.decision_style,
        controlled_state_fields=_continuous_controlled_state_fields(contract),
        magnitude_action_names=magnitude_actions,
        action_domains=action_domains,
        candidate_numerical_lanes=("qvi_pde", "control_dynamic_programming"),
        benchmark_plan=_continuous_benchmark_plan(semantic_family),
    )


def _normalized_base_track(contract: DynamicContractIR) -> str:
    return str(contract.base_track or "").strip() or "payoff_expression"


def _schedule_roles(contract: DynamicContractIR) -> tuple[str, ...]:
    roles: list[str] = []
    for bucket in contract.event_program.buckets:
        for event in bucket.events:
            role = str(getattr(event, "schedule_role", "") or "").strip()
            if role and role not in roles:
                roles.append(role)
    return tuple(roles)


def _state_update_fields(contract: DynamicContractIR) -> tuple[str, ...]:
    fields: list[str] = []
    for bucket in contract.event_program.buckets:
        for event in bucket.events:
            if isinstance(event, (CouponEvent, StateResetEvent)):
                for update in event.state_updates:
                    if update.field_name not in fields:
                        fields.append(update.field_name)
            if isinstance(event, DecisionEvent):
                for action in event.action_set:
                    for update in action.state_updates:
                        if update.field_name not in fields:
                            fields.append(update.field_name)
    return tuple(fields)


def _automatic_family(contract: DynamicContractIR) -> str:
    family = str(contract.semantic_family or "").strip().lower()
    if family in {"autocallable", "phoenix", "snowball"}:
        return "autocallable"
    if family in {"tarn", "tarf"}:
        return "tarn"
    state_fields = set(contract.state_schema.field_names)
    if "coupon_memory" in state_fields:
        return "autocallable"
    if {"accrued_coupon", "cumulative_gain"} & state_fields:
        return "tarn"
    raise DynamicLaneAdmissionError(
        "automatic event/state admission is currently bounded to autocallable/phoenix/snowball and TARN/TARF-style cohorts"
    )


def _discrete_family(contract: DynamicContractIR) -> str:
    family = str(contract.semantic_family or "").strip().lower()
    if family in {"callable_bond", "callable_note"}:
        return "callable_bond"
    if family in {"swing", "swing_option"}:
        return "swing_option"
    control_program = contract.control_program
    assert control_program is not None
    if control_program.controller_role == "issuer" and _normalized_base_track(contract) == "static_leg":
        return "callable_bond"
    if control_program.inventory_fields:
        return "swing_option"
    raise DynamicLaneAdmissionError(
        "discrete-control admission is currently bounded to callable-bond and swing-style cohorts"
    )


def _continuous_family(contract: DynamicContractIR) -> str:
    family = str(contract.semantic_family or "").strip().lower()
    if family in {"gmwb", "gmxb"}:
        return "gmwb"
    state_fields = set(contract.state_schema.field_names)
    if {"account_value", "guarantee_base"}.issubset(state_fields):
        return "gmwb"
    raise DynamicLaneAdmissionError(
        "continuous-control admission is currently bounded to GMWB-style financial-control cohorts"
    )


def _discrete_candidate_lanes(semantic_family: str) -> tuple[str, ...]:
    if semantic_family == "callable_bond":
        return ("exercise_lattice", "event_aware_pde")
    if semantic_family == "swing_option":
        return ("control_lsmc",)
    raise DynamicLaneAdmissionError(f"unsupported discrete-control cohort {semantic_family!r}")


def _continuous_controlled_state_fields(contract: DynamicContractIR) -> tuple[str, ...]:
    field_names = tuple(contract.state_schema.field_names)
    if {"account_value", "guarantee_base"}.issubset(set(field_names)):
        return ("account_value", "guarantee_base")
    return field_names


def _automatic_benchmark_plan(semantic_family: str) -> DynamicBenchmarkPlan:
    if semantic_family == "autocallable":
        return DynamicBenchmarkPlan(
            cohort_id="autocallable_note",
            proving_family="event_aware_monte_carlo",
            validation_mode="literature_benchmark",
            reference_notes=(
                "autocallable discrete-observation benchmark cohort",
                "compare event ordering, coupon memory, and early redemption semantics",
            ),
        )
    if semantic_family == "tarn":
        return DynamicBenchmarkPlan(
            cohort_id="target_redemption_note",
            proving_family="event_aware_monte_carlo",
            validation_mode="literature_benchmark",
            reference_notes=(
                "target-redemption running-state benchmark cohort",
                "compare accumulated coupon or gain state and target-triggered stopping",
            ),
        )
    raise DynamicLaneAdmissionError(f"unsupported automatic-state cohort {semantic_family!r}")


def _discrete_benchmark_plan(semantic_family: str) -> DynamicBenchmarkPlan:
    if semantic_family == "callable_bond":
        return DynamicBenchmarkPlan(
            cohort_id="callable_bond",
            proving_family="exercise_lattice",
            validation_mode="parity_plan",
            reference_notes=(
                "issuer-call backward-induction benchmark cohort",
                "preserve call timing and static coupon-leg semantics",
            ),
        )
    if semantic_family == "swing_option":
        return DynamicBenchmarkPlan(
            cohort_id="swing_option",
            proving_family="control_lsmc",
            validation_mode="benchmark_plan",
            reference_notes=(
                "inventory-constrained holder-control benchmark cohort",
                "preserve remaining-right semantics and decision timing",
            ),
        )
    raise DynamicLaneAdmissionError(f"unsupported discrete-control cohort {semantic_family!r}")


def _continuous_benchmark_plan(semantic_family: str) -> DynamicBenchmarkPlan:
    if semantic_family != "gmwb":
        raise DynamicLaneAdmissionError(f"unsupported continuous-control cohort {semantic_family!r}")
    return DynamicBenchmarkPlan(
        cohort_id="gmwb_financial_control",
        proving_family="qvi_pde",
        validation_mode="literature_benchmark",
        reference_notes=(
            "financial-control-only GMWB benchmark cohort",
            "keep mortality, lapse, and fee overlays out of the proving slice",
        ),
    )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


__all__ = [
    "AutomaticEventStateLaneAdmission",
    "ContinuousControlLaneAdmission",
    "DiscreteControlLaneAdmission",
    "DynamicBenchmarkPlan",
    "DynamicLaneAdmission",
    "DynamicLaneAdmissionError",
    "compile_dynamic_lane_admission",
]
