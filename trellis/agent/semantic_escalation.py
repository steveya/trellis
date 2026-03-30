"""Deterministic agent-role ownership for semantic extension and validation."""

from __future__ import annotations

from collections.abc import Mapping


_SEMANTIC_ROLE_MATRIX: tuple[dict[str, object], ...] = (
    {
        "stage": "gap_classification",
        "owner_role": "semantic_classifier",
        "trigger_condition": "unsupported_semantic_request",
        "artifact_kind": "SemanticGapReport",
        "can_invent_semantic_schema": False,
        "can_invent_route": False,
        "can_validate_model": False,
        "scope": "gap_classification_only",
    },
    {
        "stage": "primitive_proposal",
        "owner_role": "quant",
        "trigger_condition": "missing_runtime_primitives_or_route_helpers",
        "artifact_kind": "SemanticExtensionProposal",
        "can_invent_semantic_schema": False,
        "can_invent_route": True,
        "can_validate_model": False,
        "scope": "bounded_semantic_to_pricing_assembly",
    },
    {
        "stage": "route_assembly",
        "owner_role": "quant",
        "trigger_condition": "pricing_plan_selection",
        "artifact_kind": "GenerationPlan",
        "can_invent_semantic_schema": False,
        "can_invent_route": True,
        "can_validate_model": False,
        "scope": "bounded_semantic_to_pricing_assembly",
    },
    {
        "stage": "payoff_model_validation",
        "owner_role": "model_validator",
        "trigger_condition": "validation_risk_gate",
        "artifact_kind": "ValidationReport",
        "can_invent_semantic_schema": False,
        "can_invent_route": False,
        "can_validate_model": True,
        "scope": "model_validation_only",
    },
    {
        "stage": "trace_handoff",
        "owner_role": "knowledge_agent",
        "trigger_condition": "semantic_extension_trace_persisted",
        "artifact_kind": "semantic_extension_trace",
        "can_invent_semantic_schema": False,
        "can_invent_route": False,
        "can_validate_model": False,
        "scope": "trace_handoff_only",
    },
)


def semantic_role_matrix() -> list[dict[str, object]]:
    """Return the canonical semantic-extension role matrix."""
    return [dict(entry) for entry in _SEMANTIC_ROLE_MATRIX]


def semantic_role_ownership_summary(
    *,
    stage: str,
    semantic_gap: Mapping[str, object] | None = None,
    semantic_extension: Mapping[str, object] | None = None,
    semantic_contract: bool = False,
    review_policy=None,
    trigger_condition: str | None = None,
    artifact_kind: str | None = None,
    executed: bool | None = None,
) -> dict[str, object]:
    """Summarize which role owns one semantic-extension stage and why."""
    entry = _matrix_entry(stage)
    if entry is None:
        return _fail_closed_summary(stage=stage)

    trigger_condition = (
        trigger_condition
        or _default_trigger_condition(
            stage,
            semantic_gap=semantic_gap,
            semantic_extension=semantic_extension,
            semantic_contract=semantic_contract,
            review_policy=review_policy,
        )
        or str(entry["trigger_condition"])
    )
    artifact_kind = artifact_kind or str(entry["artifact_kind"])

    if executed is None:
        executed = stage != "payoff_model_validation" or bool(
            getattr(review_policy, "run_model_validator_llm", False)
        )

    summary = {
        "selected_stage": stage,
        "selected_role": entry["owner_role"],
        "trigger_condition": trigger_condition,
        "artifact_kind": artifact_kind,
        "executed": bool(executed),
        "fail_closed": False,
        "requires_clarification": _requires_clarification(stage, semantic_gap, semantic_extension),
        "can_invent_semantic_schema": entry["can_invent_semantic_schema"],
        "can_invent_route": entry["can_invent_route"],
        "can_validate_model": entry["can_validate_model"],
        "scope": entry["scope"],
        "role_matrix": semantic_role_matrix(),
    }
    summary["summary"] = (
        f"{stage}:{summary['selected_role']} -> {artifact_kind} because {trigger_condition}"
    )
    return summary


def semantic_role_matrix_entry(stage: str) -> dict[str, object] | None:
    """Return one matrix entry by stage, if it exists."""
    entry = _matrix_entry(stage)
    return dict(entry) if entry is not None else None


def semantic_role_ownership_stage(
    ownership: Mapping[str, object] | None,
    stage: str,
) -> dict[str, object]:
    """Return one stage-specific role assignment from an ownership summary."""
    if not isinstance(ownership, Mapping):
        return _fail_closed_stage(stage)
    matrix = ownership.get("role_matrix") or ()
    if isinstance(matrix, Mapping):
        matrix = (matrix,)
    for item in matrix:
        if isinstance(item, Mapping) and str(item.get("stage") or "") == stage:
            return dict(item)
    return _fail_closed_stage(stage)


def _matrix_entry(stage: str) -> dict[str, object] | None:
    for entry in _SEMANTIC_ROLE_MATRIX:
        if entry["stage"] == stage:
            return dict(entry)
    return None


def _fail_closed_summary(stage: str) -> dict[str, object]:
    matrix = semantic_role_matrix()
    return {
        "selected_stage": stage or "unowned",
        "selected_role": "",
        "trigger_condition": "no_safe_owner_available",
        "artifact_kind": "",
        "executed": False,
        "fail_closed": True,
        "requires_clarification": False,
        "can_invent_semantic_schema": False,
        "can_invent_route": False,
        "can_validate_model": False,
        "scope": "unowned",
        "role_matrix": matrix,
        "summary": f"unowned stage `{stage or 'unknown'}` has no safe role owner",
    }


def _fail_closed_stage(stage: str) -> dict[str, object]:
    return {
        "stage": stage,
        "owner_role": "",
        "trigger_condition": "no_safe_owner_available",
        "artifact_kind": "",
        "can_invent_semantic_schema": False,
        "can_invent_route": False,
        "can_validate_model": False,
        "scope": "unowned",
        "fail_closed": True,
    }


def _requires_clarification(
    stage: str,
    semantic_gap: Mapping[str, object] | None,
    semantic_extension: Mapping[str, object] | None,
) -> bool:
    if stage != "gap_classification":
        return False
    if isinstance(semantic_gap, Mapping):
        return bool(semantic_gap.get("requires_clarification"))
    if isinstance(semantic_extension, Mapping):
        return str(semantic_extension.get("decision") or "") == "clarification"
    return False


def _default_trigger_condition(
    stage: str,
    *,
    semantic_gap: Mapping[str, object] | None,
    semantic_extension: Mapping[str, object] | None,
    semantic_contract: bool,
    review_policy,
) -> str | None:
    if stage == "gap_classification":
        if _requires_clarification(stage, semantic_gap, semantic_extension):
            return "requires_clarification"
        if isinstance(semantic_gap, Mapping):
            gap_types = semantic_gap.get("gap_types") or ()
            if isinstance(gap_types, (list, tuple)) and gap_types:
                return str(gap_types[0])
        return "unsupported_semantic_request"

    if stage == "primitive_proposal":
        if isinstance(semantic_extension, Mapping):
            decision = str(semantic_extension.get("decision") or "").strip()
            if decision:
                if decision == "new_primitive":
                    missing_runtime_primitives = semantic_extension.get("missing_runtime_primitives") or ()
                    if isinstance(missing_runtime_primitives, (list, tuple)) and missing_runtime_primitives:
                        return str(missing_runtime_primitives[0])
                    missing_route_helpers = semantic_extension.get("missing_route_helpers") or ()
                    if isinstance(missing_route_helpers, (list, tuple)) and missing_route_helpers:
                        return str(missing_route_helpers[0])
                    return "missing_runtime_primitives_or_route_helpers"
                if decision == "mock_inputs":
                    missing_market_inputs = semantic_extension.get("missing_market_inputs") or ()
                    if isinstance(missing_market_inputs, (list, tuple)) and missing_market_inputs:
                        return str(missing_market_inputs[0])
                    return "missing_market_inputs"
                if decision == "knowledge_artifact":
                    return "missing_knowledge_artifacts"
                if decision == "clarification":
                    return "requires_clarification"
                return decision
        return "missing_runtime_primitives_or_route_helpers"

    if stage == "route_assembly":
        if semantic_contract:
            return "semantic_contract_request"
        if isinstance(semantic_extension, Mapping):
            decision = str(semantic_extension.get("decision") or "").strip()
            if decision:
                return f"pricing_plan_selection:{decision}"
        return "pricing_plan_selection"

    if stage == "payoff_model_validation":
        if review_policy is not None:
            reason = str(getattr(review_policy, "model_validator_reason", "") or "").strip()
            if reason:
                return reason
            if not bool(getattr(review_policy, "run_model_validator_llm", False)):
                return "model_validator_skipped"
        return "validation_risk_gate"

    if stage == "trace_handoff":
        if isinstance(semantic_extension, Mapping):
            decision = str(semantic_extension.get("decision") or "").strip()
            if decision == "knowledge_artifact":
                return "knowledge_artifact_trace_handoff"
        return "semantic_extension_trace_persisted"

    return None
