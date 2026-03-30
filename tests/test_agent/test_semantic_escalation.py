"""Tests for semantic escalation ownership and role matrix summaries."""

from __future__ import annotations

from types import SimpleNamespace


def test_semantic_role_ownership_summary_selects_quant_for_primitive_proposal():
    from trellis.agent.semantic_escalation import semantic_role_ownership_summary

    summary = semantic_role_ownership_summary(
        stage="primitive_proposal",
        semantic_extension={
            "decision": "new_primitive",
            "missing_runtime_primitives": ["generate_schedule"],
            "missing_route_helpers": [],
            "missing_market_inputs": [],
        },
    )

    assert summary["selected_stage"] == "primitive_proposal"
    assert summary["selected_role"] == "quant"
    assert summary["trigger_condition"] == "generate_schedule"
    assert summary["artifact_kind"] == "SemanticExtensionProposal"
    assert summary["can_invent_semantic_schema"] is False
    assert summary["can_invent_route"] is True
    assert summary["fail_closed"] is False
    assert summary["role_matrix"]


def test_semantic_role_ownership_summary_selects_model_validator_for_validation():
    from trellis.agent.semantic_escalation import semantic_role_ownership_summary

    review_policy = SimpleNamespace(
        run_model_validator_llm=True,
        model_validator_reason="high_risk_route_requires_llm_review",
    )
    summary = semantic_role_ownership_summary(
        stage="payoff_model_validation",
        review_policy=review_policy,
        artifact_kind="ValidationReport",
    )

    assert summary["selected_role"] == "model_validator"
    assert summary["trigger_condition"] == "high_risk_route_requires_llm_review"
    assert summary["artifact_kind"] == "ValidationReport"
    assert summary["can_validate_model"] is True
    assert summary["executed"] is True


def test_semantic_role_ownership_summary_tracks_trace_handoff_role():
    from trellis.agent.semantic_escalation import semantic_role_ownership_summary

    summary = semantic_role_ownership_summary(
        stage="trace_handoff",
        semantic_extension={"decision": "knowledge_artifact"},
    )

    assert summary["selected_role"] == "knowledge_agent"
    assert summary["trigger_condition"] == "knowledge_artifact_trace_handoff"
    assert summary["artifact_kind"] == "semantic_extension_trace"
    assert summary["scope"] == "trace_handoff_only"


def test_semantic_role_ownership_summary_fail_closed_for_unknown_stage():
    from trellis.agent.semantic_escalation import semantic_role_ownership_summary

    summary = semantic_role_ownership_summary(stage="not_a_stage")

    assert summary["fail_closed"] is True
    assert summary["selected_role"] == ""
    assert summary["selected_stage"] == "not_a_stage"
