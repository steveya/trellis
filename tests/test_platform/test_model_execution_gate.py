"""Tests for governed model execution-time lifecycle gating."""

from __future__ import annotations

import pytest

from trellis.platform.context import RunMode, build_execution_context


def _context(run_mode: RunMode):
    return build_execution_context(
        session_id="sess_gate_001",
        market_source="treasury_gov",
        run_mode=run_mode,
    )


def _store_with_statuses(tmp_path, statuses):
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelRegistryStore,
        ModelVersionRecord,
    )

    store = ModelRegistryStore(base_dir=tmp_path)
    store.create_model(
        ModelRecord(
            model_id="vanilla_option_analytical",
            semantic_id="vanilla_option",
            semantic_version="1.0.0",
            product_family="equity_option",
            instrument_class="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            underlier_structure="single_asset",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "underlier_spot", "black_vol_surface"),
            supported_method_families=("analytical",),
        )
    )
    for index, status in enumerate(statuses, start=1):
        version = f"v{index}"
        store.create_version(
            ModelVersionRecord(
                model_id="vanilla_option_analytical",
                version=version,
                contract_summary={"semantic_id": "vanilla_option"},
                methodology_summary={"method_family": "analytical"},
                engine_binding={"engine_id": "pricing_engine.local"},
            ),
            actor="builder",
            reason="initial_candidate",
        )
        if status is ModelLifecycleStatus.VALIDATED:
            store.transition_version(
                "vanilla_option_analytical",
                version,
                ModelLifecycleStatus.VALIDATED,
                actor="validator",
                reason="validation_bundle_passed",
            )
        elif status is ModelLifecycleStatus.APPROVED:
            store.transition_version(
                "vanilla_option_analytical",
                version,
                ModelLifecycleStatus.VALIDATED,
                actor="validator",
                reason="validation_bundle_passed",
            )
            store.transition_version(
                "vanilla_option_analytical",
                version,
                ModelLifecycleStatus.APPROVED,
                actor="reviewer",
                reason="manual_approval",
            )
        elif status is ModelLifecycleStatus.DEPRECATED:
            store.transition_version(
                "vanilla_option_analytical",
                version,
                ModelLifecycleStatus.DEPRECATED,
                actor="reviewer",
                reason="withdrawn",
            )
    return store


def test_production_gate_selects_approved_version_over_newer_validated_candidate(tmp_path):
    from trellis.platform.models import ModelLifecycleStatus, evaluate_model_execution_gate

    store = _store_with_statuses(
        tmp_path,
        (ModelLifecycleStatus.APPROVED, ModelLifecycleStatus.VALIDATED),
    )

    result = evaluate_model_execution_gate(
        registry=store,
        model_id="vanilla_option_analytical",
        execution_context=_context(RunMode.PRODUCTION),
    )

    assert result.allowed is True
    assert result.selected_model["version"] == "v1"
    assert result.selected_model["status"] == "approved"
    assert "approved" in result.allowed_statuses


def test_research_gate_allows_validated_model_when_no_approved_version_exists(tmp_path):
    from trellis.platform.models import ModelLifecycleStatus, evaluate_model_execution_gate

    store = _store_with_statuses(tmp_path, (ModelLifecycleStatus.VALIDATED,))

    result = evaluate_model_execution_gate(
        registry=store,
        model_id="vanilla_option_analytical",
        execution_context=_context(RunMode.RESEARCH),
    )

    assert result.allowed is True
    assert result.selected_model["version"] == "v1"
    assert result.selected_model["status"] == "validated"


def test_sandbox_gate_allows_draft_model_explicitly(tmp_path):
    from trellis.platform.models import ModelLifecycleStatus, evaluate_model_execution_gate

    store = _store_with_statuses(tmp_path, (ModelLifecycleStatus.DRAFT,))

    result = evaluate_model_execution_gate(
        registry=store,
        model_id="vanilla_option_analytical",
        execution_context=_context(RunMode.SANDBOX),
    )

    assert result.allowed is True
    assert result.selected_model["status"] == "draft"


def test_production_gate_rejects_validated_only_model_with_structured_reason(tmp_path):
    from trellis.platform.models import (
        ModelExecutionGateError,
        ModelLifecycleStatus,
        enforce_model_execution_gate,
        evaluate_model_execution_gate,
    )

    store = _store_with_statuses(tmp_path, (ModelLifecycleStatus.VALIDATED,))
    result = evaluate_model_execution_gate(
        registry=store,
        model_id="vanilla_option_analytical",
        execution_context=_context(RunMode.PRODUCTION),
    )

    assert result.allowed is False
    assert result.rejection_codes == ("lifecycle_not_allowed",)
    assert result.rejections[0].status == "validated"

    with pytest.raises(ModelExecutionGateError) as excinfo:
        enforce_model_execution_gate(
            registry=store,
            model_id="vanilla_option_analytical",
            execution_context=_context(RunMode.PRODUCTION),
        )

    assert excinfo.value.result.rejection_codes == ("lifecycle_not_allowed",)


def test_sandbox_gate_still_rejects_deprecated_model(tmp_path):
    from trellis.platform.models import ModelLifecycleStatus, evaluate_model_execution_gate

    store = _store_with_statuses(tmp_path, (ModelLifecycleStatus.DEPRECATED,))
    result = evaluate_model_execution_gate(
        registry=store,
        model_id="vanilla_option_analytical",
        execution_context=_context(RunMode.SANDBOX),
    )

    assert result.allowed is False
    assert result.rejection_codes == ("lifecycle_not_allowed",)
    assert result.rejections[0].status == "deprecated"
