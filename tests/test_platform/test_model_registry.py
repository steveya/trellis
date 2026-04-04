"""Tests for the governed platform model registry."""

from __future__ import annotations

import pytest


def test_model_registry_create_and_load_round_trip(tmp_path):
    from trellis.platform.models import ModelLifecycleStatus, ModelRecord, ModelRegistryStore

    store = ModelRegistryStore(base_dir=tmp_path)
    record = ModelRecord(
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
        status=ModelLifecycleStatus.DRAFT,
        tags=("equity", "governed"),
    )

    stored = store.create_model(record)
    loaded = store.get_model("vanilla_option_analytical")

    assert stored == loaded
    assert loaded.match_basis()["semantic_id"] == "vanilla_option"
    assert loaded.match_basis()["supported_method_families"] == ("analytical",)


def test_model_registry_create_version_updates_latest_and_preserves_lineage(tmp_path):
    from trellis.platform.models import (
        ModelLineage,
        ModelRecord,
        ModelRegistryStore,
        ModelVersionRecord,
    )

    store = ModelRegistryStore(base_dir=tmp_path)
    store.create_model(
        ModelRecord(
            model_id="callable_bond_tree",
            semantic_id="callable_bond",
            semantic_version="1.0.0",
            product_family="callable_bond",
            instrument_class="bond",
            payoff_family="callable_fixed_income",
            exercise_style="bermudan",
            underlier_structure="single_curve",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "vol_surface"),
            supported_method_families=("rate_tree",),
        )
    )

    version = store.create_version(
        ModelVersionRecord(
            model_id="callable_bond_tree",
            version="v1",
            contract_summary={"semantic_id": "callable_bond"},
            methodology_summary={"method_family": "rate_tree"},
            assumptions=("mean_reversion=0.1",),
            engine_binding={"engine_id": "pricing_engine.local"},
            validation_summary={"all_gates_passed": True},
            validation_refs=("task_runs/audits/standalone/rate_tree.json",),
            lineage=ModelLineage(
                source_run_id="run_001",
                source_request_id="request_001",
                source_audit_id="audit_001",
                source_audit_path="task_runs/audits/standalone/rate_tree.json",
            ),
            artifacts={"module_path": "trellis/instruments/_agent/callable_bond.py"},
        ),
        actor="builder",
        reason="initial_candidate",
    )

    model = store.get_model("callable_bond_tree")
    loaded_version = store.get_version("callable_bond_tree", "v1")

    assert model.latest_version == "v1"
    assert model.status.value == "draft"
    assert loaded_version.transitions[0].to_status.value == "draft"
    assert loaded_version.lineage.source_audit_id == "audit_001"
    assert loaded_version.validation_refs == ("task_runs/audits/standalone/rate_tree.json",)


def test_model_registry_transitions_are_explicit_and_auditable(tmp_path):
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelRegistryStore,
        ModelVersionRecord,
    )

    store = ModelRegistryStore(base_dir=tmp_path)
    store.create_model(
        ModelRecord(
            model_id="swaption_lsm",
            semantic_id="swaption",
            semantic_version="1.0.0",
            product_family="rates_option",
            instrument_class="swaption",
            payoff_family="swaption",
            exercise_style="bermudan",
            underlier_structure="single_curve",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "vol_surface"),
            supported_method_families=("monte_carlo",),
        )
    )
    store.create_version(
        ModelVersionRecord(
            model_id="swaption_lsm",
            version="v1",
            contract_summary={"semantic_id": "swaption"},
            methodology_summary={"method_family": "monte_carlo"},
        ),
        actor="builder",
        reason="initial_candidate",
    )

    validated = store.transition_version(
        "swaption_lsm",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="validation_bundle_passed",
        notes="Cross-method validation cleared",
    )
    approved = store.transition_version(
        "swaption_lsm",
        "v1",
        ModelLifecycleStatus.APPROVED,
        actor="reviewer",
        reason="manual_approval",
    )
    model = store.get_model("swaption_lsm")

    assert validated.status is ModelLifecycleStatus.VALIDATED
    assert approved.status is ModelLifecycleStatus.APPROVED
    assert [entry.to_status.value for entry in approved.transitions] == [
        "draft",
        "validated",
        "approved",
    ]
    assert approved.transitions[-1].changed_by == "reviewer"
    assert model.status is ModelLifecycleStatus.APPROVED
    assert model.latest_approved_version == "v1"


def test_model_registry_rejects_invalid_lifecycle_transition(tmp_path):
    from trellis.platform.models import (
        InvalidLifecycleTransitionError,
        ModelLifecycleStatus,
        ModelRecord,
        ModelRegistryStore,
        ModelVersionRecord,
    )

    store = ModelRegistryStore(base_dir=tmp_path)
    store.create_model(
        ModelRecord(
            model_id="barrier_analytic",
            semantic_id="barrier_option",
            semantic_version="1.0.0",
            product_family="equity_option",
            instrument_class="barrier_option",
            payoff_family="barrier_option",
            exercise_style="european",
            underlier_structure="single_asset",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "underlier_spot", "black_vol_surface"),
            supported_method_families=("analytical",),
        )
    )
    store.create_version(
        ModelVersionRecord(
            model_id="barrier_analytic",
            version="v1",
            contract_summary={"semantic_id": "barrier_option"},
            methodology_summary={"method_family": "analytical"},
        ),
        actor="builder",
        reason="initial_candidate",
    )

    with pytest.raises(InvalidLifecycleTransitionError):
        store.transition_version(
            "barrier_analytic",
            "v1",
            ModelLifecycleStatus.APPROVED,
            actor="reviewer",
            reason="skip_validation",
        )


def test_legacy_model_audit_no_longer_auto_approves_successful_builds():
    from trellis.agent.model_audit import ValidationGateResult, build_audit_record

    record = build_audit_record(
        task_id="T99",
        run_id="executor_build_20260329T120000",
        method="analytical",
        instrument_type="cds",
        source_code="class CDSPayoff:\n    pass",
        spec_schema_dict={"class_name": "CDSPayoff", "spec_name": "CDSSpec"},
        class_name="CDSPayoff",
        module_path="instruments/_agent/cdspayoff.py",
        repo_revision="abc123",
        llm_model_id="claude-sonnet-4-6",
        knowledge_hash="deadbeef01234567",
        market_state_summary={"as_of": "2024-01-15", "discount_factors": {}},
        pricing_plan_summary={"method": "analytical"},
        validation_gates=(
            ValidationGateResult(gate="import", passed=True, issues=()),
            ValidationGateResult(gate="semantic", passed=True, issues=()),
        ),
        attempt_number=1,
        total_attempts=1,
        wall_clock_seconds=4.2,
    )

    assert record.all_gates_passed is True
    assert record.approval_status == "pending_review"
