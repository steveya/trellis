"""Tests for canonical governed audit-bundle assembly."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def test_run_ledger_build_audit_bundle_for_successful_run(tmp_path):
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import (
        ArtifactReference,
        RunLedgerStore,
        build_run_record,
    )

    trace_path = _write_yaml(
        tmp_path / "platform" / "request_success.yaml",
        {
            "request_id": "request_success",
            "request_type": "term_sheet",
            "entry_point": "session",
            "action": "price_existing_instrument",
            "status": "succeeded",
            "outcome": "priced",
            "instrument_type": "european_option",
            "product_instrument": "vanilla_option",
            "route_method": "analytical",
            "measures": ["price", "delta"],
            "request_metadata": {
                "semantic_contract": {
                    "semantic_id": "vanilla_option",
                    "semantic_version": "2026.04",
                },
            },
            "semantic_checkpoint": {
                "semantic_id": "vanilla_option",
                "semantic_version": "2026.04",
                "payoff_family": "vanilla",
            },
            "generation_boundary": {
                "method": "analytical",
                "lowering": {
                    "route_id": "analytical_black76",
                    "route_family": "analytical",
                },
            },
            "validation_contract": {
                "bundle_id": "validation_bundle.vanilla_option.default",
            },
            "cycle_report": {
                "request_id": "request_success",
                "status": "succeeded",
                "outcome": "priced",
                "success": True,
                "pricing_method": "analytical",
                "validation_contract_id": "validation_bundle.vanilla_option.default",
                "stage_statuses": {
                    "quant": "passed",
                    "validation_bundle": "passed",
                    "critic": "passed",
                    "arbiter": "passed",
                    "model_validator": "skipped",
                },
                "failure_count": 0,
                "deterministic_blockers": [],
                "conceptual_blockers": [],
                "calibration_blockers": [],
                "residual_limitations": [],
                "residual_risks": [],
            },
            "details": {
                "result_type": "price_and_greeks",
            },
            "events": [
                {
                    "event": "request_succeeded",
                    "status": "ok",
                    "timestamp": "2026-04-03T23:20:00+00:00",
                    "details": {"price": 12.34},
                },
            ],
        },
    )
    audit_path = _write_json(
        tmp_path / "audits" / "model_audit.json",
        {
            "schema_version": 1,
            "audit_id": "audit_001",
            "task_id": "T13",
            "run_id": "run_success",
            "method": "analytical",
            "instrument_type": "european_option",
            "timestamp": "2026-04-03T23:19:00+00:00",
            "class_name": "EuropeanOptionPricer",
            "module_path": "trellis.models.black",
            "source_code_hash": "abc123",
            "all_gates_passed": True,
            "approval_status": "pending_review",
            "validation_gates": [{"gate": "import", "passed": True}],
            "build_metrics": {
                "attempt_number": 1,
                "total_attempts": 1,
                "wall_clock_seconds": 1.2,
            },
        },
    )
    diagnosis_path = _write_json(
        tmp_path / "diagnostics" / "run_success.json",
        {
            "schema_version": 2,
            "task": {"id": "T13", "title": "Price vanilla option"},
            "run": {"run_id": "task_run_success"},
            "outcome": {
                "success": True,
                "status": "completed",
                "failure_bucket": "success",
                "headline": "Task completed successfully.",
                "next_action": "No action required.",
            },
            "primary_failure": {},
        },
    )
    task_run_path = _write_json(
        tmp_path / "task_runs" / "history" / "T13" / "task_run_success.json",
        {
            "task_id": "T13",
            "task_kind": "pricing",
            "run_id": "task_run_success",
            "persisted_at": "2026-04-03T23:21:00+00:00",
            "summary": {
                "success": True,
                "status": "completed",
                "comparison_status": "passed",
            },
            "workflow": {"status": "completed", "next_action": "No action required."},
            "comparison": {"summary": {"status": "passed"}},
        },
    )

    context = build_execution_context(
        session_id="sess_audit_success",
        market_source="treasury_gov",
        run_mode="research",
    )
    store = RunLedgerStore(base_dir=tmp_path / "ledger")
    store.create_run(
        build_run_record(
            run_id="run_success",
            request_id="request_success",
            status="succeeded",
            action="price_existing_instrument",
            execution_context=context,
            trade_identity={
                "instrument_type": "european_option",
                "trade_id": "trade_001",
            },
            selected_model={
                "model_id": "vanilla_option_analytical",
                "version": "v2",
                "status": "validated",
            },
            selected_engine={"engine_id": "pricing_engine.local", "version": "1"},
            market_snapshot_id="snapshot_123",
            valuation_timestamp="2026-04-03T23:18:00+00:00",
            result_summary={"price": 12.34, "delta": 0.51},
            warnings=("using_cached_snapshot",),
            validation_summary={"status": "passed"},
            policy_outcome={
                "policy_id": "policy_bundle.research.default",
                "allowed": True,
                "blocker_codes": [],
                "blockers": [],
            },
            provenance={"route_family": "analytical", "method_family": "analytical"},
            artifacts=(
                ArtifactReference(
                    artifact_id="platform_trace",
                    artifact_kind="platform_trace",
                    uri=str(trace_path),
                ),
                ArtifactReference(
                    artifact_id="model_audit",
                    artifact_kind="model_audit",
                    uri=str(audit_path),
                ),
                ArtifactReference(
                    artifact_id="diagnosis_packet",
                    artifact_kind="diagnosis_packet",
                    uri=str(diagnosis_path),
                ),
                ArtifactReference(
                    artifact_id="task_run_history",
                    artifact_kind="task_run_history",
                    uri=str(task_run_path),
                ),
            ),
        )
    )

    bundle = store.build_audit_bundle("run_success")

    assert bundle.run["run_id"] == "run_success"
    assert bundle.inputs["request"]["request_type"] == "term_sheet"
    assert bundle.inputs["parsed_contract"]["semantic_checkpoint"]["semantic_id"] == "vanilla_option"
    assert bundle.inputs["provider_bindings"]["market_data"]["primary"]["provider_id"] == "market_data.treasury_gov"
    assert bundle.inputs["market_snapshot_id"] == "snapshot_123"
    assert bundle.execution["selected_model"]["model_id"] == "vanilla_option_analytical"
    assert bundle.execution["policy_outcome"]["allowed"] is True
    assert bundle.outputs["price"] == 12.34
    assert bundle.diagnostics["blocked"] is False
    assert bundle.diagnostics["blocker_codes"] == []
    assert bundle.diagnostics["diagnosis"]["headline"] == "Task completed successfully."
    assert bundle.artifacts["platform_trace"]["route_method"] == "analytical"
    assert bundle.artifacts["platform_trace"]["agent_cycle"]["status"] == "passed"
    assert bundle.artifacts["model_audit"]["all_gates_passed"] is True
    assert bundle.artifacts["task_run"]["task_id"] == "T13"


def test_run_ledger_build_audit_bundle_for_blocked_run_preserves_failure_context(tmp_path):
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import ArtifactReference, RunLedgerStore, build_run_record

    trace_path = _write_yaml(
        tmp_path / "platform" / "request_blocked.yaml",
        {
            "request_id": "request_blocked",
            "request_type": "trade",
            "entry_point": "mcp",
            "action": "price_trade",
            "status": "blocked",
            "outcome": "request_blocked",
            "instrument_type": "credit_default_swap",
            "blocker_codes": ["provider_binding_missing", "policy_denied"],
            "details": {
                "reason": "market-data provider binding is required",
            },
            "events": [
                {
                    "event": "request_blocked",
                    "status": "error",
                    "timestamp": "2026-04-03T23:40:00+00:00",
                    "details": {"reason": "market-data provider binding is required"},
                },
            ],
        },
    )

    context = build_execution_context(
        session_id="sess_audit_blocked",
        market_source="treasury_gov",
        run_mode="production",
    )
    store = RunLedgerStore(base_dir=tmp_path / "ledger")
    store.create_run(
        build_run_record(
            run_id="run_blocked",
            request_id="request_blocked",
            status="blocked",
            action="price_trade",
            execution_context=context,
            trade_identity={
                "instrument_type": "credit_default_swap",
                "trade_id": "trade_blocked",
            },
            market_snapshot_id="",
            valuation_timestamp="",
            warnings=("provider_binding_missing",),
            result_summary={"error": "policy blocked execution"},
            policy_outcome={
                "policy_id": "policy_bundle.production.default",
                "allowed": False,
                "blocker_codes": ["provider_binding_missing", "policy_denied"],
                "blockers": [
                    {
                        "code": "provider_binding_missing",
                        "message": "Explicit market-data binding is required.",
                        "requirement": "provider_disclosure",
                        "field": "provider_bindings.market_data.primary",
                    },
                ],
            },
            artifacts=(
                ArtifactReference(
                    artifact_id="platform_trace",
                    artifact_kind="platform_trace",
                    uri=str(trace_path),
                ),
            ),
        )
    )

    bundle = store.build_audit_bundle("run_blocked")

    assert bundle.run["status"] == "blocked"
    assert bundle.inputs["request"]["request_type"] == "trade"
    assert bundle.execution["policy_outcome"]["allowed"] is False
    assert bundle.diagnostics["blocked"] is True
    assert bundle.diagnostics["blocker_codes"] == [
        "policy_denied",
        "provider_binding_missing",
    ]
    assert bundle.diagnostics["failure_context"]["error"] == "policy blocked execution"
    assert bundle.diagnostics["failure_context"]["trace_reason"] == "market-data provider binding is required"
    assert bundle.diagnostics["trace_status"] == "blocked"
    assert bundle.artifacts["platform_trace"]["status"] == "blocked"
