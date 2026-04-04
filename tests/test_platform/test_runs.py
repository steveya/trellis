"""Tests for the governed platform run ledger."""

from __future__ import annotations


def test_run_ledger_round_trip_persists_canonical_provenance_fields(tmp_path):
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import (
        ArtifactReference,
        RunLedgerStore,
        build_run_record,
    )

    context = build_execution_context(
        session_id="sess_run_001",
        market_source="treasury_gov",
        run_mode="research",
    )
    store = RunLedgerStore(base_dir=tmp_path)
    record = build_run_record(
        run_id="run_001",
        request_id="request_001",
        status="succeeded",
        action="price_existing_instrument",
        execution_context=context,
        market_snapshot_id="snapshot_001",
        valuation_timestamp="2026-04-03T23:50:00+00:00",
        selected_model={"model_id": "vanilla_option_analytical", "version": "v1"},
        selected_engine={"engine_id": "pricing_engine.local", "version": "1"},
        result_summary={"price": 12.34},
        warnings=("using_cached_snapshot",),
        artifacts=(
            ArtifactReference(
                artifact_id="trace",
                artifact_kind="platform_trace",
                uri="trellis/agent/knowledge/traces/platform/request_001.yaml",
            ),
        ),
    )

    store.create_run(record)
    loaded = store.get_run("run_001")

    assert loaded is not None
    assert loaded.request_id == "request_001"
    assert loaded.run_mode == "research"
    assert loaded.policy_id == "policy_bundle.research.default"
    assert loaded.market_snapshot_id == "snapshot_001"
    assert loaded.selected_model["model_id"] == "vanilla_option_analytical"
    assert loaded.artifact_paths == (
        "trellis/agent/knowledge/traces/platform/request_001.yaml",
    )


def test_run_ledger_attach_artifacts_deduplicates_and_updates_record(tmp_path):
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import RunLedgerStore, build_run_record, legacy_artifact_refs_from_paths

    context = build_execution_context(
        session_id="sess_run_002",
        market_source="mock",
        run_mode="sandbox",
    )
    store = RunLedgerStore(base_dir=tmp_path)
    store.create_run(
        build_run_record(
            run_id="run_002",
            request_id="request_002",
            status="running",
            action="build_then_price",
            execution_context=context,
        )
    )

    updated = store.attach_artifacts(
        "run_002",
        legacy_artifact_refs_from_paths(
            platform_trace_path="/tmp/platform.yaml",
            model_audit_path="/tmp/audit.json",
            task_run_history_path="/tmp/task_runs/history/T13/run.json",
        ),
    )
    updated = store.attach_artifacts(
        "run_002",
        legacy_artifact_refs_from_paths(
            platform_trace_path="/tmp/platform.yaml",
        ),
    )

    assert [item.artifact_kind for item in updated.artifacts] == [
        "model_audit",
        "platform_trace",
        "task_run_history",
    ]
    assert updated.artifact_paths == (
        "/tmp/audit.json",
        "/tmp/platform.yaml",
        "/tmp/task_runs/history/T13/run.json",
    )


def test_legacy_artifact_refs_normalize_current_repo_artifact_surfaces():
    from trellis.platform.runs import legacy_artifact_refs_from_paths

    artifacts = legacy_artifact_refs_from_paths(
        platform_trace_path="/tmp/platform.yaml",
        model_audit_path="/tmp/audit.json",
        task_run_history_path="/tmp/task_runs/history/T13/run.json",
        task_run_latest_path="/tmp/task_runs/latest/T13.json",
        diagnosis_packet_path="/tmp/task_runs/diagnostics/latest/T13.json",
    )

    assert [item.artifact_kind for item in artifacts] == [
        "diagnosis_packet",
        "model_audit",
        "platform_trace",
        "task_run_history",
        "task_run_latest",
    ]
    assert artifacts[0].uri == "/tmp/task_runs/diagnostics/latest/T13.json"
