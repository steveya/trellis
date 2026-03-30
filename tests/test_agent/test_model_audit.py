"""Tests for ModelAuditRecord write/load/approve cycle."""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# model_audit module unit tests
# ---------------------------------------------------------------------------

def test_build_and_write_audit_record(tmp_path):
    from trellis.agent.model_audit import (
        ValidationGateResult,
        build_audit_record,
        write_model_audit_record,
        load_model_audit_record,
    )

    gates = [
        ValidationGateResult(gate="import", passed=True, issues=()),
        ValidationGateResult(gate="semantic", passed=True, issues=()),
        ValidationGateResult(gate="bundle", passed=True, issues=(),
                             details={"bundle_id": "default"}),
    ]
    rec = build_audit_record(
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
        validation_gates=gates,
        attempt_number=1,
        total_attempts=1,
        wall_clock_seconds=4.2,
    )

    assert rec.audit_id == "executor_build_20260329T120000_analytical"
    assert rec.all_gates_passed is True
    assert rec.approval_status == "auto_approved"
    assert len(rec.source_code_hash) == 16
    assert rec.build_metrics.wall_clock_seconds == pytest.approx(4.2)

    path = write_model_audit_record(rec, base_dir=tmp_path)
    assert path.exists()

    loaded = load_model_audit_record(path)
    assert loaded["task_id"] == "T99"
    assert loaded["all_gates_passed"] is True
    assert loaded["approval_status"] == "auto_approved"


def test_pending_review_when_gate_fails(tmp_path):
    from trellis.agent.model_audit import ValidationGateResult, build_audit_record, write_model_audit_record

    gates = [
        ValidationGateResult(gate="bundle", passed=False,
                             issues=("invariant failed",)),
    ]
    rec = build_audit_record(
        task_id="T1", run_id="r1", method="mc", instrument_type="option",
        source_code="pass", spec_schema_dict={}, class_name="Foo",
        module_path="foo.py", repo_revision="x", llm_model_id="m",
        knowledge_hash="k", market_state_summary={}, pricing_plan_summary={},
        validation_gates=gates, attempt_number=2, total_attempts=3,
    )
    assert rec.all_gates_passed is False
    assert rec.approval_status == "pending_review"


def test_approve_model_sidecar(tmp_path):
    from trellis.agent.model_audit import (
        ValidationGateResult,
        build_audit_record,
        write_model_audit_record,
        load_model_audit_record,
        approve_model,
    )

    gates = [ValidationGateResult(gate="bundle", passed=True, issues=())]
    rec = build_audit_record(
        task_id="T5", run_id="r5", method="analytical", instrument_type="bond",
        source_code="# bond pricer", spec_schema_dict={}, class_name="BondPayoff",
        module_path="bond.py", repo_revision="rev", llm_model_id="m",
        knowledge_hash="kh", market_state_summary={}, pricing_plan_summary={},
        validation_gates=gates, attempt_number=1, total_attempts=1,
    )
    audit_path = write_model_audit_record(rec, base_dir=tmp_path)

    sidecar = approve_model(
        audit_path, reviewer="alice", status="approved", notes="Looks good"
    )
    assert sidecar.exists()

    merged = load_model_audit_record(audit_path)
    assert "approval" in merged
    assert merged["approval"]["status"] == "approved"
    assert merged["approval"]["reviewer"] == "alice"
    assert "audit_record_hash" in merged["approval"]


def test_benchmark_sidecar_merged_on_load(tmp_path):
    from trellis.agent.model_audit import (
        ValidationGateResult,
        build_audit_record,
        write_model_audit_record,
        load_model_audit_record,
        write_benchmark_sidecar,
    )

    gates = [ValidationGateResult(gate="bundle", passed=True, issues=())]
    rec = build_audit_record(
        task_id="T10", run_id="r10", method="mc", instrument_type="swap",
        source_code="pass", spec_schema_dict={}, class_name="SwapPayoff",
        module_path="swap.py", repo_revision="r", llm_model_id="m",
        knowledge_hash="k", market_state_summary={}, pricing_plan_summary={},
        validation_gates=gates, attempt_number=1, total_attempts=1,
    )
    audit_path = write_model_audit_record(rec, base_dir=tmp_path)

    write_benchmark_sidecar(
        audit_path,
        comparison_status="pass",
        prices={"mc": 98.5, "analytical": 98.6},
        deviations_pct={"analytical": 0.1},
    )

    merged = load_model_audit_record(audit_path)
    assert "benchmark" in merged
    assert merged["benchmark"]["comparison_status"] == "pass"


# ---------------------------------------------------------------------------
# _write_build_audit_record integration (mocked build context)
# ---------------------------------------------------------------------------

def test_write_build_audit_record_does_not_raise():
    """_write_build_audit_record must be non-blocking even with bad inputs."""
    from trellis.agent.executor import _write_build_audit_record
    import time

    # Should silently succeed or silently fail — never raise
    _write_build_audit_record(
        compiled_request=None,
        model="test-model",
        pricing_plan=None,
        instrument_type=None,
        spec_schema=None,
        output_module_path="instruments/_agent/test.py",
        code="class T: pass",
        market_state=None,
        attempt_number=1,
        gate_results=[],
        build_start_time=time.time(),
    )


def test_get_repo_revision_returns_string():
    from trellis.agent.executor import _get_repo_revision
    rev = _get_repo_revision()
    assert isinstance(rev, str)
    assert len(rev) > 0
