"""Deterministic paired FpML/native task-conformance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POSITIVE_IDS = {"FPC001", "FPC002", "FPC003"}
HONEST_BLOCK_IDS = {
    "FPC101",
    "FPC102",
    "FPC103",
    "FPC104",
    "FPC105",
    "FPC106",
}


def _tasks():
    from trellis.agent.task_manifests import load_fpml_conformance_tasks

    return load_fpml_conformance_tasks(root=ROOT)


def _task_lookup():
    return {task["id"]: task for task in _tasks()}


def _unexpected_agent_call(*args, **kwargs):  # pragma: no cover - assertion helper
    raise AssertionError("FpML conformance must not invoke an agent build path")


def _run(task, tmp_path, monkeypatch):
    from trellis.agent.task_runtime import build_market_state, run_task

    monkeypatch.setenv("TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST", "1")
    return run_task(
        task,
        build_market_state(),
        build_fn=_unexpected_agent_call,
        task_run_storage_root=tmp_path,
        task_run_storage_layout="standalone",
        recovery_mode="strict",
        execution_mode_override="deterministic_replay",
    )


def test_fpml_conformance_manifest_has_paired_and_honest_block_cohorts():
    tasks = _tasks()
    by_id = {task["id"]: task for task in tasks}

    assert set(by_id) == POSITIVE_IDS | HONEST_BLOCK_IDS
    assert all(task["task_kind"] == "fpml_conformance" for task in tasks)
    assert all(task["task_corpus"] == "fpml_conformance" for task in tasks)
    assert all(task["market_scenario_id"] == "usd_rates_smile" for task in tasks)
    assert {by_id[task_id]["expected_outcome"] for task_id in POSITIVE_IDS} == {
        "pricing_success"
    }
    assert {by_id[task_id]["expected_outcome"] for task_id in HONEST_BLOCK_IDS} == {
        "honest_block"
    }


def test_native_contract_oracle_builds_existing_ir_types():
    from trellis.agent.contract_ir import ContractIR, contract_ir_economic_identity
    from trellis.agent.fpml_conformance import build_native_conformance_contract
    from trellis.agent.static_leg_contract import (
        StaticLegContractIR,
        static_leg_economic_identity,
    )

    tasks = _task_lookup()
    swap = build_native_conformance_contract(tasks["FPC001"]["native_contract"])
    swaption = build_native_conformance_contract(tasks["FPC002"]["native_contract"])
    cap = build_native_conformance_contract(tasks["FPC003"]["native_contract"])

    assert isinstance(swap, StaticLegContractIR)
    assert isinstance(swaption, ContractIR)
    assert isinstance(cap, StaticLegContractIR)
    assert static_leg_economic_identity(swap).startswith("static_leg:v1:")
    assert contract_ir_economic_identity(swaption).startswith("contract_ir:v1:")
    assert static_leg_economic_identity(cap).startswith("static_leg:v1:")


@pytest.mark.parametrize("task_id", sorted(POSITIVE_IDS))
def test_positive_pair_proves_identity_selection_price_and_provenance(
    task_id,
    tmp_path,
    monkeypatch,
):
    import trellis.agent.model_validator as model_validator

    monkeypatch.setattr(model_validator, "validate_model", _unexpected_agent_call)
    monkeypatch.setattr(
        model_validator,
        "validate_model_for_request",
        _unexpected_agent_call,
    )
    task = _task_lookup()[task_id]
    result = _run(task, tmp_path, monkeypatch)

    assert result["success"] is True
    assert result["passed_expectation"] is True
    assert result["outcome_class"] == "compare_ready"
    assert result["task_kind"] == "fpml_conformance"
    assert result["execution_mode"] == "deterministic_import_conformance"
    assert result["conformance"]["identity"]["equal"] is True
    assert result["conformance"]["economic_projection_equal"] is True
    assert result["conformance"]["selection"]["equal"] is True
    assert result["conformance"]["price"]["within_tolerance"] is True
    assert result["cross_validation"]["status"] == "passed"
    assert result["import_report"]["mapping_provenance"]
    assert result["import_report"]["clarification"] == {
        "requires_clarification": False,
        "missing_fields": [],
        "ambiguous_fields": [],
        "messages": [],
    }
    assert result["agent_calls"] == {
        "builder": False,
        "codegen": False,
        "quant_review": False,
        "model_validator": False,
        "recovery": False,
    }
    assert result["conformance"]["envelope_variants"]
    assert all(
        variant["passed"] for variant in result["conformance"]["envelope_variants"]
    )

    xml = (ROOT / task["fpml"]["fixture"]).read_text()
    assert xml not in repr(result)


@pytest.mark.parametrize("task_id", sorted(HONEST_BLOCK_IDS))
def test_expected_import_failures_are_certified_without_agent_calls(
    task_id,
    tmp_path,
    monkeypatch,
):
    import trellis.agent.model_validator as model_validator

    monkeypatch.setattr(model_validator, "validate_model", _unexpected_agent_call)
    monkeypatch.setattr(
        model_validator,
        "validate_model_for_request",
        _unexpected_agent_call,
    )
    task = _task_lookup()[task_id]
    result = _run(task, tmp_path, monkeypatch)

    assert result["success"] is False
    assert result["passed_expectation"] is True
    assert result["expected_honest_block"] is True
    assert result["outcome_class"] == "honest_block"
    assert result["conformance"]["observed_blocker_ids"] == task["expected_blocker_ids"]
    assert result["conformance"]["price"] is None
    assert result["agent_calls"] == {
        "builder": False,
        "codegen": False,
        "quant_review": False,
        "model_validator": False,
        "recovery": False,
    }


def test_conformance_results_persist_body_free_import_and_clarification_evidence(
    tmp_path,
    monkeypatch,
):
    tasks = _task_lookup()
    positive = _run(tasks["FPC001"], tmp_path / "positive", monkeypatch)
    blocked = _run(tasks["FPC101"], tmp_path / "blocked", monkeypatch)
    latest = json.loads(Path(positive["task_run_latest_path"]).read_text())
    blocked_latest = json.loads(Path(blocked["task_run_latest_path"]).read_text())

    assert latest["task_kind"] == "fpml_conformance"
    assert latest["result"]["import_report"]["mapping_provenance"]
    assert latest["result"]["import_report"]["economic_identity"]
    assert latest["result"]["agent_calls"]["model_validator"] is False
    assert blocked_latest["result"]["import_report"]["clarification"] == {
        "requires_clarification": True,
        "missing_fields": ["valuation_party_id"],
        "ambiguous_fields": [],
        "messages": ["Provide unambiguous FpML values for: valuation_party_id."],
    }
    assert (ROOT / tasks["FPC001"]["fpml"]["fixture"]).read_text() not in repr(latest)
    assert (ROOT / tasks["FPC101"]["fpml"]["fixture"]).read_text() not in repr(
        blocked_latest
    )


def test_remediation_excludes_all_certified_conformance_honest_blocks(
    tmp_path,
    monkeypatch,
):
    from scripts.remediate import analyze_failures

    tasks = _task_lookup()
    results = [
        _run(tasks[task_id], tmp_path / task_id, monkeypatch)
        for task_id in sorted(HONEST_BLOCK_IDS)
    ]

    analysis = analyze_failures(results)
    assert all(not entries for entries in analysis.values())
