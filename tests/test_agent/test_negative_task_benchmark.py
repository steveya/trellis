from __future__ import annotations

import json
from pathlib import Path

from trellis.agent.negative_task_benchmark import (
    build_negative_benchmark_report,
    evaluate_negative_task_result,
    load_negative_benchmark_tasks,
    persist_negative_benchmark_record,
    select_negative_benchmark_tasks,
)


ROOT = Path(__file__).resolve().parents[2]


def test_select_negative_benchmark_tasks_filters_requested_ids():
    tasks = load_negative_benchmark_tasks(root=ROOT)
    selected = select_negative_benchmark_tasks(tasks, requested_ids=["N001", "N004"])
    assert [task["id"] for task in selected] == ["N001", "N004"]


def test_evaluate_negative_task_result_recognizes_clarification():
    task = {
        "id": "N001",
        "expected_outcome": "clarification_requested",
        "clarification_contract": {"missing_fields": ["strike", "expiry"]},
    }
    result = {
        "success": False,
        "blocker_details": {
            "reason": "semantic_clarification_required",
            "semantic_gap": {
                "requires_clarification": True,
                "missing_contract_fields": ["strike", "expiry"],
            },
        },
    }
    evaluation = evaluate_negative_task_result(task, result)
    assert evaluation["observed_outcome"] == "clarification_requested"
    assert evaluation["passed"] is True
    assert set(evaluation["observed_missing_fields"]) == {"strike", "expiry"}


def test_evaluate_negative_task_result_recognizes_honest_block():
    task = {
        "id": "N004",
        "expected_outcome": "honest_block",
        "clarification_contract": {"expected_blockers": ["stochastic_rates"]},
    }
    result = {
        "success": False,
        "blocker_details": {
            "blocker_report": {
                "blockers": [{"category": "stochastic_rates"}],
            },
        },
    }
    evaluation = evaluate_negative_task_result(task, result)
    assert evaluation["observed_outcome"] == "honest_block"
    assert evaluation["passed"] is True
    assert evaluation["observed_blocker_categories"] == ("stochastic_rates",)


def test_persist_negative_benchmark_record_writes_history_and_latest(tmp_path):
    record = {
        "task_id": "N001",
        "run_id": "N001_20260413T120000000000Z",
        "run_started_at": "2026-04-13T12:00:00+00:00",
        "run_completed_at": "2026-04-13T12:00:02+00:00",
        "status": "clarification_requested",
    }
    paths = persist_negative_benchmark_record(record, root=tmp_path)
    history_path = Path(paths["history_path"])
    latest_path = Path(paths["latest_path"])
    assert history_path.exists()
    assert latest_path.exists()
    assert json.loads(history_path.read_text())["task_id"] == "N001"
    assert json.loads(latest_path.read_text())["run_id"] == record["run_id"]


def test_build_negative_benchmark_report_summarizes_outcomes():
    report = build_negative_benchmark_report(
        benchmark_name="negative_suite",
        git_revision="deadbeef",
        benchmark_runs=[
            {
                "task_id": "N001",
                "expected_outcome": "clarification_requested",
                "observed_outcome": "clarification_requested",
                "passed_expectation": True,
                "elapsed_seconds": 1.5,
                "token_usage_summary": {"total_tokens": 12},
            },
            {
                "task_id": "N004",
                "expected_outcome": "honest_block",
                "observed_outcome": "honest_block",
                "passed_expectation": True,
                "elapsed_seconds": 2.5,
                "token_usage_summary": {"total_tokens": 8},
            },
        ],
    )
    assert report["task_count"] == 2
    assert report["passed_count"] == 2
    assert report["expected_counts"]["clarification_requested"] == 1
    assert report["observed_counts"]["honest_block"] == 1
