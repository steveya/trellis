from __future__ import annotations

import json
from pathlib import Path

import pytest

from trellis.agent.financepy_benchmark import (
    build_financepy_benchmark_report,
    extract_trellis_benchmark_outputs,
    load_financepy_benchmark_tasks,
    persist_financepy_benchmark_record,
    select_financepy_benchmark_tasks,
)
from trellis.agent.financepy_reference import price_financepy_reference


ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("financepy")


def test_select_financepy_benchmark_tasks_filters_requested_ids():
    tasks = load_financepy_benchmark_tasks(root=ROOT)
    selected = select_financepy_benchmark_tasks(tasks, requested_ids=["F001", "F002"])
    assert [task["id"] for task in selected] == ["F001", "F002"]


def test_persist_financepy_benchmark_record_writes_history_and_latest(tmp_path):
    record = {
        "task_id": "F001",
        "run_id": "F001_20260413T120000000000Z",
        "run_started_at": "2026-04-13T12:00:00+00:00",
        "run_completed_at": "2026-04-13T12:00:05+00:00",
        "status": "priced",
    }
    paths = persist_financepy_benchmark_record(record, root=tmp_path)
    history_path = Path(paths["history_path"])
    latest_path = Path(paths["latest_path"])
    assert history_path.exists()
    assert latest_path.exists()
    assert json.loads(history_path.read_text())["task_id"] == "F001"
    assert json.loads(latest_path.read_text())["run_id"] == record["run_id"]


def test_price_financepy_reference_supports_equity_vanilla_and_fx():
    tasks = {task["id"]: task for task in load_financepy_benchmark_tasks(root=ROOT)}
    equity = price_financepy_reference(tasks["F001"], root=ROOT)
    fx = price_financepy_reference(tasks["F002"], root=ROOT)
    assert equity["outputs"]["price"] > 0.0
    assert "delta" in equity["outputs"]
    assert fx["outputs"]["price"] != 0.0
    assert "delta" in fx["outputs"]


def test_build_financepy_benchmark_report_accumulates_totals():
    report = build_financepy_benchmark_report(
        benchmark_name="financepy_suite",
        git_revision="deadbeef",
        benchmark_runs=[
            {
                "status": "priced",
                "comparison_summary": {"status": "passed"},
                "cold_agent_token_usage": {"total_tokens": 12},
                "cold_agent_elapsed_seconds": 1.5,
                "warm_agent_mean_seconds": 0.2,
                "financepy_elapsed_seconds": 0.1,
            },
            {
                "status": "failed",
                "comparison_summary": {"status": "failed"},
                "cold_agent_token_usage": {"total_tokens": 8},
                "cold_agent_elapsed_seconds": 2.5,
                "warm_agent_mean_seconds": 0.3,
                "financepy_elapsed_seconds": 0.2,
            },
        ],
    )
    assert report["task_count"] == 2
    assert report["priced_count"] == 1
    assert report["comparison_pass_count"] == 1
    assert report["token_usage_total"] == 20
    assert report["cold_elapsed_seconds_total"] == 4.0


def test_extract_trellis_benchmark_outputs_prefers_cold_run_prices_over_warm_probe():
    outputs = extract_trellis_benchmark_outputs(
        {
            "summary": {"prices": {"analytical": 10.45}},
            "comparison": {"prices": {"black_scholes": 10.45}},
            "result": {"price": 10.45},
        },
        {"last_price": 398.35},
    )

    assert outputs["price"] == 10.45
    assert outputs["analytical"] == 10.45
    assert outputs["black_scholes"] == 10.45
