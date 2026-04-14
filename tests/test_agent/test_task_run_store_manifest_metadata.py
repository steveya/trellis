from __future__ import annotations

from datetime import datetime, timezone


def test_build_task_run_record_carries_manifest_and_timestamp_metadata():
    from trellis.agent.task_run_store import build_task_run_record

    record = build_task_run_record(
        {
            "id": "F001",
            "title": "FinancePy parity task",
            "status": "pending",
            "task_corpus": "benchmark_financepy",
            "task_definition_version": 2,
            "task_definition_manifest": "TASKS_BENCHMARK_FINANCEPY.yaml",
            "market_scenario_id": "flat_usd_equity_vanilla",
        },
        {
            "success": True,
            "run_started_at": "2026-04-13T12:00:00+00:00",
            "run_completed_at": "2026-04-13T12:00:03+00:00",
            "execution_mode": "live",
            "elapsed_seconds": 3.0,
            "token_usage_summary": {"total_tokens": 10},
        },
        persisted_at=datetime(2026, 4, 13, 12, 0, 5, tzinfo=timezone.utc),
    )

    assert record["run_started_at"] == "2026-04-13T12:00:00+00:00"
    assert record["run_completed_at"] == "2026-04-13T12:00:03+00:00"
    assert record["task"]["task_corpus"] == "benchmark_financepy"
    assert record["task"]["task_definition_version"] == 2
    assert record["task"]["market_scenario_id"] == "flat_usd_equity_vanilla"
    assert record["summary"]["task_corpus"] == "benchmark_financepy"
    assert record["summary"]["task_definition_version"] == 2
