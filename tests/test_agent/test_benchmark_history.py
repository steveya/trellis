from __future__ import annotations

import json

import pytest

from trellis.agent.benchmark_history import (
    build_benchmark_history_scorecard,
    load_benchmark_history_records,
    render_benchmark_history_scorecard,
    save_benchmark_history_scorecard,
)


def test_load_benchmark_history_records_filters_campaign_and_task(tmp_path):
    history_root = tmp_path / "history"
    (history_root / "F001").mkdir(parents=True)
    (history_root / "F002").mkdir(parents=True)
    (history_root / "F001" / "run_a.json").write_text(
        json.dumps(
            {
                "task_id": "F001",
                "benchmark_campaign_id": "daily_suite",
                "run_started_at": "2026-04-14T01:00:00+00:00",
                "run_id": "run_a",
            }
        )
    )
    (history_root / "F002" / "run_b.json").write_text(
        json.dumps(
            {
                "task_id": "F002",
                "benchmark_campaign_id": "smoke",
                "run_started_at": "2026-04-14T01:05:00+00:00",
                "run_id": "run_b",
            }
        )
    )

    records = load_benchmark_history_records(
        benchmark_root=tmp_path,
        task_ids=["F001"],
        campaign_id="daily_suite",
    )

    assert len(records) == 1
    assert records[0]["task_id"] == "F001"


def test_build_benchmark_history_scorecard_tracks_improvement(tmp_path):
    records = [
        {
            "task_id": "F001",
            "title": "Vanilla",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_1",
            "run_started_at": "2026-04-14T01:00:00+00:00",
            "run_completed_at": "2026-04-14T01:00:05+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "git_sha": "aaaa1111",
            "knowledge_revision": "know1111",
            "comparison_summary": {"status": "failed"},
            "cold_agent_elapsed_seconds": 5.0,
            "cold_agent_token_usage": {"total_tokens": 200},
        },
        {
            "task_id": "F001",
            "title": "Vanilla",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_2",
            "run_started_at": "2026-04-15T01:00:00+00:00",
            "run_completed_at": "2026-04-15T01:00:03+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "git_sha": "bbbb2222",
            "knowledge_revision": "know2222",
            "comparison_summary": {"status": "passed"},
            "cold_agent_elapsed_seconds": 3.0,
            "cold_agent_token_usage": {"total_tokens": 150},
        },
        {
            "task_id": "N001",
            "title": "Clarification",
            "task_corpus": "negative",
            "run_id": "run_3",
            "run_started_at": "2026-04-15T01:00:00+00:00",
            "run_completed_at": "2026-04-15T01:00:02+00:00",
            "execution_mode": "cold_agent_negative",
            "git_sha": "bbbb2222",
            "knowledge_revision": "know2222",
            "passed_expectation": True,
            "observed_outcome": "clarification_requested",
            "elapsed_seconds": 2.0,
            "token_usage_summary": {"total_tokens": 75},
        },
    ]

    financepy_report = build_benchmark_history_scorecard(
        scorecard_name="financepy_daily_scorecard",
        benchmark_kind="financepy",
        benchmark_runs=[records[0], records[1]],
        campaign_id="daily_suite",
    )
    negative_report = build_benchmark_history_scorecard(
        scorecard_name="negative_daily_scorecard",
        benchmark_kind="negative",
        benchmark_runs=[records[2]],
        campaign_id="daily_suite",
    )

    assert financepy_report["improved_count"] == 1
    assert financepy_report["latest_pass_count"] == 1
    assert financepy_report["tasks"][0]["transition"] == "improved"
    assert financepy_report["tasks"][0]["latest"]["knowledge_revision"] == "know2222"
    assert negative_report["latest_pass_count"] == 1
    assert negative_report["tasks"][0]["latest"]["result_label"] == "passed_expectation"

    rendered = render_benchmark_history_scorecard(financepy_report)
    assert "Improved" in rendered

    artifacts = save_benchmark_history_scorecard(
        financepy_report,
        reports_root=tmp_path,
        stem="financepy_daily_scorecard",
    )
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()


def test_build_benchmark_history_scorecard_tracks_regression():
    records = [
        {
            "task_id": "F002",
            "title": "Barrier",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_1",
            "run_started_at": "2026-04-14T01:00:00+00:00",
            "run_completed_at": "2026-04-14T01:00:04+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "git_sha": "aaaa1111",
            "knowledge_revision": "know1111",
            "comparison_summary": {"status": "passed"},
            "cold_agent_elapsed_seconds": 4.0,
            "cold_agent_token_usage": {"total_tokens": 180},
        },
        {
            "task_id": "F002",
            "title": "Barrier",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_2",
            "run_started_at": "2026-04-15T01:00:00+00:00",
            "run_completed_at": "2026-04-15T01:00:06+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "git_sha": "bbbb2222",
            "knowledge_revision": "know2222",
            "comparison_summary": {"status": "failed"},
            "cold_agent_elapsed_seconds": 6.0,
            "cold_agent_token_usage": {"total_tokens": 220},
        },
    ]

    report = build_benchmark_history_scorecard(
        scorecard_name="regression_scorecard",
        benchmark_kind="financepy",
        benchmark_runs=records,
        campaign_id="daily_suite",
    )

    assert report["regressed_count"] == 1
    assert report["improved_count"] == 0
    assert report["latest_pass_count"] == 0
    assert report["tasks"][0]["transition"] == "regressed"
    assert report["tasks"][0]["elapsed_seconds_delta"] == pytest.approx(2.0)
    assert report["tasks"][0]["token_total_delta"] == 40


def test_benchmark_history_scorecard_reports_agent_cycle_behavior():
    def cycle_report(*, success: bool = True, model_validator: str = "skipped"):
        return {
            "request_id": "executor_benchmark_cycle",
            "status": "succeeded" if success else "failed",
            "outcome": "build_completed" if success else "request_failed",
            "success": success,
            "pricing_method": "analytical",
            "validation_contract_id": "validation:vanilla_option:analytical",
            "stage_statuses": {
                "quant": "passed",
                "validation_bundle": "passed",
                "critic": "passed",
                "arbiter": "passed",
                "model_validator": model_validator,
            },
            "failure_count": 0 if success else 1,
            "deterministic_blockers": (
                [] if success else [{"source": "arbiter", "check_id": "price_bound"}]
            ),
            "conceptual_blockers": [],
            "calibration_blockers": [],
            "residual_limitations": [{"risk_id": "model_validator:advisory"}],
            "residual_risks": ["model_validator:advisory"],
        }

    records = [
        {
            "task_id": "F001",
            "title": "Vanilla",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_1",
            "run_started_at": "2026-04-14T01:00:00+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "comparison_summary": {"status": "passed"},
            "cold_agent_cycle_report": cycle_report(success=True),
        },
        {
            "task_id": "F002",
            "title": "Barrier",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_2",
            "run_started_at": "2026-04-14T01:01:00+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "comparison_summary": {"status": "failed"},
            "cold_agent_cycle_report": cycle_report(
                success=False,
                model_validator="failed",
            ),
        },
        {
            "task_id": "F003",
            "title": "Missing cycle",
            "task_corpus": "benchmark_financepy",
            "run_id": "run_3",
            "run_started_at": "2026-04-14T01:02:00+00:00",
            "execution_mode": "cold_agent_plus_financepy_reference",
            "comparison_summary": {"status": "failed"},
        },
    ]

    report = build_benchmark_history_scorecard(
        scorecard_name="cycle_scorecard",
        benchmark_kind="financepy",
        benchmark_runs=records,
    )

    assert report["agent_cycle"]["available_count"] == 2
    assert report["agent_cycle"]["not_available_count"] == 1
    assert report["agent_cycle"]["passed_count"] == 1
    assert report["agent_cycle"]["failed_count"] == 1
    assert report["agent_cycle"]["stage_trigger_rates"]["model_validator"]["triggered_count"] == 2
    assert report["agent_cycle"]["blocker_counts"]["deterministic_blockers"] == 1
    assert report["tasks"][0]["latest"]["agent_cycle"]["status"] == "passed"

    rendered = render_benchmark_history_scorecard(report)
    assert "Agent Cycle Behavior" in rendered
    assert "Model validator trigger rate" in rendered


def test_benchmark_history_exposes_public_helpers():
    """The history helpers used by per-corpus scorecards are now public.

    PR #590 round-3 Copilot review: `pilot_parity_scorecard` was reaching
    into private `_history_sort_key` / `_build_task_history_summary`.
    Both names are now public; the underscored aliases stay for callers
    that imported the private names.
    """
    from trellis.agent import benchmark_history as mod

    assert hasattr(mod, "history_sort_key")
    assert hasattr(mod, "build_task_history_summary")
    # Aliases still point at the same callables so older callers don't break.
    assert mod._history_sort_key is mod.history_sort_key
    assert mod._build_task_history_summary is mod.build_task_history_summary
