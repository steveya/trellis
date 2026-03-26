from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml


def _write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_persist_task_run_record_writes_latest_and_enriches_traces(tmp_path):
    from trellis.agent.task_run_store import load_task_run_record, persist_task_run_record

    trace_root = tmp_path / "trellis" / "agent" / "knowledge" / "traces" / "platform"
    fft_trace = trace_root / "executor_build_fft.yaml"
    analytical_trace = trace_root / "executor_build_black_scholes.yaml"

    _write_trace(
        fft_trace,
        {
            "request_id": "executor_build_fft",
            "status": "failed",
            "outcome": "request_failed",
            "action": "compile_only",
            "route_method": "fft_pricing",
            "updated_at": "2026-03-25T18:00:00+00:00",
            "request_metadata": {
                "task_id": "T104",
                "task_title": "FFT vs COS comparison",
                "comparison_target": "fft",
            },
            "token_usage": {
                "call_count": 2,
                "calls_with_usage": 2,
                "calls_without_usage": 0,
                "prompt_tokens": 180,
                "completion_tokens": 70,
                "total_tokens": 250,
                "by_stage": {"code_generation": {"total_tokens": 250}},
                "by_provider": {"anthropic": {"total_tokens": 250}},
            },
            "events": [
                {
                    "event": "builder_attempt_failed",
                    "status": "error",
                    "timestamp": "2026-03-25T18:00:00+00:00",
                    "details": {"reason": "validation"},
                }
            ],
            "linear_issue": {
                "id": "lin-1",
                "identifier": "QUA-99",
                "url": "https://linear.app/quant-macro/issue/QUA-99/demo",
            },
            "github_issue": {
                "id": 123,
                "number": 77,
                "url": "https://github.com/steveya/trellis/issues/77",
                "repository": "steveya/trellis",
            },
        },
    )
    _write_trace(
        analytical_trace,
        {
            "request_id": "executor_build_black_scholes",
            "status": "succeeded",
            "outcome": "build_completed",
            "action": "compile_only",
            "route_method": "analytical",
            "updated_at": "2026-03-25T17:59:00+00:00",
            "request_metadata": {
                "task_id": "T104",
                "task_title": "FFT vs COS comparison",
                "comparison_target": "black_scholes",
            },
            "events": [
                {
                    "event": "build_completed",
                    "status": "ok",
                    "timestamp": "2026-03-25T17:59:00+00:00",
                    "details": {"attempts": 1},
                }
            ],
        },
    )

    task = {
        "id": "T104",
        "title": "FFT vs COS comparison",
        "construct": ["transforms"],
        "cross_validate": {"internal": ["fft", "cos"], "analytical": "black_scholes"},
    }
    result = {
        "task_id": "T104",
        "title": "FFT vs COS comparison",
        "success": False,
        "start_time": "2026-03-25T14:05:00",
        "comparison_task": True,
        "comparison_targets": ["fft", "cos", "black_scholes"],
        "token_usage_summary": {
            "call_count": 3,
            "calls_with_usage": 3,
            "calls_without_usage": 0,
            "prompt_tokens": 220,
            "completion_tokens": 90,
            "total_tokens": 310,
            "by_stage": {
                "decomposition": {"total_tokens": 30},
                "code_generation": {"total_tokens": 250},
                "reflection": {"total_tokens": 30},
            },
            "by_provider": {"anthropic": {"total_tokens": 310}},
        },
        "cross_validation": {
            "status": "failed",
            "prices": {"fft": 10.12, "cos": 10.02, "black_scholes": 10.0},
            "deviations_pct": {"fft": 1.2, "cos": 0.2},
            "reference_target": "black_scholes",
        },
        "artifacts": {
            "platform_trace_paths": [str(fft_trace), str(analytical_trace)],
        },
        "method_results": {
            "fft": {
                "success": False,
                "preferred_method": "fft_pricing",
                "attempts": 1,
                "failures": ["validation failed"],
                "platform_trace_path": str(fft_trace),
                "token_usage_summary": {
                    "call_count": 2,
                    "calls_with_usage": 2,
                    "calls_without_usage": 0,
                    "prompt_tokens": 180,
                    "completion_tokens": 70,
                    "total_tokens": 250,
                    "by_stage": {"code_generation": {"total_tokens": 250}},
                    "by_provider": {"anthropic": {"total_tokens": 250}},
                },
                "artifacts": {"platform_trace_paths": [str(fft_trace)]},
                "reflection": {},
            },
            "black_scholes": {
                "success": True,
                "preferred_method": "analytical",
                "attempts": 1,
                "failures": [],
                "platform_trace_path": str(analytical_trace),
                "artifacts": {"platform_trace_paths": [str(analytical_trace)]},
                "reflection": {},
            },
        },
        "failures": ["comparison failed"],
        "reflection": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 25, 18, 1, 0, tzinfo=timezone.utc),
    )

    latest = load_task_run_record(persisted["latest_path"])

    assert Path(persisted["history_path"]).exists()
    assert Path(persisted["latest_path"]).exists()
    assert Path(persisted["latest_index_path"]).exists()
    assert latest["task_id"] == "T104"
    assert latest["summary"]["comparison_status"] == "failed"
    assert latest["summary"]["prices"]["fft"] == 10.12
    assert latest["method_runs"]["fft"]["trace_summary"]["linear_issue"]["identifier"] == "QUA-99"
    assert latest["method_runs"]["fft"]["issue_refs"]["github"]["number"] == 77
    assert latest["method_runs"]["fft"]["token_usage"]["total_tokens"] == 250
    assert latest["trace_summaries"][0]["token_usage"]["total_tokens"] == 250
    assert latest["token_usage"]["task"]["total_tokens"] == 310
    assert latest["workflow"]["status"] == "failed"
    assert latest["workflow"]["linked_issues"]["linear"][0]["identifier"] == "QUA-99"
    assert latest["workflow"]["latest_trace"]["request_id"] == "executor_build_fft"


def test_persist_task_run_record_treats_terminal_result_as_not_running_even_with_running_trace(tmp_path):
    from trellis.agent.task_run_store import persist_task_run_record, load_task_run_record

    trace_root = tmp_path / "trellis" / "agent" / "knowledge" / "traces" / "platform"
    trace_path = trace_root / "executor_build_demo.yaml"
    _write_trace(
        trace_path,
        {
            "request_id": "executor_build_demo",
            "status": "running",
            "outcome": "",
            "action": "compile_only",
            "route_method": "analytical",
            "updated_at": "2026-03-25T18:31:02+00:00",
            "events": [
                {
                    "event": "planner_completed",
                    "status": "ok",
                    "timestamp": "2026-03-25T18:31:02+00:00",
                    "details": {},
                }
            ],
        },
    )

    task = {"id": "T200", "title": "Demo terminal failure", "construct": "analytical"}
    result = {
        "task_id": "T200",
        "title": "Demo terminal failure",
        "success": False,
        "failures": ["model not found"],
        "cross_validation": {"status": "insufficient_results"},
        "artifacts": {"platform_trace_paths": [str(trace_path)]},
        "reflection": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 25, 18, 32, 0, tzinfo=timezone.utc),
    )
    latest = load_task_run_record(persisted["latest_path"])

    assert latest["workflow"]["status"] == "failed"
    assert latest["workflow"]["active_trace_count"] == 1


def test_persist_task_run_record_supports_framework_task_contract(tmp_path):
    from trellis.agent.task_run_store import (
        load_latest_task_run,
        load_latest_task_run_records,
        load_task_run_record,
        persist_task_run_record,
    )

    task = {
        "id": "E17",
        "title": "Extract: FX pricing framework (GK base, barrier, digital, quanto adjustments)",
        "construct": "framework",
        "trigger_after": ["T108", "T109"],
    }
    result = {
        "task_id": "E17",
        "task_kind": "framework",
        "title": task["title"],
        "success": True,
        "framework_result": {
            "outcome_type": "extraction_candidate",
            "candidate_name": "FX pricing framework (GK base, barrier, digital, quanto adjustments)",
            "summary": "ready to review",
            "next_action": "Review the extraction candidate.",
            "related_task_ids": ["T108", "T109"],
            "related_components": ["garman_kohlhagen_formula", "fx_market_bridge"],
            "related_issue_refs": {
                "linear": [{"identifier": "QUA-101", "id": "lin-101"}],
                "github": [{"number": 88, "id": 88}],
            },
        },
        "artifacts": {
            "related_task_latest_paths": [
                str(tmp_path / "task_runs" / "latest" / "T108.json"),
                str(tmp_path / "task_runs" / "latest" / "T109.json"),
            ]
        },
        "reflection": {},
        "knowledge_summary": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 26, 20, 0, tzinfo=timezone.utc),
    )

    latest = load_task_run_record(persisted["latest_path"])
    direct = load_latest_task_run("E17", root=tmp_path)
    records = load_latest_task_run_records(root=tmp_path, task_kind="framework")

    assert latest["task_kind"] == "framework"
    assert latest["framework"]["outcome_type"] == "extraction_candidate"
    assert latest["workflow"]["status"] == "proposed"
    assert latest["issue_refs"]["linear"][0]["identifier"] == "QUA-101"
    assert latest["summary"]["framework_outcome"] == "extraction_candidate"
    assert latest["learning"]["reusable_artifact_count"] == 0
    assert direct["storage"]["latest_path"].endswith("/task_runs/latest/E17.json")
    assert len(records) == 1
    assert records[0]["task_id"] == "E17"
