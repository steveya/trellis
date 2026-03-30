from __future__ import annotations

from pathlib import Path


def _sample_record(root: Path) -> dict[str, object]:
    storage = {
        "history_path": str(root / "task_runs" / "history" / "T999" / "run.json"),
        "latest_path": str(root / "task_runs" / "latest" / "T999.json"),
        "latest_index_path": str(root / "task_results_latest.json"),
        "diagnosis_history_packet_path": str(
            root / "task_runs" / "diagnostics" / "history" / "T999" / "run.json"
        ),
        "diagnosis_history_dossier_path": str(
            root / "task_runs" / "diagnostics" / "history" / "T999" / "run.md"
        ),
        "diagnosis_latest_packet_path": str(
            root / "task_runs" / "diagnostics" / "latest" / "T999.json"
        ),
        "diagnosis_latest_dossier_path": str(
            root / "task_runs" / "diagnostics" / "latest" / "T999.md"
        ),
    }
    return {
        "task_id": "T999",
        "task_kind": "pricing",
        "run_id": "20260329T120000000000Z",
        "persisted_at": "2026-03-29T12:00:00+00:00",
        "task": {
            "id": "T999",
            "title": "Callable bond stress test",
            "construct": ["tree"],
        },
        "summary": {
            "success": False,
            "status": "failed",
            "comparison_status": "insufficient_results",
            "next_action": "Inspect which methods failed to produce valid results, then rerun the narrowest broken route.",
        },
        "workflow": {
            "status": "failed",
            "next_action": "Inspect which methods failed to produce valid results, then rerun the narrowest broken route.",
            "latest_trace": {"request_id": "executor_build_demo"},
        },
        "comparison": {
            "status": "insufficient_results",
            "reference_target": "black_scholes",
            "prices": {"psor": 1.01, "black_scholes": 1.0},
            "deviations_pct": {"psor": 1.0},
        },
        "learning": {
            "task_kind": "pricing",
            "retrieved_lesson_ids": [],
            "retrieved_lesson_titles": [],
            "captured_lesson_ids": [],
            "lesson_contract_reports": [],
            "lesson_contract_count": 0,
            "lesson_contract_outcome": "not_attempted",
            "lesson_contract_errors": [],
            "lesson_contract_warnings": [],
            "lesson_promotion_outcomes": [],
            "lessons_attributed": 0,
            "cookbook_enriched": False,
            "cookbook_candidate_paths": [],
            "promotion_candidate_paths": [],
            "knowledge_trace_paths": [],
            "knowledge_gap_log_paths": [],
            "reusable_artifact_count": 0,
            "knowledge_outcome": "blocked_without_learning",
            "knowledge_outcome_reason": "task failed before any reusable learning artifact was captured",
        },
        "trace_summaries": [
            {
                "path": str(root / "task_runs" / "platform" / "executor_build_demo.yaml"),
                "exists": True,
                "trace_kind": "platform",
                "request_id": "executor_build_demo",
                "status": "failed",
                "outcome": "request_failed",
                "route_method": "pde",
                "latest_event": "builder_attempt_failed",
                "latest_event_status": "error",
                "updated_at": "2026-03-29T12:00:00+00:00",
            }
        ],
        "method_runs": {
            "psor": {
                "success": False,
                "preferred_method": "pde",
                "attempts": 2,
                "route_method": "pde",
                "platform_trace_path": str(root / "task_runs" / "platform" / "executor_build_demo.yaml"),
                "trace_summary": {
                    "path": str(root / "task_runs" / "platform" / "executor_build_demo_method.yaml"),
                    "exists": True,
                    "trace_kind": "platform",
                    "request_id": "executor_build_demo_method",
                    "status": "failed",
                    "outcome": "request_failed",
                    "route_method": "pde",
                    "latest_event": "builder_attempt_failed",
                    "latest_event_status": "error",
                    "updated_at": "2026-03-29T12:00:00+00:00",
                },
                "cross_validation": {"status": "failed"},
                "token_usage_summary": {"total_tokens": 120},
                "failures": ["name 'absorbing' is not defined"],
                "error": "name 'absorbing' is not defined",
            },
            "black_scholes": {
                "success": True,
                "preferred_method": "analytical",
                "attempts": 1,
                "route_method": "analytical",
                "platform_trace_path": str(root / "task_runs" / "platform" / "executor_build_black_scholes.yaml"),
                "trace_summary": {
                    "path": str(root / "task_runs" / "platform" / "executor_build_black_scholes.yaml"),
                    "exists": True,
                    "trace_kind": "platform",
                    "request_id": "executor_build_black_scholes",
                    "status": "succeeded",
                    "outcome": "build_completed",
                    "route_method": "analytical",
                    "latest_event": "build_completed",
                    "latest_event_status": "ok",
                    "updated_at": "2026-03-29T12:00:00+00:00",
                },
                "cross_validation": {"status": "passed"},
                "token_usage_summary": {"total_tokens": 40},
                "failures": [],
                "error": None,
            },
        },
        "result": {
            "success": False,
            "attempts": 2,
            "elapsed_seconds": 12.3,
            "error": "comparison failed",
            "failures": ["comparison failed"],
            "blocker_details": {},
            "cross_validation": {
                "status": "insufficient_results",
                "reference_target": "black_scholes",
                "prices": {"psor": 1.01, "black_scholes": 1.0},
                "deviations_pct": {"psor": 1.0},
            },
            "method_results": {},
            "knowledge_gaps": ["missing_cookbook"],
            "gap_confidence": 0.6,
            "task_contract_error": {},
            "reflection": {},
            "artifacts": {
                "platform_trace_paths": [
                    str(root / "task_runs" / "platform" / "executor_build_demo.yaml")
                ],
                "analytical_trace_paths": [],
            },
        },
        "storage": storage,
    }


def test_build_task_diagnosis_packet_summarizes_failure(tmp_path):
    from trellis.agent.task_diagnostics import build_task_diagnosis_packet, render_task_diagnosis_dossier

    packet = build_task_diagnosis_packet(_sample_record(tmp_path))

    assert packet["schema_version"] == 1
    assert packet["task"]["id"] == "T999"
    assert packet["outcome"]["failure_bucket"] == "comparison_insufficient_results"
    assert packet["outcome"]["decision_stage"] == "comparison"
    assert packet["primary_failure"]["likely_cause"].startswith(
        "The comparison task did not produce a valid enough method set"
    )
    assert packet["method_outcomes"][0]["method"] == "psor"
    assert packet["trace_index"][0]["scope"] == "task"
    assert packet["trace_index"][1]["scope"] == "method"

    rendered = render_task_diagnosis_dossier(packet)
    assert "## Primary Diagnosis" in rendered
    assert "## Method Outcomes" in rendered
    assert "## Storage" in rendered
    assert "comparison_insufficient_results" in rendered


def test_save_task_diagnosis_artifacts_writes_packet_and_dossier(tmp_path):
    from trellis.agent.task_diagnostics import save_task_diagnosis_artifacts

    artifacts = save_task_diagnosis_artifacts(_sample_record(tmp_path), root=tmp_path)

    assert artifacts.packet_path.exists()
    assert artifacts.dossier_path.exists()
    assert artifacts.latest_packet_path.exists()
    assert artifacts.latest_dossier_path.exists()
    assert artifacts.packet_path.parent.name == "T999"
    assert artifacts.packet["outcome"]["headline"].startswith("Callable bond stress test")
    assert "Task diagnosis" in artifacts.dossier_path.read_text()
