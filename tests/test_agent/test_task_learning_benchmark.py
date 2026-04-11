from __future__ import annotations


def _task(task_id: str, title: str, *, status: str = "pending") -> dict[str, str]:
    return {"id": task_id, "title": title, "status": status}


def _result(
    *,
    task_id: str,
    success: bool,
    attempts: int,
    elapsed_seconds: float = 0.0,
    error: str | None = None,
    knowledge_gaps: list[str] | None = None,
    lesson_ids: list[str] | None = None,
    cookbook_enriched: bool = False,
    promotion_candidate_saved: str | None = None,
    total_tokens: int = 0,
) -> dict:
    return {
        "task_id": task_id,
        "success": success,
        "attempts": attempts,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
        "knowledge_gaps": knowledge_gaps or [],
        "knowledge_summary": {"lesson_ids": lesson_ids or []},
        "reflection": {
            "lesson_captured": lesson_ids or [],
            "cookbook_enriched": cookbook_enriched,
            "promotion_candidate_saved": promotion_candidate_saved,
        },
        "artifacts": {
            "platform_request_ids": [],
            "platform_trace_paths": [],
            "analytical_trace_paths": [],
            "analytical_trace_text_paths": [],
            "knowledge_trace_paths": [],
            "cookbook_candidate_paths": [],
            "promotion_candidate_paths": [],
            "knowledge_gap_log_paths": [],
        },
        "token_usage_summary": {
            "call_count": 1,
            "calls_with_usage": 1,
            "calls_without_usage": 0,
            "prompt_tokens": max(total_tokens - 10, 0),
            "completion_tokens": min(total_tokens, 10),
            "total_tokens": total_tokens,
            "by_stage": {},
            "by_provider": {},
        },
    }


def test_select_task_learning_cohort_excludes_canaries_and_blocked_tasks_by_default():
    from trellis.agent.task_learning_benchmark import select_task_learning_cohort

    tasks = [
        _task("T01", "Canary lattice task"),
        _task("T13", "Pending non-canary PDE task"),
        _task("T14", "Blocked non-canary task", status="blocked"),
        _task("T15", "Done non-canary task", status="done"),
    ]

    selected = select_task_learning_cohort(
        tasks,
        canary_task_ids={"T01"},
    )

    assert [task["id"] for task in selected] == ["T13"]


def test_select_task_learning_cohort_can_include_done_tasks_and_requested_ids():
    from trellis.agent.task_learning_benchmark import select_task_learning_cohort

    tasks = [
        _task("T13", "Pending task"),
        _task("T15", "Done task", status="done"),
        _task("T16", "Another pending task"),
    ]

    selected = select_task_learning_cohort(
        tasks,
        canary_task_ids=set(),
        allowed_statuses=("pending", "done"),
        requested_ids=("T15", "T16"),
    )

    assert [task["id"] for task in selected] == ["T15", "T16"]


def test_build_task_learning_benchmark_report_tracks_pass_deltas_and_attribution(tmp_path):
    from trellis.agent.task_learning_benchmark import (
        build_task_learning_benchmark_report,
        render_task_learning_benchmark_report,
        save_task_learning_benchmark_report,
    )

    tasks = [
        _task("T13", "European call PDE"),
        _task("T14", "American put"),
        _task("T15", "Barrier call"),
    ]
    pass_one = [
        _result(
            task_id="T13",
            success=False,
            attempts=2,
            elapsed_seconds=11.0,
            error="build failed",
            knowledge_gaps=["missing_cookbook"],
            total_tokens=220,
        ),
        _result(
            task_id="T14",
            success=False,
            attempts=1,
            elapsed_seconds=7.0,
            error="MissingCapabilityError: missing market data discount_curve",
            total_tokens=120,
        ),
        _result(
            task_id="T15",
            success=False,
            attempts=2,
            elapsed_seconds=13.0,
            error="semantic validation failed",
            total_tokens=180,
        ),
    ]
    pass_two = [
        _result(
            task_id="T13",
            success=True,
            attempts=1,
            elapsed_seconds=4.0,
            lesson_ids=["lesson-13"],
            cookbook_enriched=True,
            promotion_candidate_saved="/tmp/promotion.yaml",
            total_tokens=90,
        ),
        _result(
            task_id="T14",
            success=False,
            attempts=1,
            elapsed_seconds=6.0,
            error="MissingCapabilityError: missing market data discount_curve",
            total_tokens=100,
        ),
        _result(
            task_id="T15",
            success=False,
            attempts=2,
            elapsed_seconds=10.0,
            error="semantic validation failed",
            total_tokens=150,
        ),
    ]

    report = build_task_learning_benchmark_report(
        benchmark_name="non_canary_task_learning",
        cohort_name="non_canary_pending",
        git_revision="abc1234",
        tasks=tasks,
        pass_runs=[
            {
                "pass_number": 1,
                "label": "pass_1",
                "results": pass_one,
                "results_path": "/tmp/pass_1.json",
            },
            {
                "pass_number": 2,
                "label": "pass_2",
                "results": pass_two,
                "results_path": "/tmp/pass_2.json",
            },
        ],
        notes=["Fresh-build passes isolate knowledge carry-forward from adapter reuse."],
    )

    assert report["passes"][0]["summary"]["totals"]["successes"] == 0
    assert report["passes"][1]["summary"]["totals"]["successes"] == 1
    assert report["pairwise_comparisons"][0]["comparison"]["task_transitions"]["improved"] == 1
    assert report["pairwise_comparisons"][0]["deltas"]["elapsed_seconds_total"] == -11.0
    assert report["pairwise_comparisons"][0]["deltas"]["token_usage_total"] == -180
    assert report["attribution"]["knowledge_assisted_improvements"]["task_ids"] == ["T13"]
    assert report["attribution"]["residual_market_or_provider_noise"]["task_ids"] == ["T14"]
    assert report["attribution"]["residual_implementation_gaps"]["task_ids"] == ["T15"]

    rendered = render_task_learning_benchmark_report(report)
    assert "Task Learning Benchmark" in rendered
    assert "Knowledge-assisted improvements" in rendered
    assert "Residual implementation gaps" in rendered

    artifacts = save_task_learning_benchmark_report(
        report,
        root=tmp_path,
        stem="non_canary_task_learning",
    )
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()

