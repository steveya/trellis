from __future__ import annotations


def _task(task_id: str, title: str) -> dict[str, str]:
    return {"id": task_id, "title": title}


def _result(
    *,
    task_id: str,
    success: bool,
    attempts: int,
    failures: list[str] | None = None,
    knowledge_gaps: list[str] | None = None,
    lesson_ids: list[str] | None = None,
    cookbook_enriched: bool = False,
    promotion_candidate_saved: str | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "success": success,
        "attempts": attempts,
        "failures": failures or [],
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
        "token_usage_summary": {"total_tokens": 0},
    }


def test_build_reliability_benchmark_report_and_render(tmp_path):
    from trellis.agent.reliability_benchmark import (
        build_reliability_benchmark_report,
        render_reliability_benchmark_report,
        save_reliability_benchmark_report,
    )

    tasks = [
        _task("T97", "Digital (cash-or-nothing) option: BS formula vs MC vs COS"),
        _task("E25", "FX option (EURUSD): GK analytical vs MC"),
    ]
    baseline = [
        _result(
            task_id="T97",
            success=False,
            attempts=2,
            failures=["validation failed"],
            knowledge_gaps=["missing_cookbook"],
        ),
        _result(task_id="E25", success=True, attempts=1, lesson_ids=["lesson-1"]),
    ]
    candidate = [
        _result(
            task_id="T97",
            success=True,
            attempts=1,
            lesson_ids=["lesson-2"],
            cookbook_enriched=True,
            promotion_candidate_saved="/tmp/promotion.yaml",
        ),
        _result(
            task_id="E25",
            success=True,
            attempts=1,
            lesson_ids=["lesson-1"],
        ),
    ]

    report = build_reliability_benchmark_report(
        benchmark_name="analytical_support_reliability",
        tasks=tasks,
        baseline_results=baseline,
        candidate_results=candidate,
        notes=["fresh-build tranche"],
    )

    assert report["comparison"]["task_transitions"]["improved"] == 1
    assert report["comparison"]["task_transitions"]["regressed"] == 0
    assert report["knowledge_capture"]["tasks_with_lessons"] == ["E25", "T97"]
    assert report["knowledge_capture"]["tasks_with_cookbooks"] == ["T97"]
    assert report["knowledge_capture"]["tasks_with_promotion_candidates"] == ["T97"]
    assert report["knowledge_capture"]["follow_up_suggestions"]

    rendered = render_reliability_benchmark_report(report)
    assert "Reliability Benchmark" in rendered
    assert "T97" in rendered
    assert "Shared Knowledge" in rendered

    artifacts = save_reliability_benchmark_report(report, root=tmp_path, stem="analytical_support_reliability")
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    saved = artifacts.report
    assert "json_path" not in saved
    assert "text_path" not in saved
    assert str(tmp_path) not in artifacts.json_path.read_text()
