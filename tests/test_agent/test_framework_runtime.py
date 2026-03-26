from __future__ import annotations

from datetime import datetime, timezone


def _pricing_task(task_id: str, *, title: str, new_component: str | list[str] | None = None) -> dict:
    task = {
        "id": task_id,
        "title": title,
        "construct": "analytical",
    }
    if new_component is not None:
        task["new_component"] = new_component
    return task


def _pricing_result(success: bool = True) -> dict:
    return {
        "task_id": "unused",
        "title": "unused",
        "success": success,
        "attempts": 1,
        "reflection": {},
        "knowledge_summary": {"lesson_ids": ["num_001"]},
    }


def test_run_framework_task_returns_extraction_candidate_when_triggers_ready(tmp_path):
    from trellis.agent.framework_runtime import run_framework_task
    from trellis.agent.task_run_store import persist_task_run_record

    persist_task_run_record(
        _pricing_task("T108", title="FX vanilla option: Garman-Kohlhagen vs MC", new_component="garman_kohlhagen_formula"),
        {
            **_pricing_result(True),
            "task_id": "T108",
            "title": "FX vanilla option: Garman-Kohlhagen vs MC",
        },
        root=tmp_path,
        persisted_at=datetime(2026, 3, 26, 18, 0, tzinfo=timezone.utc),
    )
    persist_task_run_record(
        _pricing_task("T109", title="FX barrier option", new_component="fx_barrier_adapter"),
        {
            **_pricing_result(False),
            "task_id": "T109",
            "title": "FX barrier option",
            "failures": ["unsupported route"],
        },
        root=tmp_path,
        persisted_at=datetime(2026, 3, 26, 18, 5, tzinfo=timezone.utc),
    )

    result = run_framework_task(
        {
            "id": "E17",
            "title": "Extract: FX pricing framework (GK base, barrier, digital, quanto adjustments)",
            "construct": "framework",
            "trigger_after": ["T108", "T109"],
        },
        root=tmp_path,
    )

    assert result["success"] is True
    assert result["framework_result"]["outcome_type"] == "extraction_candidate"
    assert result["framework_result"]["candidate_name"].startswith("FX pricing framework")
    assert result["framework_result"]["related_task_ids"] == ["T108", "T109"]
    assert "garman_kohlhagen_formula" in result["framework_result"]["related_components"]


def test_run_framework_task_blocks_when_trigger_evidence_is_missing(tmp_path):
    from trellis.agent.framework_runtime import run_framework_task

    result = run_framework_task(
        {
            "id": "E11",
            "title": "Extract: exercise strategy protocol (American, Bermudan, autocall)",
            "construct": "framework",
            "trigger_after": ["T05", "T36", "T80"],
        },
        root=tmp_path,
    )

    assert result["success"] is False
    assert result["framework_result"]["outcome_type"] == "does_not_yet_apply"
    assert result["framework_result"]["missing_triggers"] == ["T05", "T36", "T80"]
    assert "missing trigger tasks" in result["framework_result"]["next_action"].lower()


def test_run_framework_task_supports_every_10_tasks_trigger(tmp_path):
    from trellis.agent.framework_runtime import run_framework_task
    from trellis.agent.task_run_store import persist_task_run_record

    for index in range(10):
        task_id = f"T{200 + index}"
        persist_task_run_record(
            _pricing_task(task_id, title=f"Task {task_id}"),
            {
                **_pricing_result(True),
                "task_id": task_id,
                "title": f"Task {task_id}",
            },
            root=tmp_path,
            persisted_at=datetime(2026, 3, 26, 19, index, tzinfo=timezone.utc),
        )

    result = run_framework_task(
        {
            "id": "E15",
            "title": "Consolidate LIMITATIONS.md: resolve completed, add new",
            "construct": "infrastructure",
            "trigger_after": "every_10_tasks",
        },
        root=tmp_path,
    )

    assert result["success"] is True
    assert result["framework_result"]["outcome_type"] == "infrastructure_review"
    assert result["framework_result"]["trigger_state"]["ready"] is True
