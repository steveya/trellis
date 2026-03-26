"""Dedicated runner for framework/meta tasks in ``FRAMEWORK_TASKS.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Mapping

from trellis.agent.task_run_store import (
    ROOT,
    load_latest_task_run_records,
    persist_task_run_record,
)


_FRAMEWORK_CONSTRUCTS = {"framework", "infrastructure", "experience"}
_LESSON_ENTRIES_ROOT = ROOT / "trellis" / "agent" / "knowledge" / "lessons" / "entries"


@dataclass(frozen=True)
class FrameworkTriggerState:
    """Deterministic readiness assessment for one framework task."""

    ready: bool
    reason: str
    missing_triggers: tuple[str, ...] = ()
    satisfied_triggers: tuple[str, ...] = ()
    trigger_value: str | None = None


def run_framework_task(
    task: dict[str, Any],
    *,
    root: Path = ROOT,
    timer=time,
    now_fn=datetime.now,
    latest_pricing_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one framework/meta task against persisted pricing-run evidence."""
    construct = str(task.get("construct") or "").strip().lower()
    if construct not in _FRAMEWORK_CONSTRUCTS:
        raise ValueError(f"Task {task.get('id', '<unknown>')} is not a framework/meta task")

    latest_pricing_runs = (
        list(latest_pricing_runs)
        if latest_pricing_runs is not None
        else load_latest_task_run_records(root=root, task_kind="pricing")
    )
    pricing_by_id = {
        str(record.get("task_id")): record
        for record in latest_pricing_runs
        if record.get("task_id")
    }

    t0 = timer()
    trigger_state = evaluate_framework_trigger(task, pricing_by_id=pricing_by_id)
    related_records = [
        pricing_by_id[task_id]
        for task_id in trigger_state.satisfied_triggers
        if task_id in pricing_by_id
    ]
    framework_result = build_framework_result(task, trigger_state, related_records)
    elapsed = timer() - t0
    result = {
        "task_id": task["id"],
        "task_kind": "framework",
        "title": task["title"],
        "success": framework_result["outcome_type"] not in {"does_not_yet_apply", "blocked"},
        "start_time": now_fn().isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "attempts": 1,
        "failures": list(framework_result.get("failures") or []),
        "framework_result": framework_result,
        "artifacts": {
            "related_task_latest_paths": [
                str((root / "task_runs" / "latest" / f"{record['task_id']}.json"))
                for record in related_records
            ],
        },
        "token_usage_summary": {},
        "reflection": {},
        "knowledge_summary": {},
    }
    persisted = persist_task_run_record(task, result, root=root)
    result["task_run_history_path"] = persisted["history_path"]
    result["task_run_latest_path"] = persisted["latest_path"]
    result["task_run_latest_index_path"] = persisted["latest_index_path"]
    return result


def evaluate_framework_trigger(
    task: Mapping[str, Any],
    *,
    pricing_by_id: Mapping[str, Mapping[str, Any]],
) -> FrameworkTriggerState:
    """Check whether a framework task has enough prior evidence to run."""
    trigger = task.get("trigger_after")
    if not trigger:
        return FrameworkTriggerState(True, "No explicit trigger_after requirement.")

    if isinstance(trigger, (list, tuple)):
        trigger_ids = tuple(str(item) for item in trigger if str(item).strip())
        satisfied = tuple(task_id for task_id in trigger_ids if task_id in pricing_by_id)
        missing = tuple(task_id for task_id in trigger_ids if task_id not in pricing_by_id)
        if missing:
            return FrameworkTriggerState(
                False,
                "Not enough source task evidence is available yet.",
                missing_triggers=missing,
                satisfied_triggers=satisfied,
            )
        return FrameworkTriggerState(
            True,
            "All listed trigger tasks have persisted pricing-run evidence.",
            satisfied_triggers=satisfied,
        )

    trigger_value = str(trigger).strip().lower()
    if trigger_value == "every_10_tasks":
        count = len(pricing_by_id)
        ready = count >= 10
        return FrameworkTriggerState(
            ready,
            f"Observed {count} pricing-task latest runs; framework review expects at least 10.",
            trigger_value=trigger_value,
        )
    if trigger_value == "every_10_entries":
        count = _lesson_entry_count()
        ready = count >= 10
        return FrameworkTriggerState(
            ready,
            f"Observed {count} lesson entries; experience consolidation expects at least 10.",
            trigger_value=trigger_value,
        )

    return FrameworkTriggerState(
        False,
        f"Unsupported framework trigger_after policy: {trigger!r}",
        trigger_value=trigger_value,
    )


def build_framework_result(
    task: Mapping[str, Any],
    trigger_state: FrameworkTriggerState,
    related_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the structured framework/meta result payload."""
    construct = str(task.get("construct") or "").strip().lower()
    candidate_type = _candidate_type(task)
    evidence = _summarize_related_records(related_records)

    if not trigger_state.ready:
        return {
            "outcome_type": "does_not_yet_apply",
            "summary": trigger_state.reason,
            "next_action": "Wait for the missing trigger tasks or rerun the prerequisite tranche first.",
            "trigger_state": {
                "ready": trigger_state.ready,
                "reason": trigger_state.reason,
                "missing_triggers": list(trigger_state.missing_triggers),
                "satisfied_triggers": list(trigger_state.satisfied_triggers),
                "trigger_value": trigger_state.trigger_value,
            },
            "related_task_ids": list(trigger_state.satisfied_triggers),
            "missing_triggers": list(trigger_state.missing_triggers),
            "failures": [trigger_state.reason],
            "related_issue_refs": {"linear": [], "github": []},
        }

    issue_refs = _aggregate_related_issue_refs(related_records)
    candidate_name = _candidate_name(task)
    if construct == "experience":
        outcome_type = "consolidation_candidate"
        summary = f"Experience consolidation candidate `{candidate_name}` is ready. {trigger_state.reason}"
        if evidence["task_count"] or evidence["lesson_count"]:
            summary += (
                f" Supporting evidence includes {evidence['task_count']} related runs and "
                f"{evidence['lesson_count']} retrieved lessons."
            )
        next_action = "Review the distilled experience summary and decide what should become a stable principle."
    elif construct == "infrastructure":
        outcome_type = "infrastructure_review"
        summary = f"Infrastructure review `{candidate_name}` is ready. {trigger_state.reason}"
        if evidence["task_count"] or evidence["failure_buckets"]:
            summary += (
                f" Related supporting runs: {evidence['task_count']}; "
                f"failure buckets: {evidence['failure_buckets']}."
            )
        next_action = "Review the infra findings and decide whether to open or update explicit maintenance work."
    else:
        outcome_type = "extraction_candidate"
        summary = (
            f"Extraction candidate `{candidate_name}` is supported by "
            f"{evidence['task_count']} related pricing runs across tasks {evidence['task_ids']}."
        )
        next_action = "Review the candidate scope, related components, and linked issues before promoting it to library work."

    return {
        "outcome_type": outcome_type,
        "candidate_type": candidate_type,
        "candidate_name": candidate_name,
        "summary": summary,
        "next_action": next_action,
        "trigger_state": {
            "ready": trigger_state.ready,
            "reason": trigger_state.reason,
            "missing_triggers": [],
            "satisfied_triggers": list(trigger_state.satisfied_triggers),
            "trigger_value": trigger_state.trigger_value,
        },
        "related_task_ids": evidence["task_ids"],
        "related_components": evidence["related_components"],
        "related_issue_refs": issue_refs,
        "evidence": evidence,
        "failures": [],
    }


def _candidate_type(task: Mapping[str, Any]) -> str:
    title = str(task.get("title") or "").strip().lower()
    construct = str(task.get("construct") or "").strip().lower()
    if title.startswith("extract:"):
        return "extract"
    if title.startswith("consolidate"):
        return "consolidate"
    if construct == "infrastructure":
        return "review"
    return construct or "framework"


def _candidate_name(task: Mapping[str, Any]) -> str:
    title = str(task.get("title") or "").strip()
    if ":" in title:
        return title.split(":", 1)[1].strip()
    return title


def _summarize_related_records(related_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    task_ids = [str(record.get("task_id")) for record in related_records if record.get("task_id")]
    related_components: list[str] = []
    failure_buckets: dict[str, int] = {}
    lesson_count = 0
    for record in related_records:
        task = record.get("task") or {}
        new_component = task.get("new_component")
        if isinstance(new_component, str):
            related_components.append(new_component)
        elif isinstance(new_component, list):
            related_components.extend(str(item) for item in new_component if str(item).strip())
        bucket = str((record.get("summary") or {}).get("status") or "")
        if bucket:
            failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1
        lesson_count += len(((record.get("learning") or {}).get("retrieved_lesson_ids") or []))

    return {
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "related_components": sorted({item for item in related_components if item}),
        "failure_buckets": failure_buckets,
        "lesson_count": lesson_count,
    }


def _aggregate_related_issue_refs(
    related_records: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    linear: list[dict[str, Any]] = []
    github: list[dict[str, Any]] = []
    seen_linear: set[str] = set()
    seen_github: set[str] = set()
    for record in related_records:
        issue_refs = record.get("issue_refs") or {}
        for issue in issue_refs.get("linear") or []:
            identifier = str(issue.get("identifier") or issue.get("id") or "").strip()
            if identifier and identifier not in seen_linear:
                seen_linear.add(identifier)
                linear.append(dict(issue))
        for issue in issue_refs.get("github") or []:
            number = str(issue.get("number") or issue.get("id") or "").strip()
            if number and number not in seen_github:
                seen_github.add(number)
                github.append(dict(issue))
    return {"linear": linear, "github": github}


def _lesson_entry_count() -> int:
    if not _LESSON_ENTRIES_ROOT.exists():
        return 0
    return len(list(_LESSON_ENTRIES_ROOT.glob("*.yaml")))
