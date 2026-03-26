"""Persistence helpers for rich per-task run records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
TASK_RUN_ROOT = ROOT / "task_runs"
TASK_RUN_HISTORY_ROOT = TASK_RUN_ROOT / "history"
TASK_RUN_LATEST_ROOT = TASK_RUN_ROOT / "latest"
TASK_RUN_LATEST_INDEX = ROOT / "task_results_latest.json"


def persist_task_run_record(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    root: Path = ROOT,
    persisted_at: datetime | None = None,
) -> dict[str, str]:
    """Persist one rich task-run record to history and latest locations."""
    persisted_at = persisted_at or datetime.now(timezone.utc)
    record = build_task_run_record(task, result, persisted_at=persisted_at)

    history_root = root / "task_runs" / "history" / str(task["id"])
    latest_root = root / "task_runs" / "latest"
    latest_index_path = root / "task_results_latest.json"
    history_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    history_path = history_root / f"{record['run_id']}.json"
    latest_path = latest_root / f"{task['id']}.json"

    history_path.write_text(json.dumps(record, indent=2, default=str))
    latest_path.write_text(json.dumps(record, indent=2, default=str))

    latest_index = _load_json_mapping(latest_index_path)
    latest_index[str(task["id"])] = record
    latest_index_path.write_text(json.dumps(latest_index, indent=2, default=str))

    return {
        "history_path": str(history_path),
        "latest_path": str(latest_path),
        "latest_index_path": str(latest_index_path),
    }


def load_latest_task_run(task_id: str, *, root: Path = ROOT) -> dict[str, Any] | None:
    """Load one latest task-run record by task identifier."""
    index = _load_json_mapping(root / "task_results_latest.json")
    record = index.get(str(task_id))
    if not isinstance(record, dict):
        return None
    return _attach_storage_paths(record, task_id=task_id, root=root)


def load_latest_task_run_records(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Load all latest task-run records, optionally filtered by task kind."""
    index = _load_json_mapping(root / "task_results_latest.json")
    records: list[dict[str, Any]] = []
    for task_id, record in index.items():
        if not isinstance(record, dict):
            continue
        normalized = _attach_storage_paths(record, task_id=str(task_id), root=root)
        if task_kind is not None and normalized.get("task_kind") != task_kind:
            continue
        records.append(normalized)
    return sorted(records, key=lambda item: str(item.get("task_id") or ""))


def build_task_run_record(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    persisted_at: datetime,
) -> dict[str, Any]:
    """Build the canonical persisted task-run record."""
    task_kind = infer_task_kind(task, result)
    traces = _collect_trace_summaries(result)
    method_runs = _build_method_runs(result)
    issue_refs = _aggregate_issue_refs(traces)
    run_id = _run_id(result, persisted_at)
    framework = dict(result.get("framework_result") or {})
    learning = _learning_summary(result, task_kind=task_kind)
    if task_kind == "framework":
        issue_refs = _merge_issue_refs(issue_refs, framework.get("related_issue_refs") or {})
    workflow = _workflow_summary(
        result,
        traces,
        method_runs,
        task_kind=task_kind,
        issue_refs=issue_refs,
    )

    return {
        "task_id": task["id"],
        "task_kind": task_kind,
        "run_id": run_id,
        "persisted_at": persisted_at.astimezone(timezone.utc).isoformat(),
        "task": _task_snapshot(task),
        "result": result,
        "comparison": {
            "task": bool(result.get("comparison_task")),
            "targets": list(result.get("comparison_targets") or []),
            "summary": dict(result.get("cross_validation") or {}),
        },
        "market": dict(result.get("market_context") or {}),
        "framework": framework,
        "method_runs": method_runs,
        "token_usage": {
            "task": dict(result.get("token_usage_summary") or {}),
            "methods": {
                method: dict((payload.get("token_usage_summary") or {}))
                for method, payload in method_runs.items()
                if payload.get("token_usage_summary")
            },
        },
        "learning": learning,
        "artifacts": dict(result.get("artifacts") or {}),
        "trace_summaries": traces,
        "issue_refs": issue_refs,
        "workflow": workflow,
        "summary": {
            "success": bool(result.get("success")),
            "status": workflow["status"],
            "task_kind": task_kind,
            "preferred_method": result.get("preferred_method"),
            "payoff_class": result.get("payoff_class"),
            "error": result.get("error"),
            "failures": list(result.get("failures") or []),
            "comparison_status": (result.get("cross_validation") or {}).get("status"),
            "reference_target": (result.get("cross_validation") or {}).get("reference_target"),
            "prices": dict((result.get("cross_validation") or {}).get("prices") or {}),
            "deviations_pct": dict((result.get("cross_validation") or {}).get("deviations_pct") or {}),
            "token_usage": dict(result.get("token_usage_summary") or {}),
            "framework_outcome": framework.get("outcome_type"),
            "learning": learning,
        },
    }


def load_task_run_record(path: str | Path) -> dict[str, Any]:
    """Load one persisted task-run record."""
    return json.loads(Path(path).read_text())


def _task_snapshot(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "title",
        "status",
        "construct",
        "cross_validate",
        "new_component",
        "market",
        "market_assertions",
        "blocked_by",
        "trigger_after",
    )
    return {key: task.get(key) for key in keys if key in task}


def _run_id(result: dict[str, Any], persisted_at: datetime) -> str:
    start_time = str(result.get("start_time") or "").strip()
    if start_time:
        return (
            start_time.replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("+00:00", "z")
            .replace("+", "_")
        )
    return persisted_at.strftime("%Y%m%dT%H%M%S%fZ")


def _build_method_runs(result: dict[str, Any]) -> dict[str, Any]:
    method_results = result.get("method_results") or {}
    if not isinstance(method_results, dict):
        return {}

    enriched: dict[str, Any] = {}
    for target, payload in method_results.items():
        trace = _trace_summary(payload.get("platform_trace_path"))
        trace_token_usage = trace.get("token_usage") if isinstance(trace, dict) else {}
        enriched[target] = {
            **payload,
            "token_usage": dict(payload.get("token_usage_summary") or trace_token_usage or {}),
            "trace_summary": trace,
            "issue_refs": _issue_refs_from_trace(trace),
        }
    return enriched


def _collect_trace_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    paths = []
    artifacts = result.get("artifacts") or {}
    paths.extend(artifacts.get("platform_trace_paths") or [])
    if result.get("platform_trace_path"):
        paths.append(result["platform_trace_path"])

    seen: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for path in paths:
        if not isinstance(path, str) or not path or path in seen:
            continue
        seen.add(path)
        summary = _trace_summary(path)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _trace_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    trace_path = Path(path)
    if not trace_path.exists():
        return {
            "path": path,
            "exists": False,
        }

    data = yaml.safe_load(trace_path.read_text()) or {}
    events = list(data.get("events") or [])
    latest_event = events[-1] if events else {}
    linear = data.get("linear_issue") or {}
    github = data.get("github_issue") or {}
    metadata = data.get("request_metadata") or {}
    return {
        "path": str(trace_path),
        "exists": True,
        "request_id": data.get("request_id"),
        "status": data.get("status"),
        "outcome": data.get("outcome"),
        "action": data.get("action"),
        "route_method": data.get("route_method"),
        "updated_at": data.get("updated_at"),
        "latest_event": latest_event.get("event"),
        "latest_event_status": latest_event.get("status"),
        "latest_event_details": latest_event.get("details") or {},
        "token_usage": data.get("token_usage") or {},
        "request_metadata": metadata,
        "linear_issue": linear if linear else None,
        "github_issue": github if github else None,
    }


def _issue_refs_from_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    if not trace:
        return {}
    refs: dict[str, Any] = {}
    if trace.get("linear_issue"):
        refs["linear"] = trace["linear_issue"]
    if trace.get("github_issue"):
        refs["github"] = trace["github_issue"]
    return refs


def _aggregate_issue_refs(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    linear: list[dict[str, Any]] = []
    github: list[dict[str, Any]] = []
    seen_linear: set[str] = set()
    seen_github: set[str] = set()
    for trace in traces:
        issue = trace.get("linear_issue") or {}
        issue_id = issue.get("id")
        if issue_id and issue_id not in seen_linear:
            seen_linear.add(issue_id)
            linear.append(issue)
        issue = trace.get("github_issue") or {}
        issue_number = issue.get("number")
        if issue_number and str(issue_number) not in seen_github:
            seen_github.add(str(issue_number))
            github.append(issue)
    return {"linear": linear, "github": github}


def _workflow_summary(
    result: dict[str, Any],
    traces: list[dict[str, Any]],
    method_runs: dict[str, Any],
    *,
    task_kind: str,
    issue_refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if task_kind == "framework":
        return _framework_workflow_summary(result, traces, issue_refs=issue_refs)

    comparison = result.get("cross_validation") or {}
    task_is_running = result.get("status") == "running" or bool(result.get("live"))
    running_traces = [trace for trace in traces if trace.get("status") == "running"]
    blocker_details = result.get("blocker_details") or {}
    blockers = (blocker_details.get("blocker_report") or {}).get("blockers") or []
    new_primitive_items = (blocker_details.get("new_primitive_workflow") or {}).get("items") or []

    if task_is_running:
        status = "running"
        next_action = "Agent build or validation is still in progress."
    elif blockers:
        status = "blocked"
        next_action = "Blocked on missing or unsupported primitives. See blocker report and linked issues."
    elif new_primitive_items:
        status = "needs_library_work"
        next_action = "A new primitive workflow has been defined; implementation work is required."
    elif not result.get("success"):
        status = "failed"
        next_action = (
            "Tracked externally via linked issues."
            if issue_refs["linear"] or issue_refs["github"]
            else "No automated follow-up is active yet; review failures and traces."
        )
    else:
        status = "succeeded"
        if comparison:
            next_action = "Review comparison prices and deviations for consistency across methods."
        else:
            next_action = "Completed successfully."

    latest_trace = _latest_trace(traces)
    return {
        "status": status,
        "next_action": next_action,
        "latest_trace": latest_trace,
        "active_trace_count": len(running_traces),
        "linked_issues": issue_refs,
        "comparison_status": comparison.get("status"),
        "method_count": len(method_runs),
    }


def _framework_workflow_summary(
    result: dict[str, Any],
    traces: list[dict[str, Any]],
    *,
    issue_refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build workflow status for framework/meta task runs."""
    task_is_running = result.get("status") == "running" or bool(result.get("live"))
    running_traces = [trace for trace in traces if trace.get("status") == "running"]
    framework = result.get("framework_result") or {}
    outcome_type = str(framework.get("outcome_type") or "").strip().lower()

    if task_is_running:
        status = "running"
        next_action = "Framework review or extraction analysis is still in progress."
    elif outcome_type in {"does_not_yet_apply", "blocked"}:
        status = "blocked"
        next_action = framework.get("next_action") or (
            "Gather the missing trigger evidence or narrow the framework task scope."
        )
    elif result.get("success"):
        status = "proposed"
        next_action = framework.get("next_action") or (
            "Review the generated framework candidate and decide whether to promote or track follow-up work."
        )
    else:
        status = "failed"
        next_action = (
            "Tracked externally via linked issues."
            if issue_refs["linear"] or issue_refs["github"]
            else "Review the framework task summary and supporting evidence."
        )

    latest_trace = _latest_trace(traces)
    return {
        "status": status,
        "next_action": next_action,
        "latest_trace": latest_trace,
        "active_trace_count": len(running_traces),
        "linked_issues": issue_refs,
        "comparison_status": None,
        "method_count": 0,
    }


def _learning_summary(
    result: dict[str, Any],
    *,
    task_kind: str,
) -> dict[str, Any]:
    """Summarize reusable knowledge artifacts left behind by a task run."""
    reflection = result.get("reflection") or {}
    knowledge_summary = result.get("knowledge_summary") or {}
    artifacts = result.get("artifacts") or {}
    lesson_captured = reflection.get("lesson_captured")
    if isinstance(lesson_captured, str):
        captured_ids = [lesson_captured]
    elif isinstance(lesson_captured, list):
        captured_ids = [item for item in lesson_captured if isinstance(item, str)]
    else:
        captured_ids = []
    lesson_ids = list(knowledge_summary.get("lesson_ids") or [])
    lesson_titles = list(knowledge_summary.get("lesson_titles") or [])
    cookbook_paths = list(artifacts.get("cookbook_candidate_paths") or [])
    knowledge_trace_paths = list(artifacts.get("knowledge_trace_paths") or [])
    knowledge_gap_paths = list(artifacts.get("knowledge_gap_log_paths") or [])

    return {
        "task_kind": task_kind,
        "retrieved_lesson_ids": lesson_ids,
        "retrieved_lesson_titles": lesson_titles,
        "captured_lesson_ids": captured_ids,
        "lessons_attributed": int(reflection.get("lessons_attributed") or 0),
        "cookbook_enriched": bool(reflection.get("cookbook_enriched")),
        "cookbook_candidate_paths": cookbook_paths,
        "knowledge_trace_paths": knowledge_trace_paths,
        "knowledge_gap_log_paths": knowledge_gap_paths,
        "reusable_artifact_count": (
            len(captured_ids)
            + len(cookbook_paths)
            + len(knowledge_trace_paths)
        ),
    }


def _latest_trace(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    dated = [trace for trace in traces if trace.get("updated_at")]
    if not dated:
        return traces[-1] if traces else None
    return max(dated, key=lambda trace: str(trace.get("updated_at")))


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def infer_task_kind(
    task: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None = None,
) -> str:
    """Infer whether a persisted task run belongs to the pricing or framework flow."""
    if result and str(result.get("task_kind") or "").strip():
        return str(result["task_kind"]).strip()
    if task:
        construct = task.get("construct")
        if isinstance(construct, str) and construct.strip().lower() in {
            "framework",
            "infrastructure",
            "experience",
        }:
            return "framework"
    return "pricing"


def _attach_storage_paths(record: dict[str, Any], *, task_id: str, root: Path) -> dict[str, Any]:
    """Attach canonical storage paths to one loaded latest record."""
    normalized = dict(record)
    normalized.setdefault("task_kind", infer_task_kind(normalized.get("task"), normalized.get("result")))
    storage = dict(normalized.get("storage") or {})
    storage.setdefault("latest_path", str(root / "task_runs" / "latest" / f"{task_id}.json"))
    normalized["storage"] = storage
    return normalized


def _merge_issue_refs(
    base: dict[str, list[dict[str, Any]]],
    extra: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Merge linked issue refs from traces and framework-evidence summaries."""
    merged = {
        "linear": list(base.get("linear") or []),
        "github": list(base.get("github") or []),
    }
    seen_linear = {
        str(issue.get("identifier") or issue.get("id") or "").strip()
        for issue in merged["linear"]
        if str(issue.get("identifier") or issue.get("id") or "").strip()
    }
    seen_github = {
        str(issue.get("number") or issue.get("id") or "").strip()
        for issue in merged["github"]
        if str(issue.get("number") or issue.get("id") or "").strip()
    }
    for issue in extra.get("linear") or []:
        identifier = str(issue.get("identifier") or issue.get("id") or "").strip()
        if identifier and identifier not in seen_linear:
            seen_linear.add(identifier)
            merged["linear"].append(dict(issue))
    for issue in extra.get("github") or []:
        number = str(issue.get("number") or issue.get("id") or "").strip()
        if number and number not in seen_github:
            seen_github.add(number)
            merged["github"].append(dict(issue))
    return merged
