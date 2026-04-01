"""Save and load detailed run records (inputs, outputs, timing, errors) for each pricing task."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from trellis.agent.task_diagnostics import (
    DIAGNOSIS_HISTORY_ROOT,
    DIAGNOSIS_LATEST_ROOT,
    save_task_diagnosis_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_RUN_ROOT = ROOT / "task_runs"
TASK_RUN_HISTORY_ROOT = TASK_RUN_ROOT / "history"
TASK_RUN_LATEST_ROOT = TASK_RUN_ROOT / "latest"
TASK_RUN_LATEST_INDEX = ROOT / "task_results_latest.json"
_SKIP_TASK_DIAGNOSIS_PERSIST_ENV = "TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST"


def _env_flag(name: str) -> bool:
    """Parse a boolean-like environment flag."""
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def persist_task_run_record(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    root: Path = ROOT,
    persisted_at: datetime | None = None,
) -> dict[str, str]:
    """Write a full task-run record to both the history archive and the latest-result snapshot."""
    persisted_at = persisted_at or datetime.now(timezone.utc)
    record = build_task_run_record(task, result, persisted_at=persisted_at)

    history_root = root / "task_runs" / "history" / str(task["id"])
    latest_root = root / "task_runs" / "latest"
    latest_index_path = root / "task_results_latest.json"
    history_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    history_path = history_root / f"{record['run_id']}.json"
    latest_path = latest_root / f"{task['id']}.json"
    diagnosis_history_packet_path = (
        root / DIAGNOSIS_HISTORY_ROOT.relative_to(ROOT) / str(task["id"]) / f"{record['run_id']}.json"
    )
    diagnosis_history_dossier_path = diagnosis_history_packet_path.with_suffix(".md")
    diagnosis_latest_packet_path = root / DIAGNOSIS_LATEST_ROOT.relative_to(ROOT) / f"{task['id']}.json"
    diagnosis_latest_dossier_path = diagnosis_latest_packet_path.with_suffix(".md")

    record["storage"] = {
        "history_path": str(history_path),
        "latest_path": str(latest_path),
        "latest_index_path": str(latest_index_path),
        "diagnosis_history_packet_path": str(diagnosis_history_packet_path),
        "diagnosis_history_dossier_path": str(diagnosis_history_dossier_path),
        "diagnosis_latest_packet_path": str(diagnosis_latest_packet_path),
        "diagnosis_latest_dossier_path": str(diagnosis_latest_dossier_path),
    }

    history_path.write_text(json.dumps(record, indent=2, default=str))
    latest_path.write_text(json.dumps(record, indent=2, default=str))

    latest_index = _load_json_mapping(latest_index_path)
    latest_index[str(task["id"])] = record
    latest_index_path.write_text(json.dumps(latest_index, indent=2, default=str))

    diagnosis = None
    diagnosis_error: str | None = None
    diagnosis_persist_skipped = ""
    if _env_flag(_SKIP_TASK_DIAGNOSIS_PERSIST_ENV):
        diagnosis_persist_skipped = f"env:{_SKIP_TASK_DIAGNOSIS_PERSIST_ENV}"
    else:
        try:
            diagnosis = save_task_diagnosis_artifacts(record, root=root)
        except Exception as exc:
            diagnosis_error = str(exc)[:200]

    return {
        "history_path": str(history_path),
        "latest_path": str(latest_path),
        "latest_index_path": str(latest_index_path),
        "diagnosis_packet_path": str(
            diagnosis.packet_path if diagnosis is not None else diagnosis_history_packet_path
        ),
        "diagnosis_dossier_path": str(
            diagnosis.dossier_path if diagnosis is not None else diagnosis_history_dossier_path
        ),
        "latest_diagnosis_packet_path": str(
            diagnosis.latest_packet_path if diagnosis is not None else diagnosis_latest_packet_path
        ),
        "latest_diagnosis_dossier_path": str(
            diagnosis.latest_dossier_path if diagnosis is not None else diagnosis_latest_dossier_path
        ),
        "diagnosis_headline": str(
            diagnosis.packet.get("outcome", {}).get("headline") if diagnosis is not None else ""
        ),
        "diagnosis_failure_bucket": str(
            diagnosis.packet.get("outcome", {}).get("failure_bucket") if diagnosis is not None else ""
        ),
        "diagnosis_decision_stage": str(
            diagnosis.packet.get("outcome", {}).get("decision_stage") if diagnosis is not None else ""
        ),
        "diagnosis_next_action": str(
            diagnosis.packet.get("outcome", {}).get("next_action") if diagnosis is not None else ""
        ),
        "diagnosis_persist_error": diagnosis_error or "",
        "diagnosis_persist_skipped": diagnosis_persist_skipped,
    }


def load_latest_task_run(task_id: str, *, root: Path = ROOT) -> dict[str, Any] | None:
    """Load the most recent run record for a single task, or None if it has never been run."""
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
    learning = summarize_task_learning(result, task_kind=task_kind)
    result_with_learning = dict(result)
    result_with_learning["learning"] = learning
    post_build = _post_build_summary(result_with_learning, method_runs)
    if task_kind == "framework":
        issue_refs = _merge_issue_refs(issue_refs, framework.get("related_issue_refs") or {})
    workflow = _workflow_summary(
        result,
        traces,
        method_runs,
        post_build=post_build,
        task_kind=task_kind,
        issue_refs=issue_refs,
    )

    return {
        "task_id": task["id"],
        "task_kind": task_kind,
        "run_id": run_id,
        "persisted_at": persisted_at.astimezone(timezone.utc).isoformat(),
        "task": _task_snapshot(task),
        "result": result_with_learning,
        "comparison": {
            "task": bool(result.get("comparison_task")),
            "targets": list(result.get("comparison_targets") or []),
            "summary": dict(result.get("cross_validation") or {}),
        },
        "market": dict(result.get("market_context") or {}),
        "framework": framework,
        "method_runs": method_runs,
        "post_build": post_build,
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
    paths.extend(artifacts.get("analytical_trace_paths") or [])
    if result.get("platform_trace_path"):
        paths.append(result["platform_trace_path"])
    if result.get("analytical_trace_path"):
        paths.append(result["analytical_trace_path"])

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

    if trace_path.suffix.lower() == ".json":
        data = json.loads(trace_path.read_text()) or {}
        steps = list(data.get("steps") or [])
        latest_step = steps[-1] if steps else {}
        route = data.get("route") or {}
        context = data.get("context") or {}
        generation_plan = context.get("generation_plan") or {}
        instruction_resolution = generation_plan.get("instruction_resolution") or {}
        selected_curve_names = dict(context.get("selected_curve_names") or {})
        return {
            "path": str(trace_path),
            "exists": True,
            "trace_kind": data.get("trace_type", "analytical"),
            "request_id": data.get("trace_id"),
            "status": data.get("status"),
            "outcome": data.get("status"),
            "action": route.get("name"),
            "route_method": route.get("family"),
            "updated_at": data.get("updated_at"),
            "latest_event": latest_step.get("kind"),
            "latest_event_status": latest_step.get("status"),
            "latest_event_details": {
                "label": latest_step.get("label"),
                "notes": latest_step.get("notes") or [],
            },
            "selected_curve_names": selected_curve_names,
            "instruction_resolution": instruction_resolution,
            "instruction_resolution_effective_count": len(
                instruction_resolution.get("effective_instructions") or []
            ),
            "instruction_resolution_dropped_count": len(
                instruction_resolution.get("dropped_instructions") or []
            ),
            "instruction_resolution_conflict_count": len(
                instruction_resolution.get("conflicts") or []
            ),
            "token_usage": data.get("token_usage") or {},
            "request_metadata": context,
            "linear_issue": None,
            "github_issue": None,
            "step_count": len(steps),
        }

    data = yaml.safe_load(trace_path.read_text()) or {}
    events = list(data.get("events") or [])
    latest_event = events[-1] if events else {}
    linear = data.get("linear_issue") or {}
    github = data.get("github_issue") or {}
    metadata = data.get("request_metadata") or {}
    semantic_role_ownership = (
        data.get("semantic_role_ownership")
        or metadata.get("semantic_role_ownership")
        or {}
    )
    selected_curve_names = dict(
        metadata.get("selected_curve_names")
        or (metadata.get("runtime_contract") or {}).get("selected_curve_names")
        or (metadata.get("runtime_contract") or {}).get("snapshot_reference", {}).get("selected_curve_names")
        or {}
    )
    return {
        "path": str(trace_path),
        "exists": True,
        "trace_kind": "platform",
        "request_id": data.get("request_id"),
        "status": data.get("status"),
        "outcome": data.get("outcome"),
        "action": data.get("action"),
        "route_method": data.get("route_method"),
        "updated_at": data.get("updated_at"),
        "latest_event": latest_event.get("event"),
        "latest_event_status": latest_event.get("status"),
        "latest_event_details": latest_event.get("details") or {},
        "selected_curve_names": selected_curve_names,
        "semantic_role_ownership": semantic_role_ownership,
        "semantic_role_ownership_stage": semantic_role_ownership.get("selected_stage"),
        "semantic_role_ownership_role": semantic_role_ownership.get("selected_role"),
        "semantic_role_ownership_trigger": semantic_role_ownership.get("trigger_condition"),
        "semantic_role_ownership_artifact": semantic_role_ownership.get("artifact_kind"),
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
    post_build: dict[str, Any],
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
        "post_build_latest_phase": post_build.get("latest_phase"),
        "post_build_latest_status": post_build.get("latest_status"),
        "post_build_latest_method": post_build.get("latest_method"),
        "post_build_flags": dict(post_build.get("active_flags") or {}),
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


def summarize_task_learning(
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
    lesson_contract_reports = _lesson_contract_reports(reflection.get("lesson_contract"))
    lesson_contract_errors = _unique_strings(
        error
        for report in lesson_contract_reports
        for error in (report.get("errors") or [])
        if isinstance(error, str)
    )
    lesson_contract_warnings = _unique_strings(
        warning
        for report in lesson_contract_reports
        for warning in (report.get("warnings") or [])
        if isinstance(warning, str)
    )
    lesson_promotion_outcomes = _string_list(reflection.get("lesson_promotion_outcome"))
    if lesson_contract_reports:
        if all(bool(report.get("valid")) for report in lesson_contract_reports):
            lesson_contract_outcome = "validated"
        elif any(bool(report.get("valid")) for report in lesson_contract_reports):
            lesson_contract_outcome = "mixed"
        else:
            lesson_contract_outcome = "rejected"
    else:
        lesson_contract_outcome = "not_attempted"
    lesson_ids = list(knowledge_summary.get("lesson_ids") or [])
    lesson_titles = list(knowledge_summary.get("lesson_titles") or [])
    retrieval_stages = list(knowledge_summary.get("retrieval_stages") or [])
    retrieval_sources = list(knowledge_summary.get("retrieval_sources") or [])
    cookbook_paths = list(artifacts.get("cookbook_candidate_paths") or [])
    promotion_candidate_paths = list(artifacts.get("promotion_candidate_paths") or [])
    knowledge_trace_paths = list(artifacts.get("knowledge_trace_paths") or [])
    knowledge_gap_paths = list(artifacts.get("knowledge_gap_log_paths") or [])
    method_results = result.get("method_results") or {}
    successful_method_results = False
    if isinstance(method_results, Mapping):
        successful_method_results = any(
            bool(payload.get("success"))
            for payload in method_results.values()
            if isinstance(payload, Mapping)
        )
    reusable_artifact_count = (
        len(captured_ids)
        + len(cookbook_paths)
        + len(promotion_candidate_paths)
        + len(knowledge_trace_paths)
        + len(knowledge_gap_paths)
    )
    if reusable_artifact_count > 0:
        knowledge_outcome = "captured_knowledge"
        if captured_ids:
            knowledge_outcome_reason = (
                f"captured {len(captured_ids)} lesson(s) and related reusable artifacts"
            )
        elif cookbook_paths:
            knowledge_outcome_reason = (
                f"captured cookbook artifacts ({len(cookbook_paths)})"
            )
        elif promotion_candidate_paths:
            knowledge_outcome_reason = (
                f"captured promotion candidates ({len(promotion_candidate_paths)})"
            )
        elif knowledge_trace_paths:
            knowledge_outcome_reason = (
                f"captured knowledge trace artifacts ({len(knowledge_trace_paths)})"
            )
        else:
            knowledge_outcome_reason = (
                f"captured knowledge-gap artifacts ({len(knowledge_gap_paths)})"
            )
    elif bool(result.get("success")) or successful_method_results:
        knowledge_outcome = "no_new_knowledge"
        if successful_method_results and not bool(result.get("success")):
            knowledge_outcome_reason = (
                "one or more method builds succeeded without capturing new reusable knowledge artifacts"
            )
        else:
            knowledge_outcome_reason = (
                "task succeeded without new reusable knowledge artifacts"
            )
    else:
        knowledge_outcome = "blocked_without_learning"
        knowledge_outcome_reason = "task failed before any reusable learning artifact was captured"

    return {
        "task_kind": task_kind,
        "retrieved_lesson_ids": lesson_ids,
        "retrieved_lesson_titles": lesson_titles,
        "retrieval_stages": retrieval_stages,
        "retrieval_sources": retrieval_sources,
        "captured_lesson_ids": captured_ids,
        "lesson_contract_reports": lesson_contract_reports,
        "lesson_contract_count": len(lesson_contract_reports),
        "lesson_contract_outcome": lesson_contract_outcome,
        "lesson_contract_errors": lesson_contract_errors,
        "lesson_contract_warnings": lesson_contract_warnings,
        "lesson_promotion_outcomes": lesson_promotion_outcomes,
        "lessons_attributed": int(reflection.get("lessons_attributed") or 0),
        "cookbook_enriched": bool(reflection.get("cookbook_enriched")),
        "cookbook_candidate_paths": cookbook_paths,
        "promotion_candidate_paths": promotion_candidate_paths,
        "knowledge_trace_paths": knowledge_trace_paths,
        "knowledge_gap_log_paths": knowledge_gap_paths,
        "reusable_artifact_count": reusable_artifact_count,
        "knowledge_outcome": knowledge_outcome,
        "knowledge_outcome_reason": knowledge_outcome_reason,
    }


def _post_build_snapshot(tracking: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project one post-build tracking payload into a compact summary."""
    tracking = dict(tracking or {})
    events = list(tracking.get("events") or [])
    return {
        "latest_phase": tracking.get("last_phase"),
        "latest_status": tracking.get("last_status"),
        "updated_at": tracking.get("updated_at"),
        "event_count": len(events),
        "active_flags": dict(tracking.get("active_flags") or {}),
    }


def _post_build_summary(
    result: Mapping[str, Any],
    method_runs: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate post-build tracking across top-level and per-method runs."""
    task_snapshot = _post_build_snapshot(result.get("post_build_tracking"))
    methods: dict[str, dict[str, Any]] = {}
    merged_flags: dict[str, bool] = {}
    latest_method: str | None = None
    latest_phase: str | None = task_snapshot.get("latest_phase")
    latest_status: str | None = task_snapshot.get("latest_status")
    latest_updated_at: str = str(task_snapshot.get("updated_at") or "")

    for flag, enabled in dict(task_snapshot.get("active_flags") or {}).items():
        merged_flags[str(flag)] = bool(enabled)

    for method, payload in method_runs.items():
        if not isinstance(payload, Mapping):
            continue
        snapshot = _post_build_snapshot(payload.get("post_build_tracking"))
        methods[str(method)] = snapshot
        for flag, enabled in dict(snapshot.get("active_flags") or {}).items():
            merged_flags[str(flag)] = merged_flags.get(str(flag), False) or bool(enabled)
        updated_at = str(snapshot.get("updated_at") or "")
        if updated_at and updated_at >= latest_updated_at:
            latest_updated_at = updated_at
            latest_method = str(method)
            latest_phase = snapshot.get("latest_phase")
            latest_status = snapshot.get("latest_status")

    return {
        "task": task_snapshot,
        "methods": methods,
        "latest_method": latest_method,
        "latest_phase": latest_phase,
        "latest_status": latest_status,
        "active_flags": merged_flags,
    }


def _latest_trace(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    dated = [trace for trace in traces if trace.get("updated_at")]
    if not dated:
        return traces[-1] if traces else None
    return max(dated, key=lambda trace: str(trace.get("updated_at")))


def _lesson_contract_reports(value: Any) -> list[dict[str, Any]]:
    """Normalize contract reports from a reflection payload."""
    if isinstance(value, Mapping):
        if "valid" in value and "normalized_payload" in value:
            return [dict(value)]
        reports: list[dict[str, Any]] = []
        for item in value.values():
            if isinstance(item, Mapping) and "valid" in item and "normalized_payload" in item:
                reports.append(dict(item))
        return reports
    if isinstance(value, list):
        return [
            dict(item)
            for item in value
            if isinstance(item, Mapping) and "valid" in item and "normalized_payload" in item
        ]
    if isinstance(value, tuple):
        return [
            dict(item)
            for item in value
            if isinstance(item, Mapping) and "valid" in item and "normalized_payload" in item
        ]
    return []


def _string_list(value: Any) -> list[str]:
    """Normalize a scalar or sequence into a list of strings."""
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, tuple):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _unique_strings(values) -> list[str]:
    """Deduplicate string values while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str):
            value = str(value)
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


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
    storage.setdefault("latest_index_path", str(root / "task_results_latest.json"))
    run_id = str(normalized.get("run_id") or "").strip()
    if run_id:
        storage.setdefault(
            "diagnosis_history_packet_path",
            str(root / "task_runs" / "diagnostics" / "history" / str(task_id) / f"{run_id}.json"),
        )
        storage.setdefault(
            "diagnosis_history_dossier_path",
            str(
                root
                / "task_runs"
                / "diagnostics"
                / "history"
                / str(task_id)
                / f"{run_id}.md"
            ),
        )
    storage.setdefault(
        "diagnosis_latest_packet_path",
        str(root / "task_runs" / "diagnostics" / "latest" / f"{task_id}.json"),
    )
    storage.setdefault(
        "diagnosis_latest_dossier_path",
        str(root / "task_runs" / "diagnostics" / "latest" / f"{task_id}.md"),
    )
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
