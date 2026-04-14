"""Save and load detailed run records (inputs, outputs, timing, errors) for each pricing task."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import yaml

from trellis.agent.analytical_traces import AnalyticalTrace, route_health_snapshot
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
CANARY_BATCH_ROOT = TASK_RUN_ROOT / "canary_batches"
CANARY_BATCH_HISTORY_ROOT = CANARY_BATCH_ROOT / "history"
CANARY_BATCH_LATEST_ROOT = CANARY_BATCH_ROOT / "latest"
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


def persist_canary_batch_record(
    *,
    canaries: list[dict[str, Any]],
    meta: Mapping[str, Any],
    results: list[dict[str, Any]],
    model: str,
    validation: str,
    knowledge_light: bool,
    replay: bool,
    requested_task_id: str | None,
    requested_subset: str | None,
    root: Path = ROOT,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, str]:
    """Persist one explicit canary-batch record and stable latest view."""
    batch_id = f"canary_{started_at.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    execution_mode = "cassette_replay" if replay else "live"
    batch_scope, scope_slug = _canary_batch_scope(
        requested_task_id=requested_task_id,
        requested_subset=requested_subset,
    )
    knowledge_profile = "knowledge_light" if knowledge_light else "default"
    comparison_key = f"{execution_mode}:{scope_slug}:{validation}:{knowledge_profile}:{model}"
    synthetic_source = _synthetic_canary_source(root)
    benchmark_eligible = execution_mode == "live" and not synthetic_source

    by_task_id = {
        str((payload.get("canary_id") or payload.get("task_id") or "")).strip(): dict(payload)
        for payload in results
        if isinstance(payload, Mapping)
    }
    entries = [
        _build_canary_batch_entry(
            canary=dict(canary),
            result=by_task_id.get(str(canary.get("id") or "").strip(), {}),
            batch_id=batch_id,
            execution_mode=execution_mode,
            benchmark_eligible=benchmark_eligible,
        )
        for canary in canaries
    ]
    summary = _summarize_canary_batch(
        entries,
        execution_mode=execution_mode,
        batch_scope=batch_scope,
        scope_slug=scope_slug,
        benchmark_eligible=benchmark_eligible,
        synthetic_source=synthetic_source,
        model=model,
        validation=validation,
        knowledge_profile=knowledge_profile,
        comparison_key=comparison_key,
        requested_task_id=requested_task_id,
        requested_subset=requested_subset,
        started_at=started_at,
        finished_at=finished_at,
    )
    record = {
        "batch_id": batch_id,
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": finished_at.astimezone(timezone.utc).isoformat(),
        "selection": {
            "requested_task_id": requested_task_id,
            "requested_subset": requested_subset,
        },
        "config": {
            "model": model,
            "validation": validation,
            "knowledge_profile": knowledge_profile,
            "replay": replay,
            "execution_mode": execution_mode,
            "comparison_key": comparison_key,
            "synthetic_source": synthetic_source or None,
        },
        "manifest": {
            "version": meta.get("version"),
            "refresh_cadence": meta.get("refresh_cadence"),
            "total_budget_usd": meta.get("total_budget_usd"),
        },
        "summary": summary,
        "canaries": entries,
    }

    history_root = root / CANARY_BATCH_HISTORY_ROOT.relative_to(ROOT)
    latest_root = root / CANARY_BATCH_LATEST_ROOT.relative_to(ROOT)
    history_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    history_path = history_root / f"{batch_id}.json"
    latest_path = latest_root / f"{_scope_slug_to_latest_key(scope_slug, execution_mode, validation, knowledge_profile, model)}.json"
    history_path.write_text(json.dumps(record, indent=2, default=str))
    latest_path.write_text(json.dumps(record, indent=2, default=str))
    return {
        "batch_id": batch_id,
        "history_path": str(history_path),
        "latest_path": str(latest_path),
    }


def load_canary_batch_records(
    *,
    root: Path = ROOT,
    execution_mode: str | None = None,
    benchmark_only: bool = False,
) -> list[dict[str, Any]]:
    """Load persisted canary batch records from history."""
    history_root = root / CANARY_BATCH_HISTORY_ROOT.relative_to(ROOT)
    if not history_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(history_root.glob("*.json")):
        payload = json.loads(path.read_text())
        summary = dict(payload.get("summary") or {})
        if execution_mode is not None and str(summary.get("execution_mode") or "") != execution_mode:
            continue
        if benchmark_only and not bool(summary.get("benchmark_eligible")):
            continue
        records.append(payload)
    return sorted(records, key=lambda item: str(item.get("started_at") or ""))


def load_canary_task_history(
    task_id: str,
    *,
    root: Path = ROOT,
    execution_mode: str | None = None,
    benchmark_only: bool = False,
) -> list[dict[str, Any]]:
    """Flatten persisted canary batch history for one task ID."""
    history: list[dict[str, Any]] = []
    for batch in load_canary_batch_records(
        root=root,
        execution_mode=execution_mode,
        benchmark_only=benchmark_only,
    ):
        summary = dict(batch.get("summary") or {})
        for item in batch.get("canaries") or []:
            if str(item.get("task_id") or "") != str(task_id):
                continue
            history.append(
                {
                    **dict(item),
                    "batch_id": batch.get("batch_id"),
                    "started_at": batch.get("started_at"),
                    "finished_at": batch.get("finished_at"),
                    "batch_scope": summary.get("batch_scope"),
                    "comparison_key": summary.get("comparison_key"),
                    "benchmark_eligible": bool(summary.get("benchmark_eligible")),
                }
            )
    return sorted(history, key=lambda item: str(item.get("started_at") or ""))


def _canary_batch_scope(
    *,
    requested_task_id: str | None,
    requested_subset: str | None,
) -> tuple[str, str]:
    """Return the human-readable scope plus a filename-safe slug."""
    if requested_task_id:
        task_id = str(requested_task_id).strip()
        return "single_task", f"single_task_{task_id}"
    if requested_subset:
        subset = str(requested_subset).strip()
        return f"subset:{subset}", f"subset_{subset}"
    return "full_curated", "full_curated"


def _synthetic_canary_source(root: Path) -> str:
    """Return a synthetic-source marker when a canary batch comes from pytest."""
    if os.environ.get("PYTEST_CURRENT_TEST") and root.resolve() == ROOT.resolve():
        return "pytest"
    return ""


def _scope_slug_to_latest_key(
    scope_slug: str,
    execution_mode: str,
    validation: str,
    knowledge_profile: str,
    model: str,
) -> str:
    """Return the stable latest filename stem for one canary batch scope."""
    return f"{execution_mode}__{scope_slug}__{validation}__{knowledge_profile}__{model}"


def _build_canary_batch_entry(
    *,
    canary: Mapping[str, Any],
    result: Mapping[str, Any],
    batch_id: str,
    execution_mode: str,
    benchmark_eligible: bool,
) -> dict[str, Any]:
    """Project one canary task result into the persisted batch history shape."""
    token_usage = dict(result.get("token_usage_summary") or {})
    return {
        "batch_id": batch_id,
        "task_id": str(canary.get("id") or result.get("task_id") or "").strip(),
        "engine_family": str(canary.get("engine_family") or result.get("engine_family") or "").strip(),
        "complexity": str(canary.get("complexity") or result.get("complexity") or "").strip(),
        "success": bool(result.get("success")),
        "skipped": bool(result.get("skipped")),
        "reason": str(result.get("reason") or "").strip(),
        "error": str(result.get("error") or "").strip(),
        "execution_mode": str(result.get("execution_mode") or execution_mode),
        "benchmark_eligible": benchmark_eligible and str(result.get("execution_mode") or execution_mode) == "live",
        "elapsed_seconds": round(float(result.get("elapsed_seconds") or 0.0), 4),
        "attempts": int(result.get("attempts") or 0),
        "token_usage": token_usage,
        "total_tokens": int(token_usage.get("total_tokens") or 0),
        "task_run_history_path": str(result.get("task_run_history_path") or "").strip(),
        "task_run_latest_path": str(result.get("task_run_latest_path") or "").strip(),
        "task_diagnosis_packet_path": str(result.get("task_diagnosis_packet_path") or "").strip(),
        "task_diagnosis_dossier_path": str(result.get("task_diagnosis_dossier_path") or "").strip(),
    }


def _summarize_canary_batch(
    entries: list[Mapping[str, Any]],
    *,
    execution_mode: str,
    batch_scope: str,
    scope_slug: str,
    benchmark_eligible: bool,
    synthetic_source: str,
    model: str,
    validation: str,
    knowledge_profile: str,
    comparison_key: str,
    requested_task_id: str | None,
    requested_subset: str | None,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    """Summarize one canary batch into stable aggregate metrics."""
    completed = [item for item in entries if not bool(item.get("skipped"))]
    elapsed_values = [float(item.get("elapsed_seconds") or 0.0) for item in completed]
    token_values = [int(item.get("total_tokens") or 0) for item in completed]
    attempt_values = [int(item.get("attempts") or 0) for item in completed]
    pass_count = sum(1 for item in completed if bool(item.get("success")))
    failure_count = sum(1 for item in completed if not bool(item.get("success")))
    skip_count = sum(1 for item in entries if bool(item.get("skipped")))
    completed_count = len(completed)
    return {
        "execution_mode": execution_mode,
        "batch_scope": batch_scope,
        "scope_slug": scope_slug,
        "benchmark_eligible": benchmark_eligible,
        "benchmark_exclusion_reason": (
            ""
            if benchmark_eligible
            else (f"synthetic:{synthetic_source}" if synthetic_source else "replay")
        ),
        "comparison_key": comparison_key,
        "model": model,
        "validation": validation,
        "knowledge_profile": knowledge_profile,
        "synthetic_source": synthetic_source or None,
        "requested_task_id": requested_task_id,
        "requested_subset": requested_subset,
        "task_count": len(entries),
        "completed_count": completed_count,
        "pass_count": pass_count,
        "failure_count": failure_count,
        "skip_count": skip_count,
        "pass_rate": _fraction(pass_count, completed_count),
        "total_elapsed_seconds": round(sum(elapsed_values), 4),
        "avg_elapsed_seconds": _fraction(sum(elapsed_values), completed_count),
        "median_elapsed_seconds": round(median(elapsed_values), 4) if elapsed_values else 0.0,
        "total_tokens": sum(token_values),
        "avg_tokens": _fraction(sum(token_values), completed_count),
        "median_tokens": round(median(token_values), 4) if token_values else 0.0,
        "total_attempts": sum(attempt_values),
        "avg_attempts": _fraction(sum(attempt_values), completed_count),
        "max_attempts": max(attempt_values) if attempt_values else 0,
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": finished_at.astimezone(timezone.utc).isoformat(),
    }


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
    telemetry = summarize_skill_telemetry(
        result_with_learning,
        task_kind=task_kind,
        traces=traces,
        method_runs=method_runs,
        workflow=workflow,
    )

    return {
        "task_id": task["id"],
        "task_kind": task_kind,
        "run_id": run_id,
        "persisted_at": persisted_at.astimezone(timezone.utc).isoformat(),
        "run_started_at": str(result.get("run_started_at") or result.get("start_time") or ""),
        "run_completed_at": str(result.get("run_completed_at") or ""),
        "task": _task_snapshot(task),
        "result": result_with_learning,
        "comparison": {
            "task": bool(result.get("comparison_task")),
            "targets": list(result.get("comparison_targets") or []),
            "summary": dict(result.get("cross_validation") or {}),
        },
        "execution": {
            "mode": str(result.get("execution_mode") or "live"),
            "llm_cassette": dict(result.get("llm_cassette") or {}),
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
        "telemetry": telemetry,
        "artifacts": dict(result.get("artifacts") or {}),
        "trace_summaries": traces,
        "issue_refs": issue_refs,
        "workflow": workflow,
        "summary": {
            "success": bool(result.get("success")),
            "status": workflow["status"],
            "task_kind": task_kind,
            "task_corpus": str(task.get("task_corpus") or ""),
            "task_definition_version": task.get("task_definition_version"),
            "market_scenario_id": str(task.get("market_scenario_id") or ""),
            "market_scenario_digest": str(
                dict(task.get("market") or {}).get("scenario_digest")
                or dict(result.get("market_context") or {}).get("metadata", {}).get("scenario_digest")
                or ""
            ),
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
            "execution_mode": str(result.get("execution_mode") or "live"),
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
        "task_corpus",
        "task_definition_version",
        "task_definition_manifest",
        "market_scenario_id",
        "market_scenario_digest",
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
        analytical_trace = AnalyticalTrace.from_dict(data)
        steps = list(data.get("steps") or [])
        latest_step = steps[-1] if steps else {}
        route = data.get("route") or {}
        context = data.get("context") or {}
        generation_plan = context.get("generation_plan") or {}
        instruction_resolution = generation_plan.get("instruction_resolution") or {}
        runtime_contract = dict(context.get("runtime_contract") or {})
        selected_curve_names = dict(
            context.get("selected_curve_names")
            or runtime_contract.get("selected_curve_names")
            or (runtime_contract.get("snapshot_reference") or {}).get("selected_curve_names")
            or {}
        )
        binding_health = _normalize_binding_health(
            raw_health=route_health_snapshot(analytical_trace),
            construction_identity=context.get("construction_identity") or {},
            semantic_blueprint=(context.get("semantic_blueprint") or {}),
            trace_kind=data.get("trace_type", "analytical"),
            trace_action=route.get("name"),
            trace_method=route.get("family"),
            trace_status=data.get("status"),
        )
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
            "binding_health": binding_health,
            "route_health": dict(binding_health),
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
    generation_boundary = data.get("generation_boundary") or {}
    route_binding_authority = (
        generation_boundary.get("route_binding_authority")
        or metadata.get("route_binding_authority")
        or {}
    )
    construction_identity = (
        generation_boundary.get("construction_identity")
        or metadata.get("construction_identity")
        or {}
    )
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
    semantic_blueprint = dict(metadata.get("semantic_blueprint") or {})
    operator_metadata = dict(
        route_binding_authority.get("operator_metadata")
        or construction_identity.get("operator_metadata")
        or {}
    )
    binding_health = _normalize_binding_health(
        raw_health={},
        construction_identity=construction_identity,
        route_binding_authority=route_binding_authority,
        semantic_blueprint=semantic_blueprint,
        trace_kind="platform",
        trace_action=data.get("action"),
        trace_method=data.get("route_method"),
        trace_status=data.get("status"),
        operator_metadata=operator_metadata,
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
        "construction_identity": construction_identity,
        "binding_authority": route_binding_authority,
        "route_binding_authority": route_binding_authority,
        "binding_health": binding_health,
        "route_health": dict(binding_health),
        "token_usage": data.get("token_usage") or {},
        "request_metadata": metadata,
        "linear_issue": linear if linear else None,
        "github_issue": github if github else None,
    }


def _normalize_binding_health(
    *,
    raw_health: Mapping[str, Any] | None,
    construction_identity: Mapping[str, Any] | None = None,
    route_binding_authority: Mapping[str, Any] | None = None,
    semantic_blueprint: Mapping[str, Any] | None = None,
    trace_kind: str = "",
    trace_action: Any = None,
    trace_method: Any = None,
    trace_status: Any = None,
    operator_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize trace health onto binding-first telemetry fields."""
    from trellis.agent.route_registry import should_surface_route_alias

    raw_health = dict(raw_health or {})
    construction_identity = dict(construction_identity or {})
    route_binding_authority = dict(route_binding_authority or {})
    semantic_blueprint = dict(semantic_blueprint or {})
    operator_metadata = dict(operator_metadata or {})
    authority_route_id = str(route_binding_authority.get("route_id") or "").strip()
    if authority_route_id and not should_surface_route_alias(route_binding_authority):
        authority_route_id = ""
    route_id = str(
        raw_health.get("route_id")
        or authority_route_id
        or semantic_blueprint.get("dsl_route")
        or ""
    ).strip()
    if route_id == "unknown":
        route_id = ""
    route_family = str(
        raw_health.get("route_family")
        or route_binding_authority.get("route_family")
        or semantic_blueprint.get("dsl_route_family")
        or trace_method
        or ""
    ).strip()
    binding_id = str(
        raw_health.get("binding_id")
        or raw_health.get("backend_binding_id")
        or construction_identity.get("backend_binding_id")
        or ""
    ).strip()
    binding_family = str(
        raw_health.get("binding_family")
        or construction_identity.get("backend_engine_family")
        or route_family
        or trace_method
        or ""
    ).strip()
    binding_alias = str(
        raw_health.get("binding_alias")
        or raw_health.get("route_alias")
        or construction_identity.get("route_alias")
        or route_id
        or ""
    ).strip()
    binding_display_name = str(
        raw_health.get("binding_display_name")
        or construction_identity.get("binding_display_name")
        or operator_metadata.get("display_name")
        or ""
    ).strip()
    binding_short_description = str(
        raw_health.get("binding_short_description")
        or construction_identity.get("binding_short_description")
        or operator_metadata.get("short_description")
        or ""
    ).strip()
    binding_diagnostic_label = str(
        raw_health.get("binding_diagnostic_label")
        or construction_identity.get("binding_diagnostic_label")
        or operator_metadata.get("diagnostic_label")
        or ""
    ).strip()
    primary_kind = str(
        raw_health.get("primary_kind")
        or construction_identity.get("primary_kind")
        or ""
    ).strip()
    primary_label = str(
        raw_health.get("primary_label")
        or construction_identity.get("primary_label")
        or binding_display_name
        or binding_family
        or binding_alias
        or "unknown"
    ).strip()
    return {
        "binding_id": binding_id,
        "binding_family": binding_family,
        "binding_alias": binding_alias,
        "route_id": route_id,
        "route_family": route_family,
        "route_alias": binding_alias,
        "trace_status": str(raw_health.get("trace_status") or trace_status or "").strip(),
        "effective_instruction_ids": list(raw_health.get("effective_instruction_ids") or []),
        "effective_instruction_count": int(raw_health.get("effective_instruction_count") or 0),
        "hard_constraint_count": int(raw_health.get("hard_constraint_count") or 0),
        "conflict_count": int(raw_health.get("conflict_count") or 0),
        "canary_task_ids": list(
            route_binding_authority.get("canary_task_ids")
            or raw_health.get("canary_task_ids")
            or []
        ),
        "primary_kind": primary_kind,
        "primary_label": primary_label,
        "backend_binding_id": binding_id,
        "binding_display_name": binding_display_name,
        "binding_short_description": binding_short_description,
        "binding_diagnostic_label": binding_diagnostic_label,
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
    selected_artifact_ids = list(knowledge_summary.get("selected_artifact_ids") or [])
    selected_artifact_titles = list(knowledge_summary.get("selected_artifact_titles") or [])
    selected_artifacts_by_audience = {
        str(audience): [dict(item) for item in artifacts]
        for audience, artifacts in dict(knowledge_summary.get("selected_artifacts_by_audience") or {}).items()
        if isinstance(artifacts, list) and artifacts
    }
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
        "selected_artifact_ids": selected_artifact_ids,
        "selected_artifact_titles": selected_artifact_titles,
        "selected_artifacts_by_audience": selected_artifacts_by_audience,
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


def summarize_skill_telemetry(
    result: dict[str, Any],
    *,
    task_kind: str,
    traces: list[dict[str, Any]],
    method_runs: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize selected-skill attribution and binding health for one task run."""
    learning = dict(result.get("learning") or {})
    outcome = _telemetry_run_outcome(
        result=result,
        workflow=workflow,
        method_runs=method_runs,
    )
    retry_count = _telemetry_retry_count(result=result, method_runs=method_runs)
    degraded = _telemetry_is_degraded(result=result, method_runs=method_runs)
    comparison_status = str((result.get("cross_validation") or {}).get("status") or "").strip()
    binding_observations = _route_observations(
        traces=traces,
        method_runs=method_runs,
        outcome=outcome,
        retry_count=retry_count,
        degraded=degraded,
        selected_artifact_ids=[],
    )
    selected_artifacts = _selected_artifact_observations(
        learning=learning,
        route_observations=binding_observations,
        outcome=outcome,
        retry_count=retry_count,
        degraded=degraded,
    )
    selected_artifact_ids = [item["artifact_id"] for item in selected_artifacts]
    if selected_artifact_ids:
        for item in binding_observations:
            item["selected_artifact_ids"] = list(selected_artifact_ids)
    return {
        "task_kind": task_kind,
        "run_outcome": outcome,
        "retried": retry_count > 0,
        "retry_count": retry_count,
        "degraded": degraded,
        "comparison_status": comparison_status,
        "selected_artifacts": selected_artifacts,
        "binding_observations": binding_observations,
        "route_observations": [dict(item) for item in binding_observations],
    }


def aggregate_skill_telemetry(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll up selected-skill outcome attribution across persisted task-run records."""
    aggregates: dict[str, dict[str, Any]] = {}
    for record in records:
        telemetry = dict(record.get("telemetry") or {})
        task_id = str(record.get("task_id") or "").strip()
        persisted_at = str(record.get("persisted_at") or "").strip()
        for item in telemetry.get("selected_artifacts") or []:
            if not isinstance(item, Mapping):
                continue
            artifact_id = str(item.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            aggregate = aggregates.setdefault(
                artifact_id,
                {
                    "artifact_id": artifact_id,
                    "title": str(item.get("title") or "").strip(),
                    "kind": str(item.get("kind") or "").strip(),
                    "selection_count": 0,
                    "audiences": [],
                    "binding_ids": [],
                    "binding_families": [],
                    "binding_aliases": [],
                    "route_ids": [],
                    "route_families": [],
                    **_rollup_counter_fields(),
                },
            )
            aggregate["selection_count"] += 1
            _accumulate_rollup_counters(
                aggregate,
                item,
                task_id=task_id,
                persisted_at=persisted_at,
            )
            for key in (
                "audiences",
                "binding_ids",
                "binding_families",
                "binding_aliases",
                "route_ids",
                "route_families",
            ):
                for value in item.get(key) or []:
                    if value and value not in aggregate[key]:
                        aggregate[key].append(value)
    rollup = {
        "run_count": len(records),
        "artifacts": sorted(aggregates.values(), key=lambda item: item["artifact_id"]),
    }
    rollup["ranking_inputs"] = build_skill_ranking_inputs(rollup)["artifacts"]
    return rollup


def aggregate_binding_health(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll up binding observations across persisted task-run records."""
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        telemetry = dict(record.get("telemetry") or {})
        task_id = str(record.get("task_id") or "").strip()
        persisted_at = str(record.get("persisted_at") or "").strip()
        for item in telemetry.get("binding_observations") or telemetry.get("route_observations") or []:
            if not isinstance(item, Mapping):
                continue
            binding_id = str(item.get("binding_id") or item.get("backend_binding_id") or "").strip()
            binding_family = str(item.get("binding_family") or item.get("route_family") or "").strip()
            binding_alias = str(item.get("binding_alias") or item.get("route_alias") or item.get("route_id") or "").strip()
            if not binding_id and not binding_family and not binding_alias:
                continue
            key = (binding_id, binding_family, binding_alias)
            aggregate = aggregates.setdefault(
                key,
                {
                    "binding_id": binding_id,
                    "binding_family": binding_family,
                    "binding_alias": binding_alias,
                    "route_id": str(item.get("route_id") or "").strip(),
                    "route_family": str(item.get("route_family") or "").strip(),
                    "route_alias": str(item.get("route_alias") or binding_alias).strip(),
                    "primary_label": str(item.get("primary_label") or "").strip(),
                    "binding_display_name": str(item.get("binding_display_name") or "").strip(),
                    "binding_diagnostic_label": str(item.get("binding_diagnostic_label") or "").strip(),
                    "observation_count": 0,
                    "trace_kinds": [],
                    "selected_artifact_ids": [],
                    "effective_instruction_count_total": 0,
                    "hard_constraint_count_total": 0,
                    "conflict_count_total": 0,
                    **_rollup_counter_fields(),
                },
            )
            aggregate["observation_count"] += 1
            _accumulate_rollup_counters(
                aggregate,
                item,
                task_id=task_id,
                persisted_at=persisted_at,
            )
            for key_name in ("trace_kinds", "selected_artifact_ids"):
                for value in item.get(key_name) or []:
                    if value and value not in aggregate[key_name]:
                        aggregate[key_name].append(value)
            aggregate["effective_instruction_count_total"] += int(
                item.get("effective_instruction_count") or 0
            )
            aggregate["hard_constraint_count_total"] += int(
                item.get("hard_constraint_count") or 0
            )
            aggregate["conflict_count_total"] += int(item.get("conflict_count") or 0)
    rollup = {
        "run_count": len(records),
        "bindings": sorted(
            aggregates.values(),
            key=lambda item: (
                item["binding_family"],
                item["binding_display_name"],
                item["binding_id"],
                item["binding_alias"],
            ),
        ),
    }
    rollup["routes"] = rollup["bindings"]
    rollup["ranking_inputs"] = build_binding_ranking_inputs(rollup)["bindings"]
    return rollup


def aggregate_route_health(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper for legacy route-health rollup callers."""
    return aggregate_binding_health(records)


def load_latest_skill_telemetry_rollup(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Load the latest task runs and aggregate selected-skill telemetry."""
    return load_latest_telemetry_rollups(root=root, task_kind=task_kind)["skill_telemetry"]


def load_latest_binding_health_rollup(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Load the latest task runs and aggregate binding-health observations."""
    return load_latest_telemetry_rollups(root=root, task_kind=task_kind)["binding_health"]


def load_latest_route_health_rollup(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for legacy route-health rollup callers."""
    return load_latest_binding_health_rollup(root=root, task_kind=task_kind)


def load_latest_telemetry_rollups(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Load the latest task runs once and rebuild both telemetry rollups."""
    records = load_latest_task_run_records(root=root, task_kind=task_kind)
    binding_health = aggregate_binding_health(records)
    return {
        "skill_telemetry": aggregate_skill_telemetry(records),
        "binding_health": binding_health,
        "route_health": binding_health,
    }


def build_skill_ranking_inputs(rollup: Mapping[str, Any]) -> dict[str, Any]:
    """Project skill-telemetry rollups into stable ranking inputs."""
    artifacts = []
    for item in rollup.get("artifacts") or []:
        if not isinstance(item, Mapping):
            continue
        selection_count = int(item.get("selection_count") or 0)
        artifacts.append(
            {
                "artifact_id": str(item.get("artifact_id") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "kind": str(item.get("kind") or "").strip(),
                "selection_count": selection_count,
                "success_rate": _fraction(item.get("success_count"), selection_count),
                "failure_rate": _fraction(item.get("failure_count"), selection_count),
                "blocked_rate": _fraction(item.get("blocked_count"), selection_count),
                "retry_rate": _fraction(item.get("retried_count"), selection_count),
                "degradation_rate": _fraction(item.get("degraded_count"), selection_count),
                "avg_retry_count": _fraction(item.get("retry_count_total"), selection_count),
                "last_seen_at": str(item.get("last_seen_at") or "").strip(),
                "first_seen_at": str(item.get("first_seen_at") or "").strip(),
                "binding_coverage_count": len(
                    item.get("binding_ids")
                    or item.get("binding_aliases")
                    or item.get("binding_families")
                    or []
                ),
                "route_coverage_count": len(item.get("route_ids") or []),
                "task_coverage_count": len(item.get("task_ids") or []),
            }
        )
    return {
        "run_count": int(rollup.get("run_count") or 0),
        "artifacts": sorted(artifacts, key=lambda item: item["artifact_id"]),
    }


def build_binding_ranking_inputs(rollup: Mapping[str, Any]) -> dict[str, Any]:
    """Project binding-health rollups into stable ranking inputs."""
    bindings = []
    for item in rollup.get("bindings") or rollup.get("routes") or []:
        if not isinstance(item, Mapping):
            continue
        observation_count = int(item.get("observation_count") or 0)
        bindings.append(
            {
                "binding_id": str(item.get("binding_id") or item.get("backend_binding_id") or "").strip(),
                "binding_family": str(item.get("binding_family") or item.get("route_family") or "").strip(),
                "binding_alias": str(item.get("binding_alias") or item.get("route_alias") or item.get("route_id") or "").strip(),
                "route_id": str(item.get("route_id") or "").strip(),
                "route_family": str(item.get("route_family") or "").strip(),
                "observation_count": observation_count,
                "success_rate": _fraction(item.get("success_count"), observation_count),
                "failure_rate": _fraction(item.get("failure_count"), observation_count),
                "blocked_rate": _fraction(item.get("blocked_count"), observation_count),
                "retry_rate": _fraction(item.get("retried_count"), observation_count),
                "degradation_rate": _fraction(item.get("degraded_count"), observation_count),
                "avg_retry_count": _fraction(item.get("retry_count_total"), observation_count),
                "avg_effective_instruction_count": _fraction(
                    item.get("effective_instruction_count_total"),
                    observation_count,
                ),
                "avg_hard_constraint_count": _fraction(
                    item.get("hard_constraint_count_total"),
                    observation_count,
                ),
                "avg_conflict_count": _fraction(
                    item.get("conflict_count_total"),
                    observation_count,
                ),
                "primary_label": str(item.get("primary_label") or "").strip(),
                "binding_display_name": str(item.get("binding_display_name") or "").strip(),
                "binding_diagnostic_label": str(item.get("binding_diagnostic_label") or "").strip(),
                "last_seen_at": str(item.get("last_seen_at") or "").strip(),
                "first_seen_at": str(item.get("first_seen_at") or "").strip(),
                "selected_artifact_count": len(item.get("selected_artifact_ids") or []),
                "task_coverage_count": len(item.get("task_ids") or []),
            }
        )
    result = {
        "run_count": int(rollup.get("run_count") or 0),
        "bindings": sorted(
            bindings,
            key=lambda item: (
                item["binding_family"],
                item["binding_display_name"],
                item["binding_id"],
                item["binding_alias"],
            ),
        ),
    }
    result["routes"] = result["bindings"]
    return result


def build_route_ranking_inputs(rollup: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for legacy route-ranking callers."""
    return build_binding_ranking_inputs(rollup)


def load_latest_skill_ranking_inputs(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Load the latest task runs and derive skill-ranking inputs."""
    return build_skill_ranking_inputs(
        load_latest_telemetry_rollups(root=root, task_kind=task_kind)["skill_telemetry"]
    )


def load_latest_binding_ranking_inputs(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Load the latest task runs and derive binding-ranking inputs."""
    return build_binding_ranking_inputs(
        load_latest_telemetry_rollups(root=root, task_kind=task_kind)["binding_health"]
    )


def load_latest_route_ranking_inputs(
    *,
    root: Path = ROOT,
    task_kind: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for legacy route-ranking callers."""
    return load_latest_binding_ranking_inputs(root=root, task_kind=task_kind)


def _rollup_counter_fields() -> dict[str, Any]:
    """Return the shared counter fields used by telemetry rollups."""
    return {
        "success_count": 0,
        "failure_count": 0,
        "blocked_count": 0,
        "retried_count": 0,
        "retry_count_total": 0,
        "degraded_count": 0,
        "outcome_counts": {},
        "task_ids": [],
        "first_seen_at": "",
        "last_seen_at": "",
    }


def _accumulate_rollup_counters(
    aggregate: dict[str, Any],
    item: Mapping[str, Any],
    *,
    task_id: str,
    persisted_at: str,
) -> None:
    """Accumulate the shared counters for one telemetry observation."""
    if item.get("success"):
        aggregate["success_count"] += 1
    elif str(item.get("outcome") or "") in {"blocked", "needs_library_work"}:
        aggregate["blocked_count"] += 1
    else:
        aggregate["failure_count"] += 1
    outcome = str(item.get("outcome") or "unknown")
    aggregate["outcome_counts"][outcome] = aggregate["outcome_counts"].get(outcome, 0) + 1
    if item.get("retried"):
        aggregate["retried_count"] += 1
    aggregate["retry_count_total"] += int(item.get("retry_count") or 0)
    if item.get("degraded"):
        aggregate["degraded_count"] += 1
    if task_id and task_id not in aggregate["task_ids"]:
        aggregate["task_ids"].append(task_id)
    if persisted_at:
        if not aggregate["first_seen_at"] or persisted_at < aggregate["first_seen_at"]:
            aggregate["first_seen_at"] = persisted_at
        if not aggregate["last_seen_at"] or persisted_at > aggregate["last_seen_at"]:
            aggregate["last_seen_at"] = persisted_at


def _fraction(numerator: Any, denominator: int) -> float:
    """Return a rounded ratio for ranking inputs."""
    if denominator <= 0:
        return 0.0
    return round(float(numerator or 0) / float(denominator), 4)


def _telemetry_run_outcome(
    *,
    result: Mapping[str, Any],
    workflow: Mapping[str, Any],
    method_runs: Mapping[str, Any],
) -> str:
    """Normalize one task run into a stable telemetry outcome bucket."""
    workflow_status = str(workflow.get("status") or "").strip()
    comparison_status = str((result.get("cross_validation") or {}).get("status") or "").strip()
    if workflow_status in {"blocked", "needs_library_work"}:
        return workflow_status
    if comparison_status and comparison_status != "passed":
        return f"comparison:{comparison_status}"
    if bool(result.get("success")):
        return "succeeded"
    if _telemetry_is_degraded(result=result, method_runs=method_runs):
        return "degraded"
    if workflow_status:
        return workflow_status
    return "failed"


def _telemetry_retry_count(
    *,
    result: Mapping[str, Any],
    method_runs: Mapping[str, Any],
) -> int:
    """Count retries beyond the initial attempt for one task run."""
    method_retry_count = sum(
        max(int(payload.get("attempts") or 0) - 1, 0)
        for payload in method_runs.values()
        if isinstance(payload, Mapping)
    )
    if method_retry_count:
        return method_retry_count
    return max(int(result.get("attempts") or 0) - 1, 0)


def _telemetry_is_degraded(
    *,
    result: Mapping[str, Any],
    method_runs: Mapping[str, Any],
) -> bool:
    """Report whether the run partially succeeded but did not cleanly complete."""
    successful_methods = [
        payload
        for payload in method_runs.values()
        if isinstance(payload, Mapping) and bool(payload.get("success"))
    ]
    failed_methods = [
        payload
        for payload in method_runs.values()
        if isinstance(payload, Mapping) and not bool(payload.get("success"))
    ]
    comparison_status = str((result.get("cross_validation") or {}).get("status") or "").strip().lower()
    return bool(successful_methods and failed_methods) or comparison_status == "insufficient_results"


def _selected_artifact_observations(
    *,
    learning: Mapping[str, Any],
    route_observations: list[dict[str, Any]],
    outcome: str,
    retry_count: int,
    degraded: bool,
) -> list[dict[str, Any]]:
    """Project selected-artifact telemetry for one task run."""
    by_audience = {
        str(audience): [dict(item) for item in artifacts]
        for audience, artifacts in dict(learning.get("selected_artifacts_by_audience") or {}).items()
        if isinstance(artifacts, list)
    }
    route_ids = _unique_strings(
        [str(item.get("route_id") or "").strip() for item in route_observations]
    )
    route_families = _unique_strings(
        [str(item.get("route_family") or "").strip() for item in route_observations]
    )
    binding_ids = _unique_strings(
        [
            str(item.get("binding_id") or item.get("backend_binding_id") or "").strip()
            for item in route_observations
        ]
    )
    binding_families = _unique_strings(
        [str(item.get("binding_family") or "").strip() for item in route_observations]
    )
    binding_aliases = _unique_strings(
        [
            str(item.get("binding_alias") or item.get("route_alias") or "").strip()
            for item in route_observations
        ]
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for audience, items in by_audience.items():
        for item in items:
            artifact_id = str(item.get("id") or "").strip()
            if not artifact_id:
                continue
            artifact = artifacts.setdefault(
                artifact_id,
                {
                    "artifact_id": artifact_id,
                    "title": str(item.get("title") or "").strip(),
                    "kind": str(item.get("kind") or "").strip(),
                    "audiences": [],
                    "outcome": outcome,
                    "success": outcome == "succeeded",
                    "retried": retry_count > 0,
                    "retry_count": retry_count,
                    "degraded": degraded,
                    "binding_ids": list(binding_ids),
                    "binding_families": list(binding_families),
                    "binding_aliases": list(binding_aliases),
                    "route_ids": list(route_ids),
                    "route_families": list(route_families),
                    "task_ids": [],
                },
            )
            if audience not in artifact["audiences"]:
                artifact["audiences"].append(audience)
    return sorted(artifacts.values(), key=lambda item: item["artifact_id"])


def _route_observations(
    *,
    traces: list[dict[str, Any]],
    method_runs: Mapping[str, Any],
    outcome: str,
    retry_count: int,
    degraded: bool,
    selected_artifact_ids: list[str],
) -> list[dict[str, Any]]:
    """Project binding-health observations from trace summaries and method runs."""
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for trace in traces:
        binding_health = _normalize_binding_health(
            raw_health=trace.get("binding_health") or trace.get("route_health") or {},
            construction_identity=trace.get("construction_identity") or {},
            route_binding_authority=trace.get("binding_authority") or trace.get("route_binding_authority") or {},
            semantic_blueprint=(dict(trace.get("request_metadata") or {}).get("semantic_blueprint") or {}),
            trace_kind=str(trace.get("trace_kind") or "").strip(),
            trace_action=trace.get("action"),
            trace_method=trace.get("route_method"),
            trace_status=trace.get("status"),
        )
        construction_identity = dict(trace.get("construction_identity") or {})
        metadata = dict(trace.get("request_metadata") or {})
        trace_kind = str(trace.get("trace_kind") or "").strip()
        binding_id = str(binding_health.get("binding_id") or "").strip()
        binding_family = str(binding_health.get("binding_family") or "").strip()
        binding_alias = str(binding_health.get("binding_alias") or "").strip()
        if not binding_id and not binding_family and not binding_alias:
            continue
        key = (binding_id, binding_family, binding_alias, trace_kind)
        if key in seen:
            continue
        seen.add(key)
        instruction_resolution = dict(trace.get("instruction_resolution") or {})
        instruction_ids = [
            str(item.get("id") or "").strip()
            for item in instruction_resolution.get("effective_instructions") or []
            if str(item.get("id") or "").strip()
        ]
        instruction_ids.extend(
            [
                value
                for value in binding_health.get("effective_instruction_ids") or []
                if isinstance(value, str) and value.strip() and value not in instruction_ids
            ]
        )
        effective_instruction_count = int(
            binding_health.get("effective_instruction_count")
            or trace.get("instruction_resolution_effective_count")
            or len(instruction_ids)
        )
        hard_constraint_count = int(binding_health.get("hard_constraint_count") or 0)
        conflict_count = int(
            binding_health.get("conflict_count")
            or trace.get("instruction_resolution_conflict_count")
            or 0
        )
        observations.append(
            {
                "binding_id": binding_id,
                "binding_family": binding_family,
                "binding_alias": binding_alias,
                "route_id": str(binding_health.get("route_id") or "").strip(),
                "route_family": str(binding_health.get("route_family") or "").strip(),
                "primary_kind": str(
                    binding_health.get("primary_kind")
                    or construction_identity.get("primary_kind")
                    or ""
                ).strip(),
                "primary_label": str(
                    binding_health.get("primary_label")
                    or construction_identity.get("primary_label")
                    or binding_family
                    or binding_alias
                    or "unknown"
                ).strip(),
                "backend_binding_id": str(
                    binding_health.get("backend_binding_id")
                    or construction_identity.get("backend_binding_id")
                    or ""
                ).strip(),
                "binding_display_name": str(
                    binding_health.get("binding_display_name")
                    or construction_identity.get("binding_display_name")
                    or ""
                ).strip(),
                "binding_short_description": str(
                    binding_health.get("binding_short_description")
                    or construction_identity.get("binding_short_description")
                    or ""
                ).strip(),
                "binding_diagnostic_label": str(
                    binding_health.get("binding_diagnostic_label")
                    or construction_identity.get("binding_diagnostic_label")
                    or ""
                ).strip(),
                "route_alias": str(
                    binding_health.get("route_alias")
                    or construction_identity.get("route_alias")
                    or binding_alias
                    or ""
                ).strip(),
                "trace_kind": trace_kind or "unknown",
                "trace_status": str(trace.get("status") or "").strip(),
                "outcome": outcome,
                "success": outcome == "succeeded",
                "retried": retry_count > 0,
                "retry_count": retry_count,
                "degraded": degraded,
                "selected_artifact_ids": list(selected_artifact_ids),
                "instruction_ids": instruction_ids,
                "effective_instruction_count": effective_instruction_count,
                "hard_constraint_count": hard_constraint_count,
                "conflict_count": conflict_count,
                "task_ids": list(binding_health.get("canary_task_ids") or []),
            }
        )

    for payload in method_runs.values():
        if not isinstance(payload, Mapping):
            continue
        binding_family = str(payload.get("route_method") or "").strip()
        if not binding_family:
            continue
        key = ("", binding_family, "", "method_run")
        if key in seen:
            continue
        seen.add(key)
        observations.append(
            {
                "binding_id": "",
                "binding_family": binding_family,
                "binding_alias": "",
                "route_id": "",
                "route_family": binding_family,
                "primary_kind": "method_family",
                "primary_label": binding_family,
                "backend_binding_id": "",
                "binding_display_name": "",
                "binding_short_description": "",
                "binding_diagnostic_label": "",
                "route_alias": "",
                "trace_kind": "method_run",
                "trace_status": "ok" if bool(payload.get("success")) else "error",
                "outcome": outcome,
                "success": outcome == "succeeded",
                "retried": retry_count > 0,
                "retry_count": retry_count,
                "degraded": degraded,
                "selected_artifact_ids": list(selected_artifact_ids),
                "instruction_ids": [],
                "effective_instruction_count": 0,
                "hard_constraint_count": 0,
                "conflict_count": 0,
                "task_ids": [],
            }
        )

    return observations


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
