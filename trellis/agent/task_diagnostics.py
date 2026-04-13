"""Canonical per-task diagnosis packets and human-readable dossiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS_ROOT = ROOT / "task_runs" / "diagnostics"
DIAGNOSIS_HISTORY_ROOT = DIAGNOSIS_ROOT / "history"
DIAGNOSIS_LATEST_ROOT = DIAGNOSIS_ROOT / "latest"
DIAGNOSIS_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TaskDiagnosisArtifacts:
    """Persisted files for one per-task diagnosis packet."""

    packet: dict[str, Any]
    packet_path: Path
    dossier_path: Path
    latest_packet_path: Path
    latest_dossier_path: Path


def build_task_diagnosis_packet(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical structured diagnosis packet for one task run."""
    task = dict(record.get("task") or {})
    result = dict(record.get("result") or {})
    summary = dict(record.get("summary") or {})
    workflow = dict(record.get("workflow") or {})
    learning = dict(record.get("learning") or {})
    comparison = dict(record.get("comparison") or {})
    market = dict(record.get("market") or {})
    framework = dict(record.get("framework") or {})
    telemetry = _telemetry_section(record)
    runtime_controls = dict(result.get("runtime_controls") or {})
    method_runs = dict(record.get("method_runs") or {})
    post_build = dict(record.get("post_build") or {})
    traces = list(record.get("trace_summaries") or [])
    storage = dict(record.get("storage") or {})

    method_outcomes = _method_outcomes(method_runs)
    failure_bucket = _diagnosis_failure_bucket(
        result=result,
        summary=summary,
        method_outcomes=method_outcomes,
    )
    decision_stage = _decision_stage(
        result=result,
        workflow=workflow,
        summary=summary,
        failure_bucket=failure_bucket,
    )
    trace_index = _trace_index(traces, method_runs)
    primary_failure = _primary_failure(
        result=result,
        summary=summary,
        workflow=workflow,
        comparison=comparison,
        failure_bucket=failure_bucket,
        decision_stage=decision_stage,
        method_outcomes=method_outcomes,
    )

    outcome = {
        "success": bool(summary.get("success")),
        "status": summary.get("status"),
        "failure_bucket": failure_bucket,
        "decision_stage": decision_stage,
        "comparison_status": summary.get("comparison_status"),
        "next_action": _preferred_next_action(
            workflow.get("next_action"),
            failure_bucket=failure_bucket,
            decision_stage=decision_stage,
            success=bool(summary.get("success")),
        ),
        "headline": _headline(task=task, outcome=summary, primary_failure=primary_failure),
    }

    model_audit = _model_audit_section(method_runs, record)
    consolidated_validation = _consolidated_validation_section(method_runs)

    return {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "task": {
            "id": record.get("task_id"),
            "title": task.get("title") or record.get("task_id"),
            "kind": record.get("task_kind"),
            "status": task.get("status"),
            "construct": task.get("construct"),
        },
        "run": {
            "run_id": record.get("run_id"),
            "persisted_at": record.get("persisted_at"),
            "task_run_history_path": storage.get("history_path"),
            "task_run_latest_path": storage.get("latest_path"),
            "task_run_latest_index_path": storage.get("latest_index_path"),
        },
        "outcome": outcome,
        "primary_failure": primary_failure,
        "method_outcomes": method_outcomes,
        "trace_index": trace_index,
        "learning": learning,
        "workflow": workflow,
        "comparison": comparison,
        "market": market,
        "framework": framework,
        "telemetry": telemetry,
        "runtime_controls": runtime_controls,
        "post_build": post_build,
        "evidence": _evidence(
            result=result,
            comparison=comparison,
            method_outcomes=method_outcomes,
        ),
        "model_audit": model_audit,
        "consolidated_validation": consolidated_validation,
        "storage": storage,
    }


def _diagnosis_failure_bucket(
    *,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    method_outcomes: list[dict[str, Any]],
) -> str:
    """Override the coarse task bucket when the method-level story is clearer."""
    failure_bucket = _classify_task_result(result)
    if failure_bucket != "comparison_failed":
        return failure_bucket

    comparison_status = str(summary.get("comparison_status") or "").strip().lower()
    successful_methods = [item for item in method_outcomes if item.get("success")]
    failed_methods = [item for item in method_outcomes if not item.get("success")]
    if comparison_status not in {"", "passed"}:
        return failure_bucket
    if not successful_methods or not failed_methods:
        return failure_bucket

    build_like_buckets = {
        "build_failure",
        "import_validation",
        "semantic_validation",
        "import_hallucination",
        "implementation_gap",
        "llm_response",
        "timeout",
    }
    failed_buckets = {
        str(item.get("failure_bucket") or "").strip().lower()
        for item in failed_methods
    }
    if failed_buckets and failed_buckets <= build_like_buckets:
        return "comparator_build_failure"
    return failure_bucket


def render_task_diagnosis_dossier(packet: Mapping[str, Any]) -> str:
    """Render one diagnosis packet as a human-readable Markdown dossier."""
    task = dict(packet.get("task") or {})
    run = dict(packet.get("run") or {})
    outcome = dict(packet.get("outcome") or {})
    primary_failure = dict(packet.get("primary_failure") or {})
    learning = dict(packet.get("learning") or {})
    workflow = dict(packet.get("workflow") or {})
    comparison = dict(packet.get("comparison") or {})
    telemetry = dict(packet.get("telemetry") or {})
    runtime_controls = dict(packet.get("runtime_controls") or {})
    post_build = dict(packet.get("post_build") or {})
    evidence = dict(packet.get("evidence") or {})
    storage = dict(packet.get("storage") or {})

    lines: list[str] = [
        f"# Task diagnosis: `{task.get('id', '')}` {task.get('title', '')}",
        "",
        "## Summary",
        f"- Task kind: `{task.get('kind', '')}`",
        f"- Run id: `{run.get('run_id', '')}`",
        f"- Persisted at: `{run.get('persisted_at', '')}`",
        f"- Outcome: `{outcome.get('status', '')}`",
        f"- Failure bucket: `{outcome.get('failure_bucket', '')}`",
        f"- Decision stage: `{outcome.get('decision_stage', '')}`",
        f"- Headline: {outcome.get('headline', '')}",
        f"- Next action: {outcome.get('next_action', '')}",
        "",
        "## Primary Diagnosis",
        f"- Likely cause: {primary_failure.get('likely_cause', '')}",
        f"- Confidence: `{primary_failure.get('confidence', '')}`",
    ]
    signals = list(primary_failure.get("signals") or [])
    if signals:
        lines.append("- Signals:")
        lines.extend(f"  - {signal}" for signal in signals)
    else:
        lines.append("- Signals: none recorded")
    lines.append("")

    if comparison:
        lines.extend(
            [
                "## Comparison",
                f"- Comparison status: `{comparison.get('summary', {}).get('status', comparison.get('status', ''))}`",
                f"- Reference target: `{comparison.get('summary', {}).get('reference_target', comparison.get('reference_target', ''))}`",
            ]
        )
        prices = dict(comparison.get("summary", {}).get("prices") or comparison.get("prices") or {})
        if prices:
            lines.append("- Prices:")
            for key, value in prices.items():
                lines.append(f"  - `{key}`: `{_format_value(value)}`")
        deviations = dict(comparison.get("summary", {}).get("deviations_pct") or comparison.get("deviations_pct") or {})
        if deviations:
            lines.append("- Deviations pct:")
            for key, value in deviations.items():
                lines.append(f"  - `{key}`: `{_format_value(value)}`")
        lines.append("")

    if runtime_controls:
        lines.extend(
            [
                "## Runtime Controls",
                f"- Skip post-build reflection: `{runtime_controls.get('skip_post_build_reflection', '')}`",
                f"- Skip post-build consolidation: `{runtime_controls.get('skip_post_build_consolidation', '')}`",
                f"- Skip diagnosis persist: `{runtime_controls.get('skip_task_diagnosis_persist', '')}`",
                f"- LLM wait log path: `{runtime_controls.get('llm_wait_log_path', '') or 'not set'}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Method Outcomes",
            *_render_table(
                headers=("Method", "Success", "Attempts", "Bucket", "Post-build", "Trace"),
                rows=[
                    (
                        outcome_row.get("method"),
                        _yes_no(outcome_row.get("success")),
                        outcome_row.get("attempts"),
                        outcome_row.get("failure_bucket"),
                        outcome_row.get("post_build_latest_phase"),
                        outcome_row.get("trace_path"),
                    )
                    for outcome_row in packet.get("method_outcomes", [])
                ],
            ),
            "",
            "## Trace Index",
            *_render_table(
                headers=("Scope", "Name", "Status", "Construction", "Latest Event", "Path"),
                rows=[
                    (
                        trace_row.get("scope"),
                        trace_row.get("name"),
                        trace_row.get("status"),
                        trace_row.get("construction_label"),
                        trace_row.get("latest_event"),
                        trace_row.get("path"),
                    )
                    for trace_row in packet.get("trace_index", [])
                ],
            ),
            "",
            "## Learning",
            f"- Knowledge outcome: `{learning.get('knowledge_outcome', '')}`",
            f"- Knowledge outcome reason: {learning.get('knowledge_outcome_reason', '')}",
            f"- Lessons captured: `{_join_or_none(learning.get('captured_lesson_ids'))}`",
            f"- Lessons retrieved: `{_join_or_none(learning.get('retrieved_lesson_ids'))}`",
            f"- Retrieval stages: `{_join_or_none(learning.get('retrieval_stages'))}`",
            f"- Retrieval sources: `{_join_or_none(learning.get('retrieval_sources'))}`",
            f"- Selected artifacts: `{_join_or_none(learning.get('selected_artifact_ids'))}`",
            f"- Cookbook paths: `{_join_or_none(learning.get('cookbook_candidate_paths'))}`",
            f"- Promotion candidate paths: `{_join_or_none(learning.get('promotion_candidate_paths'))}`",
            "",
            "## Skill Telemetry",
            f"- Run outcome: `{telemetry.get('run_outcome', '')}`",
            f"- Retries beyond first attempt: `{telemetry.get('retry_count', 0)}`",
            f"- Degraded: `{_yes_no(telemetry.get('degraded'))}`",
        ]
    )
    selected_artifacts = list(telemetry.get("selected_artifacts") or [])
    if selected_artifacts:
        lines.extend(
            [
                *_render_table(
                    headers=("Artifact", "Kind", "Audience", "Outcome", "Routes"),
                    rows=[
                        (
                            item.get("artifact_id"),
                            item.get("kind"),
                            _join_or_none(item.get("audiences")),
                            item.get("outcome"),
                            _join_or_none(item.get("route_ids")),
                        )
                        for item in selected_artifacts
                    ],
                ),
            ]
        )
    else:
        lines.append("- Selected artifacts: none recorded")
    route_observations = list(telemetry.get("route_observations") or [])
    if route_observations:
        lines.extend(
            [
                *_render_table(
                    headers=("Primary", "Diagnostic", "Outcome", "Health", "Trace"),
                    rows=[
                        (
                            item.get("primary_label"),
                            item.get("binding_diagnostic_label")
                            or item.get("route_alias")
                            or item.get("route_family"),
                            item.get("outcome"),
                            (
                                "effective="
                                f"{item.get('effective_instruction_count', 0)}, "
                                "hard="
                                f"{item.get('hard_constraint_count', 0)}, "
                                "conflicts="
                                f"{item.get('conflict_count', 0)}"
                            ),
                            item.get("trace_kind"),
                        )
                        for item in route_observations
                    ],
                ),
                "",
                "## Evidence",
            ]
        )
    else:
        lines.extend(
            [
                "- Route observations: none recorded",
                "",
                "## Evidence",
            ]
        )
    for label in (
        ("Top-level error", evidence.get("top_level_error")),
        ("Top-level failures", evidence.get("top_level_failures")),
        ("Blocker details", evidence.get("blocker_details")),
        ("Task contract error", evidence.get("task_contract_error")),
        ("Cross-validation", evidence.get("cross_validation")),
        ("Knowledge gaps", evidence.get("knowledge_gaps")),
    ):
        lines.append(f"- {label[0]}: {_stringify(label[1])}")
    model_audit = dict(packet.get("model_audit") or {})
    if model_audit.get("audit_record_paths"):
        lines.append("")
        lines.append("## Model Audit")
        env = dict(model_audit.get("environment") or {})
        lines.append(f"- Repo revision: `{env.get('repo_revision', '')}`")
        lines.append(f"- Knowledge hash: `{env.get('knowledge_hash', '')}`")
        models_used = env.get("llm_models_used") or []
        lines.append(f"- LLM models used: `{', '.join(models_used) or 'unknown'}`")
        lines.append("- Audit records:")
        for path in model_audit.get("audit_record_paths") or []:
            lines.append(f"  - `{path}`")

    lines.extend(
        [
            "",
            "## Post-build",
            f"- Latest phase: `{post_build.get('latest_phase', '')}`",
            f"- Latest status: `{post_build.get('latest_status', '')}`",
            f"- Latest method: `{post_build.get('latest_method', '')}`",
            f"- Active flags: `{_join_or_none(sorted(flag for flag, enabled in dict(post_build.get('active_flags') or {}).items() if enabled))}`",
        ]
    )
    method_post_build = dict(post_build.get("methods") or {})
    if method_post_build:
        lines.append("- Method checkpoints:")
        for method, summary in method_post_build.items():
            summary = dict(summary or {})
            lines.append(
                "  - "
                f"`{method}`: phase=`{summary.get('latest_phase', '')}`, "
                f"status=`{summary.get('latest_status', '')}`, "
                f"events=`{summary.get('event_count', '')}`"
            )

    lines.extend(
        [
            "",
            "## Workflow",
            f"- Status: `{workflow.get('status', '')}`",
            f"- Next action: {workflow.get('next_action', '')}",
            f"- Latest trace request id: `{dict(workflow.get('latest_trace') or {}).get('request_id', '')}`",
            "",
            "## Storage",
            f"- Run record: `{storage.get('history_path', '')}`",
            f"- Latest run record: `{storage.get('latest_path', '')}`",
            f"- Diagnosis packet: `{storage.get('diagnosis_history_packet_path', '')}`",
            f"- Diagnosis dossier: `{storage.get('diagnosis_history_dossier_path', '')}`",
            f"- Latest diagnosis packet: `{storage.get('diagnosis_latest_packet_path', '')}`",
            f"- Latest diagnosis dossier: `{storage.get('diagnosis_latest_dossier_path', '')}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_task_diagnosis_artifacts(
    record: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> TaskDiagnosisArtifacts:
    """Persist a packet and dossier alongside the run record."""
    packet = build_task_diagnosis_packet(record)
    task_id = str(packet.get("task", {}).get("id") or record.get("task_id") or "").strip()
    run_id = str(packet.get("run", {}).get("run_id") or record.get("run_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is required to persist diagnosis artifacts")
    if not run_id:
        raise ValueError("run_id is required to persist diagnosis artifacts")

    history_root = root / "task_runs" / "diagnostics" / "history" / task_id
    latest_root = root / "task_runs" / "diagnostics" / "latest"
    history_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    packet_path = history_root / f"{run_id}.json"
    dossier_path = history_root / f"{run_id}.md"
    latest_packet_path = latest_root / f"{task_id}.json"
    latest_dossier_path = latest_root / f"{task_id}.md"

    storage = dict(packet.get("storage") or {})
    storage.update(
        {
            "diagnosis_history_packet_path": str(packet_path),
            "diagnosis_history_dossier_path": str(dossier_path),
            "diagnosis_latest_packet_path": str(latest_packet_path),
            "diagnosis_latest_dossier_path": str(latest_dossier_path),
        }
    )
    packet["storage"] = storage
    packet["run"]["task_run_history_path"] = storage.get("history_path")
    packet["run"]["task_run_latest_path"] = storage.get("latest_path")
    packet["run"]["task_run_latest_index_path"] = storage.get("latest_index_path")
    packet["run"]["diagnosis_history_packet_path"] = storage.get("diagnosis_history_packet_path")
    packet["run"]["diagnosis_history_dossier_path"] = storage.get("diagnosis_history_dossier_path")
    packet["run"]["diagnosis_latest_packet_path"] = storage.get("diagnosis_latest_packet_path")
    packet["run"]["diagnosis_latest_dossier_path"] = storage.get("diagnosis_latest_dossier_path")

    json_payload = json.dumps(packet, indent=2, default=str)
    dossier = render_task_diagnosis_dossier(packet)
    packet_path.write_text(json_payload)
    dossier_path.write_text(dossier)
    latest_packet_path.write_text(json_payload)
    latest_dossier_path.write_text(dossier)
    return TaskDiagnosisArtifacts(
        packet=packet,
        packet_path=packet_path,
        dossier_path=dossier_path,
        latest_packet_path=latest_packet_path,
        latest_dossier_path=latest_dossier_path,
    )


def _model_audit_section(
    method_runs: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract audit record paths and environment snapshot from method run data."""
    audit_paths: list[str] = []
    llm_models: set[str] = set()
    repo_revisions: set[str] = set()
    knowledge_hashes: set[str] = set()

    for payload in method_runs.values():
        if not isinstance(payload, Mapping):
            continue
        path = payload.get("audit_record_path")
        if path:
            audit_paths.append(str(path))
        env = dict(payload.get("audit_environment") or {})
        if env.get("llm_model_id"):
            llm_models.add(str(env["llm_model_id"]))
        if env.get("repo_revision"):
            repo_revisions.add(str(env["repo_revision"]))
        if env.get("knowledge_hash"):
            knowledge_hashes.add(str(env["knowledge_hash"]))

    # Also accept top-level audit_record_paths list (populated by task_run_store)
    top_level = list(record.get("audit_record_paths") or [])
    for p in top_level:
        if str(p) not in audit_paths:
            audit_paths.append(str(p))

    return {
        "audit_record_paths": audit_paths,
        "environment": {
            "repo_revision": next(iter(repo_revisions), None),
            "knowledge_hash": next(iter(knowledge_hashes), None),
            "llm_models_used": sorted(llm_models),
        },
    }


def _consolidated_validation_section(
    method_runs: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize per-gate validation outcomes for each method."""
    result: dict[str, Any] = {}
    for method, payload in method_runs.items():
        if not isinstance(payload, Mapping):
            continue
        gates_raw = list(payload.get("validation_gates") or [])
        gates = [
            {
                "gate": g.get("gate") if isinstance(g, Mapping) else str(g),
                "passed": bool(g.get("passed")) if isinstance(g, Mapping) else False,
                "issue_count": len(g.get("issues") or []) if isinstance(g, Mapping) else 0,
            }
            for g in gates_raw
        ]
        all_passed = all(g["passed"] for g in gates) if gates else None
        result[str(method)] = {"gates": gates, "all_passed": all_passed}
    return result


def _telemetry_section(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return telemetry from the persisted record or derive a minimal fallback view."""
    telemetry = record.get("telemetry")
    if isinstance(telemetry, Mapping) and telemetry:
        return _enrich_telemetry_route_observations(dict(telemetry), record)

    learning = dict(record.get("learning") or {})
    workflow = dict(record.get("workflow") or {})
    selected_artifacts = []
    titles = {
        str(artifact_id): str(title)
        for artifact_id, title in zip(
            learning.get("selected_artifact_ids") or [],
            learning.get("selected_artifact_titles") or [],
        )
    }
    by_audience = dict(learning.get("selected_artifacts_by_audience") or {})
    if by_audience:
        seen: set[str] = set()
        for audience, artifacts in by_audience.items():
            if not isinstance(artifacts, list):
                continue
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                artifact_id = str(artifact.get("id") or "").strip()
                if not artifact_id or artifact_id in seen:
                    continue
                seen.add(artifact_id)
                selected_artifacts.append(
                    {
                        "artifact_id": artifact_id,
                        "title": str(artifact.get("title") or "").strip(),
                        "kind": str(artifact.get("kind") or "").strip(),
                        "audiences": [str(audience)],
                        "outcome": str(workflow.get("status") or "").strip(),
                        "success": bool(record.get("summary", {}).get("success")),
                        "retried": False,
                        "retry_count": 0,
                        "degraded": False,
                        "route_ids": [],
                        "route_families": [],
                    }
                )
    else:
        selected_artifacts = [
            {
                "artifact_id": str(artifact_id),
                "title": titles.get(str(artifact_id), ""),
                "kind": "",
                "audiences": [],
                "outcome": str(workflow.get("status") or "").strip(),
                "success": bool(record.get("summary", {}).get("success")),
                "retried": False,
                "retry_count": 0,
                "degraded": False,
                "route_ids": [],
                "route_families": [],
            }
            for artifact_id in learning.get("selected_artifact_ids") or []
        ]

    route_observations = []
    for trace in record.get("trace_summaries") or []:
        if not isinstance(trace, Mapping):
            continue
        route_health = dict(trace.get("route_health") or {})
        construction_identity = dict(trace.get("construction_identity") or {})
        route_binding_authority = dict(trace.get("route_binding_authority") or {})
        trace_kind = str(trace.get("trace_kind") or "").strip()
        route_id = str(route_health.get("route_id") or "").strip()
        if not route_id and trace_kind != "platform":
            route_id = str(trace.get("action") or "").strip()
        route_family = str(
            route_health.get("route_family")
            or trace.get("route_method")
            or ""
        ).strip()
        if not route_id and not route_family:
            continue
        route_observations.append(
            {
                "route_id": route_id,
                "route_family": route_family,
                "primary_kind": str(
                    route_health.get("primary_kind")
                    or construction_identity.get("primary_kind")
                    or ""
                ).strip(),
                "primary_label": str(
                    route_health.get("primary_label")
                    or construction_identity.get("primary_label")
                    or route_family
                    or route_id
                    or "unknown"
                ).strip(),
                "backend_binding_id": str(
                    route_health.get("backend_binding_id")
                    or construction_identity.get("backend_binding_id")
                    or ""
                ).strip(),
                "binding_display_name": str(
                    route_health.get("binding_display_name")
                    or construction_identity.get("binding_display_name")
                    or ""
                ).strip(),
                "binding_short_description": str(
                    route_health.get("binding_short_description")
                    or construction_identity.get("binding_short_description")
                    or ""
                ).strip(),
                "binding_diagnostic_label": str(
                    route_health.get("binding_diagnostic_label")
                    or construction_identity.get("binding_diagnostic_label")
                    or ""
                ).strip(),
                "route_alias": str(
                    route_health.get("route_alias")
                    or construction_identity.get("route_alias")
                    or route_id
                    or ""
                ).strip(),
                "trace_kind": str(trace.get("trace_kind") or "").strip(),
                "trace_status": str(trace.get("status") or "").strip(),
                "outcome": str(workflow.get("status") or "").strip(),
                "success": bool(record.get("summary", {}).get("success")),
                "retried": False,
                "retry_count": 0,
                "degraded": False,
                "selected_artifact_ids": [
                    item.get("artifact_id")
                    for item in selected_artifacts
                    if item.get("artifact_id")
                ],
                "instruction_ids": list(route_health.get("effective_instruction_ids") or []),
                "effective_instruction_count": int(route_health.get("effective_instruction_count") or 0),
                "hard_constraint_count": int(route_health.get("hard_constraint_count") or 0),
                "conflict_count": int(route_health.get("conflict_count") or 0),
                "task_ids": list(
                    route_binding_authority.get("canary_task_ids")
                    or route_health.get("canary_task_ids")
                    or []
                ),
            }
        )

    return {
        "task_kind": str(record.get("task_kind") or "").strip(),
        "run_outcome": str(workflow.get("status") or "").strip(),
        "retried": False,
        "retry_count": 0,
        "degraded": False,
        "comparison_status": str(record.get("summary", {}).get("comparison_status") or "").strip(),
        "selected_artifacts": selected_artifacts,
        "route_observations": route_observations,
    }


def _enrich_telemetry_route_observations(
    telemetry: dict[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Backfill route-authority canary IDs into persisted telemetry when available."""
    traces = list(record.get("trace_summaries") or [])
    route_task_ids: dict[tuple[str, str], list[str]] = {}
    route_construction: dict[tuple[str, str], dict[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        route_health = dict(trace.get("route_health") or {})
        construction_identity = dict(trace.get("construction_identity") or {})
        route_binding_authority = dict(trace.get("route_binding_authority") or {})
        trace_kind = str(trace.get("trace_kind") or "").strip()
        route_id = str(route_health.get("route_id") or "").strip()
        if not route_id and trace_kind != "platform":
            route_id = str(trace.get("action") or "").strip()
        route_family = str(route_health.get("route_family") or trace.get("route_method") or "").strip()
        task_ids = list(
            route_binding_authority.get("canary_task_ids")
            or route_health.get("canary_task_ids")
            or []
        )
        if route_id or route_family:
            route_task_ids[(route_id, route_family)] = task_ids
            route_construction[(route_id, route_family)] = construction_identity

    observations = []
    for observation in telemetry.get("route_observations") or []:
        if not isinstance(observation, Mapping):
            continue
        item = dict(observation)
        route_id = str(item.get("route_id") or "").strip()
        route_family = str(item.get("route_family") or "").strip()
        construction_identity = dict(route_construction.get((route_id, route_family)) or {})
        item["primary_kind"] = str(
            item.get("primary_kind")
            or construction_identity.get("primary_kind")
            or ""
        ).strip()
        item["primary_label"] = str(
            item.get("primary_label")
            or construction_identity.get("primary_label")
            or route_family
            or route_id
            or "unknown"
        ).strip()
        item["backend_binding_id"] = str(
            item.get("backend_binding_id")
            or construction_identity.get("backend_binding_id")
            or ""
        ).strip()
        item["binding_display_name"] = str(
            item.get("binding_display_name")
            or construction_identity.get("binding_display_name")
            or ""
        ).strip()
        item["binding_short_description"] = str(
            item.get("binding_short_description")
            or construction_identity.get("binding_short_description")
            or ""
        ).strip()
        item["binding_diagnostic_label"] = str(
            item.get("binding_diagnostic_label")
            or construction_identity.get("binding_diagnostic_label")
            or ""
        ).strip()
        item["route_alias"] = str(
            item.get("route_alias")
            or construction_identity.get("route_alias")
            or route_id
            or ""
        ).strip()
        item["task_ids"] = list(
            item.get("task_ids")
            or route_task_ids.get((route_id, route_family))
            or []
        )
        observations.append(item)
    telemetry["route_observations"] = observations
    return telemetry


def _primary_failure(
    *,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    workflow: Mapping[str, Any],
    comparison: Mapping[str, Any],
    failure_bucket: str,
    decision_stage: str,
    method_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    signals = _failure_signals(result=result, method_outcomes=method_outcomes)
    likely_cause = _likely_cause(
        result=result,
        workflow=workflow,
        comparison=comparison,
        failure_bucket=failure_bucket,
        method_outcomes=method_outcomes,
    )
    confidence = _confidence_for(failure_bucket=failure_bucket, signals=signals)
    headline = _headline_from_bucket(failure_bucket, decision_stage, result, method_outcomes)
    if summary.get("comparison_status") and summary.get("comparison_status") != "passed":
        signals.append(f"comparison_status={summary.get('comparison_status')}")
    return {
        "bucket": failure_bucket,
        "stage": decision_stage,
        "headline": headline,
        "likely_cause": likely_cause,
        "confidence": confidence,
        "signals": _unique_strings(signals),
    }


def _decision_stage(
    *,
    result: Mapping[str, Any],
    workflow: Mapping[str, Any],
    summary: Mapping[str, Any],
    failure_bucket: str,
) -> str:
    if summary.get("success") or result.get("success"):
        return "completed"
    if result.get("task_contract_error"):
        return "task_contract"
    if result.get("blocker_details"):
        return "blocked"
    comparison_status = str(summary.get("comparison_status") or "").strip().lower()
    if comparison_status and comparison_status != "passed":
        return "comparison"
    if failure_bucket == "comparator_build_failure":
        return "comparison"
    if failure_bucket in {"missing_market_data"}:
        return "market_data"
    if failure_bucket in {"missing_cookbook", "missing_decomposition", "import_hallucination"}:
        return "knowledge"
    if failure_bucket in {"implementation_gap"}:
        return "implementation"
    if failure_bucket in {"validation_failure"}:
        return "validation"
    if failure_bucket in {"llm_response", "timeout", "rate_limit"}:
        return "infrastructure"
    if str(workflow.get("status") or "").strip().lower() == "blocked":
        return "blocked"
    return "build"


def _method_outcomes(method_runs: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for method, payload in method_runs.items():
        if not isinstance(payload, Mapping):
            continue
        trace = dict(payload.get("trace_summary") or {})
        outcome = {
            "method": method,
            "success": bool(payload.get("success")),
            "attempts": int(payload.get("attempts") or 0),
            "preferred_method": payload.get("preferred_method"),
            "failure_bucket": _classify_task_result(payload),
            "route_method": trace.get("route_method") or payload.get("route_method"),
            "trace_path": payload.get("platform_trace_path"),
            "trace_status": trace.get("status"),
            "latest_event": trace.get("latest_event"),
            "comparison_status": (payload.get("cross_validation") or {}).get("status"),
            "token_usage": dict(payload.get("token_usage_summary") or {}),
            "error": payload.get("error"),
            "failures": list(payload.get("failures") or []),
            "post_build_latest_phase": dict(payload.get("post_build_tracking") or {}).get("last_phase"),
            "post_build_latest_status": dict(payload.get("post_build_tracking") or {}).get("last_status"),
        }
        outcomes.append(outcome)
    return outcomes


def _trace_index(
    traces: list[dict[str, Any]],
    method_runs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_entry(*, scope: str, name: str, ordinal: int, trace: Mapping[str, Any]) -> None:
        path = str(trace.get("path") or "").strip()
        if not path or path in seen:
            return
        seen.add(path)
        construction_identity = dict(trace.get("construction_identity") or {})
        entries.append(
            {
                "scope": scope,
                "name": name,
                "ordinal": ordinal,
                "path": path,
                "trace_kind": trace.get("trace_kind"),
                "request_id": trace.get("request_id"),
                "status": trace.get("status"),
                "outcome": trace.get("outcome"),
                "route_method": trace.get("route_method"),
                "construction_label": (
                    str(construction_identity.get("primary_label") or "").strip()
                    or str(trace.get("route_method") or "").strip()
                ),
                "latest_event": trace.get("latest_event"),
                "latest_event_status": trace.get("latest_event_status"),
                "updated_at": trace.get("updated_at"),
            }
        )

    for ordinal, trace in enumerate(traces, start=1):
        if isinstance(trace, Mapping):
            add_entry(scope="task", name="task trace", ordinal=ordinal, trace=trace)

    for ordinal, (method, payload) in enumerate(method_runs.items(), start=1):
        if not isinstance(payload, Mapping):
            continue
        trace = payload.get("trace_summary")
        if isinstance(trace, Mapping):
            add_entry(scope="method", name=str(method), ordinal=ordinal, trace=trace)
    return entries


def _evidence(
    *,
    result: Mapping[str, Any],
    comparison: Mapping[str, Any],
    method_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    blocker_details = result.get("blocker_details") or {}
    task_contract_error = result.get("task_contract_error") or {}
    cross_validation = result.get("cross_validation") or {}
    return {
        "top_level_error": result.get("error"),
        "top_level_failures": list(result.get("failures") or []),
        "blocker_details": dict(blocker_details) if isinstance(blocker_details, Mapping) else blocker_details,
        "task_contract_error": dict(task_contract_error) if isinstance(task_contract_error, Mapping) else task_contract_error,
        "cross_validation": dict(cross_validation) if isinstance(cross_validation, Mapping) else cross_validation,
        "knowledge_gaps": list(result.get("knowledge_gaps") or []),
        "gap_confidence": result.get("gap_confidence"),
        "method_failures": [
            {
                "method": item.get("method"),
                "failure_bucket": item.get("failure_bucket"),
                "error": item.get("error"),
                "failures": item.get("failures"),
                "trace_path": item.get("trace_path"),
                "post_build_latest_phase": item.get("post_build_latest_phase"),
                "post_build_latest_status": item.get("post_build_latest_status"),
            }
            for item in method_outcomes
            if not item.get("success")
        ],
        "comparison": {
            "status": comparison.get("status"),
            "reference_target": comparison.get("reference_target"),
            "prices": dict(comparison.get("prices") or {}),
            "deviations_pct": dict(comparison.get("deviations_pct") or {}),
        },
    }


def _failure_signals(
    *,
    result: Mapping[str, Any],
    method_outcomes: list[dict[str, Any]],
) -> list[str]:
    signals: list[str] = []
    if result.get("error"):
        signals.extend(_signal_strings(result.get("error")))
    signals.extend(_signal_strings(result.get("failures")))
    signals.extend(_signal_strings(result.get("blocker_details")))
    task_contract_error = result.get("task_contract_error") or {}
    if task_contract_error:
        signals.extend(_signal_strings(task_contract_error))
    cross_validation = result.get("cross_validation") or {}
    if isinstance(cross_validation, Mapping):
        status = str(cross_validation.get("status") or "").strip()
        if status and status != "passed":
            signals.append(f"cross_validation status={status}")
    for item in method_outcomes:
        if not item.get("success"):
            signals.append(f"{item.get('method')}: {item.get('failure_bucket')}")
            signals.extend(_signal_strings(item.get("error")))
            signals.extend(_signal_strings(item.get("failures")))
            if item.get("comparison_status") and item.get("comparison_status") != "passed":
                signals.append(f"{item.get('method')}: comparison_status={item.get('comparison_status')}")
    return _unique_strings(signals)


def _likely_cause(
    *,
    result: Mapping[str, Any],
    workflow: Mapping[str, Any],
    comparison: Mapping[str, Any],
    failure_bucket: str,
    method_outcomes: list[dict[str, Any]],
) -> str:
    comparison_status = str(
        (result.get("cross_validation") or {}).get("status")
        or (comparison.get("summary") or {}).get("status")
        or comparison.get("status")
        or ""
    ).strip().lower()
    if result.get("task_contract_error"):
        return "The task contract itself is invalid, so the run failed before a usable pricing path could be assembled."
    if result.get("blocker_details"):
        return "The run was intentionally blocked by explicit blocker details."
    if failure_bucket == "comparator_build_failure":
        failed_methods = [item.get("method") for item in method_outcomes if not item.get("success")]
        if failed_methods:
            return (
                "One comparison/comparator lane failed to build while other methods completed; "
                f"broken lanes: {', '.join(str(method) for method in failed_methods if method)}."
            )
        return "One comparison/comparator lane failed to build while the rest of the task completed."
    if failure_bucket == "comparison_failure" or comparison_status not in {"", "passed"}:
        failed_methods = [item.get("method") for item in method_outcomes if not item.get("success")]
        if failed_methods:
            return (
                "The comparison task did not produce a valid enough method set to compare; "
                f"failing methods: {', '.join(str(method) for method in failed_methods if method)}."
            )
        return "The comparison task did not produce enough valid results to compare."

    if failure_bucket == "import_hallucination":
        return "The generated code imported a path or symbol that Trellis does not expose."
    if failure_bucket == "missing_market_data":
        return "The task could not assemble the required market state or data capability."
    if failure_bucket == "missing_cookbook":
        return "The planner did not have a cookbook route for the requested method."
    if failure_bucket == "missing_decomposition":
        return "The semantic decomposition for the request was still too weak or incomplete."
    if failure_bucket == "implementation_gap":
        return "The repository is missing a primitive, symbol, or route contract that the build expected."
    if failure_bucket == "validation_failure":
        return "The build ran, but validation rejected the produced result."
    if failure_bucket == "llm_response":
        return "The model returned malformed or unusable output."
    if failure_bucket == "timeout":
        return "One or more steps timed out before producing a usable result."
    if failure_bucket == "rate_limit":
        return "An upstream provider rate limit interrupted the run."
    if str(workflow.get("status") or "").strip().lower() == "blocked":
        return "The task was blocked by the current workflow state."
    return "The run failed without a narrower bucketed diagnosis."


def _confidence_for(*, failure_bucket: str, signals: list[str]) -> str:
    if failure_bucket in {"import_hallucination", "missing_market_data", "llm_response", "timeout", "rate_limit"}:
        return "high"
    if failure_bucket == "comparator_build_failure":
        return "high"
    if failure_bucket in {"comparison_failure", "validation_failure", "implementation_gap"}:
        return "medium"
    if signals:
        return "medium"
    return "low"


def _headline(
    *,
    task: Mapping[str, Any],
    outcome: Mapping[str, Any],
    primary_failure: Mapping[str, Any],
) -> str:
    title = str(task.get("title") or task.get("id") or "task").strip()
    if bool(outcome.get("success")):
        return f"{title} completed successfully."
    return f"{title} failed in `{primary_failure.get('stage', '')}` with `{primary_failure.get('bucket', '')}`."


def _headline_from_bucket(
    failure_bucket: str,
    decision_stage: str,
    result: Mapping[str, Any],
    method_outcomes: list[dict[str, Any]],
) -> str:
    if result.get("success"):
        return "Run completed successfully."
    if failure_bucket == "comparator_build_failure":
        failing_methods = [item.get("method") for item in method_outcomes if not item.get("success")]
        if failing_methods:
            return f"Comparator build failed for {', '.join(str(method) for method in failing_methods if method)}."
        return "Comparator build failed before the task could finish."
    if failure_bucket == "comparison_failure":
        failing_methods = [item.get("method") for item in method_outcomes if not item.get("success")]
        if failing_methods:
            return f"Comparison failed after {len(failing_methods)} method failure(s)."
        return "Comparison failed before enough methods returned valid results."
    if failure_bucket == "implementation_gap":
        return f"Implementation gap surfaced during {decision_stage}."
    if failure_bucket == "missing_market_data":
        return f"Missing market data surfaced during {decision_stage}."
    if failure_bucket == "validation_failure":
        return f"Validation failed during {decision_stage}."
    if failure_bucket == "timeout":
        return f"Timeout surfaced during {decision_stage}."
    if failure_bucket == "llm_response":
        return f"Model response failed during {decision_stage}."
    return f"Task failed during {decision_stage}."


def _default_next_action(*, failure_bucket: str, decision_stage: str) -> str:
    if failure_bucket == "comparator_build_failure":
        return "Repair the failing comparator route or scaffold, then rerun the broken comparator lane before rerunning the full task."
    if failure_bucket == "comparison_failure":
        return "Inspect which methods failed to produce valid results, then rerun the narrowest broken route."
    if failure_bucket == "implementation_gap":
        return "Fix the missing primitive or symbol in the implementation path and rerun the task."
    if failure_bucket == "missing_market_data":
        return "Fill in the missing market capability or mock data before rerunning."
    if failure_bucket in {"import_hallucination", "missing_cookbook", "missing_decomposition"}:
        return "Update the knowledge layer so the planner can choose an exact route and import path."
    if failure_bucket in {"llm_response", "timeout", "rate_limit"}:
        return "Rerun after the model or provider issue clears."
    if decision_stage == "completed":
        return "No action required."
    return "Inspect the recorded evidence and rerun once the blocking issue is resolved."


def _preferred_next_action(
    workflow_next_action: Any,
    *,
    failure_bucket: str,
    decision_stage: str,
    success: bool,
) -> str:
    """Prefer the diagnosis-specific next step when the workflow text is generic."""
    text = str(workflow_next_action or "").strip()
    if success:
        if text and not _is_generic_next_action(text):
            return text
        return "No action required."
    if text and not _is_generic_next_action(text):
        return text
    return _default_next_action(failure_bucket=failure_bucket, decision_stage=decision_stage)


def _render_table(*, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    if not rows:
        return ["- None"]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [header_row, separator_row]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_escape_table_cell(cell) for cell in row)
            + " |"
        )
    return lines


def _is_generic_next_action(text: str) -> bool:
    normalized = text.strip().lower()
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "no automated follow-up is active yet",
            "tracked externally via linked issues",
            "review the framework task summary",
            "review comparison prices and deviations",
            "inspect which methods failed to produce valid results",
            "completed successfully",
        )
    )


def _escape_table_cell(value: Any) -> str:
    text = _stringify(value)
    return text.replace("|", "\\|")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set)):
        parts = [part for part in (_stringify(item) for item in value) if part]
        return ", ".join(parts)
    return str(value)


def _signal_strings(value: Any) -> list[str]:
    """Flatten one evidence value into whole-string signals."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        text = json.dumps(value, sort_keys=True, default=str)
        return [text] if text and text != "{}" else []
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(_signal_strings(item))
        return parts
    text = str(value).strip()
    return [text] if text else []


def _join_or_none(values: Any) -> str:
    if isinstance(values, (list, tuple, set)):
        items = [str(item).strip() for item in values if str(item).strip()]
        return ", ".join(items) if items else "none"
    text = str(values or "").strip()
    return text or "none"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _classify_task_result(result: Mapping[str, Any]) -> str:
    """Lazy import the shared task-result classifier to avoid import cycles."""
    from trellis.agent.evals import classify_task_result

    return classify_task_result(result)


__all__ = [
    "DIAGNOSIS_HISTORY_ROOT",
    "DIAGNOSIS_LATEST_ROOT",
    "DIAGNOSIS_ROOT",
    "DIAGNOSIS_SCHEMA_VERSION",
    "TaskDiagnosisArtifacts",
    "build_task_diagnosis_packet",
    "render_task_diagnosis_dossier",
    "save_task_diagnosis_artifacts",
]
