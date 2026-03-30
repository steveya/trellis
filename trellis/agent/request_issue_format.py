"""Shared formatting helpers for request-audit Linear and GitHub issues."""

from __future__ import annotations

from typing import Any, Mapping


def build_issue_title(trace: dict[str, Any], compiled_request) -> str:
    """Build an informative issue title with task and method context when available."""
    parts: list[str] = []

    task_label = _task_label(compiled_request)
    if task_label:
        parts.append(task_label)
    else:
        parts.append(_request_label(trace, compiled_request))

    comparison_target = _metadata_value(compiled_request, trace, "comparison_target")
    if comparison_target:
        parts.append(str(comparison_target))

    route_method = trace.get("route_method") or _metadata_value(compiled_request, trace, "preferred_method")
    if route_method:
        parts.append(str(route_method))

    entry_point = getattr(compiled_request.request, "entry_point", None)
    request_type = getattr(compiled_request.request, "request_type", None)
    if entry_point or request_type:
        parts.append("/".join(part for part in (entry_point, request_type) if part))

    title = "Trellis audit: " + " · ".join(_compact(part, 72) for part in parts if part)
    return _compact(title, 180)


def build_issue_body(
    trace: dict[str, Any],
    compiled_request,
    event_record: dict[str, Any] | None = None,
) -> str:
    """Render the initial issue body from request, task, trace, and trigger metadata."""
    request = compiled_request.request
    lines: list[str] = ["## Why This Issue Exists"]

    if event_record is not None:
        lines.append(f"- trigger_event: `{event_record.get('event', 'unknown')}`")
        lines.append(f"- trigger_status: `{event_record.get('status', 'info')}`")

    task_id = _metadata_value(compiled_request, trace, "task_id")
    task_title = _metadata_value(compiled_request, trace, "task_title")
    if task_id or task_title:
        task_line = f"`{task_id}`" if task_id else "`unknown`"
        if task_title:
            task_line += f" — {task_title}"
        lines.append(f"- task: {task_line}")

    comparison_target = _metadata_value(compiled_request, trace, "comparison_target")
    if comparison_target:
        lines.append(f"- comparison_target: `{comparison_target}`")

    preferred_method = _metadata_value(compiled_request, trace, "preferred_method")
    if preferred_method:
        lines.append(f"- preferred_method: `{preferred_method}`")

    semantic_role_ownership = _metadata_value(compiled_request, trace, "semantic_role_ownership")
    if isinstance(semantic_role_ownership, Mapping) and semantic_role_ownership:
        lines.extend(["", "## Ownership"])
        lines.append(
            f"- selected_stage: `{semantic_role_ownership.get('selected_stage', '')}`"
        )
        lines.append(
            f"- selected_role: `{semantic_role_ownership.get('selected_role', '')}`"
        )
        lines.append(
            f"- trigger_condition: `{semantic_role_ownership.get('trigger_condition', '')}`"
        )
        lines.append(
            f"- artifact_kind: `{semantic_role_ownership.get('artifact_kind', '')}`"
        )

    route_method = trace.get("route_method")
    if route_method:
        lines.append(f"- route_method: `{route_method}`")

    lines.extend(
        [
            f"- request_id: `{request.request_id}`",
            f"- entry_point: `{request.entry_point}`",
            f"- request_type: `{request.request_type}`",
            f"- action: `{trace.get('action', 'unknown')}`",
            f"- instrument_type: `{request.instrument_type or 'unknown'}`",
            f"- requires_build: `{bool(trace.get('requires_build'))}`",
        ]
    )

    blocker_codes = list(trace.get("blocker_codes") or [])
    if blocker_codes:
        lines.append(f"- blocker_codes: `{', '.join(blocker_codes)}`")

    if event_record is not None:
        details = event_record.get("details") or {}
        if details:
            lines.extend(["", "## Trigger Details"])
            for key in sorted(details):
                lines.append(f"- {key}: `{details[key]}`")

    description = (request.description or "").strip()
    if description:
        lines.extend(["", "## Request Description", description])

    events = trace.get("events", [])
    if events:
        lines.extend(["", "## Recent Events"])
        for item in events[-8:]:
            lines.append(
                f"- `{item.get('timestamp', '')}` `{item.get('event', '')}` "
                f"({item.get('status', 'info')})"
            )
    return "\n".join(lines)


def build_event_comment(
    trace: dict[str, Any],
    event_record: dict[str, Any],
) -> str:
    """Render a lifecycle comment with the same task/method context as the issue body."""
    event = event_record.get("event", "")
    status = event_record.get("status", "info")
    lines = [f"### {_event_title(event)}", f"- status: `{status}`"]

    task_id = _metadata_value(None, trace, "task_id")
    task_title = _metadata_value(None, trace, "task_title")
    if task_id or task_title:
        task_line = f"`{task_id}`" if task_id else "`unknown`"
        if task_title:
            task_line += f" — {task_title}"
        lines.append(f"- task: {task_line}")

    comparison_target = _metadata_value(None, trace, "comparison_target")
    if comparison_target:
        lines.append(f"- comparison_target: `{comparison_target}`")

    route_method = trace.get("route_method") or _metadata_value(None, trace, "preferred_method")
    if route_method:
        lines.append(f"- route_method: `{route_method}`")

    semantic_role_ownership = _metadata_value(None, trace, "semantic_role_ownership")
    if isinstance(semantic_role_ownership, Mapping) and semantic_role_ownership:
        lines.append(
            f"- ownership_stage: `{semantic_role_ownership.get('selected_stage', '')}`"
        )
        lines.append(
            f"- ownership_role: `{semantic_role_ownership.get('selected_role', '')}`"
        )
        lines.append(
            f"- ownership_trigger: `{semantic_role_ownership.get('trigger_condition', '')}`"
        )
        lines.append(
            f"- ownership_artifact: `{semantic_role_ownership.get('artifact_kind', '')}`"
        )

    details = event_record.get("details") or {}
    for key in sorted(details):
        lines.append(f"- {key}: `{details[key]}`")

    outcome = trace.get("outcome")
    if outcome:
        lines.append(f"- trace_outcome: `{outcome}`")
    return "\n".join(lines)


def _event_title(event: str) -> str:
    mapping = {
        "build_started": "Build started",
        "quant_selected_method": "Quant selected method",
        "planner_completed": "Planner completed",
        "builder_attempt_started": "Builder attempt started",
        "builder_attempt_failed": "Builder attempt failed",
        "builder_attempt_succeeded": "Builder attempt succeeded",
        "critic_completed": "Critic completed",
        "arbiter_completed": "Arbiter completed",
        "model_validator_completed": "Model validator completed",
        "model_validator_skipped": "Model validator skipped",
        "build_completed": "Build completed",
        "existing_generated_module_reused": "Existing generated module reused",
        "request_blocked": "Request blocked",
        "request_failed": "Request failed",
        "request_succeeded": "Request succeeded",
    }
    return mapping.get(event, event.replace("_", " ").title())


def _request_label(trace: dict[str, Any], compiled_request) -> str:
    request = compiled_request.request
    description = (request.description or "").strip()
    if description:
        return " ".join(description.split())[:96]
    instrument = request.instrument_type or trace.get("product_instrument")
    if instrument:
        return f"{request.request_type} {instrument}"
    return request.request_id


def _task_label(compiled_request) -> str | None:
    task_id = _metadata_value(compiled_request, None, "task_id")
    task_title = _metadata_value(compiled_request, None, "task_title")
    if task_id and task_title:
        return f"{task_id} — {task_title}"
    if task_id:
        return str(task_id)
    if task_title:
        return str(task_title)
    return None


def _metadata_value(compiled_request, trace: dict[str, Any] | None, key: str) -> object | None:
    if compiled_request is not None:
        request = getattr(compiled_request, "request", None)
        metadata = getattr(request, "metadata", None)
        if isinstance(metadata, Mapping) and key in metadata:
            return metadata.get(key)
    if trace is not None:
        metadata = trace.get("request_metadata") or {}
        if isinstance(metadata, Mapping):
            return metadata.get(key)
    return None


def _compact(text: object, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"
