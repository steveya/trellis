"""Appendable audit storage for unified platform requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TRACE_ROOT = Path(__file__).parent / "knowledge" / "traces" / "platform"

_TERMINAL_OUTCOMES = {
    "priced",
    "agent_priced",
    "ask_priced_existing",
    "ask_built_and_priced",
    "pipeline_priced",
    "greeks_computed",
    "analytics_computed",
    "request_blocked",
    "request_failed",
    "price_failed",
    "greeks_failed",
    "analytics_failed",
    "pipeline_failed",
    "ask_failed",
}


@dataclass(frozen=True)
class PlatformTraceEvent:
    """One timestamped lifecycle event appended to a platform request trace."""
    event: str
    status: str
    timestamp: str
    details: dict[str, Any]


@dataclass(frozen=True)
class PlatformTrace:
    """Structured view of a persisted platform request trace file."""
    request_id: str
    request_type: str
    entry_point: str
    action: str
    success: bool | None
    outcome: str
    status: str
    timestamp: str
    updated_at: str
    measures: tuple[str, ...] = ()
    instrument_type: str | None = None
    product_instrument: str | None = None
    route_method: str | None = None
    sensitivity_support: dict[str, Any] | None = None
    requires_build: bool = False
    blocker_codes: tuple[str, ...] = ()
    knowledge_summary: dict[str, Any] | None = None
    simulation_identity: dict[str, Any] | None = None
    simulation_seed: int | None = None
    sample_source: dict[str, Any] | None = None
    sample_indexing: dict[str, Any] | None = None
    simulation_stream_id: str | None = None
    semantic_role_ownership: dict[str, Any] | None = None
    details: dict[str, Any] | None = None
    events: tuple[PlatformTraceEvent, ...] = ()
    linear_issue_id: str | None = None
    linear_issue_identifier: str | None = None
    linear_issue_url: str | None = None
    github_issue_number: int | None = None
    github_issue_url: str | None = None
    github_issue_repository: str | None = None
    trace_path: str | None = None
    token_usage: dict[str, Any] | None = None


def ensure_platform_trace(
    compiled_request,
    *,
    details: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Ensure a trace file exists for the request and refresh core metadata."""
    root = root or TRACE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{compiled_request.request.request_id}.yaml"

    trace = _load_trace_dict(path)
    base = _base_trace_dict(compiled_request)
    if not trace:
        trace = base
    else:
        for key in (
            "request_id",
            "request_type",
            "entry_point",
            "action",
            "measures",
            "instrument_type",
            "product_instrument",
            "route_method",
            "sensitivity_support",
            "requires_build",
            "blocker_codes",
            "knowledge_summary",
            "simulation_identity",
            "simulation_seed",
            "sample_source",
            "sample_indexing",
            "simulation_stream_id",
            "semantic_role_ownership",
            "request_metadata",
        ):
            trace[key] = base[key]
        trace.setdefault("success", None)
        trace.setdefault("outcome", "")
        trace.setdefault("status", "running")
        trace.setdefault("timestamp", base["timestamp"])
    trace["updated_at"] = _now_utc()
    trace["details"] = _merge_details(trace.get("details"), details)
    trace.setdefault("events", [])
    trace.setdefault("linear_issue", {})
    trace.setdefault("github_issue", {})
    _write_trace_dict(path, trace)
    return path


def append_platform_trace_event(
    compiled_request,
    event: str,
    *,
    status: str = "info",
    success: bool | None = None,
    outcome: str | None = None,
    details: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Append one lifecycle event to the request trace."""
    path = ensure_platform_trace(compiled_request, root=root)
    trace = _load_trace_dict(path)

    event_record = {
        "event": event,
        "status": status,
        "timestamp": _now_utc(),
        "details": details or {},
    }
    trace.setdefault("events", []).append(event_record)
    trace["updated_at"] = event_record["timestamp"]

    if outcome is not None:
        trace["outcome"] = outcome
        trace["success"] = success
        trace["details"] = _merge_details(trace.get("details"), details)
        trace["status"] = _trace_status_for(outcome, success)
    elif success is not None:
        trace["success"] = success
        trace["status"] = "succeeded" if success else "failed"

    issue_refs = _sync_issue_trackers(trace, compiled_request, event_record)
    for key, issue_ref in issue_refs.items():
        if issue_ref:
            trace[key] = issue_ref

    _write_trace_dict(path, trace)
    return path


def record_platform_trace(
    compiled_request,
    *,
    success: bool,
    outcome: str,
    details: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Persist the terminal outcome for a platform request."""
    event = _event_name_for_outcome(outcome, success)
    return append_platform_trace_event(
        compiled_request,
        event,
        status="ok" if success else "error",
        success=success,
        outcome=outcome,
        details=details,
        root=root,
    )


def attach_platform_trace_token_usage(
    trace_path: str | Path,
    token_usage: dict[str, Any],
) -> Path:
    """Persist aggregated token-usage telemetry onto an existing trace file."""
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(path)
    trace = _load_trace_dict(path)
    trace["token_usage"] = dict(token_usage or {})
    trace["updated_at"] = _now_utc()
    _write_trace_dict(path, trace)
    return path


def load_platform_traces(*, root: Path | None = None) -> list[PlatformTrace]:
    """Load request traces from disk."""
    root = root or TRACE_ROOT
    if not root.exists():
        return []

    traces: list[PlatformTrace] = []
    for path in sorted(root.glob("*.yaml")):
        data = _load_trace_dict(path)
        issue = data.get("linear_issue") or {}
        github_issue = data.get("github_issue") or {}
        traces.append(
            PlatformTrace(
                request_id=data.get("request_id", path.stem),
                request_type=data.get("request_type", "unknown"),
                entry_point=data.get("entry_point", "unknown"),
                action=data.get("action", "unknown"),
                success=data.get("success"),
                outcome=data.get("outcome", ""),
                status=data.get("status", "unknown"),
                timestamp=data.get("timestamp", ""),
                updated_at=data.get("updated_at", data.get("timestamp", "")),
                measures=tuple(data.get("measures", [])),
                instrument_type=data.get("instrument_type"),
                product_instrument=data.get("product_instrument"),
                route_method=data.get("route_method"),
                sensitivity_support=data.get("sensitivity_support") or {},
                requires_build=bool(data.get("requires_build")),
                blocker_codes=tuple(data.get("blocker_codes", [])),
                knowledge_summary=data.get("knowledge_summary") or {},
                simulation_identity=data.get("simulation_identity") or {},
                simulation_seed=data.get("simulation_seed"),
                sample_source=data.get("sample_source") or {},
                sample_indexing=data.get("sample_indexing") or {},
                simulation_stream_id=data.get("simulation_stream_id"),
                semantic_role_ownership=data.get("semantic_role_ownership") or {},
                details=data.get("details") or {},
                events=tuple(
                    PlatformTraceEvent(
                        event=item.get("event", ""),
                        status=item.get("status", "info"),
                        timestamp=item.get("timestamp", ""),
                        details=item.get("details") or {},
                    )
                    for item in data.get("events", [])
                ),
                linear_issue_id=issue.get("id"),
                linear_issue_identifier=issue.get("identifier"),
                linear_issue_url=issue.get("url"),
                github_issue_number=github_issue.get("number"),
                github_issue_url=github_issue.get("url"),
                github_issue_repository=github_issue.get("repository"),
                trace_path=str(path),
                token_usage=data.get("token_usage") or {},
            )
        )
    return traces


def summarize_platform_traces(traces: list[PlatformTrace]) -> dict[str, int]:
    """Count traces by execution action."""
    summary: dict[str, int] = {}
    for trace in traces:
        summary[trace.action] = summary.get(trace.action, 0) + 1
    return summary


def _base_trace_dict(compiled_request) -> dict[str, Any]:
    """Build the canonical top-level YAML payload for a compiled request trace."""
    blocker_codes: list[str] = []
    blocker_report = getattr(compiled_request, "blocker_report", None)
    if blocker_report is not None:
        blocker_codes = [blocker.id for blocker in blocker_report.blockers]

    now = _now_utc()
    request = compiled_request.request
    execution_plan = compiled_request.execution_plan
    request_metadata = dict(request.metadata or {})
    runtime_contract = dict(request_metadata.get("runtime_contract") or {})
    simulation_identity = dict(
        runtime_contract.get("simulation_identity")
        or request_metadata.get("simulation_identity")
        or {}
    )
    def _first_present(*values):
        for value in values:
            if value is not None:
                return value
        return None

    sample_source = dict(
        simulation_identity.get("sample_source")
        or runtime_contract.get("sample_source")
        or request_metadata.get("sample_source")
        or {}
    )
    sample_indexing = dict(
        simulation_identity.get("sample_indexing")
        or runtime_contract.get("sample_indexing")
        or request_metadata.get("sample_indexing")
        or {}
    )
    return {
        "request_id": request.request_id,
        "request_type": request.request_type,
        "entry_point": request.entry_point,
        "action": execution_plan.action,
        "success": None,
        "outcome": "",
        "status": "running",
        "timestamp": now,
        "updated_at": now,
        "measures": list(execution_plan.measures),
        "instrument_type": request.instrument_type,
        "product_instrument": getattr(compiled_request.product_ir, "instrument", None),
        "route_method": execution_plan.route_method,
        "sensitivity_support": (
            compiled_request.pricing_plan.sensitivity_support.to_dict()
            if getattr(compiled_request, "pricing_plan", None) is not None
            and getattr(compiled_request.pricing_plan, "sensitivity_support", None) is not None
            else {}
        ),
        "requires_build": bool(getattr(execution_plan, "requires_build", False)),
        "blocker_codes": blocker_codes,
        "knowledge_summary": dict(getattr(compiled_request, "knowledge_summary", {}) or {}),
        "simulation_identity": simulation_identity,
        "simulation_seed": _first_present(
            simulation_identity.get("seed"),
            runtime_contract.get("simulation_seed"),
            request_metadata.get("simulation_seed"),
        ),
        "sample_source": sample_source,
        "sample_indexing": sample_indexing,
        "simulation_stream_id": _first_present(
            simulation_identity.get("simulation_stream_id"),
            runtime_contract.get("simulation_stream_id"),
            request_metadata.get("simulation_stream_id"),
        ),
        "semantic_role_ownership": dict(
            request_metadata.get("semantic_role_ownership") or {}
        ),
        "request_metadata": request_metadata,
        "token_usage": {},
    }


def _load_trace_dict(path: Path) -> dict[str, Any]:
    """Load a trace YAML file, returning an empty mapping when absent."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def _write_trace_dict(path: Path, trace: dict[str, Any]) -> None:
    """Persist a trace dictionary to YAML using stable readable formatting."""
    with open(path, "w") as fh:
        yaml.safe_dump(
            _normalize_yaml_value(trace),
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _normalize_yaml_value(value: Any) -> Any:
    """Convert tuples and nested containers into safe YAML-friendly values."""
    if isinstance(value, dict):
        return {
            str(key): _normalize_yaml_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_normalize_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_yaml_value(item) for item in value]
    return value


def _merge_details(
    base: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge trace detail dictionaries with incoming keys taking precedence."""
    merged = dict(base or {})
    merged.update(incoming or {})
    return merged


def _trace_status_for(outcome: str, success: bool | None) -> str:
    """Map a terminal outcome plus success flag onto the coarse trace status."""
    if outcome in {"request_blocked"}:
        return "blocked"
    if success is True:
        return "succeeded"
    if success is False:
        return "failed"
    if outcome in _TERMINAL_OUTCOMES:
        return "completed"
    return "running"


def _event_name_for_outcome(outcome: str, success: bool) -> str:
    """Choose the synthetic terminal event name written for an outcome."""
    if outcome in {"request_blocked"}:
        return "request_blocked"
    if success:
        return "request_succeeded"
    return "request_failed"


def _sync_issue_trackers(
    trace: dict[str, Any],
    compiled_request,
    event_record: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Best-effort sync to external issue trackers and return any discovered refs."""
    from trellis.agent.config import issue_tracker_sync_enabled

    if not issue_tracker_sync_enabled():
        return {}

    refs: dict[str, dict[str, Any]] = {}

    try:
        from trellis.agent.linear_tracker import sync_request_issue

        issue_ref = sync_request_issue(trace, compiled_request, event_record)
        if issue_ref:
            refs["linear_issue"] = issue_ref
    except Exception:
        pass

    try:
        from trellis.agent.github_tracker import sync_request_issue

        issue_ref = sync_request_issue(trace, compiled_request, event_record)
        if issue_ref:
            refs["github_issue"] = issue_ref
    except Exception:
        pass

    return refs


def _now_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()
