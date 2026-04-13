"""Appendable audit storage for unified platform requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
    semantic_checkpoint: dict[str, Any] | None = None
    generation_boundary: dict[str, Any] | None = None
    semantic_role_ownership: dict[str, Any] | None = None
    validation_contract: dict[str, Any] | None = None
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

    trace = _prepare_trace_summary_for_write(path, _load_trace_summary_dict(path))
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
            "semantic_checkpoint",
            "generation_boundary",
            "semantic_role_ownership",
            "validation_contract",
            "request_metadata",
        ):
            trace[key] = base[key]
        trace.setdefault("success", None)
        trace.setdefault("outcome", "")
        trace.setdefault("status", "running")
        trace.setdefault("timestamp", base["timestamp"])
    trace["updated_at"] = _now_utc()
    trace["details"] = _merge_details(trace.get("details"), details)
    trace.setdefault("linear_issue", {})
    trace.setdefault("github_issue", {})
    trace.setdefault("token_usage", {})
    _write_trace_summary_dict(path, trace)
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
    trace = _prepare_trace_summary_for_write(path, _load_trace_summary_dict(path))

    event_record = _normalize_event_record({
        "event": event,
        "status": status,
        "timestamp": _now_utc(),
        "details": details or {},
    })
    _append_trace_event(path, event_record)
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

    _write_trace_summary_dict(path, trace)
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
    trace = _prepare_trace_summary_for_write(path, _load_trace_summary_dict(path))
    trace["token_usage"] = dict(token_usage or {})
    trace["updated_at"] = _now_utc()
    _write_trace_summary_dict(path, trace)
    return path


def load_platform_trace_boundary(trace_path: str | Path) -> dict[str, Any]:
    """Load the compact semantic/route/validation boundary from a trace file."""
    data = _load_trace_summary_dict(Path(trace_path))
    return {
        "semantic_checkpoint": dict(data.get("semantic_checkpoint") or {}),
        "generation_boundary": dict(data.get("generation_boundary") or {}),
        "validation_contract": dict(data.get("validation_contract") or {}),
        "request_metadata": dict(data.get("request_metadata") or {}),
        "route_method": data.get("route_method"),
        "instrument_type": data.get("instrument_type"),
        "product_instrument": data.get("product_instrument"),
    }


def load_platform_trace_payload(trace_path: str | Path) -> dict[str, Any]:
    """Load the full normalized trace payload for one persisted platform trace."""
    return _load_trace_payload_dict(Path(trace_path))


def load_platform_trace_events(trace_path: str | Path) -> tuple[PlatformTraceEvent, ...]:
    """Load the full lifecycle event history for one platform trace."""
    return tuple(
        PlatformTraceEvent(
            event=item.get("event", ""),
            status=item.get("status", "info"),
            timestamp=item.get("timestamp", ""),
            details=item.get("details") or {},
        )
        for item in _load_trace_event_dicts(Path(trace_path))
    )

def load_platform_traces(
    *,
    root: Path | None = None,
    include_events: bool = False,
) -> list[PlatformTrace]:
    """Load request traces from disk.

    By default this returns the cheaper summary view and skips event hydration.
    Call ``load_platform_trace_events()`` or set ``include_events=True`` when
    full lifecycle history is needed.
    """
    root = root or TRACE_ROOT
    if not root.exists():
        return []

    traces: list[PlatformTrace] = []
    for path in sorted(root.glob("*.yaml")):
        data = _load_trace_summary_dict(path)
        issue = data.get("linear_issue") or {}
        github_issue = data.get("github_issue") or {}
        events = load_platform_trace_events(path) if include_events else ()
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
                semantic_checkpoint=data.get("semantic_checkpoint") or {},
                generation_boundary=data.get("generation_boundary") or {},
                semantic_role_ownership=data.get("semantic_role_ownership") or {},
                validation_contract=data.get("validation_contract") or {},
                details=data.get("details") or {},
                events=events,
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
    semantic_checkpoint = _semantic_checkpoint_summary(
        request=request,
        request_metadata=request_metadata,
    )
    generation_boundary = _generation_boundary_summary(
        compiled_request,
        request_metadata=request_metadata,
    )
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
        "semantic_checkpoint": semantic_checkpoint,
        "generation_boundary": generation_boundary,
        "semantic_role_ownership": dict(
            request_metadata.get("semantic_role_ownership") or {}
        ),
        "validation_contract": dict(
            request_metadata.get("validation_contract") or {}
        ),
        "request_metadata": request_metadata,
        "token_usage": {},
    }


def _semantic_checkpoint_summary(
    *,
    request,
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the canonical semantic identity and wrapper status."""
    semantic_contract = dict(request_metadata.get("semantic_contract") or {})
    if not semantic_contract:
        return {}
    concept = dict(semantic_contract.get("semantic_concept") or {})
    product = dict(semantic_contract.get("product") or {})
    methods = dict(semantic_contract.get("methods") or {})
    market_data = dict(semantic_contract.get("market_data") or {})
    requested_instrument = str(getattr(request, "instrument_type", "") or "")
    compatibility_wrappers = tuple(
        str(item)
        for item in (concept.get("compatibility_wrappers") or ())
        if str(item).strip()
    )
    bridge_status = _compatibility_bridge_status(
        requested_instrument=requested_instrument,
        semantic_id=semantic_contract.get("semantic_id"),
        compatibility_wrappers=compatibility_wrappers,
    )
    return {
        "semantic_id": semantic_contract.get("semantic_id"),
        "semantic_version": semantic_contract.get("semantic_version"),
        "requested_instrument_type": requested_instrument or None,
        "product_instrument_class": product.get("instrument_class"),
        "payoff_family": product.get("payoff_family"),
        "underlier_structure": product.get("underlier_structure"),
        "preferred_method": methods.get("preferred_method"),
        "required_market_inputs": list(market_data.get("required_inputs") or ()),
        "optional_market_inputs": list(market_data.get("optional_inputs") or ()),
        "compatibility_bridge_status": bridge_status,
        "matched_wrapper": (
            requested_instrument
            if bridge_status == "thin_compatibility_wrapper"
            else ""
        ),
    }


def _generation_boundary_summary(
    compiled_request,
    *,
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize valuation, lowering, and approved-module route boundaries."""
    from trellis.agent.route_registry import route_binding_authority_summary

    generation_plan = getattr(compiled_request, "generation_plan", None)
    semantic_blueprint = dict(request_metadata.get("semantic_blueprint") or {})
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    lane_plan = dict(semantic_blueprint.get("lane_plan") or {})
    family_ir_payload = dict(semantic_blueprint.get("dsl_family_ir") or {})
    route_binding_authority = (
        route_binding_authority_summary(
            getattr(generation_plan, "route_binding_authority", None)
        )
        or dict(request_metadata.get("route_binding_authority") or {})
    )
    backend_binding = dict(route_binding_authority.get("backend_binding") or {})
    lowering_route_id = (
        str(getattr(generation_plan, "lowering_route_id", "") or "").strip()
        or str(semantic_blueprint.get("dsl_route") or "").strip()
        or str(route_binding_authority.get("route_id") or "").strip()
        or None
    )
    lowering_route_family = (
        str(semantic_blueprint.get("dsl_route_family") or "").strip()
        or str(getattr(primitive_plan, "route_family", "") or "").strip()
        or str(getattr(primitive_plan, "engine_family", "") or "").strip()
        or str(route_binding_authority.get("route_family") or "").strip()
        or str(route_binding_authority.get("engine_family") or "").strip()
        or str(backend_binding.get("engine_family") or "").strip()
        or None
    )
    lowering = {
        "route_id": lowering_route_id,
        "route_family": lowering_route_family,
        "primitive_routes": list(semantic_blueprint.get("primitive_routes") or ()),
        "route_modules": list(semantic_blueprint.get("route_modules") or ()),
        "expr_kind": (
            getattr(generation_plan, "lowering_expr_kind", "")
            or semantic_blueprint.get("dsl_expr_kind")
        ),
        "family_ir_type": (
            getattr(generation_plan, "lowering_family_ir_type", "")
            or semantic_blueprint.get("dsl_family_ir_type")
        ),
        "family_ir_summary": _family_ir_trace_summary(family_ir_payload),
        "helper_refs": list(
            getattr(generation_plan, "lowering_helper_refs", ())
            or semantic_blueprint.get("dsl_helper_refs")
            or ()
        ),
        "target_bindings": list(semantic_blueprint.get("dsl_target_bindings") or ()),
        "lowering_errors": list(semantic_blueprint.get("dsl_lowering_errors") or ()),
    }
    lane_summary = (
        {
            "lane_family": getattr(generation_plan, "lane_family", "") or lane_plan.get("lane_family"),
            "plan_kind": getattr(generation_plan, "lane_plan_kind", "") or lane_plan.get("plan_kind"),
            "timeline_roles": list(
                getattr(generation_plan, "lane_timeline_roles", ()) or lane_plan.get("timeline_roles") or ()
            ),
            "market_requirements": list(
                getattr(generation_plan, "lane_market_requirements", ()) or lane_plan.get("market_requirements") or ()
            ),
            "state_obligations": list(
                getattr(generation_plan, "lane_state_obligations", ()) or lane_plan.get("state_obligations") or ()
            ),
            "control_obligations": list(
                getattr(generation_plan, "lane_control_obligations", ()) or lane_plan.get("control_obligations") or ()
            ),
            "construction_steps": list(
                getattr(generation_plan, "lane_construction_steps", ()) or lane_plan.get("construction_steps") or ()
            ),
            "exact_target_refs": list(
                getattr(generation_plan, "lane_exact_binding_refs", ()) or lane_plan.get("exact_target_refs") or ()
            ),
            "unresolved_primitives": list(
                getattr(generation_plan, "lane_unresolved_primitives", ()) or lane_plan.get("unresolved_primitives") or ()
            ),
        }
        if generation_plan is not None or lane_plan
        else {}
    )
    summary = {
        "method": (
            getattr(generation_plan, "method", "")
            or getattr(getattr(compiled_request, "execution_plan", None), "route_method", None)
        ),
        "approved_modules": list(getattr(generation_plan, "approved_modules", ()) or ()),
        "inspected_modules": list(getattr(generation_plan, "inspected_modules", ()) or ()),
        "symbols_to_reuse": list(getattr(generation_plan, "symbols_to_reuse", ()) or ()),
        "valuation_context": dict(semantic_blueprint.get("valuation_context") or {}),
        "required_data_spec": dict(semantic_blueprint.get("required_data_spec") or {}),
        "market_binding_spec": dict(semantic_blueprint.get("market_binding_spec") or {}),
        "lane_plan": lane_summary,
        "lowering": lowering,
        "construction_identity": _construction_identity_summary(
            lane_plan=lane_summary,
            lowering=lowering,
            route_binding_authority=route_binding_authority,
        ),
        "route_binding_authority": route_binding_authority,
        "primitive_plan": (
            {
                "route": getattr(primitive_plan, "route", ""),
                "engine_family": getattr(primitive_plan, "engine_family", ""),
                "route_family": getattr(primitive_plan, "route_family", ""),
                "backend_binding_id": getattr(primitive_plan, "backend_binding_id", ""),
                "backend_exact_target_refs": list(
                    getattr(primitive_plan, "backend_exact_target_refs", ()) or ()
                ),
                "backend_helper_refs": list(
                    getattr(primitive_plan, "backend_helper_refs", ()) or ()
                ),
                "adapters": list(getattr(primitive_plan, "adapters", ()) or ()),
                "blockers": list(getattr(primitive_plan, "blockers", ()) or ()),
            }
            if primitive_plan is not None
            else {}
        ),
    }
    if not any(summary.values()):
        return {}
    return summary


def _construction_identity_summary(
    *,
    lane_plan: dict[str, Any],
    lowering: dict[str, Any],
    route_binding_authority: dict[str, Any],
) -> dict[str, Any]:
    """Project the family-first construction identity used by diagnostics."""
    from trellis.agent.route_registry import should_surface_route_alias

    backend_binding = dict(route_binding_authority.get("backend_binding") or {})
    operator_metadata = dict(route_binding_authority.get("operator_metadata") or {})
    exact_backend_fit = bool(
        backend_binding.get("exact_backend_fit")
        if "exact_backend_fit" in backend_binding
        else route_binding_authority.get("exact_backend_fit")
    )
    lane_family = str(lane_plan.get("lane_family") or "").strip() or None
    family_ir_type = str(lowering.get("family_ir_type") or "").strip() or None
    route_family = str(lowering.get("route_family") or "").strip() or None
    raw_route_alias = (
        str(route_binding_authority.get("route_id") or "").strip()
        or str(lowering.get("route_id") or "").strip()
        or None
    )
    route_alias = raw_route_alias if should_surface_route_alias(route_binding_authority) else None
    backend_binding_id = str(backend_binding.get("binding_id") or "").strip() or None
    backend_engine_family = (
        str(backend_binding.get("engine_family") or "").strip()
        or str(route_binding_authority.get("engine_family") or "").strip()
        or None
    )
    binding_display_name = str(operator_metadata.get("display_name") or "").strip() or None
    binding_short_description = str(operator_metadata.get("short_description") or "").strip() or None
    binding_diagnostic_label = str(operator_metadata.get("diagnostic_label") or "").strip() or None

    if exact_backend_fit and backend_binding_id:
        primary_kind = "backend_binding"
        primary_label = binding_display_name or backend_binding_id
    elif family_ir_type:
        primary_kind = "family_ir"
        primary_label = family_ir_type
    elif lane_family:
        primary_kind = "lane_family"
        primary_label = lane_family
    elif route_family:
        primary_kind = "method_family"
        primary_label = route_family
    else:
        primary_kind = "unknown"
        primary_label = "unknown"

    return {
        "primary_kind": primary_kind,
        "primary_label": primary_label,
        "lane_family": lane_family,
        "plan_kind": str(lane_plan.get("plan_kind") or "").strip() or None,
        "family_ir_type": family_ir_type,
        "backend_binding_id": backend_binding_id,
        "backend_engine_family": backend_engine_family,
        "backend_exact_fit": exact_backend_fit,
        "binding_display_name": binding_display_name,
        "binding_short_description": binding_short_description,
        "binding_diagnostic_label": binding_diagnostic_label,
        "operator_metadata": operator_metadata or None,
        "route_alias": route_alias,
        "route_alias_policy": str(route_binding_authority.get("compatibility_alias_policy") or "").strip() or None,
        "route_authority_kind": str(route_binding_authority.get("authority_kind") or "").strip() or None,
        "state_obligations": list(lane_plan.get("state_obligations") or ()),
        "control_obligations": list(lane_plan.get("control_obligations") or ()),
    }


def _family_ir_trace_summary(family_ir_payload: dict[str, Any]) -> dict[str, Any]:
    """Project a compact trace-friendly summary from the YAML-safe family IR payload."""
    if not family_ir_payload:
        return {}

    state_spec = dict(family_ir_payload.get("state_spec") or {})
    characteristic_spec = dict(family_ir_payload.get("characteristic_spec") or {})
    process_spec = dict(family_ir_payload.get("process_spec") or {})
    operator_spec = dict(family_ir_payload.get("operator_spec") or {})
    control_spec = dict(family_ir_payload.get("control_spec") or {})
    control_program = dict(family_ir_payload.get("control_program") or {})
    boundary_spec = dict(family_ir_payload.get("boundary_spec") or {})
    path_requirement_spec = dict(family_ir_payload.get("path_requirement_spec") or {})
    payoff_reducer_spec = dict(family_ir_payload.get("payoff_reducer_spec") or {})
    event_program = dict(family_ir_payload.get("event_program") or {})
    compatibility_wrapper = str(family_ir_payload.get("compatibility_wrapper") or "").strip()

    event_transform_kinds: list[str] = []
    for transform in family_ir_payload.get("event_transforms") or ():
        kind = str((transform or {}).get("transform_kind") or "").strip()
        if kind and kind not in event_transform_kinds:
            event_transform_kinds.append(kind)
    if not event_transform_kinds:
        for bucket in family_ir_payload.get("event_timeline") or ():
            for transform in (bucket or {}).get("transforms") or ():
                kind = str((transform or {}).get("transform_kind") or "").strip()
                if kind and kind not in event_transform_kinds:
                    event_transform_kinds.append(kind)

    event_dates: list[str] = []
    for bucket in family_ir_payload.get("event_timeline") or ():
        event_date = str((bucket or {}).get("event_date") or "").strip()
        if event_date and event_date not in event_dates:
            event_dates.append(event_date)

    event_kinds: list[str] = []
    for event in family_ir_payload.get("event_specs") or ():
        kind = str((event or {}).get("event_kind") or "").strip()
        if kind and kind not in event_kinds:
            event_kinds.append(kind)
    if not event_kinds:
        for bucket in family_ir_payload.get("event_timeline") or ():
            for event in (bucket or {}).get("events") or ():
                kind = str((event or {}).get("event_kind") or "").strip()
                if kind and kind not in event_kinds:
                    event_kinds.append(kind)

    semantic_event_kinds: list[str] = []
    semantic_transform_kinds: list[str] = []
    semantic_event_dates: list[str] = []
    for bucket in event_program.get("timeline") or ():
        event_date = str((bucket or {}).get("event_date") or "").strip()
        if event_date and event_date not in semantic_event_dates:
            semantic_event_dates.append(event_date)
        for event in (bucket or {}).get("events") or ():
            kind = str((event or {}).get("event_kind") or "").strip()
            transform_kind = str((event or {}).get("transform_kind") or "").strip()
            if kind and kind not in semantic_event_kinds:
                semantic_event_kinds.append(kind)
            if transform_kind and transform_kind not in semantic_transform_kinds:
                semantic_transform_kinds.append(transform_kind)

    summary = {
        "state_variable": state_spec.get("state_variable"),
        "dimension": state_spec.get("dimension"),
        "state_tags": list(state_spec.get("state_tags") or ()),
        "model_family": characteristic_spec.get("model_family"),
        "characteristic_family": characteristic_spec.get("characteristic_family"),
        "supported_transform_methods": list(characteristic_spec.get("supported_methods") or ()),
        "transform_backend_capability": characteristic_spec.get("backend_capability"),
        "process_family": process_spec.get("process_family"),
        "simulation_scheme": process_spec.get("simulation_scheme"),
        "operator_family": operator_spec.get("operator_family"),
        "solver_family": operator_spec.get("solver_family"),
        "control_style": control_spec.get("control_style"),
        "controller_role": control_spec.get("controller_role"),
        "semantic_control_style": control_program.get("control_style"),
        "semantic_controller_role": control_program.get("controller_role"),
        "semantic_decision_phase": control_program.get("decision_phase"),
        "semantic_schedule_role": control_program.get("schedule_role"),
        "path_requirement_kind": path_requirement_spec.get("requirement_kind"),
        "reducer_kind": payoff_reducer_spec.get("reducer_kind"),
        "terminal_payoff_kind": family_ir_payload.get("terminal_payoff_kind"),
        "strike_semantics": family_ir_payload.get("strike_semantics"),
        "quote_semantics": family_ir_payload.get("quote_semantics"),
        "event_kinds": event_kinds,
        "semantic_event_kinds": semantic_event_kinds,
        "terminal_condition_kind": boundary_spec.get("terminal_condition_kind"),
        "event_transform_kinds": event_transform_kinds,
        "semantic_transform_kinds": semantic_transform_kinds,
        "event_dates": event_dates,
        "semantic_event_dates": semantic_event_dates,
        "helper_symbol": family_ir_payload.get("helper_symbol"),
        "market_mapping": family_ir_payload.get("market_mapping"),
        "compatibility_wrapper": compatibility_wrapper or None,
        "compatibility_status": (
            "transitional_wrapper"
            if compatibility_wrapper
            else "native_transform_family_ir"
            if characteristic_spec
            else "native_event_aware"
        ),
        "end_state": (
            "migrate_to_plain_EventAwarePDEIR"
            if compatibility_wrapper
            else "native_transform_family_ir"
            if characteristic_spec
            else "native_event_aware"
        ),
    }
    return summary


def _compatibility_bridge_status(
    *,
    requested_instrument: str,
    semantic_id: object,
    compatibility_wrappers: tuple[str, ...],
) -> str:
    """Classify whether the request came through a compatibility wrapper."""
    normalized_request = _normalize_semantic_token(requested_instrument)
    if not normalized_request:
        return "implicit_semantic_request"
    if normalized_request == _normalize_semantic_token(semantic_id):
        return "canonical_semantic"
    wrapper_tokens = {
        _normalize_semantic_token(item)
        for item in compatibility_wrappers
        if _normalize_semantic_token(item)
    }
    if normalized_request in wrapper_tokens:
        return "thin_compatibility_wrapper"
    return "request_alias"


def _normalize_semantic_token(value: object) -> str:
    """Normalize semantic identifiers for stable wrapper comparisons."""
    return str(value or "").strip().lower().replace(" ", "_")


def _load_trace_dict(path: Path) -> dict[str, Any]:
    """Load a trace payload with events, returning an empty mapping when absent."""
    return _load_trace_payload_dict(path)


def _load_trace_summary_dict(path: Path) -> dict[str, Any]:
    """Load one trace summary YAML file, returning an empty mapping when absent."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def _write_trace_summary_dict(path: Path, trace: dict[str, Any]) -> None:
    """Persist a trace summary dictionary to YAML using stable readable formatting."""
    with open(path, "w") as fh:
        yaml.safe_dump(
            _normalize_yaml_value(trace),
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _load_trace_payload_dict(path: Path) -> dict[str, Any]:
    """Load one full trace payload by combining summary YAML with event history."""
    summary = _load_trace_summary_dict(path)
    if not summary:
        return {}
    payload = dict(summary)
    payload["events"] = _load_trace_event_dicts(path, trace=summary)
    return payload


def _load_trace_event_dicts(
    path: Path,
    *,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load event records from the append-only log or legacy inline YAML."""
    events_path = _trace_events_path(path)
    if events_path.exists():
        events: list[dict[str, Any]] = []
        for line in events_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(_normalize_event_record(payload))
        return events

    summary = trace if trace is not None else _load_trace_summary_dict(path)
    return [
        _normalize_event_record(item)
        for item in summary.get("events", [])
        if isinstance(item, dict)
    ]


def _append_trace_event(path: Path, event_record: dict[str, Any]) -> None:
    """Append one normalized lifecycle event to the trace NDJSON log."""
    events_path = _trace_events_path(path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "a") as fh:
        fh.write(json.dumps(_normalize_yaml_value(event_record), sort_keys=False))
        fh.write("\n")


def _write_trace_events(path: Path, events: list[dict[str, Any]]) -> None:
    """Persist a full normalized event list into the append-only trace log."""
    events_path = _trace_events_path(path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w") as fh:
        for event in events:
            fh.write(json.dumps(_normalize_yaml_value(_normalize_event_record(event)), sort_keys=False))
            fh.write("\n")


def _prepare_trace_summary_for_write(path: Path, trace: dict[str, Any]) -> dict[str, Any]:
    """Externalize legacy inline events before mutating or rewriting a summary."""
    normalized = dict(trace or {})
    legacy_events = normalized.pop("events", None) or []
    if legacy_events and not _trace_events_path(path).exists():
        _write_trace_events(path, list(legacy_events))
    return normalized


def _trace_events_path(path: Path) -> Path:
    """Return the append-only event log path for one summary YAML trace."""
    return path.with_suffix(".events.ndjson")


def _normalize_event_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one persisted event record into stable JSON/YAML primitives."""
    return {
        "event": str(record.get("event", "") or ""),
        "status": str(record.get("status", "info") or "info"),
        "timestamp": str(record.get("timestamp", "") or ""),
        "details": _normalize_yaml_value(record.get("details") or {}),
    }


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
