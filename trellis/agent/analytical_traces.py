"""Structured analytical build traces for route assembly and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trellis.agent.codegen_guardrails import (
    GenerationPlan,
    PrimitivePlan,
    PrimitiveRef,
    render_generation_plan,
    render_generation_route_card,
)


TRACE_ROOT = Path(__file__).parent / "knowledge" / "traces" / "analytical"


@dataclass(frozen=True)
class AnalyticalTraceRoute:
    """Compact route identity for an analytical build trace."""

    family: str
    name: str
    model: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticalTraceRoute":
        return cls(
            family=str(data.get("family") or "unknown"),
            name=str(data.get("name") or "unknown"),
            model=str(data.get("model") or "unknown"),
        )


@dataclass(frozen=True)
class AnalyticalTraceStep:
    """One structured analytical build step."""

    id: str
    parent_id: str | None
    kind: str
    label: str
    status: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticalTraceStep":
        return cls(
            id=str(data.get("id") or ""),
            parent_id=data.get("parent_id"),
            kind=str(data.get("kind") or "step"),
            label=str(data.get("label") or ""),
            status=str(data.get("status") or "unknown"),
            inputs=dict(data.get("inputs") or {}),
            outputs=dict(data.get("outputs") or {}),
            notes=tuple(data.get("notes") or ()),
        )


@dataclass(frozen=True)
class AnalyticalTrace:
    """Canonical machine-readable analytical build trace."""

    trace_id: str
    route: AnalyticalTraceRoute
    status: str
    created_at: str
    updated_at: str
    steps: tuple[AnalyticalTraceStep, ...]
    trace_type: str = "analytical"
    task_id: str | None = None
    issue_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    json_path: str | None = None
    text_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticalTrace":
        return cls(
            trace_id=str(data.get("trace_id") or ""),
            route=AnalyticalTraceRoute.from_dict(dict(data.get("route") or {})),
            status=str(data.get("status") or "unknown"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or data.get("created_at") or ""),
            steps=tuple(
                AnalyticalTraceStep.from_dict(step)
                for step in data.get("steps") or ()
            ),
            trace_type=str(data.get("trace_type") or "analytical"),
            task_id=data.get("task_id"),
            issue_id=data.get("issue_id"),
            context=dict(data.get("context") or {}),
            json_path=data.get("json_path"),
            text_path=data.get("text_path"),
        )


@dataclass(frozen=True)
class AnalyticalTraceArtifacts:
    """Persisted files for one analytical trace."""

    trace: AnalyticalTrace
    json_path: Path
    text_path: Path


def route_health_snapshot(trace: AnalyticalTrace) -> dict[str, Any]:
    """Return a compact route-health observation for one analytical trace."""
    generation_plan = dict(trace.context.get("generation_plan") or {})
    instruction_resolution = dict(generation_plan.get("instruction_resolution") or {})
    effective = list(instruction_resolution.get("effective_instructions") or [])
    conflicts = list(instruction_resolution.get("conflicts") or [])
    return {
        "route_id": trace.route.name,
        "route_family": trace.route.family,
        "trace_status": trace.status,
        "effective_instruction_ids": [
            str(item.get("id") or "").strip()
            for item in effective
            if str(item.get("id") or "").strip()
        ],
        "effective_instruction_count": len(effective),
        "hard_constraint_count": sum(
            1 for item in effective if str(item.get("instruction_type") or "") == "hard_constraint"
        ),
        "conflict_count": len(conflicts),
    }


def build_analytical_trace_from_generation_plan(
    plan: GenerationPlan,
    *,
    trace_id: str | None = None,
    task_id: str | None = None,
    issue_id: str | None = None,
    route_family: str | None = None,
    model: str | None = None,
    status: str | None = None,
    context: dict[str, Any] | None = None,
) -> AnalyticalTrace:
    """Construct an analytical trace from a deterministic generation plan."""
    primitive_plan = plan.primitive_plan
    route_family = route_family or plan.method
    route_name = primitive_plan.route if primitive_plan is not None else "unknown"
    model = model or (primitive_plan.engine_family if primitive_plan is not None else plan.method)
    trace_id = trace_id or _default_trace_id(route_family, route_name, plan.instrument_type)
    created_at = _now_utc()
    resolved_context = dict(context or {})
    instruction_resolution = _instruction_resolution_context(plan)
    resolved_context.setdefault(
        "generation_plan",
        _generation_plan_context(plan, instruction_resolution=instruction_resolution),
    )
    resolved_context.setdefault("route_card", render_generation_route_card(plan))
    resolved_context.setdefault("route_plan", render_generation_plan(plan))

    root_id = f"{trace_id}:root"
    steps = (
        AnalyticalTraceStep(
            id=root_id,
            parent_id=None,
            kind="trace",
            label="Analytical build",
            status=status or _default_trace_status(plan, primitive_plan),
            inputs={
                "task_id": task_id,
                "issue_id": issue_id,
                "route_family": route_family,
                "route_name": route_name,
                "model": model,
            },
            outputs={
                "route": {
                    "family": route_family,
                    "name": route_name,
                    "model": model,
                },
                "status": status or _default_trace_status(plan, primitive_plan),
            },
            notes=(
                "The trace mirrors the deterministic GenerationPlan used to assemble the route.",
            ),
        ),
        AnalyticalTraceStep(
            id=f"{trace_id}:semantic_resolution",
            parent_id=root_id,
            kind="semantic_resolution",
            label="Resolve contract and route",
            status="ok",
            inputs={
                "method": plan.method,
                "instrument_type": plan.instrument_type,
                "repo_revision": plan.repo_revision,
                "inspected_modules": list(plan.inspected_modules),
                "approved_modules": list(plan.approved_modules[:40]),
                "symbols_to_reuse": list(plan.symbols_to_reuse[:40]),
                "uncertainty_flags": list(plan.uncertainty_flags),
            },
            outputs={
                "route_family": route_family,
                "route_name": route_name,
                "model": model,
                "primitive_plan_score": primitive_plan.score if primitive_plan else None,
                "instruction_resolution": instruction_resolution,
            },
            notes=(
                "Record the semantic contract that drives route selection, not just the final code path.",
            ),
        ),
        AnalyticalTraceStep(
            id=f"{trace_id}:instruction_lifecycle",
            parent_id=root_id,
            kind="instruction_lifecycle",
            label="Resolve route guidance lifecycle",
            status="ok" if not instruction_resolution["conflict_count"] else "warning",
            inputs={
                "route": instruction_resolution["route"],
                "effective_instruction_count": instruction_resolution["effective_instruction_count"],
                "dropped_instruction_count": instruction_resolution["dropped_instruction_count"],
                "conflict_count": instruction_resolution["conflict_count"],
            },
            outputs={
                "instruction_resolution": instruction_resolution,
            },
            notes=(
                "List the effective, dropped, and conflicting route guidance records so replay consumers can see what actually governed the build.",
            ),
        ),
        AnalyticalTraceStep(
            id=f"{trace_id}:decomposition",
            parent_id=root_id,
            kind="decomposition",
            label="Select reusable kernels",
            status="ok" if primitive_plan is None or not primitive_plan.blockers else "warning",
            inputs={
                "primitives": _primitive_refs_to_dicts(primitive_plan.primitives if primitive_plan else ()),
                "adapters": list(primitive_plan.adapters if primitive_plan else ()),
                "blockers": list(primitive_plan.blockers if primitive_plan else ()),
                "notes": list(primitive_plan.notes if primitive_plan else ()),
            },
            outputs={
                "selected_primitives": _primitive_refs_to_dicts(primitive_plan.primitives if primitive_plan else ()),
                "reuse_decision": "exact_decomposition" if _has_terminal_basis_assembly(plan) else "route_local",
            },
            notes=(
                "Capture the reusable valuation components and any exact basis-claim assembly.",
            ),
        ),
        AnalyticalTraceStep(
            id=f"{trace_id}:assembly",
            parent_id=root_id,
            kind="assembly",
            label="Assemble route from kernels",
            status="ok" if primitive_plan is None or not primitive_plan.blockers else "warning",
            inputs={
                "approved_modules": list(plan.approved_modules[:40]),
                "route_helper": _route_helper_name(primitive_plan),
                "adapters": list(primitive_plan.adapters if primitive_plan else ()),
            },
            outputs={
                "assembly_card": render_generation_route_card(plan),
                "route_helper": _route_helper_name(primitive_plan),
                "helper_modules": _primitive_modules(primitive_plan),
            },
            notes=(
                "Prefer thin orchestration around existing analytical kernels and route helpers.",
            ),
        ),
        AnalyticalTraceStep(
            id=f"{trace_id}:validation",
            parent_id=root_id,
            kind="validation",
            label="Validate route and fallbacks",
            status="ok" if not plan.uncertainty_flags and (primitive_plan is None or not primitive_plan.blockers) else "warning",
            inputs={
                "proposed_tests": list(plan.proposed_tests),
                "uncertainty_flags": list(plan.uncertainty_flags),
                "blockers": list(primitive_plan.blockers if primitive_plan else ()),
                "validation": resolved_context.get("validation"),
            },
            outputs={
                "validation_state": resolved_context.get("validation_state", "planned"),
                "blocker_report_present": plan.blocker_report is not None,
                "new_primitive_workflow_present": plan.new_primitive_workflow is not None,
            },
            notes=(
                "Record proposed tests, blocker state, and any fallback or reuse notes.",
            ),
        ),
        AnalyticalTraceStep(
            id=f"{trace_id}:output",
            parent_id=root_id,
            kind="output",
            label="Final analytical artifact",
            status=status or _default_trace_status(plan, primitive_plan),
            inputs={
                "route": route_name,
                "model": model,
            },
            outputs={
                "route_card": render_generation_route_card(plan),
                "route_plan": render_generation_plan(plan),
                "trace_type": "analytical",
            },
            notes=(
                "Persist both the machine-readable trace and the text rendering from the same source of truth.",
            ),
        ),
    )

    return AnalyticalTrace(
        trace_id=trace_id,
        route=AnalyticalTraceRoute(
            family=route_family,
            name=route_name,
            model=model,
        ),
        status=status or _default_trace_status(plan, primitive_plan),
        created_at=created_at,
        updated_at=created_at,
        steps=steps,
        context=resolved_context,
        trace_type="analytical",
        task_id=task_id,
        issue_id=issue_id,
    )


def save_analytical_trace(
    trace: AnalyticalTrace,
    *,
    root: Path | None = None,
) -> AnalyticalTraceArtifacts:
    """Persist a trace as JSON plus a human-readable Markdown rendering."""
    root = root or TRACE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{trace.trace_id}.json"
    text_path = root / f"{trace.trace_id}.md"
    trace_to_write = replace(
        trace,
        json_path=str(json_path),
        text_path=str(text_path),
        updated_at=_now_utc(),
    )
    payload = trace_to_write.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_analytical_trace(trace_to_write))
    return AnalyticalTraceArtifacts(
        trace=trace_to_write,
        json_path=json_path,
        text_path=text_path,
    )


def emit_analytical_trace_from_generation_plan(
    plan: GenerationPlan,
    *,
    trace_id: str | None = None,
    task_id: str | None = None,
    issue_id: str | None = None,
    route_family: str | None = None,
    model: str | None = None,
    status: str | None = None,
    context: dict[str, Any] | None = None,
    root: Path | None = None,
) -> AnalyticalTraceArtifacts:
    """Build and persist an analytical trace from a generation plan."""
    trace = build_analytical_trace_from_generation_plan(
        plan,
        trace_id=trace_id,
        task_id=task_id,
        issue_id=issue_id,
        route_family=route_family,
        model=model,
        status=status,
        context=context,
    )
    return save_analytical_trace(trace, root=root)


def load_analytical_traces(*, root: Path | None = None) -> list[AnalyticalTrace]:
    """Load persisted analytical traces from disk."""
    root = root or TRACE_ROOT
    if not root.exists():
        return []
    traces: list[AnalyticalTrace] = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        traces.append(AnalyticalTrace.from_dict(data))
    return traces


def render_analytical_trace(trace: AnalyticalTrace) -> str:
    """Render a text trace from the structured analytical trace object."""
    lines = [
        f"# Analytical Trace: `{trace.trace_id}`",
        f"- Trace type: `{trace.trace_type}`",
        f"- Route family: `{trace.route.family}`",
        f"- Route name: `{trace.route.name}`",
        f"- Model: `{trace.route.model}`",
        f"- Status: `{trace.status}`",
        f"- Created at: `{trace.created_at}`",
        f"- Updated at: `{trace.updated_at}`",
    ]
    if trace.task_id:
        lines.append(f"- Task ID: `{trace.task_id}`")
    if trace.issue_id:
        lines.append(f"- Issue ID: `{trace.issue_id}`")
    if trace.context:
        lines.append("")
        lines.append("## Context")
        for key in sorted(trace.context):
            lines.append(f"- `{key}`: {trace.context[key]!r}")
    lines.append("")
    lines.append("## Steps")
    for step in trace.steps:
        depth = 0 if step.parent_id is None else 1
        prefix = "  " * depth
        lines.append(f"{prefix}- **{step.kind}** `{step.id}`")
        lines.append(f"{prefix}  - Label: {step.label}")
        lines.append(f"{prefix}  - Status: `{step.status}`")
        if step.parent_id is not None:
            lines.append(f"{prefix}  - Parent: `{step.parent_id}`")
        if step.notes:
            lines.append(f"{prefix}  - Notes:")
            lines.extend(f"{prefix}    - {note}" for note in step.notes)
        if step.inputs:
            lines.append(f"{prefix}  - Inputs:")
            lines.extend(
                f"{prefix}    - `{key}`: {_stringify(value)}"
                for key, value in sorted(step.inputs.items())
            )
        if step.outputs:
            lines.append(f"{prefix}  - Outputs:")
            lines.extend(
                f"{prefix}    - `{key}`: {_stringify(value)}"
                for key, value in sorted(step.outputs.items())
            )
    return "\n".join(lines)


def _default_trace_id(route_family: str, route_name: str, instrument_type: str | None) -> str:
    parts = [
        _slug(route_family),
        _slug(route_name),
    ]
    if instrument_type:
        parts.append(_slug(instrument_type))
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    return "_".join(part for part in parts if part)


def _default_trace_status(plan: GenerationPlan, primitive_plan: PrimitivePlan | None) -> str:
    if primitive_plan is not None and primitive_plan.blockers:
        return "warning"
    if plan.uncertainty_flags:
        return "warning"
    return "ok"


def _generation_plan_context(
    plan: GenerationPlan,
    *,
    instruction_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primitive_plan = plan.primitive_plan
    return {
        "method": plan.method,
        "instrument_type": plan.instrument_type,
        "inspected_modules": list(plan.inspected_modules),
        "approved_modules": list(plan.approved_modules),
        "symbols_to_reuse": list(plan.symbols_to_reuse),
        "proposed_tests": list(plan.proposed_tests),
        "uncertainty_flags": list(plan.uncertainty_flags),
        "repo_revision": plan.repo_revision,
        "instruction_resolution": instruction_resolution or _instruction_resolution_context(plan),
        "primitive_plan": {
            "route": primitive_plan.route if primitive_plan else None,
            "engine_family": primitive_plan.engine_family if primitive_plan else None,
            "route_family": primitive_plan.route_family if primitive_plan else None,
            "score": primitive_plan.score if primitive_plan else None,
        },
    }


def _instruction_resolution_context(plan: GenerationPlan) -> dict[str, Any]:
    resolved = getattr(plan, "resolved_instructions", None)
    if resolved is None:
        from trellis.agent.codegen_guardrails import _resolve_generation_instructions

        resolved = _resolve_generation_instructions(plan)
    return {
        "route": resolved.route,
        "effective_instruction_count": len(resolved.effective_instructions),
        "dropped_instruction_count": len(resolved.dropped_instructions),
        "conflict_count": len(resolved.conflicts),
        "effective_instructions": [asdict(instruction) for instruction in resolved.effective_instructions],
        "dropped_instructions": [asdict(instruction) for instruction in resolved.dropped_instructions],
        "conflicts": [asdict(conflict) for conflict in resolved.conflicts],
    }


def _has_terminal_basis_assembly(plan: GenerationPlan) -> bool:
    primitive_plan = plan.primitive_plan
    if primitive_plan is None:
        return False
    route = primitive_plan.route
    return route in {"analytical_black76", "analytical_garman_kohlhagen", "quanto_adjustment_analytical"}


def _route_helper_name(primitive_plan: PrimitivePlan | None) -> str | None:
    if primitive_plan is None:
        return None
    for primitive in primitive_plan.primitives:
        if primitive.role == "route_helper":
            return f"{primitive.module}.{primitive.symbol}"
    return None


def _primitive_refs_to_dicts(primitives: tuple[PrimitiveRef, ...]) -> list[dict[str, Any]]:
    return [asdict(primitive) for primitive in primitives]


def _primitive_modules(primitive_plan: PrimitivePlan | None) -> list[str]:
    if primitive_plan is None:
        return []
    return [primitive.module for primitive in primitive_plan.primitives]


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or "trace"


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    return str(value)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
