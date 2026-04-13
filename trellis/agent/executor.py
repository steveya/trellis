"""Top-level agent executor: plan → build → fetch → price → return."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import textwrap
import time
from dataclasses import asdict, dataclass, is_dataclass, replace as replace_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from types import SimpleNamespace

from trellis.agent.codegen_guardrails import (
    GeneratedSourceSanitizationReport,
    build_generation_plan,
    render_generation_route_card,
    sanitize_generated_source,
    validate_generated_imports,
)
from trellis.agent.blocker_planning import render_blocker_report
from trellis.agent.introspection import (
    find_symbol,
    get_package_tree,
    list_module_exports,
    read_module_source,
    search_lessons,
    search_package,
    search_tests,
)
from trellis.agent.analytical_traces import emit_analytical_trace_from_generation_plan
from trellis.agent.semantic_validation import validate_semantics
from trellis.agent.builder import write_module, run_tests
from trellis.agent.knowledge.import_registry import resolve_import_candidates
from trellis.agent.knowledge.api_map import format_api_map_for_prompt


# Sentinel in the skeleton that gets replaced by the LLM-generated body.
EVALUATE_SENTINEL = '        raise NotImplementedError("evaluate not yet implemented")'
_ROUTE_GUESSING_BLOCKER_REASON = "route guessing"

_REPO_REVISION: str | None = None


def _get_repo_revision() -> str:
    """Return the current git commit SHA (first 16 chars) or 'unknown'. Cached."""
    global _REPO_REVISION
    if _REPO_REVISION is not None:
        return _REPO_REVISION
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            _REPO_REVISION = result.stdout.strip()[:16]
            return _REPO_REVISION
    except Exception:
        pass
    _REPO_REVISION = "unknown"
    return _REPO_REVISION


@dataclass(frozen=True)
class GeneratedModuleResult:
    """Successful generated source plus its sanitation report."""

    raw_code: str
    sanitized_code: str
    code: str
    source_report: GeneratedSourceSanitizationReport


@dataclass(frozen=True)
class KnowledgeRetrievalRequest:
    """One stage-aware knowledge retrieval request issued during a build."""

    audience: str
    stage: str
    attempt_number: int
    knowledge_surface: str
    prompt_surface: str
    retry_reason: str | None
    instrument_type: str | None
    pricing_method: str | None
    product_ir: Any = None
    compiled_request: Any = None
    recent_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeContextResult:
    """Resolved knowledge text plus retrieval metadata for one attempt."""

    text: str
    knowledge_surface: str
    retrieval_stage: str
    retrieval_source: str


class GeneratedModuleSourceError(RuntimeError):
    """Raised when generated source cannot be sanitized for execution."""

    def __init__(
        self,
        message: str,
        *,
        source_report: GeneratedSourceSanitizationReport,
    ) -> None:
        super().__init__(message)
        self.source_report = source_report


def _llm_stage_metadata(
    *,
    compiled_request,
    model: str,
    attempt: int | None = None,
    instrument_type: str | None = None,
) -> dict[str, Any]:
    """Attach stable request metadata to LLM stage logs."""
    request = getattr(compiled_request, "request", None)
    request_metadata = getattr(request, "metadata", None) or {}
    metadata = {
        "model": model,
        "attempt": attempt,
        "instrument_type": instrument_type,
        "request_id": getattr(request, "request_id", None),
        "task_id": request_metadata.get("task_id"),
        "comparison_target": request_metadata.get("comparison_target"),
        "preferred_method": request_metadata.get("preferred_method"),
    }
    return {
        key: value
        for key, value in metadata.items()
        if value not in {None, ""}
    }


def _render_spec_default_value(field_type: str, default: str) -> str:
    """Render a spec default using valid Python syntax for the declared field type."""
    normalized_type = field_type.replace(" ", "").lower()
    if "str" in normalized_type:
        if default == "None":
            return "None"
        try:
            parsed_default = ast.literal_eval(default)
        except (SyntaxError, ValueError):
            parsed_default = default
        if isinstance(parsed_default, str):
            return repr(parsed_default)
        return repr(default)
    return default


def _hydrate_spec_schema_defaults_from_semantics(
    spec_schema,
    *,
    semantic_contract=None,
):
    """Overlay semantic term-field defaults onto a deterministic spec schema.

    Static planner schemas intentionally stay generic. When the semantic layer
    has already recovered contract conventions, hydrate those defaults here so
    skeleton generation, smoke tests, and validation all operate on the same
    contract surface.
    """
    if spec_schema is None or semantic_contract is None:
        return spec_schema

    product = getattr(semantic_contract, "product", None)
    term_fields = dict(getattr(product, "term_fields", {}) or {})
    if not term_fields:
        return spec_schema

    spec_name = str(getattr(spec_schema, "spec_name", "") or "")
    if spec_name not in {"SwaptionSpec", "BermudanSwaptionSpec"}:
        return spec_schema

    from trellis.agent.planner import FieldDef, SpecSchema

    def _enum_default(prefix: str, raw_value: object | None) -> str | None:
        if raw_value in {None, ""}:
            return None
        text = str(raw_value).strip()
        if not text:
            return None
        if text.startswith(f"{prefix}."):
            return text
        return f"{prefix}.{text}"

    overrides: dict[str, str] = {}
    day_count_default = (
        _enum_default("DayCountConvention", term_fields.get("fixed_leg_day_count"))
        or _enum_default("DayCountConvention", term_fields.get("day_count"))
    )
    if day_count_default is not None:
        overrides["day_count"] = day_count_default

    swap_frequency_default = _enum_default(
        "Frequency",
        term_fields.get("payment_frequency") or term_fields.get("swap_frequency"),
    )
    if swap_frequency_default is not None:
        overrides["swap_frequency"] = swap_frequency_default

    rate_index = term_fields.get("rate_index")
    if rate_index not in {None, ""}:
        overrides["rate_index"] = str(rate_index).strip()

    if not overrides:
        return spec_schema

    fields = []
    changed = False
    for field in getattr(spec_schema, "fields", ()):
        default = overrides.get(field.name, field.default)
        if default != field.default:
            changed = True
        fields.append(
            FieldDef(
                name=field.name,
                type=field.type,
                description=field.description,
                default=default,
            )
        )

    if not changed:
        return spec_schema

    return SpecSchema(
        class_name=spec_schema.class_name,
        spec_name=spec_schema.spec_name,
        requirements=list(getattr(spec_schema, "requirements", ())),
        fields=fields,
    )


def _handle_tool_call(name: str, input_data: dict) -> str:
    """Dispatch a tool call from the LLM agent."""
    if name == "inspect_api_map":
        return format_api_map_for_prompt(compact=True)

    if name == "inspect_library":
        tree = get_package_tree()
        return json.dumps(tree, indent=2, default=str)

    elif name == "read_module":
        try:
            src = read_module_source(input_data["module_path"])
            return src
        except Exception as e:
            return f"Error reading module: {e}"

    elif name == "find_symbol":
        try:
            matches = find_symbol(input_data["symbol"])
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error finding symbol: {e}"

    elif name == "list_exports":
        try:
            exports = list_module_exports(input_data["module_path"])
            return json.dumps(exports, indent=2)
        except Exception as e:
            return f"Error listing exports: {e}"

    elif name == "resolve_import_candidates":
        try:
            candidates = resolve_import_candidates(input_data["symbols"])
            return json.dumps(candidates, indent=2)
        except Exception as e:
            return f"Error resolving imports: {e}"

    elif name == "lookup_primitive_route":
        try:
            from trellis.agent.assembly_tools import lookup_primitive_route

            payload = lookup_primitive_route(
                description=input_data["description"],
                instrument_type=input_data.get("instrument_type"),
                preferred_method=input_data.get("preferred_method"),
            )
            return json.dumps(asdict(payload), indent=2)
        except Exception as e:
            return f"Error building primitive route plan: {e}"

    elif name == "build_thin_adapter_plan":
        try:
            from trellis.agent.assembly_tools import build_thin_adapter_plan
            from trellis.agent.codegen_guardrails import build_generation_plan
            from trellis.agent.knowledge.decompose import decompose_to_ir
            from trellis.agent.quant import (
                select_pricing_method,
                select_pricing_method_for_product_ir,
            )

            description = input_data["description"]
            instrument_type = input_data.get("instrument_type")
            preferred_method = input_data.get("preferred_method")
            class_name = input_data.get("class_name", "GeneratedPayoff")

            product_ir = decompose_to_ir(description, instrument_type=instrument_type)
            if preferred_method:
                pricing_plan = select_pricing_method_for_product_ir(
                    product_ir,
                    preferred_method=preferred_method,
                    context_description=description,
                )
            else:
                pricing_plan = select_pricing_method(description, instrument_type=instrument_type)
            generation_plan = build_generation_plan(
                pricing_plan=pricing_plan,
                instrument_type=instrument_type or getattr(product_ir, "instrument", None),
                inspected_modules=tuple(pricing_plan.method_modules),
                product_ir=product_ir,
            )
            spec_schema = SimpleNamespace(class_name=class_name, fields=[])
            payload = build_thin_adapter_plan(
                spec_schema,
                pricing_plan=pricing_plan,
                generation_plan=generation_plan,
            )
            return json.dumps(asdict(payload), indent=2)
        except Exception as e:
            return f"Error building thin adapter plan: {e}"

    elif name == "select_invariant_pack":
        try:
            from trellis.agent.assembly_tools import select_invariant_pack

            payload = select_invariant_pack(
                instrument_type=input_data.get("instrument_type"),
                method=input_data["method"],
            )
            return json.dumps(asdict(payload), indent=2)
        except Exception as e:
            return f"Error selecting invariant pack: {e}"

    elif name == "build_comparison_harness":
        try:
            from trellis.agent.assembly_tools import build_comparison_harness_plan

            payload = build_comparison_harness_plan(input_data["task"])
            return json.dumps(
                {
                    "targets": [asdict(target) for target in payload.targets],
                    "reference_target": payload.reference_target,
                    "tolerance_pct": payload.tolerance_pct,
                },
                indent=2,
            )
        except Exception as e:
            return f"Error building comparison harness plan: {e}"

    elif name == "capture_cookbook_candidate":
        try:
            from trellis.agent.assembly_tools import build_cookbook_candidate_payload

            payload = build_cookbook_candidate_payload(
                method=input_data["method"],
                description=input_data["description"],
                code=input_data["code"],
            )
            return json.dumps(payload or {}, indent=2)
        except Exception as e:
            return f"Error extracting cookbook candidate: {e}"

    elif name == "search_repo":
        try:
            matches = search_package(
                input_data["pattern"],
                limit=int(input_data.get("limit", 20)),
            )
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error searching repo: {e}"

    elif name == "search_tests":
        try:
            matches = search_tests(
                input_data["pattern"],
                limit=int(input_data.get("limit", 20)),
            )
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error searching tests: {e}"

    elif name == "search_lessons":
        try:
            matches = search_lessons(
                input_data["pattern"],
                limit=int(input_data.get("limit", 20)),
            )
            return json.dumps(matches, indent=2)
        except Exception as e:
            return f"Error searching lessons: {e}"

    elif name == "write_module":
        path = write_module(input_data["file_path"], input_data["content"])
        return f"Module written to {path}"

    elif name == "run_tests":
        result = run_tests(input_data.get("test_path"))
        return json.dumps(result, indent=2)

    elif name == "fetch_market_data":
        source = input_data.get("source", "fred")
        as_of_str = input_data.get("as_of")
        as_of = datetime.strptime(as_of_str, "%Y-%m-%d").date() if as_of_str else None

        if source == "fred":
            from trellis.data.fred import FredDataProvider
            data = FredDataProvider().fetch_yields(as_of)
        else:
            from trellis.data.treasury_gov import TreasuryGovDataProvider
            data = TreasuryGovDataProvider().fetch_yields(as_of)
        return json.dumps(data, indent=2)

    elif name == "execute_pricing":
        from trellis.instruments.bond import Bond
        from trellis.curves.yield_curve import YieldCurve
        from trellis.engine.pricer import price_instrument

        params = input_data["params"]
        curve_data = {float(k): float(v) for k, v in input_data["curve_data"].items()}
        curve = YieldCurve.from_treasury_yields(curve_data)

        instrument_type = input_data.get("instrument_type", "Bond")
        if instrument_type == "Bond":
            if "maturity_date" in params and isinstance(params["maturity_date"], str):
                params["maturity_date"] = datetime.strptime(params["maturity_date"], "%Y-%m-%d").date()
            bond = Bond(**params)
        else:
            return f"Unknown instrument type: {instrument_type}"

        result = price_instrument(bond, curve)
        return json.dumps({
            "clean_price": result.clean_price,
            "dirty_price": result.dirty_price,
            "accrued_interest": result.accrued_interest,
            "greeks": result.greeks,
        }, indent=2, default=str)

    return f"Unknown tool: {name}"


def _record_platform_event(
    compiled_request,
    event: str,
    *,
    status: str = "info",
    success: bool | None = None,
    outcome: str | None = None,
    details: dict | None = None,
) -> None:
    """Best-effort request audit event emission."""
    if compiled_request is None:
        return
    try:
        from trellis.agent.platform_traces import append_platform_trace_event

        append_platform_trace_event(
            compiled_request,
            event,
            status=status,
            success=success,
            outcome=outcome,
            details=details,
        )
    except Exception:
        pass


def _append_agent_observation(
    build_meta: dict | None,
    agent: str,
    kind: str,
    summary: str,
    *,
    severity: str = "info",
    attempt: int | None = None,
    details=None,
) -> None:
    """Best-effort structured memory capture for cross-agent learning."""
    if build_meta is None:
        return
    observation = {
        "agent": agent,
        "kind": kind,
        "summary": summary,
        "severity": severity,
    }
    if attempt is not None:
        observation["attempt"] = attempt
    if details is not None:
        observation["details"] = details
    build_meta.setdefault("agent_observations", []).append(observation)


def _semantic_role_ownership_details(
    *,
    stage: str,
    compiled_request=None,
    trigger_condition: str | None = None,
    artifact_kind: str | None = None,
    review_policy=None,
    executed: bool | None = None,
) -> dict[str, object]:
    """Build a compact role-ownership summary for a trace event."""
    from trellis.agent.semantic_escalation import semantic_role_ownership_summary

    request = getattr(compiled_request, "request", None)
    metadata = getattr(request, "metadata", None)
    semantic_gap = None
    semantic_extension = None
    if isinstance(metadata, Mapping):
        semantic_gap = metadata.get("semantic_gap")
        semantic_extension = metadata.get("semantic_extension")

    summary = semantic_role_ownership_summary(
        stage=stage,
        semantic_gap=semantic_gap if isinstance(semantic_gap, Mapping) else None,
        semantic_extension=semantic_extension if isinstance(semantic_extension, Mapping) else None,
        semantic_contract=bool(getattr(compiled_request, "semantic_contract", None)),
        review_policy=review_policy,
        trigger_condition=trigger_condition,
        artifact_kind=artifact_kind,
        executed=executed,
    )
    return {
        "ownership_stage": summary.get("selected_stage", stage),
        "owner_role": summary.get("selected_role", ""),
        "ownership_trigger_condition": summary.get("trigger_condition", ""),
        "ownership_artifact_kind": summary.get("artifact_kind", ""),
        "ownership_scope": summary.get("scope", ""),
        "ownership_executed": summary.get("executed", False),
        "ownership_summary": summary.get("summary", ""),
    }


def _emit_analytical_trace_metadata(
    *,
    build_meta: dict | None,
    generation_plan,
    compiled_request,
    spec_schema,
    market_state=None,
):
    """Persist analytical trace metadata into build tracking state."""
    selected_curve_names = dict(
        getattr(market_state, "selected_curve_names", None) or {}
    )
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    route_family = (
        getattr(primitive_plan, "route_family", None)
        if primitive_plan is not None
        else None
    ) or getattr(generation_plan, "method", None)
    analytical_trace = emit_analytical_trace_from_generation_plan(
        generation_plan,
        trace_id=getattr(getattr(compiled_request, "request", None), "request_id", None),
        task_id=getattr(getattr(compiled_request, "request", None), "request_id", None),
        issue_id=getattr(compiled_request, "linear_issue_identifier", None),
        route_family=route_family,
        model=getattr(primitive_plan, "engine_family", None) or getattr(generation_plan, "method", None),
        context={
            "spec_name": getattr(spec_schema, "spec_name", None),
            "class_name": getattr(spec_schema, "class_name", None),
            "route_card": render_generation_route_card(generation_plan),
            "selected_curve_names": selected_curve_names,
        },
    )
    if build_meta is not None:
        build_meta["analytical_trace_id"] = analytical_trace.trace.trace_id
        build_meta["analytical_trace_path"] = str(analytical_trace.json_path)
        build_meta["analytical_trace_text_path"] = str(analytical_trace.text_path)
    return analytical_trace


def _finalizes_in_executor(compiled_request) -> bool:
    """Standalone build requests terminate inside the executor."""
    if compiled_request is None:
        return False
    return getattr(compiled_request.request, "request_type", None) == "build"


def _semantic_clarification_blocker_details(compiled_request) -> dict[str, object] | None:
    """Return structured details when the compiled request still needs clarification."""
    request = getattr(compiled_request, "request", None)
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None

    semantic_gap = metadata.get("semantic_gap")
    semantic_extension = metadata.get("semantic_extension")
    semantic_gap_data = dict(semantic_gap) if isinstance(semantic_gap, Mapping) else None
    semantic_extension_data = (
        dict(semantic_extension) if isinstance(semantic_extension, Mapping) else None
    )
    requires_clarification = bool(
        semantic_gap_data and semantic_gap_data.get("requires_clarification")
    )
    clarification_decision = bool(
        semantic_extension_data and semantic_extension_data.get("decision") == "clarification"
    )
    if not (requires_clarification or clarification_decision):
        return None

    summary = ""
    if semantic_extension_data and semantic_extension_data.get("summary"):
        summary = str(semantic_extension_data["summary"])
    elif semantic_gap_data and semantic_gap_data.get("summary"):
        summary = str(semantic_gap_data["summary"])

    return {
        "reason": "semantic_clarification_required",
        "summary": summary,
        "semantic_gap": semantic_gap_data,
        "semantic_extension": semantic_extension_data,
    }


def _pre_generation_gate_blocker_details(
    generation_plan,
    *,
    gate_decision=None,
) -> dict[str, object] | None:
    """Project generation-plan blockers into a persisted structured payload.

    Pre-generation gate blocks happen before the later primitive-blocker branch,
    so persist the same blocker taxonomy here when it is already available.
    """
    details: dict[str, object] = {}

    def _serialize_structured_entry(entry: object) -> dict[str, object]:
        if is_dataclass(entry):
            return asdict(entry)
        if isinstance(entry, Mapping):
            return dict(entry)
        if hasattr(entry, "__dict__"):
            return dict(vars(entry))
        raise TypeError(f"Unsupported structured blocker entry: {type(entry)!r}")

    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    raw_blockers = tuple(getattr(primitive_plan, "blockers", ()) or ())
    if raw_blockers:
        details["blockers"] = list(raw_blockers)

    blocker_report = getattr(generation_plan, "blocker_report", None)
    if blocker_report is not None:
        details["blocker_codes"] = [
            blocker.id for blocker in getattr(blocker_report, "blockers", ())
        ]
        details["blocker_report"] = {
            "summary": getattr(blocker_report, "summary", ""),
            "should_block": bool(getattr(blocker_report, "should_block", False)),
            "blockers": [
                _serialize_structured_entry(blocker)
                for blocker in getattr(blocker_report, "blockers", ())
            ],
        }

    new_primitive_workflow = getattr(generation_plan, "new_primitive_workflow", None)
    if new_primitive_workflow is not None:
        details["new_primitive_workflow"] = {
            "summary": getattr(new_primitive_workflow, "summary", ""),
            "items": [
                _serialize_structured_entry(item)
                for item in getattr(new_primitive_workflow, "items", ())
            ],
        }

    if gate_decision is not None:
        details["gate_decision"] = asdict(gate_decision)
        route_admissibility = tuple(
            getattr(gate_decision, "route_admissibility_failures", ()) or ()
        )
        if route_admissibility:
            details["route_admissibility_failures"] = list(route_admissibility)
        reason = str(getattr(gate_decision, "reason", "") or "").strip()
        if (
            reason
            and "blocker_report" not in details
            and _ROUTE_GUESSING_BLOCKER_REASON in reason.lower()
        ):
            details["blocker_codes"] = ["missing_binding_surface:lane_without_exact_binding"]
            details["blocker_report"] = {
                "summary": reason,
                "should_block": True,
                "blockers": [
                    {
                        "id": "missing_binding_surface:lane_without_exact_binding",
                        "category": "missing_binding_surface",
                        "primitive_kind": "backend_binding",
                        "severity": "high",
                        "summary": reason,
                    }
                ],
            }

    return details or None


# ---------------------------------------------------------------------------
# Audit record writer (non-blocking helper called at build success)
# ---------------------------------------------------------------------------

def _write_build_audit_record(
    *,
    compiled_request,
    model: str | None,
    pricing_plan,
    instrument_type: str | None,
    spec_schema,
    output_module_path: str,
    code: str,
    market_state,
    attempt_number: int,
    gate_results: list,
    build_start_time: float,
) -> "Path | None":
    """Write a ModelAuditRecord at build success. Non-blocking — exceptions are swallowed."""
    path_out: Path | None = None
    try:
        from trellis.agent.model_audit import build_audit_record, write_model_audit_record, ValidationGateResult
        from trellis.agent.knowledge.store import KnowledgeStore

        request = getattr(compiled_request, "request", None)
        run_id = getattr(request, "request_id", None) or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        task_id = "standalone"

        market_summary = {}
        if market_state is not None:
            try:
                market_summary = market_state.summarize_for_audit()
            except Exception:
                pass

        pricing_plan_summary = {
            "method": getattr(pricing_plan, "method", ""),
            "required_market_data": sorted(getattr(pricing_plan, "required_market_data", [])),
            "method_modules": list(getattr(pricing_plan, "method_modules", [])),
            "selection_reason": getattr(pricing_plan, "selection_reason", ""),
            "modeling_requirements": list(getattr(pricing_plan, "modeling_requirements", [])),
        }

        spec_dict = {
            "class_name": getattr(spec_schema, "class_name", ""),
            "spec_name": getattr(spec_schema, "spec_name", ""),
            "requirements": list(getattr(spec_schema, "requirements", [])),
            "fields": [getattr(f, "name", str(f)) for f in getattr(spec_schema, "fields", [])],
        }

        knowledge_hash = "unknown"
        try:
            knowledge_hash = KnowledgeStore.instance().compute_knowledge_hash()
        except Exception:
            pass

        # Convert flat tuples to ValidationGateResult objects
        vgr_list = []
        for entry in gate_results:
            if len(entry) == 3:
                gate, passed, issues = entry
                details: dict = {}
            else:
                gate, passed, issues, details = entry
            vgr_list.append(ValidationGateResult(
                gate=gate,
                passed=passed,
                issues=tuple(str(i) for i in issues),
                details=dict(details),
            ))

        audit_rec = build_audit_record(
            task_id=task_id,
            run_id=run_id,
            method=getattr(pricing_plan, "method", "unknown"),
            instrument_type=instrument_type or "",
            source_code=code,
            spec_schema_dict=spec_dict,
            class_name=getattr(spec_schema, "class_name", ""),
            module_path=output_module_path,
            repo_revision=_get_repo_revision(),
            llm_model_id=model or "",
            knowledge_hash=knowledge_hash,
            market_state_summary=market_summary,
            pricing_plan_summary=pricing_plan_summary,
            validation_gates=vgr_list,
            attempt_number=attempt_number,
            total_attempts=attempt_number,
            wall_clock_seconds=time.time() - build_start_time,
        )
        path_out = write_model_audit_record(audit_rec)
    except Exception:
        pass  # audit write is best-effort and must never block a successful build
    return path_out


# ---------------------------------------------------------------------------
# Structured payoff builder (two-step pipeline)
# ---------------------------------------------------------------------------

def build_payoff(
    payoff_description: str,
    requirements: set[str] | None = None,
    model: str | None = None,
    max_retries: int = 3,
    force_rebuild: bool = False,
    fresh_build: bool = False,
    validation: str = "standard",
    market_state=None,
    instrument_type: str | None = None,
    preferred_method: str | None = None,
    request_metadata: Mapping[str, object] | None = None,
    compiled_request=None,
    build_meta: dict | None = None,
    gap_report=None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None = None,
) -> type:
    """Build a Payoff class via the multi-agent pipeline.

    Pipeline:
    1. **Quant agent** selects the pricing method and data requirements
    2. **Data check** verifies required market data is available
    3. **Planner** determines spec schema and module path
    4. **Builder agent** generates the code using the prescribed method
    5. **Critic agent** reviews the code
    6. **Arbiter** validates with invariants
    """
    from trellis.agent.planner import plan_build
    from trellis.agent.quant import (
        check_data_availability,
        select_pricing_method,
        select_pricing_method_for_product_ir,
    )
    from trellis.agent.builder import dynamic_import, ensure_agent_package
    from trellis.agent.config import (
        enforce_llm_token_budget,
        get_default_model,
        get_model_for_stage,
        llm_usage_stage,
        summarize_llm_usage,
    )
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.platform_requests import compile_build_request
    from trellis.core.market_state import MissingCapabilityError

    model = model or get_default_model()
    product_ir = None
    if compiled_request is None:
        try:
            compiled_request = compile_build_request(
                payoff_description,
                instrument_type=instrument_type,
                market_snapshot=getattr(market_state, "market_snapshot", None),
                settlement=getattr(market_state, "settlement", None),
                model=model,
                preferred_method=preferred_method,
                metadata=request_metadata,
            )
            product_ir = compiled_request.product_ir
        except Exception:
            try:
                product_ir = decompose_to_ir(
                    payoff_description,
                    instrument_type=instrument_type,
                )
            except Exception:
                product_ir = None
    else:
        product_ir = compiled_request.product_ir

    if compiled_request is not None:
        clarification_blocker = _semantic_clarification_blocker_details(compiled_request)
        if clarification_blocker is not None:
            if build_meta is not None:
                build_meta["blocker_details"] = clarification_blocker
            _record_platform_event(
                compiled_request,
                "request_blocked" if _finalizes_in_executor(compiled_request) else "build_blocked",
                status="error",
                success=False if _finalizes_in_executor(compiled_request) else None,
                outcome="request_blocked" if _finalizes_in_executor(compiled_request) else None,
                details=clarification_blocker,
            )
            raise RuntimeError(
                "Cannot generate payoff because the semantic request still requires clarification: "
                + clarification_blocker.get("summary", "unsupported request")
            )

    if build_meta is not None and compiled_request is not None:
        build_meta.setdefault(
            "knowledge_summary",
            dict(getattr(compiled_request, "knowledge_summary", {}) or {}),
        )

    # Step 1: Quant agent selects method + data requirements
    pricing_plan = (
        compiled_request.pricing_plan if compiled_request is not None else None
    ) or (
        select_pricing_method_for_product_ir(
            product_ir,
            preferred_method=preferred_method,
            context_description=payoff_description,
        )
        if product_ir is not None and preferred_method is not None
        else select_pricing_method(
            payoff_description,
            instrument_type=instrument_type,
            model=model,
        )
    )
    _record_platform_event(
        compiled_request,
        "quant_selected_method",
        status="ok",
        details={
            **_semantic_role_ownership_details(
                stage="primitive_proposal",
                compiled_request=compiled_request,
                trigger_condition=pricing_plan.selection_reason or "pricing_plan_selection",
                artifact_kind="PricingPlan",
            ),
            "method": pricing_plan.method,
            "selection_reason": pricing_plan.selection_reason,
            "assumption_summary": list(pricing_plan.assumption_summary),
            "required_market_data": sorted(pricing_plan.required_market_data),
            "sensitivity_support": (
                pricing_plan.sensitivity_support.to_dict()
                if pricing_plan.sensitivity_support is not None
                else {}
            ),
        },
    )
    _append_agent_observation(
        build_meta,
        "quant",
        "decision",
        f"Selected pricing method `{pricing_plan.method}`",
        details={
            "semantic_role_ownership": _semantic_role_ownership_details(
                stage="primitive_proposal",
                compiled_request=compiled_request,
                trigger_condition=pricing_plan.selection_reason or "pricing_plan_selection",
                artifact_kind="PricingPlan",
            ),
            "required_market_data": sorted(pricing_plan.required_market_data),
            "method_modules": list(pricing_plan.method_modules),
            "reasoning": pricing_plan.reasoning,
            "selection_reason": pricing_plan.selection_reason,
            "assumption_summary": list(pricing_plan.assumption_summary),
            "sensitivity_support": (
                pricing_plan.sensitivity_support.to_dict()
                if pricing_plan.sensitivity_support is not None
                else {}
            ),
        },
    )

    # Step 2: Check market data availability (early — before writing code)
    if market_state is not None:
        data_errors = check_data_availability(pricing_plan, market_state)
        if data_errors:
            _record_platform_event(
                compiled_request,
                "request_failed" if _finalizes_in_executor(compiled_request) else "build_failed",
                status="error",
                success=False if _finalizes_in_executor(compiled_request) else None,
                outcome="request_failed" if _finalizes_in_executor(compiled_request) else None,
                details={
                    "reason": "missing_market_data",
                    "data_errors": data_errors,
                },
            )
            raise MissingCapabilityError(
                pricing_plan.required_market_data - market_state.available_capabilities,
                market_state.available_capabilities,
                details=data_errors,
            )

    # Use quant agent's data requirements if caller didn't specify
    if requirements is None:
        requirements = pricing_plan.required_market_data

    # Step 3: Plan (spec schema + module path)
    # Extract spec_schema_hint from compiled request blueprint if available
    _spec_hint = None
    _bp = getattr(compiled_request, "semantic_blueprint", None) if compiled_request is not None else None
    if _bp is not None:
        _spec_hint = getattr(_bp, "spec_schema_hint", None)
    plan = plan_build(
        payoff_description,
        requirements,
        model=model,
        instrument_type=instrument_type or getattr(product_ir, "instrument", None),
        preferred_method=pricing_plan.method,
        spec_schema_hint=_spec_hint,
    )
    hydrated_spec_schema = _hydrate_spec_schema_defaults_from_semantics(
        getattr(plan, "spec_schema", None),
        semantic_contract=(
            getattr(compiled_request, "semantic_contract", None)
            if compiled_request is not None
            else None
        ),
    )
    if hydrated_spec_schema is not getattr(plan, "spec_schema", None):
        plan = replace_dataclass(plan, spec_schema=hydrated_spec_schema)
    _record_platform_event(
        compiled_request,
        "planner_completed",
        status="ok",
        details={
            **_semantic_role_ownership_details(
                stage="route_assembly",
                compiled_request=compiled_request,
                trigger_condition=pricing_plan.selection_reason or "pricing_plan_selection",
                artifact_kind="GenerationPlan",
            ),
            "module_path": plan.steps[0].module_path if plan.steps else "",
            "spec_name": plan.spec_schema.spec_name if plan.spec_schema is not None else "",
            "class_name": plan.spec_schema.class_name if plan.spec_schema is not None else "",
        },
    )

    # Step 3b: Reuse supported deterministic adapters and explicit cached builds.
    existing = _try_import_existing(plan)
    reuse_reason = None
    if existing is not None:
        if _is_deterministic_supported_route(plan) and not fresh_build:
            reuse_reason = "deterministic_supported_route"
        elif not force_rebuild:
            reuse_reason = "cached_generated_module"
    if existing is not None and reuse_reason is not None:
        _record_platform_event(
            compiled_request,
            "existing_generated_module_reused",
            status="ok",
            details={
                "module_path": plan.steps[0].module_path if plan.steps else "",
                "class_name": getattr(existing, "__name__", type(existing).__name__),
                "reason": reuse_reason,
            },
        )
        generation_plan = (
            compiled_request.generation_plan if compiled_request is not None else None
        ) or build_generation_plan(
            pricing_plan=pricing_plan,
            instrument_type=instrument_type,
            inspected_modules=tuple(
                module_path
                for module_path, _ in _reference_modules(
                    pricing_plan,
                    instrument_type=instrument_type or getattr(product_ir, "instrument", None),
                )
            ),
            product_ir=product_ir,
        )
        _emit_analytical_trace_metadata(
            build_meta=build_meta,
            generation_plan=generation_plan,
            compiled_request=compiled_request,
            spec_schema=getattr(plan, "spec_schema", None),
            market_state=market_state,
        )
        return existing
    if existing is not None and _is_deterministic_supported_route(plan) and fresh_build:
        _record_platform_event(
            compiled_request,
            "existing_generated_module_bypassed",
            status="info",
            details={
                "module_path": plan.steps[0].module_path if plan.steps else "",
                "class_name": getattr(existing, "__name__", type(existing).__name__),
                "reason": "fresh_build",
            },
        )

    # Step 4: Design spec
    if plan.spec_schema is not None:
        spec_schema = plan.spec_schema
        _record_platform_event(
            compiled_request,
            "spec_design_skipped",
            status="info",
            details={
                "reason": "deterministic_spec_schema",
                "spec_name": spec_schema.spec_name,
                "class_name": spec_schema.class_name,
                "field_count": len(spec_schema.fields),
            },
        )
    else:
        stage_model = get_model_for_stage("spec_design", model)
        _record_platform_event(
            compiled_request,
            "spec_design_started",
            status="info",
            details={
                "model": stage_model,
                "requirement_count": len(requirements),
                "description_chars": len(payoff_description),
            },
        )
        import time as _time

        _SPEC_DESIGN_MAX_RETRIES = 2
        _last_spec_exc = None
        for _spec_attempt in range(_SPEC_DESIGN_MAX_RETRIES + 1):
            try:
                with llm_usage_stage(
                    "spec_design",
                    metadata=_llm_stage_metadata(
                        compiled_request=compiled_request,
                        model=stage_model,
                        attempt=_spec_attempt + 1,
                        instrument_type=instrument_type,
                    ),
                ) as usage_records:
                    spec_schema = _design_spec(payoff_description, requirements, stage_model)
                _record_platform_event(
                    compiled_request,
                    "spec_design_completed",
                    status="ok",
                    details={
                        "model": stage_model,
                        "spec_name": spec_schema.spec_name,
                        "class_name": spec_schema.class_name,
                        "field_count": len(spec_schema.fields),
                        "token_usage": summarize_llm_usage(usage_records),
                        "attempt": _spec_attempt + 1,
                    },
                )
                _last_spec_exc = None
                break
            except Exception as exc:
                _last_spec_exc = exc
                if _spec_attempt < _SPEC_DESIGN_MAX_RETRIES:
                    _backoff = 0.5 * (2 ** _spec_attempt)
                    _record_platform_event(
                        compiled_request,
                        "spec_design_retry",
                        status="warning",
                        details={
                            "model": stage_model,
                            "attempt": _spec_attempt + 1,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "backoff_seconds": _backoff,
                        },
                    )
                    _time.sleep(_backoff)
                else:
                    _record_platform_event(
                        compiled_request,
                        "spec_design_failed",
                        status="error",
                        details={
                            "model": stage_model,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "attempts": _SPEC_DESIGN_MAX_RETRIES + 1,
                        },
                    )
        if _last_spec_exc is not None:
            raise _last_spec_exc
        enforce_llm_token_budget(stage="spec_design")

    generation_plan = (
        compiled_request.generation_plan if compiled_request is not None else None
    ) or build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=tuple(
            module_path
            for module_path, _ in _reference_modules(
                pricing_plan,
                instrument_type=instrument_type or getattr(product_ir, "instrument", None),
            )
        ),
        product_ir=product_ir,
    )
    _emit_analytical_trace_metadata(
        build_meta=build_meta,
        generation_plan=generation_plan,
        compiled_request=compiled_request,
        spec_schema=spec_schema,
        market_state=market_state,
    )

    # Step 4b: Pre-generation build gate
    from trellis.agent.build_gate import evaluate_pre_generation_gate
    _pre_gen_gate = evaluate_pre_generation_gate(
        gap_report,
        generation_plan,
        semantic_blueprint=getattr(compiled_request, "semantic_blueprint", None),
    )
    if build_meta is not None:
        from dataclasses import asdict as _gate_asdict
        build_meta["build_gate_decision"] = _gate_asdict(_pre_gen_gate)
    if _pre_gen_gate.decision == "block":
        blocker_details = _pre_generation_gate_blocker_details(
            generation_plan,
            gate_decision=_pre_gen_gate,
        )
        if build_meta is not None and blocker_details is not None:
            build_meta["blocker_details"] = blocker_details
        _record_platform_event(
            compiled_request,
            "build_gate_blocked",
            status="error",
            success=False,
            details=blocker_details or {
                "gate_decision": _pre_gen_gate.decision,
                "reason": _pre_gen_gate.reason,
            },
        )
        raise RuntimeError(
            f"Build gate blocked pre-generation: {_pre_gen_gate.reason}"
        )
    if _pre_gen_gate.decision == "clarify":
        _record_platform_event(
            compiled_request,
            "build_gate_clarify",
            status="error",
            success=False,
            details={"gate_decision": _pre_gen_gate.decision, "reason": _pre_gen_gate.reason},
        )
        raise RuntimeError(
            f"Build gate requires clarification: {_pre_gen_gate.reason}"
        )

    # Step 5: Generate skeleton
    skeleton = _generate_skeleton(
        spec_schema,
        payoff_description,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
    )

    # Step 6-9: Generate code with method guidance, validate, retry
    reference_modules = _reference_modules(
        pricing_plan,
        instrument_type=instrument_type or getattr(product_ir, "instrument", None),
    )
    reference_sources = _gather_references(reference_modules)
    if generation_plan.primitive_plan is not None and generation_plan.primitive_plan.blockers:
        blocker_details = {
            "blockers": list(generation_plan.primitive_plan.blockers),
        }
        blocker_chunks = []
        if generation_plan.blocker_report is not None:
            blocker_details["blocker_codes"] = [
                blocker.id for blocker in generation_plan.blocker_report.blockers
            ]
            blocker_details["blocker_report"] = {
                "summary": generation_plan.blocker_report.summary,
                "should_block": generation_plan.blocker_report.should_block,
                "blockers": [asdict(blocker) for blocker in generation_plan.blocker_report.blockers],
            }
            blocker_chunks.append(render_blocker_report(generation_plan.blocker_report))
        else:
            blocker_chunks.append(", ".join(generation_plan.primitive_plan.blockers))
        if generation_plan.new_primitive_workflow is not None:
            from trellis.agent.new_primitive_workflow import render_new_primitive_workflow

            blocker_details["new_primitive_workflow"] = {
                "summary": generation_plan.new_primitive_workflow.summary,
                "items": [asdict(item) for item in generation_plan.new_primitive_workflow.items],
            }
            blocker_chunks.append(
                render_new_primitive_workflow(generation_plan.new_primitive_workflow)
            )
        if build_meta is not None:
            build_meta["blocker_details"] = blocker_details
        _record_platform_event(
            compiled_request,
            "request_blocked" if _finalizes_in_executor(compiled_request) else "build_blocked",
            status="error",
            success=False if _finalizes_in_executor(compiled_request) else None,
            outcome="request_blocked" if _finalizes_in_executor(compiled_request) else None,
            details=blocker_details,
        )
        raise RuntimeError(
            "Cannot generate payoff because primitive planning blockers remain:\n"
            + "\n\n".join(blocker_chunks)
        )

    ensure_agent_package()
    step = plan.steps[0]
    output_module_path = step.module_path
    if fresh_build and _is_deterministic_supported_route(plan):
        output_module_path = _fresh_build_module_path(step.module_path)
    module_name = f"trellis.{output_module_path.replace('/', '.').replace('.py', '')}"
    _record_platform_event(
        compiled_request,
        "build_started",
        status="info",
        details={
            "module_name": module_name,
            "validation": validation,
            "max_retries": max_retries,
        },
    )
    if validation != "thorough":
        _record_platform_event(
            compiled_request,
            "model_validator_skipped",
            status="info",
            details={"validation": validation},
        )

    # Retrieve all relevant knowledge for this task (single call)
    validation_feedback = ""
    payoff_cls = None
    previous_failures = []
    retry_reason = None
    build_start_time = time.time()
    for attempt in range(max_retries):
        attempt_number = attempt + 1
        _attempt_gate_results: list = []
        prompt_surface = _builder_prompt_surface_for_attempt(
            attempt_number=attempt_number,
            retry_reason=retry_reason,
        )
        knowledge_context = _resolve_knowledge_context_for_attempt(
            audience="builder",
            pricing_plan=pricing_plan,
            instrument_type=instrument_type,
            attempt_number=attempt_number,
            retry_reason=retry_reason,
            compiled_request=compiled_request,
            product_ir=product_ir,
            prompt_surface=prompt_surface,
            build_meta=build_meta,
            knowledge_retriever=knowledge_retriever,
            previous_failures=previous_failures,
        )
        knowledge_text = knowledge_context.text
        knowledge_surface = knowledge_context.knowledge_surface
        _record_platform_event(
            compiled_request,
            "builder_attempt_started",
            status="info",
            details={
                "attempt": attempt_number,
                "prompt_surface": prompt_surface,
                "knowledge_surface": knowledge_surface,
                "retrieval_stage": knowledge_context.retrieval_stage,
                "retrieval_source": knowledge_context.retrieval_source,
                "retry_reason": retry_reason,
                "knowledge_context_chars": len(knowledge_text),
            },
        )
        stage_model = get_model_for_stage("code_generation", model)
        generated_module: GeneratedModuleResult | None = None
        generation_token_usage: dict[str, object] = {}
        try:
            generated_module = _materialize_deterministic_exact_binding_module(
                skeleton,
                generation_plan,
                semantic_blueprint=(
                    getattr(compiled_request, "semantic_blueprint", None)
                    if compiled_request is not None
                    else None
                ),
                comparison_target=(
                    (
                        getattr(getattr(compiled_request, "request", None), "metadata", None)
                        or {}
                    ).get("comparison_target")
                    if compiled_request is not None
                    else None
                ),
            )
            if generated_module is None:
                with llm_usage_stage(
                    "code_generation",
                    metadata=_llm_stage_metadata(
                        compiled_request=compiled_request,
                        model=stage_model,
                        attempt=attempt_number,
                        instrument_type=instrument_type,
                    ),
                ) as usage_records:
                    generated_module = _generate_module(
                        skeleton, spec_schema, reference_sources, stage_model, 1,
                        extra_context=validation_feedback,
                        pricing_plan=pricing_plan,
                        knowledge_context=knowledge_text,
                        generation_plan=generation_plan,
                        prompt_surface=prompt_surface,
                    )
                    if isinstance(generated_module, str):
                        source_report = sanitize_generated_source(generated_module)
                        generated_module = GeneratedModuleResult(
                            raw_code=generated_module,
                            sanitized_code=source_report.sanitized_source,
                            code=source_report.sanitized_source.expandtabs(4),
                            source_report=source_report,
                        )
                generation_token_usage = summarize_llm_usage(usage_records)
                enforce_llm_token_budget(stage="code_generation")
            code = generated_module.code
        except Exception as exc:
            failure_text = f"attempt {attempt_number}: {type(exc).__name__}: {exc}"
            previous_failures.append(failure_text)
            source_report = getattr(exc, "source_report", None)
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "code_generation",
                    "model": stage_model,
                    "failure_count": 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "parse_status": (
                        "parse_failed"
                        if type(exc).__name__ in {"SyntaxError", "GeneratedModuleSourceError"}
                        else "failed"
                    ),
                    "source_sanitization": asdict(source_report)
                    if source_report is not None
                    else None,
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## CODE GENERATION FAILURE (your previous module could not be produced):\n"
                f"- {failure_text}\n\n"
                "Regenerate the full module from the canonical scaffold. Keep the route-local raw kernel/adaptor split, "
                "use only approved Trellis imports, and return valid Python that compiles on the first pass."
            )
            retry_reason = "code_generation"
            continue
        _record_platform_event(
            compiled_request,
            "builder_attempt_generated",
            status="ok",
            details={
                "attempt": attempt_number,
                "prompt_surface": prompt_surface,
                "knowledge_surface": knowledge_surface,
                "retrieval_stage": knowledge_context.retrieval_stage,
                "retrieval_source": knowledge_context.retrieval_source,
                "retry_reason": retry_reason,
                "knowledge_context_chars": len(knowledge_text),
                "parse_status": "compiled",
                "source_sanitization": asdict(generated_module.source_report)
                if generated_module is not None
                else None,
                "token_usage": generation_token_usage,
            },
        )

        import_report = validate_generated_imports(code, generation_plan)
        if not import_report.ok:
            failures = list(import_report.errors)
            previous_failures = failures
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "import_validation",
                    "failure_count": len(failures),
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## IMPORT VALIDATION FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nUse only approved, registry-backed Trellis imports."
            )
            retry_reason = "import_validation"
            continue
        _attempt_gate_results.append(("import", True, ()))

        semantic_report = validate_semantics(
            code,
            product_ir=product_ir,
            generation_plan=generation_plan,
        )
        if not semantic_report.ok:
            failures = list(semantic_report.errors)
            previous_failures = failures
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "semantic_validation",
                    "failure_count": len(failures),
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## SEMANTIC VALIDATION FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nMatch the generated code to the product semantics and real Trellis engine contracts."
            )
            retry_reason = "semantic_validation"
            continue
        _attempt_gate_results.append(("semantic", True, ()))

        # Gate 3b: Semantic validators (market data, parameter binding, algorithm contract)
        from trellis.agent.semantic_validators import validate_generated_semantics
        semantic_ext_report = validate_generated_semantics(
            code, generation_plan,
        )
        if semantic_ext_report.findings:
            _record_platform_event(
                compiled_request,
                "semantic_validators_completed",
                status="warning" if semantic_ext_report.ok else "error",
                details={
                    "attempt": attempt_number,
                    "finding_count": len(semantic_ext_report.findings),
                    "mode": semantic_ext_report.mode,
                    "findings": [
                        {"validator": f.validator, "severity": f.severity, "category": f.category}
                        for f in semantic_ext_report.findings[:10]
                    ],
                },
            )
        if not semantic_ext_report.ok:
            # Truncate feedback to 150 tokens (~600 chars) to preserve prompt budget
            finding_lines = [
                f"- [{f.severity}] {f.message}" for f in semantic_ext_report.errors[:5]
            ]
            feedback_text = "\n".join(finding_lines)[:600]
            failures = [f.message for f in semantic_ext_report.errors]
            previous_failures = failures
            validation_feedback = (
                "\n\n## SEMANTIC CONTRACT FAILURES (your previous code had these issues):\n"
                + feedback_text
                + "\n\nFix the above semantic issues in your implementation."
            )
            retry_reason = "semantic_validation"
            continue
        _attempt_gate_results.append(("semantic_validators", True, ()))

        from trellis.agent.lite_review import review_generated_code

        lite_review_report = review_generated_code(
            code,
            pricing_plan=pricing_plan,
            product_ir=product_ir,
            generation_plan=generation_plan,
        )
        _record_platform_event(
            compiled_request,
            "lite_review_completed",
            status="ok" if lite_review_report.ok else "error",
            details={
                "attempt": attempt_number,
                "issue_count": len(lite_review_report.issues),
                "issue_codes": [issue.code for issue in lite_review_report.issues],
            },
        )
        if not lite_review_report.ok:
            failures = list(lite_review_report.errors)
            previous_failures = failures
            for issue in lite_review_report.issues:
                _append_agent_observation(
                    build_meta,
                    "lite_reviewer",
                    "failure",
                    issue.message,
                    severity="error",
                    attempt=attempt_number,
                    details={"code": issue.code},
                )
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "lite_review",
                    "failure_count": len(failures),
                    "token_usage": generation_token_usage,
                },
            )
            validation_feedback = (
                "\n\n## DETERMINISTIC LITE REVIEW FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nRemove hardcoded market inputs and read required pricing inputs from `market_state`."
            )
            retry_reason = "lite_review"
            continue
        _attempt_gate_results.append(("lite_review", True, tuple(
            issue.code for issue in lite_review_report.issues
        )))

        file_path = write_module(output_module_path, code)
        mod = dynamic_import(file_path, module_name)
        payoff_cls = getattr(mod, spec_schema.class_name)

        actual_market_failures = _smoke_test_actual_market_state(
            payoff_cls,
            spec_schema,
            market_state,
        )
        if actual_market_failures:
            failures = actual_market_failures
            previous_failures = failures
            _record_platform_event(
                compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "actual_market_smoke",
                    "failure_count": len(failures),
                },
            )
            validation_feedback = (
                "\n\n## ACTUAL MARKET SMOKE FAILURES (your previous code had these issues):\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nThe generated payoff must also run against the real task market_state, not just synthetic validation fixtures."
            )
            retry_reason = "actual_market_smoke"
            continue

        if validation == "fast":
            _record_platform_event(
                compiled_request,
                "builder_attempt_succeeded",
                status="ok",
                details={"attempt": attempt_number},
            )
            _record_platform_event(
                compiled_request,
                "build_completed",
                status="ok",
                success=True if _finalizes_in_executor(compiled_request) else None,
                outcome="build_completed" if _finalizes_in_executor(compiled_request) else None,
                details={"attempts": attempt_number},
            )
            _audit_path = _write_build_audit_record(
                compiled_request=compiled_request,
                model=model,
                pricing_plan=pricing_plan,
                instrument_type=instrument_type,
                spec_schema=spec_schema,
                output_module_path=output_module_path,
                code=code,
                market_state=market_state,
                attempt_number=attempt_number,
                gate_results=_attempt_gate_results,
                build_start_time=build_start_time,
            )
            if build_meta is not None and _audit_path is not None:
                build_meta["audit_record_path"] = str(_audit_path)
            return payoff_cls

        failures, failure_details = _validate_build(
            payoff_cls, code, payoff_description, spec_schema,
            validation=validation,
            model=model,
            compiled_request=compiled_request,
            pricing_plan=pricing_plan,
            product_ir=product_ir,
            build_meta=build_meta,
            attempt_number=attempt_number,
            return_failure_details=True,
            gate_results_out=_attempt_gate_results,
            knowledge_retriever=knowledge_retriever,
        )

        if not failures:
            _record_platform_event(
                compiled_request,
                "builder_attempt_succeeded",
                status="ok",
                details={"attempt": attempt_number},
            )
            # Success — record lessons from any failures resolved this round
            if previous_failures:
                _record_resolved_failures(
                    previous_failures, payoff_description, pricing_plan, model,
                )
            _record_platform_event(
                compiled_request,
                "build_completed",
                status="ok",
                success=True if _finalizes_in_executor(compiled_request) else None,
                outcome="build_completed" if _finalizes_in_executor(compiled_request) else None,
                details={"attempts": attempt_number},
            )
            _audit_path = _write_build_audit_record(
                compiled_request=compiled_request,
                model=model,
                pricing_plan=pricing_plan,
                instrument_type=instrument_type,
                spec_schema=spec_schema,
                output_module_path=output_module_path,
                code=code,
                market_state=market_state,
                attempt_number=attempt_number,
                gate_results=_attempt_gate_results,
                build_start_time=build_start_time,
            )
            if build_meta is not None and _audit_path is not None:
                build_meta["audit_record_path"] = str(_audit_path)
            return payoff_cls

        # Diagnose failures and enrich feedback for next attempt
        diagnosis_text = _diagnose_and_enrich(failures)
        previous_failures = failures
        _record_platform_event(
            compiled_request,
                "builder_attempt_failed",
                status="error",
                details={
                    "attempt": attempt_number,
                    "reason": "validation",
                    "failure_count": len(failures),
                    "failure_details": [asdict(detail) for detail in failure_details],
                    "parse_status": "compiled" if generated_module is not None else "unknown",
                    "source_sanitization": asdict(generated_module.source_report)
                    if generated_module is not None
                    else None,
                },
            )

        # Record run trace (cold storage)
        _record_trace(instrument_type, pricing_plan, payoff_description,
                      attempt, code, failures)

        validation_feedback = _format_validation_failure_feedback(
            failures=failures,
            failure_details=failure_details,
        ) + diagnosis_text + "\n\nFix ALL of the above issues in your implementation."
        retry_reason = "validation"

    failure_text = "; ".join(previous_failures) if previous_failures else "unknown build failure"
    _record_platform_event(
        compiled_request,
        "request_failed" if _finalizes_in_executor(compiled_request) else "build_failed",
        status="error",
        success=False if _finalizes_in_executor(compiled_request) else None,
        outcome="request_failed" if _finalizes_in_executor(compiled_request) else None,
        details={
            "reason": "max_retries_exhausted",
            "failure_count": len(previous_failures),
        },
    )
    raise RuntimeError(
        f"Failed to build payoff after {max_retries} attempts: {failure_text}"
    )


def _validate_build(
    payoff_cls,
    code: str,
    description: str,
    spec_schema,
    validation: str = "standard",
    model: str | None = None,
    compiled_request=None,
    pricing_plan=None,
    product_ir=None,
    build_meta: dict | None = None,
    attempt_number: int | None = None,
    return_failure_details: bool = False,
    gate_results_out: list | None = None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None = None,
) -> list[str] | tuple[list[str], tuple[object, ...]]:
    """Run validation checks on a built payoff."""
    from trellis.agent.review_policy import determine_review_policy
    from trellis.agent.route_registry import route_binding_authority_summary
    from trellis.agent.validation_contract import (
        compile_validation_contract,
        validation_contract_summary,
    )
    from trellis.agent.reference_oracles import (
        execute_reference_oracle,
        reference_oracle_summary,
        select_reference_oracle,
    )
    from trellis.agent.config import (
        enforce_llm_token_budget,
        get_model_for_stage,
        llm_usage_stage,
        summarize_llm_usage,
    )
    from trellis.agent.validation_bundles import (
        execute_validation_bundle,
        select_validation_bundle,
    )
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol
    from trellis.core.payoff import DeterministicCashflowPayoff
    from trellis.instruments.bond import Bond

    settle = date(2024, 11, 15)
    failures = []
    failure_details: list[object] = []
    itype = _resolve_lower_layer_instrument_type(
        description,
        compiled_request=compiled_request,
        product_ir=product_ir,
    )
    required_market_data = set(getattr(pricing_plan, "required_market_data", ()) or ())
    validation_contract = getattr(compiled_request, "validation_contract", None)
    if validation_contract is None:
        validation_contract = compile_validation_contract(
            request=getattr(compiled_request, "request", None),
            product_ir=product_ir,
            pricing_plan=pricing_plan,
            generation_plan=getattr(compiled_request, "generation_plan", None),
            semantic_blueprint=getattr(compiled_request, "semantic_blueprint", None),
            comparison_spec=getattr(compiled_request, "comparison_spec", None),
            instrument_type=itype,
        )
    validation_contract_data = validation_contract_summary(validation_contract)
    route_binding_authority_data = route_binding_authority_summary(
        getattr(getattr(compiled_request, "generation_plan", None), "route_binding_authority", None)
    )
    if route_binding_authority_data is None and compiled_request is not None:
        route_binding_authority_data = dict(
            getattr(getattr(compiled_request, "request", None), "metadata", {}).get("route_binding_authority") or {}
        )
    review_policy = determine_review_policy(
        validation=validation,
        method=(pricing_plan.method if pricing_plan is not None else "unknown"),
        instrument_type=itype,
        product_ir=product_ir,
        validation_contract=validation_contract,
    )
    review_knowledge_text = ""
    review_prompt_surface = "none"

    # Try to instantiate the payoff with default test parameters
    try:
        test_payoff = _make_test_payoff(payoff_cls, spec_schema, settle)
    except Exception as e:
        failures.append(f"Cannot instantiate payoff for validation: {e}")
        if return_failure_details:
            return failures, tuple()
        return failures

    def payoff_factory():
        """Instantiate a fresh payoff under the generated spec schema."""
        return _make_test_payoff(payoff_cls, spec_schema, settle)

    bond = Bond(
        face=100, coupon=0.05,
        maturity_date=date(2034, 11, 15),
        maturity=10, frequency=2,
    )

    def reference_factory():
        """Build the straight-bond reference payoff used for bounding checks."""
        return DeterministicCashflowPayoff(bond)

    def _build_validation_market_state(rate=0.05, vol=0.20, corr=0.35, extra_requirements=None):
        """Create a simple market state matching the plan's capability contract.

        Uses the union of the plan's ``required_market_data`` and any
        ``extra_requirements`` supplied by the caller (e.g. the payoff's own
        ``requirements`` property discovered post-instantiation).
        """
        effective_requirements = set(required_market_data)
        if extra_requirements:
            effective_requirements.update(extra_requirements)

        discount_curve = YieldCurve.flat(rate)
        payload = {
            "as_of": settle,
            "settlement": settle,
            "discount": discount_curve,
        }
        if "forward_curve" in effective_requirements:
            from trellis.curves.forward_curve import ForwardCurve

            foreign_curve = YieldCurve.flat(max(rate - 0.02, 0.005))
            payload["forecast_curves"] = {
                "EUR-DISC": foreign_curve,
                "EUR": foreign_curve,
            }
            payload["forward_curve"] = ForwardCurve(discount_curve)
        if "black_vol_surface" in effective_requirements:
            payload["vol_surface"] = FlatVol(vol)
        if "fx_rates" in effective_requirements:
            from trellis.instruments.fx import FXRate

            payload["fx_rates"] = {
                "EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR"),
            }
        if "spot" in effective_requirements:
            payload["spot"] = 100.0
            payload["underlier_spots"] = {
                "SPX": 100.0,
                "NDX": 101.5,
                "RUT": 98.5,
                "EUR": 100.0,
            }
        if "local_vol_surface" in effective_requirements:
            def local_vol_surface(spot, time, level=vol):
                from trellis.core.differentiable import get_numpy

                np = get_numpy()
                spot_array = np.asarray(spot, dtype=float)
                time_array = np.asarray(time, dtype=float)
                smile = 1.0 + 0.10 * np.abs(np.log(np.maximum(spot_array, 1e-8) / 100.0))
                term = 1.0 + 0.05 * np.minimum(np.maximum(time_array, 0.0), 5.0) / 5.0
                return level * smile * term

            payload["local_vol_surface"] = local_vol_surface
            payload["local_vol_surfaces"] = {"spx_local_vol": local_vol_surface}
            payload.setdefault("spot", 100.0)
            payload.setdefault("underlier_spots", {"SPX": 100.0})
        if "credit_curve" in effective_requirements:
            from trellis.curves.credit_curve import CreditCurve

            payload["credit_curve"] = CreditCurve.flat(0.02)
        if "model_parameters" in effective_requirements:
            payload["model_parameters"] = {"quanto_correlation": corr}
        return MarketState(**payload)

    # Build the initial market state from the plan's required_market_data.
    # After the payoff is instantiated, rebuild using its declared requirements
    # so the validation state always covers what the payoff actually needs.
    ms = _build_validation_market_state()
    if test_payoff is not None:
        try:
            payoff_reqs = set(getattr(test_payoff, "requirements", set()) or set())
            ms = _build_validation_market_state(extra_requirements=payoff_reqs)
        except Exception:
            pass  # fall back to plan-derived market state

    def ms_factory(rate=0.05, vol=0.20, corr=0.35):
        """Create simple market states for invariant checks."""
        payoff_reqs: set[str] = set()
        try:
            payoff_reqs = set(getattr(test_payoff, "requirements", set()) or set())
        except Exception:
            pass
        return _build_validation_market_state(rate=rate, vol=vol, corr=corr, extra_requirements=payoff_reqs)

    validation_bundle = select_validation_bundle(
        instrument_type=itype,
        method=(pricing_plan.method if pricing_plan is not None else "unknown"),
        product_ir=product_ir,
        semantic_blueprint=getattr(compiled_request, "semantic_blueprint", None),
    )
    _record_platform_event(
        compiled_request,
        "validation_bundle_selected",
        status="info",
        details={
            "bundle_id": validation_bundle.bundle_id,
            "checks": list(validation_bundle.checks),
            "categories": {
                key: list(value) for key, value in validation_bundle.categories.items()
            },
            "validation_contract": validation_contract_data,
            "route_binding_authority": route_binding_authority_data,
        },
    )
    validation_check_relations = {}
    if validation_contract is not None:
        validation_check_relations = {
            check.check_id: check.relation
            for check in getattr(validation_contract, "deterministic_checks", ())
            if getattr(check, "relation", None)
        }
    bundle_execution = execute_validation_bundle(
        validation_bundle,
        validation_level=validation,
        test_payoff=test_payoff,
        market_state=ms,
        payoff_factory=payoff_factory,
        market_state_factory=ms_factory,
        reference_factory=reference_factory,
        check_relations=validation_check_relations,
        semantic_blueprint=getattr(compiled_request, "semantic_blueprint", None),
    )
    failures.extend(bundle_execution.failures)
    failure_details.extend(bundle_execution.failure_details)
    _record_platform_event(
        compiled_request,
        "validation_bundle_executed",
        status="ok" if not bundle_execution.failures else "error",
        details={
            "bundle_id": validation_bundle.bundle_id,
            "executed_checks": list(bundle_execution.executed_checks),
            "skipped_checks": list(bundle_execution.skipped_checks),
            "failure_count": len(bundle_execution.failures),
            "failure_details": [asdict(detail) for detail in bundle_execution.failure_details],
            "validation_contract": validation_contract_data,
            "route_binding_authority": route_binding_authority_data,
        },
    )

    if gate_results_out is not None:
        gate_results_out.append(("bundle", not bool(bundle_execution.failures), tuple(
            bundle_execution.failures
        ), {
            "bundle_id": validation_bundle.bundle_id,
            "validation_contract_id": (
                None if validation_contract is None else validation_contract.contract_id
            ),
            "route_binding_authority": route_binding_authority_data,
        }))

    oracle_execution = None
    if not failures and _should_run_reference_oracle(compiled_request):
        oracle_spec = select_reference_oracle(
            instrument_type=itype,
            method=(pricing_plan.method if pricing_plan is not None else "unknown"),
            product_ir=product_ir,
            semantic_blueprint=getattr(compiled_request, "semantic_blueprint", None),
        )
        if oracle_spec is not None:
            oracle_execution = execute_reference_oracle(
                oracle_spec,
                payoff_factory=payoff_factory,
                market_state_factory=ms_factory,
                reference_factory=reference_factory,
                semantic_blueprint=getattr(compiled_request, "semantic_blueprint", None),
            )
            oracle_summary = reference_oracle_summary(oracle_execution)
            _record_platform_event(
                compiled_request,
                "reference_oracle_executed",
                status="ok" if (oracle_execution and oracle_execution.passed) else "error",
                details={
                    "oracle": oracle_summary,
                    "validation_contract": validation_contract_data,
                    "route_binding_authority": route_binding_authority_data,
                },
            )
            if gate_results_out is not None:
                gate_results_out.append((
                    "reference_oracle",
                    bool(oracle_execution and oracle_execution.passed),
                    tuple(
                        []
                        if oracle_execution is None or oracle_execution.failure_message is None
                        else [oracle_execution.failure_message]
                    ),
                    {
                        "oracle_id": None if oracle_execution is None else oracle_execution.oracle_id,
                        "relation": None if oracle_execution is None else oracle_execution.relation,
                        "source": None if oracle_execution is None else oracle_execution.source,
                    },
                ))
            if oracle_execution is not None and not oracle_execution.passed:
                failures.append(
                    oracle_execution.failure_message
                    or f"Reference oracle `{oracle_execution.oracle_id}` failed."
                )
                failure_details.append(oracle_execution)

    deterministic_gate_failed = bool(failures)
    deterministic_gate_reason = (
        "deterministic_validation_failed" if deterministic_gate_failed else None
    )
    critic_mode = getattr(
        review_policy,
        "critic_mode",
        "required" if getattr(review_policy, "run_critic", False) else "skip",
    )
    critic_json_max_retries = getattr(review_policy, "critic_json_max_retries", None)
    critic_allow_text_fallback = getattr(review_policy, "critic_allow_text_fallback", True)
    critic_text_max_retries = getattr(review_policy, "critic_text_max_retries", None)

    # Standard: run critic
    if (
        validation in ("standard", "thorough")
        and getattr(review_policy, "run_critic", False)
        and not deterministic_gate_failed
    ):
        critic_error = None
        review_context = _resolve_knowledge_context_for_attempt(
            audience="review",
            pricing_plan=pricing_plan,
            instrument_type=itype,
            attempt_number=attempt_number or 1,
            compiled_request=compiled_request,
            product_ir=product_ir,
            prompt_surface="critic_review",
            build_meta=build_meta,
            knowledge_retriever=knowledge_retriever,
        )
        review_knowledge_text = review_context.text
        review_prompt_surface = review_context.knowledge_surface
        try:
            from trellis.agent.critic import available_critic_checks, critique
            from trellis.agent.arbiter import run_critic_tests
            critic_checks = available_critic_checks(
                instrument_type=itype,
                method=getattr(pricing_plan, "method", None),
                product_ir=product_ir,
                validation_contract=validation_contract,
            )
            stage_model = get_model_for_stage("critic", model)
            with llm_usage_stage(
                "critic",
                metadata=_llm_stage_metadata(
                    compiled_request=compiled_request,
                    model=stage_model,
                    attempt=attempt_number,
                    instrument_type=itype,
                ),
            ) as usage_records:
                concerns = critique(
                    code,
                    description,
                    knowledge_context=review_knowledge_text,
                    model=stage_model,
                    generation_plan=getattr(compiled_request, "generation_plan", None),
                    available_checks=critic_checks,
                    json_max_retries=critic_json_max_retries,
                    allow_text_fallback=critic_allow_text_fallback,
                    text_max_retries=critic_text_max_retries,
                )
            enforce_llm_token_budget(stage="critic")
            _record_platform_event(
                compiled_request,
                "critic_completed",
                status="ok",
                details={
                    "concern_count": len(concerns),
                    "prompt_surface": review_prompt_surface,
                    "retrieval_stage": review_context.retrieval_stage,
                    "retrieval_source": review_context.retrieval_source,
                    "knowledge_context_chars": len(review_knowledge_text),
                    "available_check_ids": [check.check_id for check in critic_checks],
                    "critic_mode": critic_mode,
                    "json_max_retries": critic_json_max_retries,
                    "allow_text_fallback": critic_allow_text_fallback,
                    "text_max_retries": critic_text_max_retries,
                    "token_usage": summarize_llm_usage(usage_records),
                },
            )
            for concern in concerns:
                _append_agent_observation(
                    build_meta,
                    "critic",
                    "concern",
                    concern.description,
                    severity=concern.severity,
                    attempt=attempt_number,
                    details={
                        "check_id": concern.check_id,
                        "status": concern.status,
                        "evidence": concern.evidence,
                        "remediation": concern.remediation,
                    },
                )
            critic_failures = run_critic_tests(
                concerns,
                test_payoff,
                allowed_check_ids={check.check_id for check in critic_checks},
            )
            failures.extend(critic_failures)
            for failure in critic_failures:
                _append_agent_observation(
                    build_meta,
                    "arbiter",
                    "failure",
                    failure.splitlines()[0],
                    severity="error",
                    attempt=attempt_number,
                    details={"message": failure},
                )
        except Exception as e:
            critic_error = e
            import logging
            logging.getLogger(__name__).warning(f"Critic validation error (non-blocking): {e}")
            _record_platform_event(
                compiled_request,
                "critic_failed",
                status="error",
                details={
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "critic_mode": critic_mode,
                    "reason": getattr(review_policy, "critic_reason", ""),
                    "json_max_retries": critic_json_max_retries,
                    "allow_text_fallback": critic_allow_text_fallback,
                    "text_max_retries": critic_text_max_retries,
                },
            )
        if gate_results_out is not None:
            _critic_fails = locals().get("critic_failures", [])
            _critic_passed = not bool(_critic_fails) and critic_error is None
            _critic_details = {
                "critic_mode": critic_mode,
                "json_max_retries": critic_json_max_retries,
                "allow_text_fallback": critic_allow_text_fallback,
                "text_max_retries": critic_text_max_retries,
            }
            if critic_error is not None:
                _critic_details["error_type"] = type(critic_error).__name__
                _critic_details["error"] = str(critic_error)
                _critic_details["advisory"] = critic_mode == "advisory"
            gate_results_out.append(
                ("critic", _critic_passed, tuple(_critic_fails[:3]), _critic_details)
            )
    elif validation in ("standard", "thorough"):
        _record_platform_event(
            compiled_request,
            "critic_skipped",
            status="info",
            details={
                "risk_level": review_policy.risk_level,
                "critic_mode": critic_mode,
                "reason": deterministic_gate_reason or review_policy.critic_reason,
                "failure_count": len(failures),
            },
        )

    _record_platform_event(
        compiled_request,
        "arbiter_completed",
        status="ok" if not failures else "error",
        details={
            "validation": validation,
            "failure_count": len(failures),
        },
    )

    # Thorough: run model validator (MRM-style)
    if validation == "thorough" and not deterministic_gate_failed:
        try:
            from trellis.agent.model_validator import validate_model

            if review_policy.run_model_validator_llm and not review_knowledge_text:
                review_context = _resolve_knowledge_context_for_attempt(
                    audience="review",
                    pricing_plan=pricing_plan,
                    instrument_type=itype,
                    attempt_number=attempt_number or 1,
                    compiled_request=compiled_request,
                    product_ir=product_ir,
                    prompt_surface="model_validator_review",
                    build_meta=build_meta,
                    knowledge_retriever=knowledge_retriever,
                )
                review_knowledge_text = review_context.text
                review_prompt_surface = review_context.knowledge_surface

            usage_summary = None
            if review_policy.run_model_validator_llm:
                stage_model = get_model_for_stage("model_validator", model)
                with llm_usage_stage(
                    "model_validator",
                    metadata=_llm_stage_metadata(
                        compiled_request=compiled_request,
                        model=stage_model,
                        attempt=attempt_number,
                        instrument_type=itype,
                    ),
                ) as usage_records:
                    report = validate_model(
                        payoff_factory=payoff_factory,
                        market_state_factory=ms_factory,
                        code=code,
                        instrument_type=itype,
                        method=spec_schema.requirements[0] if hasattr(spec_schema, 'requirements') else "unknown",
                        knowledge_context=review_knowledge_text,
                        model=stage_model,
                        product_ir=product_ir,
                        generation_plan=getattr(compiled_request, "generation_plan", None),
                        validation_contract=validation_contract,
                        review_reason=review_policy.model_validator_reason,
                        run_llm_review=True,
                    )
                enforce_llm_token_budget(stage="model_validator")
                usage_summary = summarize_llm_usage(usage_records)
            else:
                _record_platform_event(
                    compiled_request,
                    "model_validator_llm_review_skipped",
                    status="info",
                    details={
                        **_semantic_role_ownership_details(
                            stage="payoff_model_validation",
                            compiled_request=compiled_request,
                            trigger_condition=review_policy.model_validator_reason,
                            artifact_kind="ValidationReport",
                            review_policy=review_policy,
                            executed=False,
                        ),
                        "risk_level": review_policy.risk_level,
                        "reason": review_policy.model_validator_reason,
                    },
                )
                report = validate_model(
                    payoff_factory=payoff_factory,
                    market_state_factory=ms_factory,
                    code=code,
                    instrument_type=itype,
                    method=spec_schema.requirements[0] if hasattr(spec_schema, 'requirements') else "unknown",
                    knowledge_context="",
                    model=model,
                    product_ir=product_ir,
                    generation_plan=getattr(compiled_request, "generation_plan", None),
                    validation_contract=validation_contract,
                    review_reason=review_policy.model_validator_reason,
                    run_llm_review=False,
                )

            for finding in report.findings:
                _append_agent_observation(
                    build_meta,
                    "model_validator",
                    "finding",
                    finding.description,
                    severity=finding.severity,
                    attempt=attempt_number,
                    details={
                        "category": finding.category,
                        "evidence": finding.evidence,
                        "remediation": finding.remediation,
                    },
                )
                if finding.severity in ("critical", "high"):
                    failures.append(
                        f"[MODEL VALIDATION {finding.severity.upper()}] {finding.id}: "
                        f"{finding.description}\n"
                        f"  Evidence: {finding.evidence}\n"
                        f"  Remediation: {finding.remediation}"
                    )
            blocker_findings = [
                finding for finding in report.findings
                if finding.severity in ("critical", "high")
            ]
            _record_platform_event(
                compiled_request,
                "model_validator_completed",
                status="ok" if not blocker_findings else "error",
                details={
                    **_semantic_role_ownership_details(
                        stage="payoff_model_validation",
                        compiled_request=compiled_request,
                        trigger_condition=(
                            review_policy.model_validator_reason
                            or "validation_risk_gate"
                        ),
                        artifact_kind="ValidationReport",
                        review_policy=review_policy,
                        executed=review_policy.run_model_validator_llm,
                    ),
                    "finding_count": len(report.findings),
                    "blocker_count": len(blocker_findings),
                    "approved": report.approved,
                    "llm_review": review_policy.run_model_validator_llm,
                    "risk_level": review_policy.risk_level,
                    "skip_reason": (
                        None if review_policy.run_model_validator_llm
                        else review_policy.model_validator_reason
                    ),
                    "prompt_surface": review_prompt_surface if review_policy.run_model_validator_llm else "deterministic_only",
                    "knowledge_context_chars": len(review_knowledge_text) if review_policy.run_model_validator_llm else 0,
                    "token_usage": usage_summary,
                },
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Model validation error (non-blocking): {e}")
        if gate_results_out is not None:
            _mv_blockers = locals().get("blocker_findings", [])
            _mv_issues = tuple(
                f"[{f.severity.upper()}] {f.id}: {f.description}"
                for f in _mv_blockers
            )
            gate_results_out.append(("model_validator", not bool(_mv_issues), _mv_issues))
    elif validation == "thorough":
        _record_platform_event(
            compiled_request,
            "model_validator_llm_review_skipped",
            status="info",
            details={
                "risk_level": review_policy.risk_level,
                "reason": deterministic_gate_reason,
                "failure_count": len(failures),
            },
        )

    if return_failure_details:
        return failures, tuple(failure_details)
    return failures


def _extract_instrument_type(description: str) -> str:
    """Extract instrument type keyword from a description."""
    from trellis.agent.instrument_identity import resolve_instrument_identity

    resolution = resolve_instrument_identity(
        description,
        inferred_source="executor.description_ingress",
    )
    return resolution.instrument_type or "unknown"


def _resolve_lower_layer_instrument_type(
    description: str,
    *,
    compiled_request=None,
    product_ir=None,
    explicit_instrument_type: str | None = None,
) -> str:
    """Resolve family identity for lower layers without rediscovering known families."""
    from trellis.agent.instrument_identity import normalize_instrument_type

    request = getattr(compiled_request, "request", None)
    request_metadata = dict(getattr(request, "metadata", None) or {})
    runtime_contract = dict(request_metadata.get("runtime_contract") or {})
    candidates = (
        explicit_instrument_type,
        getattr(request, "instrument_type", None),
        request_metadata.get("instrument_type"),
        runtime_contract.get("instrument_type"),
        getattr(product_ir, "instrument", None),
    )
    for candidate in candidates:
        normalized = normalize_instrument_type(candidate)
        if normalized:
            return normalized
    return _extract_instrument_type(description)


def _make_test_payoff(payoff_cls, spec_schema, settle: date, market_state=None):
    """Create a test payoff instance from the spec schema with default values."""
    import sys
    from dataclasses import is_dataclass, replace as replace_dataclass

    spec_cls = None
    for attr_name in ("__init__", "evaluate"):
        func = getattr(payoff_cls, attr_name, None)
        globals_dict = getattr(func, "__globals__", None)
        if isinstance(globals_dict, dict):
            candidate = globals_dict.get(spec_schema.spec_name)
            if candidate is not None:
                spec_cls = candidate
                break

    module = sys.modules.get(getattr(payoff_cls, "__module__", ""))
    if spec_cls is None and module is not None and hasattr(module, spec_schema.spec_name):
        spec_cls = getattr(module, spec_schema.spec_name)
    elif spec_cls is None:
        for _, mod in list(sys.modules.items()):
            if mod and hasattr(mod, spec_schema.spec_name):
                spec_cls = getattr(mod, spec_schema.spec_name)
                break
    if spec_cls is None:
        raise RuntimeError(f"Cannot find {spec_schema.spec_name} in loaded modules")

    # Build kwargs from field definitions with test defaults
    from trellis.core.types import DayCountConvention, Frequency

    kwargs = {}
    type_defaults = {
        "float": 100.0,
        "int": 10,
        "str": "test",
        "bool": True,
        "date": date(2034, 11, 15),
        "str | None": None,
        "float | None": None,
        "int | None": None,
        "tuple[date, ...]": (
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 6, 1),
        ),
        "tuple[date, ...] | None": None,
        "Frequency": Frequency.SEMI_ANNUAL,
        "DayCountConvention": DayCountConvention.ACT_360,
    }
    # More specific field-name defaults
    name_defaults = {
        "notional": 100.0,
        "coupon": 0.05,
        "strike": 0.05,
        "underlyings": "AAPL,MSFT,NVDA",
        "constituents": "AAPL,MSFT,NVDA",
        "underliers": "SPX,NDX",
        "spots": "100.0,95.0",
        "weights": "1.0,-1.0",
        "vols": "0.20,0.20",
        "dividend_yields": "0.0,0.0",
        "basket_style": "weighted_sum",
        "observation_dates": (
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 6, 1),
        ),
        "expiry_date": date(2025, 11, 15),
        "swap_start": date(2025, 11, 15),
        "swap_end": date(2034, 11, 15),
        "start_date": settle,
        "end_date": date(2034, 11, 15),
        "is_payer": True,
        "call_dates": (
            date(2027, 11, 15),
            date(2029, 11, 15),
            date(2031, 11, 15),
        ),
        "put_dates": (
            date(2027, 11, 15),
            date(2029, 11, 15),
            date(2031, 11, 15),
        ),
        "exercise_dates": (
            date(2027, 11, 15),
            date(2029, 11, 15),
            date(2031, 11, 15),
        ),
        "call_price": 100.0,
        "put_price": 100.0,
        "call_schedule": "2027-11-15,2029-11-15,2031-11-15",
        "barrier": 80.0,
        "barrier_type": "down_and_out",
        "option_type": "call",
        "spot": 100.0,
        "fx_pair": "EURUSD",
        "foreign_discount_key": "EUR-DISC",
        "n_names": 5,
        "n_th": 1,
        "attachment": 0.03,
        "detachment": 0.07,
        "correlation": 0.3,
        "recovery": 0.4,
    }
    basket_context = " ".join(
        part.strip()
        for part in (
            getattr(payoff_cls, "__doc__", "") or "",
            getattr(module, "__doc__", "") or "",
        )
        if part and str(part).strip()
    ).lower()
    has_spot_field = any(
        field.name in {"spot", "s0", "underlier_spot"} for field in spec_schema.fields
    )
    has_strike_field = any(field.name == "strike" for field in spec_schema.fields)
    if spec_schema.spec_name == "FXVanillaOptionSpec":
        name_defaults["notional"] = 10.0
        name_defaults["strike"] = 1.08
        name_defaults["spot"] = 1.10
    elif spec_schema.spec_name == "QuantoOptionSpec":
        name_defaults["notional"] = 10.0
        name_defaults["strike"] = 100.0
        name_defaults["spot"] = 100.0
        name_defaults["underlier_currency"] = "EUR"
        name_defaults["domestic_currency"] = "USD"
        name_defaults["fx_pair"] = "EURUSD"
        name_defaults["quanto_correlation_key"] = None
    elif spec_schema.spec_name == "BasketOptionSpec":
        is_spread_basket = any(
            token in basket_context
            for token in ("spread option", "kirk_spread", "mc_spread_2d", "fft_spread_2d")
        )
        name_defaults["notional"] = 10.0
        name_defaults["strike"] = 5.0 if is_spread_basket else 100.0
        name_defaults["underliers"] = "SPX,NDX"
        name_defaults["spots"] = "100.0,95.0"
        if is_spread_basket:
            name_defaults["weights"] = "1.0,-1.0"
            name_defaults["basket_style"] = "spread"
        else:
            name_defaults.pop("weights", None)
            name_defaults["basket_style"] = "best_of"
        name_defaults.pop("vols", None)
        name_defaults["dividend_yields"] = "0.0,0.0"
        name_defaults["correlation"] = "1.0,0.35;0.35,1.0"
    elif has_spot_field and has_strike_field:
        # Spot-based option specs should be instantiated near-the-money so the
        # smoke tests exercise a representative valuation instead of a deeply
        # in-the-money payoff created by the generic rate-like strike default.
        name_defaults["notional"] = 10.0
        name_defaults["strike"] = name_defaults["spot"]

    description = getattr(payoff_cls, "__doc__", "") or getattr(module, "__doc__", "") or ""
    description_defaults = _description_spec_defaults(
        spec_schema,
        description=description,
    )
    if description_defaults:
        name_defaults.update(description_defaults)

    for field in spec_schema.fields:
        if field.name in name_defaults:
            kwargs[field.name] = name_defaults[field.name]
        elif field.default is not None:
            pass  # let the dataclass default handle it
        elif field.type in type_defaults:
            kwargs[field.name] = type_defaults[field.type]

    try:
        spec = spec_cls(**kwargs)
    except TypeError:
        # If we're missing required fields, try with all defaults
        spec = spec_cls(**{f.name: name_defaults.get(f.name, type_defaults.get(f.type, ""))
                          for f in spec_schema.fields if f.default is None})

    if market_state is not None and spec_schema.spec_name == "QuantoOptionSpec":
        try:
            from trellis.models.resolution.quanto import resolve_quanto_inputs

            resolved = resolve_quanto_inputs(market_state, spec)
            atm_strike = float(resolved.spot)
            if is_dataclass(spec):
                spec = replace_dataclass(spec, strike=atm_strike)
            elif hasattr(spec, "__dict__"):
                payload = dict(vars(spec))
                payload["strike"] = atm_strike
                spec = spec_cls(**payload)
        except Exception:
            pass

    return payoff_cls(spec)


def _description_spec_defaults(spec_schema, *, description: str) -> dict[str, object]:
    """Extract deterministic spec defaults from structured task descriptions.

    Title-only proof tasks are bootstrapped into canonical prose descriptions.
    Reuse those explicit terms during smoke tests and comparison pricing so the
    harness instantiates the same contract surface the builder just implemented.
    """
    text = str(description or "")
    if not text:
        return {}

    spec_name = str(getattr(spec_schema, "spec_name", "") or "")
    if spec_name not in {"AgentCapSpec", "AgentFloorSpec"}:
        return {}

    from trellis.core.types import DayCountConvention, Frequency

    overrides: dict[str, object] = {}

    def _match(pattern: str) -> str | None:
        matched = re.search(pattern, text, re.IGNORECASE)
        if matched is None:
            return None
        return matched.group(1).strip()

    def _parse_number(raw: str | None) -> float | None:
        if raw in {None, ""}:
            return None
        try:
            return float(str(raw).replace(",", "").replace("_", ""))
        except ValueError:
            return None

    def _parse_percent(raw: str | None) -> float | None:
        numeric = _parse_number(raw)
        if numeric is None:
            return None
        return numeric / 100.0

    def _parse_date(raw: str | None) -> date | None:
        if raw in {None, ""}:
            return None
        try:
            return date.fromisoformat(str(raw))
        except ValueError:
            return None

    frequency_map = {
        "annual": Frequency.ANNUAL,
        "semi-annual": Frequency.SEMI_ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    day_count_map = {
        "act/360": DayCountConvention.ACT_360,
        "act/365": DayCountConvention.ACT_365,
        "30/360": DayCountConvention.THIRTY_360,
    }

    notional = _parse_number(_match(r"Notional:\s*([0-9][0-9,_.]*)"))
    if notional is not None:
        overrides["notional"] = notional

    strike = _parse_percent(_match(r"Strike:\s*([0-9][0-9,_.]*)%"))
    if strike is not None:
        overrides["strike"] = strike

    start_date = _parse_date(_match(r"Start date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"))
    if start_date is not None:
        overrides["start_date"] = start_date

    end_date = _parse_date(_match(r"End date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"))
    if end_date is not None:
        overrides["end_date"] = end_date

    frequency = _match(r"Frequency:\s*([A-Za-z-]+)")
    if frequency is not None:
        normalized_frequency = frequency.strip().lower()
        enum_value = frequency_map.get(normalized_frequency)
        if enum_value is not None:
            overrides["frequency"] = enum_value

    day_count = _match(r"Day count:\s*([A-Za-z0-9/]+)")
    if day_count is not None:
        normalized_day_count = day_count.strip().lower()
        enum_value = day_count_map.get(normalized_day_count)
        if enum_value is not None:
            overrides["day_count"] = enum_value

    rate_index = _match(r"Rate index:\s*([A-Za-z0-9._/-]+)")
    if rate_index is not None:
        overrides["rate_index"] = rate_index.rstrip(".,;:")

    return overrides


def _smoke_test_actual_market_state(
    payoff_cls,
    spec_schema,
    market_state,
) -> list[str]:
    """Run a lightweight pricing smoke test against the actual task market state."""
    if market_state is None:
        return []
    from trellis.engine.payoff_pricer import price_payoff

    settle = getattr(market_state, "settlement", date(2024, 11, 15))
    try:
        payoff = _make_test_payoff(payoff_cls, spec_schema, settle)
        price_payoff(payoff, market_state)
    except Exception as exc:
        return [f"Actual market state smoke test failed: {exc}"]
    return []


def _design_spec(
    payoff_description: str,
    requirements: set[str],
    model: str,
):
    """LLM call #1: design the spec schema via structured JSON output."""
    from trellis.agent.config import llm_generate_json, ALLOWED_FIELD_TYPES
    from trellis.agent.prompts import spec_design_prompt
    from trellis.agent.planner import SpecSchema, FieldDef

    prompt = spec_design_prompt(payoff_description, requirements)
    data = llm_generate_json(prompt, model=model)

    fields = []
    for f in data["fields"]:
        ftype = f["type"]
        if ftype not in ALLOWED_FIELD_TYPES:
            ftype = "str"  # fallback
        fields.append(FieldDef(
            name=f["name"],
            type=ftype,
            description=f.get("description", ""),
            default=f.get("default"),
        ))

    return SpecSchema(
        class_name=data["class_name"],
        spec_name=data["spec_name"],
        requirements=data["requirements"],
        fields=fields,
    )


def _generate_skeleton(
    spec_schema,
    description: str,
    *,
    pricing_plan=None,
    generation_plan=None,
) -> str:
    """Deterministically generate the full module skeleton from the spec schema."""
    instrument_type = (
        getattr(generation_plan, "instrument_type", None)
        or getattr(pricing_plan, "model_to_build", None)
        or ""
    ).strip().lower().replace(" ", "_")
    method = (
        getattr(generation_plan, "method", None)
        or getattr(pricing_plan, "method", None)
        or ""
    ).strip().lower().replace(" ", "_")

    required = [f for f in spec_schema.fields if f.default is None]
    optional = [f for f in spec_schema.fields if f.default is not None]
    field_lines = []
    for f in required + optional:
        if f.default is None:
            field_lines.append(f"    {f.name}: {f.type}")
        else:
            rendered_default = _render_spec_default_value(f.type, f.default)
            field_lines.append(f"    {f.name}: {f.type} = {rendered_default}")
    fields_block = "\n".join(field_lines)

    requirements_str = ", ".join(f'"{r}"' for r in sorted(spec_schema.requirements))
    import_lines = list(_skeleton_type_import_lines(spec_schema))
    import_lines.extend(_skeleton_exact_binding_import_lines(generation_plan))
    semantic_helper_imports, semantic_helper_lines = _skeleton_semantic_helper_hints(
        instrument_type,
        method,
    )
    import_lines.extend(semantic_helper_imports)
    evaluate_preamble_lines = ["        spec = self._spec", *semantic_helper_lines]
    extra_imports = "\n".join(dict.fromkeys(import_lines))
    if extra_imports:
        extra_imports = f"{extra_imports}\n"
    evaluate_preamble = "\n".join(evaluate_preamble_lines) + "\n"

    return f'''"""Agent-generated payoff: {description}."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
{extra_imports}


@dataclass(frozen=True)
class {spec_schema.spec_name}:
    """Specification for {description}."""
{fields_block}


class {spec_schema.class_name}:
    """{description}."""

    def __init__(self, spec: {spec_schema.spec_name}):
        self._spec = spec

    @property
    def spec(self) -> {spec_schema.spec_name}:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {{{requirements_str}}}

    def evaluate(self, market_state: MarketState) -> float:
{evaluate_preamble}{EVALUATE_SENTINEL}
'''


def _skeleton_type_import_lines(spec_schema) -> tuple[str, ...]:
    """Return the minimal type imports required by the spec fields."""
    field_types = [str(getattr(field, "type", "") or "") for field in spec_schema.fields]
    names: list[str] = []
    if any("DayCountConvention" in field_type for field_type in field_types):
        names.append("DayCountConvention")
    if any("Frequency" in field_type for field_type in field_types):
        names.append("Frequency")
    if not names:
        return ()
    return (f"from trellis.core.types import {', '.join(names)}",)


def _skeleton_semantic_helper_hints(
    instrument_type: str,
    method: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return semantic-facing helper imports and commented evaluate hints."""
    helper_hints = {
        ("quanto_option", "analytical"): (
            ("from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state",),
            ("        # return float(price_quanto_option_analytical_from_market_state(market_state, spec))",),
        ),
        ("quanto_option", "monte_carlo"): (
            ("from trellis.models.quanto_option import price_quanto_option_monte_carlo_from_market_state",),
            ("        # return float(price_quanto_option_monte_carlo_from_market_state(market_state, spec))",),
        ),
    }
    return helper_hints.get((instrument_type, method), ((), ()))


def _skeleton_exact_binding_import_lines(generation_plan) -> tuple[str, ...]:
    """Return import lines for compiler-selected exact bindings."""
    if generation_plan is None:
        return ()

    refs: list[str] = list(_exact_binding_refs(generation_plan))
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is not None:
        refs.extend(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in getattr(primitive_plan, "primitives", ()) or ()
            if (
                getattr(primitive, "required", False)
                or getattr(primitive, "role", "") in {"route_helper", "pricing_kernel", "schedule_builder"}
            )
            and getattr(primitive, "module", "")
            and getattr(primitive, "symbol", "")
        )

    imports_by_module: dict[str, list[str]] = {}
    for ref in refs:
        module, _, symbol = str(ref or "").rpartition(".")
        if not module or not symbol:
            continue
        module_symbols = imports_by_module.setdefault(module, [])
        if symbol not in module_symbols:
            module_symbols.append(symbol)

    return tuple(
        f"from {module} import {', '.join(symbols)}"
        for module, symbols in sorted(imports_by_module.items())
    )


def _exact_binding_refs(generation_plan) -> tuple[str, ...]:
    """Return the normalized exact/helper binding refs declared by the generation plan."""
    if generation_plan is None:
        return ()
    refs: list[str] = list(getattr(generation_plan, "lane_exact_binding_refs", ()) or ())
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is not None:
        refs.extend(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in getattr(primitive_plan, "primitives", ()) or ()
            if getattr(primitive, "role", "") == "route_helper"
            and getattr(primitive, "module", "")
            and getattr(primitive, "symbol", "")
        )

    normalized: list[str] = []
    for ref in refs:
        text = str(ref or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    for ref in _semantic_exact_binding_refs(tuple(normalized)):
        if ref not in normalized:
            normalized.append(ref)
    return tuple(normalized)


def _semantic_exact_binding_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    """Return semantic-facing helper refs that supersede lower-level route helpers."""
    extra: list[str] = []
    raw_to_semantic = {
        "trellis.models.analytical.quanto.price_quanto_option_analytical": (
            "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
        ),
        "trellis.models.monte_carlo.quanto.price_quanto_option_monte_carlo": (
            "trellis.models.quanto_option.price_quanto_option_monte_carlo_from_market_state"
        ),
        "trellis.models.analytical.fx.garman_kohlhagen_price_raw": (
            "trellis.models.fx_vanilla.price_fx_vanilla_analytical"
        ),
    }
    for ref in refs:
        semantic = raw_to_semantic.get(ref)
        if semantic and semantic not in extra:
            extra.append(semantic)
    return tuple(extra)


def _swaption_comparison_helper_kwargs(semantic_blueprint) -> str:
    """Return deterministic helper kwargs for explicit swaption comparison regimes."""
    valuation_context = getattr(semantic_blueprint, "valuation_context", None)
    engine_model_spec = getattr(valuation_context, "engine_model_spec", None)
    if engine_model_spec is None or getattr(engine_model_spec, "model_name", "") != "hull_white_1f":
        return ""
    overrides = dict(getattr(engine_model_spec, "parameter_overrides", {}) or {})
    mean_reversion = overrides.get("mean_reversion")
    sigma = overrides.get("sigma")
    if mean_reversion is None or sigma is None:
        return ""
    return f", mean_reversion={float(mean_reversion)!r}, sigma={float(sigma)!r}"


def _vanilla_equity_monte_carlo_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for vanilla-equity Monte Carlo comparison targets."""
    target = str(comparison_target or "").strip().lower()
    if target == "euler":
        return ', scheme="euler"'
    if target == "milstein":
        return ', scheme="milstein"'
    if target == "exact":
        return ', scheme="exact"'
    if target == "log_euler":
        return ', scheme="log_euler"'
    if target == "plain_mc":
        return ', scheme="exact", variance_reduction="none"'
    if target == "antithetic_mc":
        return ', scheme="exact", variance_reduction="antithetic"'
    if target == "control_variate_mc":
        return ', scheme="exact", variance_reduction="control_variate"'
    return ""


def _vanilla_equity_transform_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for transform comparison targets."""
    target = str(comparison_target or "").strip().lower()
    if target == "fft":
        return ', method="fft"'
    if target == "cos":
        return ', method="cos"'
    return ""


def _zcb_option_tree_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for ZCB-option tree comparison targets."""
    target = str(comparison_target or "").strip().lower()
    if target == "ho_lee_tree":
        return ', model="ho_lee"'
    if target == "hull_white_tree":
        return ', model="hull_white"'
    return ""


def _credit_basket_tranche_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for tranche-copula comparison targets."""
    target = str(comparison_target or "").strip().lower()
    if target == "student_t_copula":
        return ', copula_family="student_t", degrees_of_freedom=5.0, n_paths=40000, seed=42'
    return ', copula_family="gaussian"'


def _basket_option_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for typed basket-option helpers."""
    target = str(comparison_target or "").strip()
    if not target:
        return ""
    return f', comparison_target="{target}"'


def _deterministic_exact_binding_evaluate_body(
    generation_plan,
    *,
    semantic_blueprint=None,
    comparison_target: str | None = None,
) -> str | None:
    """Return a deterministic evaluate body for supported exact helper-backed routes."""
    refs = set(_exact_binding_refs(generation_plan))
    swaption_comparison_kwargs = _swaption_comparison_helper_kwargs(semantic_blueprint)
    vanilla_equity_mc_kwargs = _vanilla_equity_monte_carlo_helper_kwargs(comparison_target)
    vanilla_equity_transform_kwargs = _vanilla_equity_transform_helper_kwargs(comparison_target)
    zcb_option_tree_kwargs = _zcb_option_tree_helper_kwargs(comparison_target)
    credit_basket_tranche_kwargs = _credit_basket_tranche_helper_kwargs(comparison_target)
    basket_option_kwargs = _basket_option_helper_kwargs(comparison_target)
    if (
        comparison_target == "black_scholes"
        and "trellis.models.black.black76_call" in refs
        and "trellis.models.black.black76_put" in refs
    ):
        return textwrap.dedent(
            """\
            spec = self._spec
            if market_state.discount is None:
                raise ValueError("market_state.discount is required for Black-Scholes comparison")
            if market_state.vol_surface is None:
                raise ValueError("market_state.vol_surface is required for Black-Scholes comparison")
            T = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)
            spot = float(spec.spot)
            strike = float(spec.strike)
            option_type = str(spec.option_type or "call").strip().lower()
            if T <= 0.0:
                intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
                return float(spec.notional) * intrinsic
            df = float(market_state.discount.discount(T))
            sigma = float(market_state.vol_surface.black_vol(max(T, 1e-6), strike))
            forward = spot / max(df, 1e-12)
            if option_type == "call":
                undiscounted = black76_call(forward, strike, sigma, T)
            elif option_type == "put":
                undiscounted = black76_put(forward, strike, sigma, T)
            else:
                raise ValueError(f"Unsupported option_type {spec.option_type!r}")
            return float(spec.notional) * df * float(undiscounted)
            """
        ).rstrip()
    helper_bodies = {
        "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state": (
            "return float(price_quanto_option_analytical_from_market_state(market_state, spec))"
        ),
        "trellis.models.quanto_option.price_quanto_option_monte_carlo_from_market_state": (
            "return float(price_quanto_option_monte_carlo_from_market_state(market_state, spec))"
        ),
        "trellis.models.fx_vanilla.price_fx_vanilla_analytical": (
            "return float(price_fx_vanilla_analytical(market_state, spec))"
        ),
        "trellis.models.callable_bond_pde.price_callable_bond_pde": (
            "return float(price_callable_bond_pde(market_state, spec))"
        ),
        "trellis.models.callable_bond_tree.price_callable_bond_tree": (
            'return float(price_callable_bond_tree(market_state, spec, model="hull_white"))'
        ),
        "trellis.models.rate_style_swaption.price_swaption_black76": (
            "return float(price_swaption_black76(market_state, spec"
            f"{swaption_comparison_kwargs}))"
        ),
        "trellis.models.rate_style_swaption_tree.price_swaption_tree": (
            "return float(price_swaption_tree(market_state, spec"
            f"{swaption_comparison_kwargs}))"
        ),
        "trellis.models.rate_style_swaption.price_swaption_monte_carlo": (
            "return float(price_swaption_monte_carlo("
            "market_state, spec, n_paths=20000, seed=42"
            f"{swaption_comparison_kwargs}))"
        ),
        "trellis.models.equity_option_monte_carlo.price_vanilla_equity_option_monte_carlo": (
            "return float(price_vanilla_equity_option_monte_carlo("
            f"market_state, spec{vanilla_equity_mc_kwargs}))"
        ),
        "trellis.models.equity_option_transforms.price_vanilla_equity_option_transform": (
            "return float(price_vanilla_equity_option_transform("
            f"market_state, spec{vanilla_equity_transform_kwargs}))"
        ),
        "trellis.models.zcb_option_tree.price_zcb_option_tree": (
            "return float(price_zcb_option_tree("
            f"market_state, spec{zcb_option_tree_kwargs}))"
        ),
        "trellis.models.credit_basket_copula.price_credit_basket_tranche": (
            "return float(price_credit_basket_tranche("
            f"market_state, spec{credit_basket_tranche_kwargs}))"
        ),
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_recursive": (
            "return float(price_credit_portfolio_loss_distribution_recursive("
            'market_state, spec, copula_family="gaussian"))'
        ),
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_transform_proxy": (
            "return float(price_credit_portfolio_loss_distribution_transform_proxy("
            'market_state, spec, copula_family="gaussian"))'
        ),
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_monte_carlo": (
            "return float(price_credit_portfolio_loss_distribution_monte_carlo("
            'market_state, spec, copula_family="gaussian", n_paths=40000, seed=42))'
        ),
        "trellis.models.basket_option.price_basket_option_analytical": (
            "return float(price_basket_option_analytical("
            f"market_state, spec{basket_option_kwargs}))"
        ),
        "trellis.models.basket_option.price_basket_option_monte_carlo": (
            "return float(price_basket_option_monte_carlo("
            f"market_state, spec{basket_option_kwargs}, seed=42))"
        ),
        "trellis.models.basket_option.price_basket_option_transform_proxy": (
            "return float(price_basket_option_transform_proxy("
            f"market_state, spec{basket_option_kwargs}))"
        ),
    }
    for ref, body in helper_bodies.items():
        if ref in refs:
            return body
    return None


def _materialize_deterministic_exact_binding_module(
    skeleton: str,
    generation_plan,
    *,
    semantic_blueprint=None,
    comparison_target: str | None = None,
) -> GeneratedModuleResult | None:
    """Build a thin helper-backed module without invoking the LLM when the route is exact."""
    body = _deterministic_exact_binding_evaluate_body(
        generation_plan,
        semantic_blueprint=semantic_blueprint,
        comparison_target=comparison_target,
    )
    if body is None:
        return None
    rendered = skeleton.replace(
        EVALUATE_SENTINEL,
        textwrap.indent(body, "        "),
    )
    source_report = sanitize_generated_source(rendered)
    return GeneratedModuleResult(
        raw_code=rendered,
        sanitized_code=source_report.sanitized_source,
        code=source_report.sanitized_source.expandtabs(4),
        source_report=source_report,
    )


def _generate_module(
    skeleton: str,
    spec_schema,
    reference_sources: dict[str, str],
    model: str,
    max_retries: int,
    extra_context: str = "",
    pricing_plan=None,
    knowledge_context: str = "",
    generation_plan=None,
    prompt_surface: str = "expanded",
) -> GeneratedModuleResult:
    """LLM call #2: generate the complete module with evaluate() filled in."""
    from trellis.agent.config import llm_generate
    from trellis.agent.prompts import evaluate_prompt

    prompt = evaluate_prompt(skeleton, spec_schema, reference_sources,
                             pricing_plan=pricing_plan,
                             knowledge_context=knowledge_context,
                             generation_plan=generation_plan,
                             prompt_surface=prompt_surface)
    if extra_context:
        prompt += extra_context

    last_error = ""
    last_code = ""
    last_parse_candidate = ""
    for attempt in range(max_retries):
        if attempt > 0:
            full_prompt = prompt + f"\n\n## Previous attempt had error:\n{last_error}\nFix the code."
        else:
            full_prompt = prompt

        try:
            raw_code = llm_generate(full_prompt, model=model)
            last_code = raw_code

            source_report = sanitize_generated_source(raw_code)
            if not source_report.ok:
                raise GeneratedModuleSourceError(
                    "; ".join(source_report.errors),
                    source_report=source_report,
                )

            sanitized_code = source_report.sanitized_source
            code = sanitized_code.expandtabs(4)
            last_parse_candidate = code
            if not code.strip():
                raise RuntimeError("LLM returned empty module body")
            if not _module_has_expected_structure(code, spec_schema):
                recovered = _recover_generated_module_from_module_like_source(
                    code,
                    skeleton=skeleton,
                    spec_schema=spec_schema,
                )
                if recovered is None:
                    recovered = _recover_generated_module_from_fragment(
                        code,
                        skeleton=skeleton,
                        spec_schema=spec_schema,
                    )
                if recovered is not None:
                    code = recovered
                    last_parse_candidate = code
            try:
                ast.parse(code)
            except SyntaxError:
                recovered = _recover_generated_module_from_module_like_source(
                    code,
                    skeleton=skeleton,
                    spec_schema=spec_schema,
                )
                if recovered is None:
                    recovered = _recover_generated_module_from_fragment(
                        code,
                        skeleton=skeleton,
                        spec_schema=spec_schema,
                    )
                if recovered is None:
                    raise
                code = recovered
                last_parse_candidate = code
                ast.parse(code)
            if not _module_has_expected_structure(code, spec_schema):
                raise RuntimeError(
                    "LLM returned code without the expected spec/payoff module structure."
                )
            return GeneratedModuleResult(
                raw_code=raw_code,
                sanitized_code=sanitized_code,
                code=code,
                source_report=source_report,
            )
        except SyntaxError as e:
            last_error = _format_code_generation_error(
                e,
                code=last_parse_candidate or last_code,
                error_type="SyntaxError",
            )
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Agent failed to produce valid module after {max_retries} attempts: {last_error}"
                ) from e
        except RuntimeError as e:
            last_error = str(e)
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Agent failed to produce valid module after {max_retries} attempts: {last_error}"
                ) from e

    raise RuntimeError("Unreachable")


def _format_code_generation_error(exc: SyntaxError, *, code: str, error_type: str) -> str:
    """Render actionable syntax/code-generation errors with a short code preview."""
    line_no = getattr(exc, "lineno", None)
    offset = getattr(exc, "offset", None)
    msg = getattr(exc, "msg", str(exc))
    preview = _code_preview_for_error(code, line_no=line_no)
    location = ""
    if line_no is not None:
        location = f" at line {line_no}"
        if offset is not None:
            location += f", column {offset}"
    return f"{error_type}{location}: {msg}\nCode preview:\n{preview}"


def _code_preview_for_error(code: str, *, line_no: int | None, context: int = 2) -> str:
    """Return a short numbered preview around the failing source line."""
    lines = (code or "").splitlines()
    if not lines:
        return "<no code returned>"
    if line_no is None:
        preview_lines = lines[: min(len(lines), 5)]
        start = 1
    else:
        start = max(1, line_no - context)
        end = min(len(lines), line_no + context)
        preview_lines = lines[start - 1:end]
    return "\n".join(
        f"{start + index:>4}: {line}"
        for index, line in enumerate(preview_lines)
    )


def _normalize_indent(code: str, target: int = 8) -> str:
    """Re-indent code so the base level is *target* spaces, preserving relative indent.

    Uses textwrap.dedent to strip common leading whitespace first,
    then prepends *target* spaces to each line.
    """
    import textwrap
    dedented = textwrap.dedent(code)
    lines = dedented.split("\n")
    new_lines = []
    for line in lines:
        if line.strip():
            new_lines.append(" " * target + line)
        else:
            new_lines.append("")
    return "\n".join(new_lines)


def _combine_skeleton_and_body(skeleton: str, evaluate_body: str) -> str:
    """Replace the evaluate sentinel in the skeleton with the generated body."""
    if EVALUATE_SENTINEL not in skeleton:
        raise ValueError("Skeleton does not contain evaluate sentinel")
    return skeleton.replace(EVALUATE_SENTINEL, evaluate_body)


def _inject_top_level_imports(skeleton: str, import_lines: list[str]) -> str:
    """Insert additional top-level imports into the skeleton header without duplicates."""
    normalized_imports = []
    existing_imports = {
        line.strip()
        for line in skeleton.splitlines()
        if line.strip().startswith(("import ", "from "))
    }
    for line in import_lines:
        stripped = line.strip()
        if stripped.startswith("from __future__ import"):
            continue
        if stripped and stripped not in existing_imports:
            normalized_imports.append(stripped)
            existing_imports.add(stripped)
    if not normalized_imports:
        return skeleton

    lines = skeleton.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ")):
            insert_at = index + 1
    lines[insert_at:insert_at] = normalized_imports
    updated = "\n".join(lines)
    if skeleton.endswith("\n"):
        updated += "\n"
    return updated


def _recover_generated_module_from_fragment(
    code: str,
    *,
    skeleton: str,
    spec_schema,
) -> str | None:
    """Recover a full module when repair output degenerates to imports plus evaluate-body lines."""
    if EVALUATE_SENTINEL not in skeleton or _module_has_expected_structure(code, spec_schema):
        return None

    import_lines: list[str] = []
    body_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            body_lines.append("")
            continue
        if not line.startswith((" ", "\t")) and stripped.startswith(("import ", "from ")):
            import_lines.append(stripped)
            continue
        body_lines.append(line)

    body_text = _extract_fragment_body(body_lines)
    if not body_text.strip():
        return None

    return _combine_skeleton_and_body(
        _inject_top_level_imports(skeleton, import_lines),
        _normalize_indent(body_text, target=8),
    )


def _recover_generated_module_from_module_like_source(
    code: str,
    *,
    skeleton: str,
    spec_schema,
) -> str | None:
    """Recover by extracting a valid evaluate() body from malformed full-module output."""
    if EVALUATE_SENTINEL not in skeleton:
        return None

    if spec_schema.class_name not in code and "def evaluate(" not in code:
        return None

    body_text = _extract_evaluate_body_from_module_text(code)
    if not body_text.strip():
        return None

    import_lines = [
        line.strip()
        for line in code.splitlines()
        if line.strip() and not line.startswith((" ", "\t")) and line.strip().startswith(("import ", "from "))
    ]

    return _combine_skeleton_and_body(
        _inject_top_level_imports(skeleton, import_lines),
        _normalize_indent(body_text, target=8),
    )


def _module_has_expected_structure(code: str, spec_schema) -> bool:
    """Whether the generated code defines the expected spec and payoff classes."""
    import ast

    try:
        module = ast.parse(code)
    except SyntaxError:
        return False

    class_names = {
        node.name
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    return (
        spec_schema.spec_name in class_names
        and spec_schema.class_name in class_names
    )


def _extract_fragment_body(body_lines: list[str]) -> str:
    """Normalize repair fragments into the body expected by the skeleton evaluate()."""
    import textwrap

    body_text = "\n".join(body_lines).strip("\n")
    if not body_text.strip():
        return ""

    dedented = textwrap.dedent(body_text)
    dedented_lines = dedented.splitlines()
    while dedented_lines and not dedented_lines[0].strip():
        dedented_lines.pop(0)
    if not dedented_lines:
        return ""

    dedented_lines = _dedent_fragment_tail(dedented_lines)

    normalized_text = "\n".join(dedented_lines)
    first_line = dedented_lines[0].strip()
    if first_line.startswith("def evaluate(") or first_line.startswith("async def evaluate("):
        function_body = textwrap.dedent("\n".join(dedented_lines[1:])).strip("\n")
        function_body = _repair_orphan_indentation(function_body)
        return function_body
    return _repair_orphan_indentation(normalized_text)


def _extract_evaluate_body_from_module_text(code: str) -> str:
    """Extract the evaluate() body from module-like source text without parsing."""
    import textwrap

    lines = code.splitlines()
    start_index = None
    function_indent = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def evaluate(") or stripped.startswith("async def evaluate("):
            start_index = index + 1
            function_indent = len(line) - len(stripped)
            break
    if start_index is None:
        return ""

    body_lines: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())
        if stripped and current_indent <= function_indent:
            break
        body_lines.append(line)

    if not body_lines:
        return ""

    return _repair_orphan_indentation(textwrap.dedent("\n".join(body_lines)).strip("\n"))


def _dedent_fragment_tail(lines: list[str]) -> list[str]:
    """Normalize fragments where only the first top-level line lost its base indent.

    Some model outputs start with one unindented top-level statement and then keep
    the rest of the method body indented as if the base method indent were still
    present. When that shape is stitched into the skeleton verbatim, later
    top-level statements remain trapped under the first block opener and the
    generated method silently returns ``None`` on the happy path.
    """
    if len(lines) < 2:
        return lines

    first = lines[0]
    first_stripped = first.strip()
    first_indent = len(first) - len(first.lstrip())
    if first_indent != 0:
        return lines
    if first_stripped.endswith(":") or first_stripped.endswith(("\\", "(", "[", "{", ",")):
        return lines

    positive_indents = [
        len(line) - len(line.lstrip())
        for line in lines[1:]
        if line.strip() and (len(line) - len(line.lstrip())) > 0
    ]
    if not positive_indents:
        return lines

    tail_base = min(positive_indents)
    repaired = [first.lstrip()]
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            repaired.append("")
            continue
        indent = max((len(line) - len(stripped)) - tail_base, 0)
        repaired.append((" " * indent) + stripped)
    return repaired


def _repair_orphan_indentation(body_text: str) -> str:
    """Flatten indentation jumps that are not introduced by a block opener.

    Repair output from the model sometimes contains an already-indented
    ``evaluate()`` body where one statement is accidentally indented as if it
    lived under a nonexistent block. We only normalize those orphan jumps in the
    recovery path; accepted full-module output still goes through normal parsing.
    """
    lines = body_text.splitlines()
    if not lines:
        return ""

    repaired: list[str] = []
    previous_stripped = ""
    previous_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            repaired.append("")
            continue

        current_indent = len(line) - len(stripped)
        if repaired and stripped.startswith(("elif ", "else:", "except ", "except:", "finally:")):
            current_indent = max(previous_indent - 4, 0)
        elif (
            repaired
            and previous_stripped
            and previous_stripped.endswith(":")
            and current_indent <= previous_indent
            and not stripped.startswith(("elif ", "else:", "except ", "except:", "finally:"))
        ):
            current_indent = previous_indent + 4
        elif (
            repaired
            and current_indent > previous_indent
            and previous_stripped
            and previous_stripped.endswith(":")
            and current_indent > previous_indent + 4
        ):
            current_indent = previous_indent + 4
        elif (
            repaired
            and current_indent > previous_indent
            and previous_stripped
            and not previous_stripped.endswith(":")
            and not previous_stripped.endswith(("\\", "(", "[", "{", ","))
        ):
            current_indent = previous_indent

        repaired.append((" " * current_indent) + stripped)
        previous_stripped = stripped
        previous_indent = current_indent

    return "\n".join(repaired)


def _try_import_existing(plan) -> type | None:
    """Try to import a previously built payoff class."""
    from trellis.agent.builder import TRELLIS_ROOT, dynamic_import

    for step in plan.steps:
        file_path = TRELLIS_ROOT / step.module_path
        if not file_path.exists():
            return None

    last_step = plan.steps[-1]
    file_path = TRELLIS_ROOT / last_step.module_path
    module_name = f"trellis.{last_step.module_path.replace('/', '.').replace('.py', '')}"

    try:
        mod = dynamic_import(file_path, module_name)
        return getattr(mod, plan.payoff_class_name, None)
    except Exception:
        return None


def _deterministic_reuse_module_paths() -> frozenset[str]:
    """Return checked-in module paths that are explicitly reusable by route metadata."""
    try:
        from trellis.agent.route_registry import load_route_registry

        registry = load_route_registry()
        return frozenset(
            path
            for route in registry.routes
            for path in getattr(route, "reuse_module_paths", ())
            if path
        )
    except Exception:
        return frozenset()


def _is_deterministic_supported_route(plan) -> bool:
    """Whether the plan targets a checked-in deterministic adapter."""
    supported_paths = _deterministic_reuse_module_paths()
    if not supported_paths:
        return False
    return bool(plan.steps) and all(
        step.module_path in supported_paths
        for step in plan.steps
    )


def _should_run_reference_oracle(compiled_request) -> bool:
    """Return whether the current build is eligible for a single-method oracle."""
    if compiled_request is None:
        return True
    if getattr(compiled_request, "comparison_spec", None) is not None:
        return False
    comparison_method_plans = getattr(compiled_request, "comparison_method_plans", ()) or ()
    return not bool(comparison_method_plans)


def _fresh_build_module_path(module_path: str) -> str:
    """Map a deterministic route path to an isolated scratch module for proving runs."""
    path = Path(module_path)
    return str(path.parent / "_fresh" / path.name)


def _reference_modules(
    pricing_plan=None,
    instrument_type: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Select authoritative reference modules for prompt grounding."""
    from trellis.agent.knowledge.methods import normalize_method

    modules = [
        ("trellis.core.payoff", "Payoff protocol + Cashflows/PresentValue return types"),
        ("trellis.core.date_utils", "Date utilities used by generated payoffs"),
        ("trellis.core.market_state", "MarketState capabilities and access patterns"),
        ("trellis.core.types", "Frequency/day-count types"),
        ("trellis.models.black", "Black-style analytical helpers"),
    ]
    normalized_instrument = str(instrument_type or "").strip().lower()

    if pricing_plan:
        method = normalize_method(pricing_plan.method)
        if method == "analytical":
            modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))
        elif method == "rate_tree":
            modules.append(("trellis.instruments.callable_bond", "CallableBondPayoff (tree reference)"))
            modules.append(("trellis.models.callable_bond_tree", "Callable bond lattice/tree helper"))
            modules.append(("trellis.models.trees", "Tree package exports"))
            modules.append(("trellis.models.trees.lattice", "Generic/calibrated lattice builders"))
            modules.append(("trellis.models.trees.models", "Tree model registry for BDT/Hull-White selection"))
            modules.append(("trellis.models.trees.control", "Lattice exercise/control helpers"))
        elif method == "monte_carlo":
            if normalized_instrument not in {"credit_default_swap", "cds"}:
                modules.append(("trellis.instruments.barrier_option", "BarrierOptionPayoff (MC reference)"))
                modules.append(("trellis.models.monte_carlo", "Monte Carlo package exports"))
        elif method == "qmc":
            modules.append(("trellis.models.qmc", "QMC package exports"))
            modules.append(("trellis.models.monte_carlo", "Monte Carlo package exports"))
            modules.append(("trellis.models.processes.gbm", "GBM process reference"))
        elif method == "copula":
            modules.append(("trellis.instruments.nth_to_default", "NthToDefaultPayoff (copula reference)"))
            modules.append(("trellis.models.credit_basket_copula", "Credit basket copula helper"))
            modules.append(("trellis.models.copulas.factor", "Factor copula kernel"))
            modules.append(("trellis.models.copulas.gaussian", "Gaussian copula kernel"))
            modules.append(("trellis.models.copulas.student_t", "Student-t copula kernel"))
        elif method == "pde_solver":
            modules.append(("trellis.models.pde", "PDE package exports"))
            modules.append(("trellis.models.pde.theta_method", "Theta-method solver reference"))
            modules.append(("trellis.models.pde.grid", "PDE grid reference"))
            modules.append(("trellis.models.pde.operator", "PDE operator reference"))
            if normalized_instrument == "european_option":
                modules.append(("trellis.models.equity_option_pde", "Vanilla equity PDE helper"))
        elif method == "fft_pricing":
            modules.append(("trellis.models.transforms.fft_pricer", "FFT transform pricer"))
            modules.append(("trellis.models.transforms.cos_method", "COS transform pricer"))
            modules.append(("trellis.models.transforms.single_state_diffusion", "Transform diffusion contracts"))
            if normalized_instrument == "european_option":
                modules.append(("trellis.models.equity_option_transforms", "Vanilla equity transform helper"))
            modules.append(("trellis.models.processes.heston", "Heston process reference"))
        elif method == "waterfall":
            modules.append(("trellis.models.cashflow_engine.waterfall", "Cashflow waterfall reference"))
        else:
            modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))
    else:
        modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))

    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized_instrument == "quanto_option":
        modules.append(
            ("trellis.models.resolution.quanto", "Quanto input-resolution helpers")
        )
        modules.append(
            ("trellis.models.analytical.quanto", "Quanto analytical route helpers")
        )
        modules.append(
            ("trellis.models.monte_carlo.quanto", "Quanto Monte Carlo route helpers")
        )
    if normalized_instrument == "zcb_option":
        modules.append(
            ("trellis.models.zcb_option", "Jamshidian ZCB option helper")
        )
        modules.append(
            ("trellis.models.zcb_option_tree", "ZCB option tree helper")
        )

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for module_path, label in modules:
        if module_path in seen:
            continue
        seen.add(module_path)
        deduped.append((module_path, label))
    return tuple(deduped)


def _gather_references(reference_modules: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """Read reference implementations for the code generation prompt.

    Includes method-specific references based on the quant agent's plan.
    """
    refs = {}
    for mod_path, label in reference_modules:
        try:
            refs[label] = read_module_source(mod_path)
        except Exception:
            pass
    return refs


# ---------------------------------------------------------------------------
# Test resolution workflow — diagnose failures and record lessons
# ---------------------------------------------------------------------------

def _diagnose_and_enrich(failures: list[str]) -> str:
    """Run heuristic diagnosis on validation failures and return enriched feedback.

    This gives the builder agent concrete guidance beyond just "fix it",
    drawing on accumulated lessons from past debugging sessions.
    """
    from trellis.agent.test_resolution import TestFailure, diagnose_failure

    enrichments = []
    for failure_msg in failures:
        tf = TestFailure(
            test_name="build_validation",
            test_file="executor.py",
            error_type="ValidationFailure",
            error_message=failure_msg,
            expected=None,
            actual=None,
            traceback="",
        )
        diagnosis = diagnose_failure(tf)

        if diagnosis.category != "unknown":
            enrichments.append(
                f"\n**Diagnosis ({diagnosis.category}, {diagnosis.magnitude}):** "
                f"{diagnosis.root_cause}"
            )
            if diagnosis.related_lessons:
                enrichments.append(
                    f"  Related past lessons: {', '.join(diagnosis.related_lessons)}"
                )

    # Also match against YAML-driven failure signatures
    try:
        from trellis.agent.knowledge.signatures import diagnose_from_signatures
        from trellis.agent.knowledge import get_store
        sig_text = diagnose_from_signatures(
            failures, get_store()._failure_signatures
        )
        if sig_text:
            enrichments.append(sig_text)
    except Exception:
        pass

    if enrichments:
        return "\n\n## AUTOMATED DIAGNOSIS:\n" + "\n".join(enrichments)
    return ""


def _format_validation_failure_feedback(
    *,
    failures: list[str],
    failure_details: tuple[object, ...] | list[object] | None = None,
) -> str:
    """Render machine-readable validation details into builder repair guidance."""
    lines = ["\n\n## VALIDATION FAILURES (your previous code had these issues):"]
    if not failure_details:
        lines.extend(f"- {failure}" for failure in failures)
        return "\n".join(lines) + "\n"

    for detail in failure_details:
        check = getattr(detail, "check", "unknown_check")
        message = getattr(detail, "message", str(detail))
        lines.append(f"- [{check}] {message}")
        actual = getattr(detail, "actual", None)
        if actual is not None:
            lines.append(f"  Actual: {actual}")
        expected = getattr(detail, "expected", None)
        if expected is not None:
            lines.append(f"  Expected: {expected}")
        exception_type = getattr(detail, "exception_type", None)
        exception_message = getattr(detail, "exception_message", None)
        if exception_type or exception_message:
            exception_text = (
                f"{exception_type}: {exception_message}"
                if exception_type and exception_message
                else str(exception_type or exception_message)
            )
            lines.append(f"  Exception: {exception_text}")
        context = getattr(detail, "context", None) or {}
        if context:
            context_parts = []
            for key in sorted(context):
                value = context[key]
                context_parts.append(f"{key}={value}")
            lines.append(f"  Context: {', '.join(context_parts)}")
    return "\n".join(lines) + "\n"


def _record_resolved_failures(
    failures: list[str],
    description: str,
    pricing_plan,
    model: str,
) -> None:
    """After a successful retry, ask the LLM to distill the lesson learned.

    This records the lesson into the canonical knowledge store so future builds
    can retrieve it through normal lesson selection.
    """
    import logging
    import os

    from trellis.agent.config import get_provider, llm_generate_json
    from trellis.agent.test_resolution import Lesson, record_lesson

    provider = get_provider()
    credential_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "")
    if credential_env and not os.environ.get(credential_env):
        logging.getLogger(__name__).warning(
            "Skipping resolved-failure lesson distillation because %s is not configured.",
            credential_env,
        )
        return

    prompt = f"""You fixed these validation failures for a {description}
({pricing_plan.method if pricing_plan else 'unknown'} method):

{chr(10).join(f'- {f}' for f in failures)}

Distill ONE concise lesson learned. Return JSON:
{{
  "category": "calibration|volatility|backward_induction|finite_differences|monte_carlo|market_data",
  "title": "Short title (max 10 words)",
  "mistake": "What went wrong (1 sentence)",
  "why": "Why it happens — the mental model failure (1-2 sentences)",
  "detect": "How to detect this issue (1 sentence)",
  "fix": "How to fix it (1 sentence)"
}}

Only return JSON, no markdown."""

    data = llm_generate_json(prompt, model=model)
    required = {"category", "title", "mistake", "why", "detect", "fix"}
    missing = required - set(data.keys())
    if missing:
        raise RuntimeError(f"LLM lesson output missing fields: {sorted(missing)}")

    lesson = Lesson(**data)
    from trellis.agent.knowledge.decompose import decompose as kn_decompose

    features = []
    try:
        decomp = kn_decompose(description, instrument_type=None)
        features = list(decomp.features)
    except Exception:
        pass

    try:
        record_lesson(
            lesson,
            severity="high",
            validation=f"Resolved during build of {description}",
            method=pricing_plan.method if pricing_plan else None,
            features=features,
            confidence=0.5,
        )
    except Exception as exc:
        raise RuntimeError(f"Lesson recording failed for {lesson.title!r}") from exc


# ---------------------------------------------------------------------------
# Knowledge system helpers
# ---------------------------------------------------------------------------

def _builder_knowledge_context_for_attempt(
    pricing_plan,
    instrument_type: str | None,
    *,
    attempt_number: int,
    retry_reason: str | None = None,
    compiled_request=None,
    product_ir=None,
    prompt_surface: str | None = None,
    build_meta: dict | None = None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None = None,
) -> tuple[str, str]:
    """Return builder knowledge text and surface for one attempt."""
    context = _resolve_knowledge_context_for_attempt(
        audience="builder",
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        attempt_number=attempt_number,
        retry_reason=retry_reason,
        compiled_request=compiled_request,
        product_ir=product_ir,
        prompt_surface=prompt_surface or _builder_prompt_surface_for_attempt(
            attempt_number=attempt_number,
            retry_reason=retry_reason,
        ),
        build_meta=build_meta,
        knowledge_retriever=knowledge_retriever,
    )
    return context.text, context.knowledge_surface


def _review_knowledge_context_for_attempt(
    pricing_plan,
    instrument_type: str | None,
    *,
    attempt_number: int,
    compiled_request=None,
    product_ir=None,
    prompt_surface: str | None = None,
    build_meta: dict | None = None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None = None,
) -> tuple[str, str]:
    """Return reviewer knowledge with the same compact-first policy."""
    context = _resolve_knowledge_context_for_attempt(
        audience="review",
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        attempt_number=attempt_number,
        compiled_request=compiled_request,
        product_ir=product_ir,
        prompt_surface=prompt_surface or "review",
        build_meta=build_meta,
        knowledge_retriever=knowledge_retriever,
    )
    return context.text, context.knowledge_surface


def _builder_prompt_surface_for_attempt(
    *,
    attempt_number: int,
    retry_reason: str | None,
) -> str:
    """Select the retry prompt surface from the previous failure type."""
    if attempt_number <= 1:
        return "compact"
    if retry_reason == "code_generation":
        return "compact"
    if retry_reason == "import_validation":
        return "import_repair"
    if retry_reason in {"semantic_validation", "lite_review", "actual_market_smoke", "comparison_insufficient_results"}:
        return "semantic_repair"
    if retry_reason == "validation":
        return "compact"
    return "expanded"


def _resolve_knowledge_context_for_attempt(
    *,
    audience: str,
    pricing_plan,
    instrument_type: str | None,
    attempt_number: int,
    retry_reason: str | None = None,
    compiled_request=None,
    product_ir=None,
    prompt_surface: str,
    build_meta: dict | None = None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None = None,
    previous_failures: list[str] | tuple[str, ...] | None = None,
) -> KnowledgeContextResult:
    """Resolve knowledge text and retrieval metadata for one build/review step."""
    retrieval_stage = _knowledge_retrieval_stage(
        audience=audience,
        attempt_number=attempt_number,
        retry_reason=retry_reason,
    )
    knowledge_surface = _knowledge_surface_for_stage(
        audience=audience,
        attempt_number=attempt_number,
        retry_reason=retry_reason,
    )
    compact = knowledge_surface != "expanded"
    request = KnowledgeRetrievalRequest(
        audience=audience,
        stage=retrieval_stage,
        attempt_number=attempt_number,
        knowledge_surface=knowledge_surface,
        prompt_surface=prompt_surface,
        retry_reason=retry_reason,
        instrument_type=instrument_type,
        pricing_method=getattr(pricing_plan, "method", None),
        product_ir=product_ir,
        compiled_request=compiled_request,
        recent_failures=tuple(
            str(item) for item in (previous_failures or ()) if str(item).strip()
        ),
    )

    if knowledge_retriever is not None:
        callback_text = knowledge_retriever(request)
        if isinstance(callback_text, str) and callback_text:
            callback_text, selected_artifacts = _augment_retrieval_text(
                callback_text,
                request=request,
            )
            _record_knowledge_retrieval(
                build_meta,
                audience=audience,
                retrieval_stage=retrieval_stage,
                knowledge_surface=knowledge_surface,
                retrieval_source="callback",
                attempt_number=attempt_number,
                prompt_surface=prompt_surface,
                retry_reason=retry_reason,
                chars=len(callback_text),
                selected_artifacts=selected_artifacts,
            )
            return KnowledgeContextResult(
                text=callback_text,
                knowledge_surface=knowledge_surface,
                retrieval_stage=retrieval_stage,
                retrieval_source="callback",
            )

    if compiled_request is not None and getattr(compiled_request, "knowledge", None) is not None:
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload

        knowledge_profile = str(
            (getattr(compiled_request, "knowledge_summary", {}) or {}).get("knowledge_profile") or ""
        ).strip()
        if knowledge_profile == "knowledge_light":
            stored_text = (
                compiled_request.knowledge_text
                if audience == "builder"
                else compiled_request.review_knowledge_text
            )
            if stored_text:
                text, selected_artifacts = _augment_retrieval_text(
                    stored_text,
                    request=request,
                )
                _record_knowledge_retrieval(
                    build_meta,
                    audience=audience,
                    retrieval_stage=retrieval_stage,
                    knowledge_surface=knowledge_surface,
                    retrieval_source="compiled_request",
                    attempt_number=attempt_number,
                    prompt_surface=prompt_surface,
                    retry_reason=retry_reason,
                    chars=len(text),
                    selected_artifacts=selected_artifacts,
                )
                return KnowledgeContextResult(
                    text=text,
                    knowledge_surface=knowledge_surface,
                    retrieval_stage=retrieval_stage,
                    retrieval_source="compiled_request",
                )

        if compact:
            if audience == "builder" and compiled_request.knowledge_text:
                text, selected_artifacts = _augment_retrieval_text(
                    compiled_request.knowledge_text,
                    request=request,
                )
                _record_knowledge_retrieval(
                    build_meta,
                    audience=audience,
                    retrieval_stage=retrieval_stage,
                    knowledge_surface=knowledge_surface,
                    retrieval_source="compiled_request",
                    attempt_number=attempt_number,
                    prompt_surface=prompt_surface,
                    retry_reason=retry_reason,
                    chars=len(text),
                    selected_artifacts=selected_artifacts,
                )
                return KnowledgeContextResult(
                    text=text,
                    knowledge_surface=knowledge_surface,
                    retrieval_stage=retrieval_stage,
                    retrieval_source="compiled_request",
                )
            if audience == "review" and compiled_request.review_knowledge_text:
                text, selected_artifacts = _augment_retrieval_text(
                    compiled_request.review_knowledge_text,
                    request=request,
                )
                _record_knowledge_retrieval(
                    build_meta,
                    audience=audience,
                    retrieval_stage=retrieval_stage,
                    knowledge_surface=knowledge_surface,
                    retrieval_source="compiled_request",
                    attempt_number=attempt_number,
                    prompt_surface=prompt_surface,
                    retry_reason=retry_reason,
                    chars=len(text),
                    selected_artifacts=selected_artifacts,
                )
                return KnowledgeContextResult(
                    text=text,
                    knowledge_surface=knowledge_surface,
                    retrieval_stage=retrieval_stage,
                    retrieval_source="compiled_request",
                )

        payload = build_shared_knowledge_payload(compiled_request.knowledge)
        if audience == "builder":
            key = "builder_text_expanded" if not compact else "builder_text_distilled"
        elif audience == "review":
            key = "review_text_expanded" if not compact else "review_text_distilled"
        else:
            raise ValueError(f"Unsupported knowledge audience: {audience}")
        text, selected_artifacts = _augment_retrieval_text(
            payload.get(key, ""),
            request=request,
        )
        _record_knowledge_retrieval(
            build_meta,
            audience=audience,
            retrieval_stage=retrieval_stage,
            knowledge_surface=knowledge_surface,
            retrieval_source="compiled_request_payload",
            attempt_number=attempt_number,
            prompt_surface=prompt_surface,
            retry_reason=retry_reason,
            chars=len(text),
            selected_artifacts=selected_artifacts,
        )
        return KnowledgeContextResult(
            text=text,
            knowledge_surface=knowledge_surface,
            retrieval_stage=retrieval_stage,
            retrieval_source="compiled_request_payload",
        )

    if audience == "builder":
        text = _retrieve_knowledge(
            pricing_plan,
            instrument_type,
            product_ir=product_ir,
            compact=compact,
        )
        text, selected_artifacts = _augment_retrieval_text(text, request=request)
        _record_knowledge_retrieval(
            build_meta,
            audience=audience,
            retrieval_stage=retrieval_stage,
            knowledge_surface=knowledge_surface,
            retrieval_source="live_retrieval",
            attempt_number=attempt_number,
            prompt_surface=prompt_surface,
            retry_reason=retry_reason,
            chars=len(text),
            selected_artifacts=selected_artifacts,
        )
        return KnowledgeContextResult(
            text=text,
            knowledge_surface=knowledge_surface,
            retrieval_stage=retrieval_stage,
            retrieval_source="live_retrieval",
        )
    if audience == "review":
        text = _retrieve_review_knowledge(
            pricing_plan,
            instrument_type,
            product_ir=product_ir,
            compact=compact,
        )
        text, selected_artifacts = _augment_retrieval_text(text, request=request)
        _record_knowledge_retrieval(
            build_meta,
            audience=audience,
            retrieval_stage=retrieval_stage,
            knowledge_surface=knowledge_surface,
            retrieval_source="live_retrieval",
            attempt_number=attempt_number,
            prompt_surface=prompt_surface,
            retry_reason=retry_reason,
            chars=len(text),
            selected_artifacts=selected_artifacts,
        )
        return KnowledgeContextResult(
            text=text,
            knowledge_surface=knowledge_surface,
            retrieval_stage=retrieval_stage,
            retrieval_source="live_retrieval",
        )
    raise ValueError(f"Unsupported knowledge audience: {audience}")


def _knowledge_retrieval_stage(
    *,
    audience: str,
    attempt_number: int,
    retry_reason: str | None,
) -> str:
    """Normalize builder/reviewer attempt state into a stable retrieval stage label."""
    if audience == "review":
        return "critic_review_after_retry" if attempt_number > 1 else "critic_review"
    if attempt_number <= 1:
        return "initial_build"
    if retry_reason == "code_generation":
        return "code_generation_failed"
    if retry_reason == "import_validation":
        return "import_validation_failed"
    if retry_reason == "semantic_validation":
        return "semantic_validation_failed"
    if retry_reason == "lite_review":
        return "lite_review_failed"
    if retry_reason == "actual_market_smoke":
        return "actual_market_smoke_failed"
    if retry_reason == "validation":
        return "validation_failed"
    if retry_reason == "comparison_insufficient_results":
        return "comparison_insufficient_results"
    return "retry_build"


def _knowledge_surface_for_stage(
    *,
    audience: str,
    attempt_number: int,
    retry_reason: str | None,
) -> str:
    """Select the prompt-budget surface for the current retry stage."""
    if audience == "review":
        return "expanded" if attempt_number > 1 else "compact"
    if attempt_number <= 1:
        return "compact"
    if retry_reason in {"semantic_validation", "lite_review", "actual_market_smoke", "comparison_insufficient_results"}:
        return "expanded"
    return "compact"


def _record_knowledge_retrieval(
    build_meta: dict | None,
    *,
    audience: str,
    retrieval_stage: str,
    knowledge_surface: str,
    retrieval_source: str,
    attempt_number: int,
    prompt_surface: str,
    retry_reason: str | None,
    chars: int,
    selected_artifacts: list[dict[str, str]] | None = None,
) -> None:
    """Persist one structured retrieval decision for later diagnosis."""
    if build_meta is None:
        return
    decision = {
        "audience": audience,
        "stage": retrieval_stage,
        "knowledge_surface": knowledge_surface,
        "retrieval_source": retrieval_source,
        "attempt": attempt_number,
        "prompt_surface": prompt_surface,
        "retry_reason": retry_reason,
        "chars": chars,
    }
    if selected_artifacts:
        decision["selected_artifacts"] = [dict(item) for item in selected_artifacts]
    build_meta.setdefault("knowledge_retrieval_history", []).append(decision)
    summary = build_meta.setdefault("knowledge_summary", {})
    for key, value in (
        ("retrieval_stages", retrieval_stage),
        ("retrieval_sources", retrieval_source),
    ):
        existing = [
            item
            for item in summary.get(key, ())
            if isinstance(item, str) and item
        ]
        if value not in existing:
            existing.append(value)
        summary[key] = existing
    if selected_artifacts:
        for key, field in (
            ("selected_artifact_ids", "id"),
            ("selected_artifact_titles", "title"),
        ):
            existing = [
                item
                for item in summary.get(key, ())
                if isinstance(item, str) and item
            ]
            for artifact in selected_artifacts:
                value = str(artifact.get(field) or "").strip()
                if value and value not in existing:
                    existing.append(value)
            summary[key] = existing


def _augment_retrieval_text(
    text: str,
    *,
    request: KnowledgeRetrievalRequest,
) -> tuple[str, list[dict[str, str]]]:
    """Append stage-aware retry guidance and instrument disambiguation notes."""
    sections: list[str] = []
    selected_artifacts = _prune_selected_artifacts_for_prompt(
        text,
        _stage_aware_skill_artifacts(request),
        knowledge_surface=request.knowledge_surface,
    )
    if selected_artifacts:
        sections.append("## Stage-Aware Skills")
        sections.extend(
            f"- [{artifact['kind']}] {artifact['title']}: {artifact['summary']}"
            for artifact in selected_artifacts
        )

    retry_lines = _retry_focus_lines(request.stage)
    if retry_lines:
        sections.append("## Retry Focus")
        sections.extend(f"- {line}" for line in retry_lines)

    route_specific_lines = _route_specific_retry_lines(request)
    if route_specific_lines:
        sections.append("## Route-Specific Recovery")
        sections.extend(f"- {line}" for line in route_specific_lines)

    disambiguation_lines = _instrument_disambiguation_lines(request)
    if disambiguation_lines:
        sections.append("## Instrument Disambiguation")
        sections.extend(f"- {line}" for line in disambiguation_lines)

    if not sections:
        return text, selected_artifacts

    suffix = "\n".join(sections)
    if text.strip():
        return f"{text}\n\n{suffix}", selected_artifacts
    return suffix, selected_artifacts


def _prune_selected_artifacts_for_prompt(
    text: str,
    artifacts: list[dict[str, str]],
    *,
    knowledge_surface: str,
) -> list[dict[str, str]]:
    """Drop duplicate stage-aware guidance and enforce a small prompt budget."""
    normalized_text = " ".join(str(text or "").lower().split())
    budget = 900 if knowledge_surface == "expanded" else 420
    selected: list[dict[str, str]] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    used_chars = 0

    for artifact in artifacts:
        artifact_id = str(artifact.get("id") or "").strip()
        kind = str(artifact.get("kind") or "").strip()
        title = " ".join(str(artifact.get("title") or "").split())
        summary = " ".join(str(artifact.get("summary") or "").split())
        if not summary:
            continue
        signature = (artifact_id, title.lower(), summary.lower())
        if signature in seen_signatures:
            continue
        normalized_title = title.lower()
        normalized_summary = summary.lower()
        if normalized_summary in normalized_text:
            continue
        if normalized_title and normalized_title in normalized_text:
            continue
        projected = used_chars + len(title) + len(summary) + 8
        if selected and projected > budget:
            break
        selected.append(
            {
                "id": artifact_id,
                "kind": kind,
                "title": title,
                "summary": summary,
            }
        )
        seen_signatures.add(signature)
        used_chars = projected
    return selected


def _stage_aware_skill_artifacts(
    request: KnowledgeRetrievalRequest,
) -> list[dict[str, str]]:
    """Select deterministic skill-index guidance for retry-time prompt repair."""
    if request.audience == "builder" and request.attempt_number <= 1:
        return []
    if request.audience == "review" and request.stage == "critic_review":
        return []
    try:
        from trellis.agent.knowledge import load_skill_index
    except Exception:
        return []

    instrument_tokens = _candidate_instrument_tokens(request)
    method_token = _normalize_retrieval_token(request.pricing_method)
    route_ids = _candidate_route_ids(request)
    route_families = _candidate_route_families(request)
    kind_order = _skill_kind_order_for_stage(request.stage, audience=request.audience)
    limit = 5 if request.knowledge_surface == "expanded" else 3

    records = []
    for record in load_skill_index().records:
        if record.kind not in kind_order:
            continue
        if record.status and _normalize_retrieval_token(record.status) not in {
            "active",
            "validated",
            "promoted",
            "fresh",
        }:
            continue
        if not _skill_record_matches_request(
            record,
            instrument_tokens=instrument_tokens,
            method_token=method_token,
            route_ids=route_ids,
            route_families=route_families,
        ):
            continue
        records.append(record)

    records.sort(
        key=lambda record: _skill_record_rank(
            record,
            kind_order=kind_order,
            instrument_tokens=instrument_tokens,
            method_token=method_token,
            route_ids=route_ids,
            route_families=route_families,
        )
    )

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        if record.skill_id in seen:
            continue
        seen.add(record.skill_id)
        summary = " ".join(str(record.summary or "").split())
        if not summary:
            continue
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        selected.append(
            {
                "id": record.skill_id,
                "kind": record.kind,
                "title": record.title,
                "summary": summary,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _skill_kind_order_for_stage(
    stage: str,
    *,
    audience: str,
) -> tuple[str, ...]:
    """Return the preferred skill kinds for the current retrieval stage."""
    if audience == "review":
        return ("principle", "lesson", "route_hint")
    if stage == "code_generation_failed":
        return ("route_hint", "cookbook", "principle")
    if stage == "import_validation_failed":
        return ("route_hint", "principle", "lesson")
    if stage in {
        "semantic_validation_failed",
        "lite_review_failed",
        "actual_market_smoke_failed",
        "validation_failed",
        "comparison_insufficient_results",
    }:
        return ("route_hint", "lesson", "principle", "cookbook")
    return ("cookbook", "principle", "lesson")


def _skill_record_matches_request(
    record,
    *,
    instrument_tokens: tuple[str, ...],
    method_token: str,
    route_ids: tuple[str, ...],
    route_families: tuple[str, ...],
) -> bool:
    """Apply deterministic scope matching for one skill record."""
    record_methods = {_normalize_retrieval_token(item) for item in record.method_families}
    if method_token and record_methods and method_token not in record_methods:
        return False

    record_instruments = {_normalize_retrieval_token(item) for item in record.instrument_types}
    record_route_families = {_normalize_retrieval_token(item) for item in record.route_families}
    record_tags = {_normalize_retrieval_token(item) for item in record.tags}
    route_id_tags = {f"route:{route_id}" for route_id in route_ids if route_id}

    if record.kind == "route_hint":
        if route_id_tags & record_tags:
            return True
        if route_families and record_route_families and set(route_families) & record_route_families:
            return True
        if instrument_tokens and record_instruments and set(instrument_tokens) & record_instruments:
            return True
        return False

    if instrument_tokens and record_instruments:
        return bool(set(instrument_tokens) & record_instruments)
    return True


def _skill_record_rank(
    record,
    *,
    kind_order: tuple[str, ...],
    instrument_tokens: tuple[str, ...],
    method_token: str,
    route_ids: tuple[str, ...],
    route_families: tuple[str, ...],
) -> tuple[object, ...]:
    """Order matched skill records by route specificity, stage fit, and stability."""
    record_instruments = {_normalize_retrieval_token(item) for item in record.instrument_types}
    record_methods = {_normalize_retrieval_token(item) for item in record.method_families}
    record_route_families = {_normalize_retrieval_token(item) for item in record.route_families}
    record_tags = {_normalize_retrieval_token(item) for item in record.tags}
    route_id_score = int(bool({f"route:{route_id}" for route_id in route_ids if route_id} & record_tags))
    route_family_score = int(bool(set(route_families) & record_route_families))
    instrument_score = int(bool(set(instrument_tokens) & record_instruments))
    method_score = int(bool(method_token and method_token in record_methods))
    hard_constraint_score = int(str(getattr(record, "instruction_type", "") or "") == "hard_constraint")
    return (
        -hard_constraint_score,
        -instrument_score,
        -method_score,
        -route_family_score,
        -route_id_score,
        kind_order.index(record.kind),
        -int(getattr(record, "precedence_rank", 0) or 0),
        -float(getattr(record, "confidence", 0.0) or 0.0),
        str(getattr(record, "skill_id", "") or ""),
    )


def _candidate_instrument_tokens(
    request: KnowledgeRetrievalRequest,
) -> tuple[str, ...]:
    """Return normalized instrument aliases usable for skill matching."""
    raw_values = [
        request.instrument_type,
        getattr(request.product_ir, "instrument", None),
    ]
    tokens = {
        _normalize_retrieval_token(value)
        for value in raw_values
        if _normalize_retrieval_token(value)
    }
    if "credit_default_swap" in tokens:
        tokens.add("cds")
    if "cds" in tokens:
        tokens.add("credit_default_swap")
    return tuple(sorted(tokens))


def _candidate_route_ids(
    request: KnowledgeRetrievalRequest,
) -> tuple[str, ...]:
    """Return exact route identifiers from the compiled semantic/generation boundary."""
    compiled_request = request.compiled_request
    generation_plan = getattr(compiled_request, "generation_plan", None)
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    request_metadata = (
        getattr(getattr(compiled_request, "request", None), "metadata", None) or {}
    )
    semantic_blueprint = dict(request_metadata.get("semantic_blueprint") or {})
    values = {
        _normalize_retrieval_token(getattr(generation_plan, "lowering_route_id", None)),
        _normalize_retrieval_token(getattr(primitive_plan, "route", None)),
        _normalize_retrieval_token(semantic_blueprint.get("dsl_route")),
    }
    return tuple(sorted(value for value in values if value))


def _candidate_route_families(
    request: KnowledgeRetrievalRequest,
) -> tuple[str, ...]:
    """Return route-family tokens from the semantic and generation boundary."""
    compiled_request = request.compiled_request
    generation_plan = getattr(compiled_request, "generation_plan", None)
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    request_metadata = (
        getattr(getattr(compiled_request, "request", None), "metadata", None) or {}
    )
    semantic_blueprint = dict(request_metadata.get("semantic_blueprint") or {})
    values = {
        _normalize_retrieval_token(getattr(generation_plan, "lowering_route_family", None)),
        _normalize_retrieval_token(getattr(primitive_plan, "route_family", None)),
        _normalize_retrieval_token(semantic_blueprint.get("dsl_route_family")),
    }
    route_families = getattr(request.product_ir, "route_families", ()) or ()
    for value in route_families:
        normalized = _normalize_retrieval_token(value)
        if normalized:
            values.add(normalized)
    return tuple(sorted(value for value in values if value))


def _normalize_retrieval_token(value: object) -> str:
    """Normalize free-form metadata tokens for deterministic matching."""
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_")


def _retry_focus_lines(stage: str) -> tuple[str, ...]:
    """Return narrow retry instructions for the current failure stage."""
    if stage == "code_generation_failed":
        return (
            "Regenerate from the canonical scaffold; do not improvise a new module shape.",
            "Keep indentation, imports, and class/spec names aligned with the approved route skeleton.",
        )
    if stage == "import_validation_failed":
        return (
            "Use only import-registry and API-map-backed modules and symbols.",
            "Do not swap route families or import adjacent credit/copula helpers just because the names sound similar.",
        )
    if stage == "actual_market_smoke_failed":
        return (
            "Recover against the real task market contract: required curves, spots, and volatility surfaces must be read from the live market_state.",
            "Do not fix actual-market failures by hard-coding fixture values or widening fallback defaults inside evaluate().",
        )
    if stage in {"semantic_validation_failed", "validation_failed"}:
        return (
            "Match the runtime contract exactly: required curves, units, and payoff-leg signs must be explicit in code.",
            "Prefer a smaller, explicit implementation over a broad helper abstraction when recovering from validation failures.",
        )
    if stage == "lite_review_failed":
        return (
            "Address the flagged semantic/risk issues directly instead of broadening the route or changing the product family.",
        )
    if stage == "comparison_insufficient_results":
        return (
            "Rebuild only the missing comparison lane and keep it semantically aligned with the canonical route contract.",
            "Prefer thin adapters over novel product-specific scaffolding when a comparison target is missing or invalid.",
        )
    return ()


def _route_specific_retry_lines(
    request: KnowledgeRetrievalRequest,
) -> tuple[str, ...]:
    """Return narrow route-repair guidance for fragile families."""
    instrument = (request.instrument_type or "").strip().lower()
    if not instrument and request.product_ir is not None:
        instrument = str(getattr(request.product_ir, "instrument", "") or "").strip().lower()
    pricing_method = str(request.pricing_method or "").strip().lower()
    stage = str(request.stage or "").strip().lower()
    request_metadata = (
        getattr(getattr(request.compiled_request, "request", None), "metadata", None) or {}
    )
    comparison_target = str(request_metadata.get("comparison_target") or "").strip().lower()

    if (
        instrument in {"credit_default_swap", "cds"}
        and pricing_method in {"monte_carlo", "qmc"}
        and stage in {"code_generation_failed", "import_validation_failed", "validation_failed", "actual_market_smoke_failed"}
    ):
        return (
            "Single-name CDS Monte Carlo does not need an equity price process, spot diffusion, or volatility path.",
            "Do not import `trellis.models.processes.gbm` or any adjacent equity-process fallback just to make Monte Carlo compile.",
            "Do not import or instantiate `MonteCarloEngine` for a single-name CDS route. That engine expects a diffusion process and is the wrong scaffold here.",
            "Stay within the approved CDS route backbone: credit-curve default-time sampling, discounting, schedule generation, and leg aggregation.",
            "Prefer `from trellis.models.credit_default_swap import build_cds_schedule, price_cds_monte_carlo` and delegate to those helpers from the adapter.",
            "Prefer `build_period_schedule(spec.start_date, spec.end_date, spec.frequency, day_count=spec.day_count, time_origin=spec.start_date)` so the route iterates over explicit periods instead of rebuilding coupon boundaries by hand.",
            "Use `from trellis.core.differentiable import get_numpy`, `np = get_numpy()`, and direct `np.random.default_rng(...)` draws for default times instead.",
            "Track accrual dates and survival/default times separately: use `prev_date` for `year_fraction(prev_date, pay_date, ...)` and `prev_t` for survival/default-time thresholds.",
            "Do not compare float year-fractions to `date` objects, and do not pass floats into the date positions of `year_fraction(...)`.",
            "Use `period.payment_date`, `period.accrual_fraction`, and `period.t_payment` from that schedule object so the Monte Carlo leg covers the full CDS horizon without reconstructing payment_dates manually.",
            "This route must price a Monte Carlo expectation over many paths. Use `n_paths = ...`, `alive = np.ones(n_paths, dtype=bool)`, and vectorized `default_in_interval` arrays.",
            "Do not hard-code `n_paths=50000` for a comparison-quality single-name CDS build. If the spec exposes `n_paths`, pass `spec.n_paths` through to `price_cds_monte_carlo(...)`; otherwise use a comparison-stable path count such as `250000`.",
            "Keep the CDS comparison build reproducible with `seed=42` unless the spec explicitly carries another seed.",
            "Do not collapse the Monte Carlo CDS leg to scalar `alive`, a single `rng.random()` draw per payment date, or a one-scenario loop that breaks after default.",
            "Compute interval default probability from survival ratios: `default_prob = max(0.0, min(1.0, 1.0 - s_pay / s_prev))` using `survival_probability(prev_t)` and `survival_probability(t_pay)`.",
            "Do not replace that interval default probability with a midpoint-hazard shortcut like `1.0 - exp(-hazard * dt)` when survival probabilities are available.",
            "For this comparison route, keep protection-leg discounting aligned with the analytical schedule loop: accrue interval default mass with the payment-date discount factor `discount(t_pay)`.",
            "Do not discount protection at sampled default times `tau` or replace interval default mass with sampled settlement-time discounting in the comparison build.",
            "Use `spec.start_date` as the time origin for Monte Carlo schedule times. Do not switch this route to `market_state.as_of` while the analytical comparator uses `spec.start_date`.",
            "Carry a persistent `alive` indicator across the schedule; do not overwrite the default state from scratch inside each interval.",
            "Use per-interval conditional default draws: `default_in_interval = alive & (u < conditional_default_prob)`, accrue protection on that interval only, then update `alive &= ~default_in_interval` before the next coupon date.",
            "Update `alive` before premium accrual. The premium leg should use the fraction of paths still alive through the payment date, not the start-of-interval alive state, so the Monte Carlo leg timing matches the analytical schedule loop in expectation.",
            "Normalize the running spread immediately with `spread = float(spec.spread)` and `if spread > 1.0: spread *= 1e-4` before any premium-leg accrual.",
            "After that normalization step, use only the local `spread` variable; do not read raw `spec.spread` again inside the loop.",
            "Validation contract: semantically equivalent quotes `100` and `0.01` must produce the same CDS PV up to numerical tolerance.",
        )
    if (
        instrument in {"credit_default_swap", "cds"}
        and pricing_method == "analytical"
        and stage in {"code_generation_failed", "import_validation_failed", "validation_failed", "actual_market_smoke_failed"}
    ):
        return (
            "Single-name CDS analytical pricing should stay on the same explicit schedule convention as the Monte Carlo comparator.",
            "Normalize the running spread immediately with `spread = float(spec.spread)` and `if spread > 1.0: spread *= 1e-4`, then use only the local `spread` variable.",
            "Prefer `from trellis.models.credit_default_swap import build_cds_schedule, price_cds_analytical` and delegate to those helpers from the adapter.",
            "Do not write `from trellis.models import black`; if you genuinely need Black helpers, import concrete symbols from `trellis.models.black`, but this CDS route should normally only need `trellis.models.credit_default_swap` plus core helpers.",
            "Prefer `build_period_schedule(spec.start_date, spec.end_date, spec.frequency, day_count=spec.day_count, time_origin=spec.start_date)` and iterate over explicit periods instead of rebuilding date lists by hand.",
            "Use `spec.start_date` as the time origin for `year_fraction(...)` schedule times so the analytical and Monte Carlo legs share the same `t` convention.",
            "For each payment date, discount both the premium leg and the interval default mass with `market_state.discount.discount(pay_t)`.",
            "Keep the premium leg to `spread * accrual * df * survival` only. Do not add a separate accrued-on-default premium adjustment such as `0.5 * spread * accrual * df * (prev_survival - survival)`.",
            "Do not average adjacent discount factors, trapezoid the protection leg, or introduce `0.5 * (prev_discount + discount)`.",
            "Keep the body as one schedule loop with `premium_leg += ... * df * surv` and `protection_leg += ... * df * max(prev_survival - survival, 0.0)`.",
            "Return a full Python module with the spec class and payoff class. Do not emit only an `evaluate()` fragment, markdown bullets, or an indented class body snippet.",
        )
    if (
        instrument == "callable_bond"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed"}
    ):
        return (
            "Callable rate-tree routes must keep both rate-vol access and schedule-dependent lattice control explicit in the generated body.",
            "Prefer the checked-in helper surface in `trellis.models.callable_bond_tree`: `price_callable_bond_tree(market_state, spec, model=\"hull_white\"|\"bdt\")` or the lower-level `build_callable_bond_lattice(...)` + `price_callable_bond_on_lattice(...)`.",
            "Read the short-rate seed from `market_state.discount.zero_rate(...)`; do not call `market_state.zero_rate(...)` or invent a curve accessor on MarketState itself.",
            "Read the short-rate volatility proxy from `market_state.vol_surface.black_vol(...)` and keep that access visible in `evaluate()`; do not replace it with a hard-coded fallback or hide it behind a stale helper.",
            "A good shape is `black_vol = float(market_state.vol_surface.black_vol(max(T / 2.0, 1e-6), max(r0, 1e-6)))`, followed by `sigma_hw = black_vol * max(abs(r0), 1e-6)`.",
            "Do not drop `market_state.vol_surface` access just because the route later converts Black vol into a Hull-White or BDT tree volatility.",
            "For BDT or explicit Hull-White comparison routes, prefer `price_callable_bond_tree(..., model=\"bdt\"|\"hull_white\")`. If you must build the tree directly, import `build_generic_lattice` from `trellis.models.trees.lattice` and `MODEL_REGISTRY` from `trellis.models.trees.models`, then choose `MODEL_REGISTRY[\"bdt\"]` or `MODEL_REGISTRY[\"hull_white\"]` explicitly.",
            "If you use the plain Hull-White helper route, call `build_rate_lattice(r0, sigma_hw, mean_reversion, T, n_steps, discount_curve=market_state.discount)` with positional `r0, sigma_hw, mean_reversion, T, n_steps`.",
            "Treat `trellis.models.trees.control` as the lattice-timeline facade: it exports `build_payment_timeline(...)`, `build_exercise_timeline_from_dates(...)`, `lattice_steps_from_timeline(...)`, and `lattice_step_from_time(...)`.",
            "Build payment and exercise timelines with that control facade; map the multi-date exercise timeline with `lattice_steps_from_timeline(..., dt=lattice.dt, n_steps=lattice.n_steps)` and map individual coupon/maturity times with `lattice_step_from_time(..., dt=lattice.dt, n_steps=lattice.n_steps, allow_terminal_step=True)`.",
            "Resolve callable exercise with `exercise_policy = resolve_lattice_exercise_policy(\"issuer_call\", exercise_steps=...)`.",
            "Call `lattice_backward_induction(lattice, terminal_payoff, exercise_value=..., cashflow_at_node=..., exercise_policy=...)`; do not fall back to legacy names like `terminal_value=` or `exercise_value_fn=`.",
            "Make `terminal_payoff(step, node, lattice)` return principal plus the final coupon. Do not key maturity cashflows off the node index or collapse terminal value to a coupon-only lookup.",
            "Coupon cashflows are step-based, not node-based. Build a `coupon_by_step` map from the payment timeline and use `lattice_step_from_time(...)` for any single coupon or maturity event you later index directly.",
            "Pass `exercise_policy=exercise_policy` into `lattice_backward_induction(...)` instead of open-coding holder-style exercise semantics.",
            "Keep the callable route thin: explicit coupon cashflows, explicit exercise value, explicit vol-surface access, checked-in lattice control, and no bespoke exercise convention logic.",
        )
    if (
        instrument == "puttable_bond"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Puttable bond rate-tree routes should stay on the checked-in embedded-bond helper surface whenever possible.",
            "Prefer `from trellis.models.callable_bond_tree import price_callable_bond_tree` and delegate to that helper from the adapter. The helper accepts puttable specs with `put_dates` and `put_price`.",
            "If you need the lower-level lattice path, resolve holder exercise with `exercise_policy = resolve_lattice_exercise_policy(\"holder_put\", exercise_steps=...)`.",
            "Do not pass `exercise_fn=` into `resolve_lattice_exercise_policy(...)`. That function only accepts `exercise_style` and `exercise_steps`; the holder-max objective is already built into the `holder_put` policy.",
            "Do not pass a second bespoke holder exercise function into `lattice_backward_induction(...)` when you already pass `exercise_policy=exercise_policy`.",
            "Treat `trellis.models.trees.control` as the lattice-timeline facade: build the exercise schedule from explicit `put_dates`, map it with `lattice_steps_from_timeline(...)`, and keep the policy object explicit.",
            "Keep `market_state.vol_surface.black_vol(...)` and `market_state.discount.zero_rate(...)` visible in `evaluate()` before delegating so the route keeps its calibration inputs explicit.",
            "Puttable bonds are holder-maximizing Bermudan exercise problems. Do not drift to issuer-minimizing semantics or `exercise_fn=min` anywhere in the route.",
        )
    if (
        instrument == "bermudan_swaption"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Bermudan swaption rate-tree routes should stay on the checked-in helper surface.",
            "Prefer `from trellis.models.bermudan_swaption_tree import price_bermudan_swaption_tree` and delegate to that helper from the adapter.",
            "Keep `market_state.vol_surface.black_vol(...)` visible in `evaluate()` before delegating so the route makes its calibration input explicit.",
            "Read the short-rate seed from `market_state.discount.zero_rate(...)`; do not invent `market_state.zero_rate(...)` or hide the discount curve behind a stale wrapper.",
            "Treat `trellis.models.trees.control` as the lattice-timeline facade: use `build_exercise_timeline_from_dates(...)`, `lattice_steps_from_timeline(...)`, and `resolve_lattice_exercise_policy(\"bermudan\", exercise_steps=...)`.",
            "Do not reuse `price_callable_bond_tree(...)` for Bermudan swaptions. Callable bonds and swaptions are separate helper routes with different exercise payoffs.",
            "If you must build the tree directly, use `build_generic_lattice(MODEL_REGISTRY[\"hull_white\"|\"bdt\"], r0=..., sigma=..., a=..., T=..., n_steps=..., discount_curve=market_state.discount)` and keep the swaption payoff as a Bermudan option on node-wise swap values.",
            "Keep the route thin: helper-backed lattice builder, explicit exercise schedule, explicit swaption direction (`is_payer`), and no bespoke analytical fallback inside the rate-tree build.",
        )
    if (
        instrument == "bermudan_swaption"
        and pricing_method == "analytical"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Bermudan swaption analytical comparators should stay on the checked-in lower-bound helper surface.",
            "Prefer `from trellis.models.rate_style_swaption import price_bermudan_swaption_black76_lower_bound` and delegate to that helper from the adapter.",
            "Interpret `black76_european_lower_bound` as the European swaption exercisable only on the final Bermudan date.",
            "Do not sum one European Black76 price per exercise date, and do not rebuild forward-swap-rate or annuity loops inline when the checked-in helper already owns the route.",
            "Keep the adapter minimal: validate `market_state.discount` and `market_state.vol_surface`, then call the helper and return `float(...)`.",
        )
    if (
        instrument == "swaption"
        and pricing_method == "monte_carlo"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "European rate-style swaption Monte Carlo routes should stay on the checked family helper instead of assembling the event-aware problem inline.",
            "Prefer `from trellis.models.rate_style_swaption import price_swaption_monte_carlo` and delegate to that helper from the adapter.",
            "Keep the route thin: validate discount/vol access, preserve the contract conventions on `self._spec`, then call the helper and return `float(...)`.",
            "do not hardcode `sigma = 0.01` and do not synthesize a GBM equity path. The helper resolves the Hull-White process from `market_state` on the bounded calibration/model path.",
            "If you truly need the lower-level runtime for debugging, the authoritative pieces remain `resolve_hull_white_monte_carlo_process_inputs(...)`, `build_discounted_swap_pv_payload(...)`, `build_short_rate_discount_reducer(...)`, and `price_event_aware_monte_carlo(...)` in `trellis.models.monte_carlo.event_aware`.",
        )
    if (
        instrument == "swaption"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "European rate-style swaption tree routes should stay on the checked-in helper-backed surface.",
            "Prefer `from trellis.models.rate_style_swaption_tree import price_swaption_tree` and delegate to that helper from the adapter.",
            "This helper-backed tree route is for the single-exercise European comparison surface where `swap_start == expiry_date`.",
            "Do not rebuild exercise-step selection, swap rollback, or lattice payoff glue inline when the checked-in helper already owns the route.",
            "Keep cap/floor-style period loops separate. This route is for a single-exercise European swaption comparison target, not for caplet or floorlet strips.",
        )
    if (
        instrument == "swaption"
        and pricing_method == "analytical"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "European rate-style swaption analytical routes should stay on the checked helper-backed Black76 surface.",
            "Prefer `from trellis.models.rate_style_swaption import price_swaption_black76` and delegate to that helper from the adapter.",
            "When the request supplies explicit Hull-White comparison parameters, pass `mean_reversion=` and `sigma=` into `price_swaption_black76(...)` so the analytical lane uses a Hull-White-implied Black vol instead of drifting to an unrelated market vol surface.",
            "Do not rebuild annuity, forward-swap-rate, expiry year-fraction, payment-count loops, or swaption-vol normalization inline when the checked helper already owns that binding.",
            "Keep cap/floor-style period loops separate. This helper-backed shortcut is for single-exercise European swaptions, not for caplet or floorlet strips.",
        )
    if (
        instrument == "zcb_option"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Zero-coupon bond option tree routes should stay on the checked-in helper surface.",
            "Prefer `from trellis.models.zcb_option_tree import price_zcb_option_tree` and delegate to that helper from the adapter.",
            "Map `implementation_target=ho_lee_tree` to `model=\"ho_lee\"` and `implementation_target=hull_white_tree` to `model=\"hull_white\"`.",
            "If you must build the tree directly, use `build_generic_lattice(MODEL_REGISTRY[...], r0=..., sigma=..., a=..., T=..., n_steps=..., discount_curve=market_state.discount)`.",
            "Do not call `build_rate_lattice(...)` with invented keyword forms such as `market_state=...`, `maturity=...`, or `steps=...`.",
            "The tree horizon must run to `spec.bond_maturity_date`, not merely to option expiry, because the route needs the bond price at expiry first.",
        )
    if (
        instrument == "zcb_option"
        and pricing_method == "analytical"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Jamshidian analytical routes should use the checked-in helper instead of rebuilding forward-bond Black76 inline.",
            "Prefer `from trellis.models.zcb_option import price_zcb_option_jamshidian` and delegate to that helper from the adapter.",
            "If you need the resolved-input lane under that wrapper, use `resolve_zcb_option_hw_inputs(...)` and pass `resolved.jamshidian` into `zcb_option_hw_raw(...)` from `trellis.models.analytical.jamshidian`.",
            "Treat `ResolvedJamshidianInputs` as the traced closed-form contract after date and strike normalization.",
            "Normalize strike quotes to unit face before the closed-form kernel: treat `63` on `100` face as `0.63`.",
            "Use `spec.expiry_date` and `spec.bond_maturity_date`, and validate that the bond maturity is strictly after expiry.",
        )
    if (
        instrument == "european_option"
        and pricing_method == "pde_solver"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Vanilla European PDE routes should stay on the checked-in helper surface whenever the contract is plain call/put Black-Scholes.",
            "Prefer `from trellis.models.equity_option_pde import price_vanilla_equity_option_pde` and delegate to that helper from the adapter.",
            "Map `implementation_target=theta_0.5` to `theta=0.5` and `implementation_target=theta_1.0` to `theta=1.0`.",
            "Do not rebuild terminal intrinsic branches, boundary callables, scalar interpolation, or discount-to-rate glue inline when the checked-in helper already matches the route.",
            "Use the lower-level `Grid`, `BlackScholesOperator`, and `theta_method_1d` primitives only if the payoff or boundary contract genuinely differs from plain vanilla European call/put.",
        )
    if (
        instrument == "european_option"
        and pricing_method == "analytical"
        and comparison_target == "black_scholes"
        and stage in {"code_generation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "This retry is only for the plain Black-Scholes / Black76 comparator lane, not a general analytical decomposition route.",
            "Keep the module minimal: compute `T`, `df`, `sigma`, and `forward = spec.spot / max(df, 1e-12)`, then call `black76_call` or `black76_put`.",
            "Do not import or call `terminal_vanilla_from_basis`, `black76_asset_or_nothing_*`, or `black76_cash_or_nothing_*` for this comparator.",
            "Return a full Python module with the spec class and payoff class. Do not emit only an `evaluate()` fragment, markdown bullets, or an indented class body snippet.",
        )
    return ()


def _instrument_disambiguation_lines(
    request: KnowledgeRetrievalRequest,
) -> tuple[str, ...]:
    """Return route-family disambiguation notes for ambiguous instrument families."""
    instrument = (request.instrument_type or "").strip()
    if not instrument and request.product_ir is not None:
        instrument = str(getattr(request.product_ir, "instrument", "") or "").strip()

    if instrument in {"credit_default_swap", "cds"}:
        return (
            "Treat this request as a single-name CDS / credit_default_swap contract.",
            "Do not reinterpret CDS here as nth_to_default, basket CDS, first-to-default, or any multi-name credit product.",
            "Do not import copula or Gaussian-copula machinery unless the request explicitly says nth-to-default, first-to-default, basket CDS, or multiple reference names.",
        )
    if instrument == "nth_to_default":
        return (
            "Treat this request as nth_to_default / multi-name credit, not a single-name CDS.",
            "Copula and default-correlation primitives are valid here; single-name CDS shortcuts are not enough.",
        )
    return ()


def _retrieve_knowledge(
    pricing_plan,
    instrument_type: str | None,
    *,
    product_ir=None,
    compact: bool = True,
) -> str:
    """Retrieve all relevant knowledge for a task as formatted prompt text."""
    try:
        from trellis.agent.knowledge import (
            build_shared_knowledge_payload,
            retrieve_for_product_ir,
            retrieve_for_task,
        )
        from trellis.agent.knowledge.decompose import decompose

        if product_ir is not None:
            knowledge = retrieve_for_product_ir(
                product_ir,
                preferred_method=pricing_plan.method,
            )
            payload = build_shared_knowledge_payload(knowledge)
            return payload["builder_text_distilled" if compact else "builder_text_expanded"]

        features: list[str] = []
        try:
            decomp = decompose(
                instrument_type or "",
                instrument_type=instrument_type,
            )
            features = list(decomp.features)
        except Exception:
            pass

        knowledge = retrieve_for_task(
            method=pricing_plan.method,
            features=features,
            instrument=instrument_type,
        )
        payload = build_shared_knowledge_payload(knowledge)
        return payload["builder_text_distilled" if compact else "builder_text_expanded"]
    except Exception:
        return ""


def _retrieve_review_knowledge(
    pricing_plan,
    instrument_type: str | None,
    *,
    product_ir=None,
    compact: bool = True,
) -> str:
    """Retrieve shared knowledge for reviewer-style agent prompts."""
    try:
        from trellis.agent.knowledge import (
            build_shared_knowledge_payload,
            retrieve_for_product_ir,
            retrieve_for_task,
        )
        from trellis.agent.knowledge.decompose import decompose

        if product_ir is not None:
            knowledge = retrieve_for_product_ir(
                product_ir,
                preferred_method=pricing_plan.method if pricing_plan is not None else None,
            )
            payload = build_shared_knowledge_payload(knowledge)
            return payload["review_text_distilled" if compact else "review_text_expanded"]

        features: list[str] = []
        try:
            decomp = decompose(
                instrument_type or "",
                instrument_type=instrument_type,
            )
            features = list(decomp.features)
        except Exception:
            pass

        knowledge = retrieve_for_task(
            method=pricing_plan.method if pricing_plan is not None else None,
            features=features,
            instrument=instrument_type,
        )
        payload = build_shared_knowledge_payload(knowledge)
        return payload["review_text_distilled" if compact else "review_text_expanded"]
    except Exception:
        return ""


def _record_trace(
    instrument_type: str | None,
    pricing_plan,
    description: str,
    attempt: int,
    code: str,
    failures: list[str],
) -> None:
    """Record a run trace to cold storage (best-effort)."""
    try:
        from trellis.agent.knowledge.promotion import record_trace
        record_trace(
            instrument=instrument_type or "unknown",
            method=pricing_plan.method if pricing_plan else "unknown",
            description=description,
            pricing_plan=(
                {
                    "method": pricing_plan.method,
                    "selection_reason": pricing_plan.selection_reason,
                    "assumption_summary": list(pricing_plan.assumption_summary),
                }
                if pricing_plan
                else {}
            ),
            attempt=attempt,
            code=code,
            validation_failures=failures,
            resolved=False,
        )
    except Exception:
        pass
