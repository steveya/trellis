"""Top-level agent executor: plan → build → fetch → price → return."""

from __future__ import annotations

import ast
import inspect
import json
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass, replace as replace_dataclass
from datetime import date, datetime
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, get_args, get_origin
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
from trellis.agent.generation_policy import (
    ArtifactOrigin,
    GenerationPolicy,
    GenerationPolicyError,
    normalize_generation_policy,
    record_generation_evidence,
    validate_builder_synthesis_context,
)
from trellis.agent.semantic_validation import validate_semantics
from trellis.agent.builder import write_module, run_tests
from trellis.agent.knowledge.import_registry import resolve_import_candidates
from trellis.agent.knowledge.api_map import ApiMapQuery, format_api_map_for_prompt
from trellis.agent.role_orientation import role_orientation_summary


# Sentinel in the skeleton that gets replaced by the LLM-generated body.
EVALUATE_SENTINEL = '        raise NotImplementedError("evaluate not yet implemented")'
_ROUTE_GUESSING_BLOCKER_REASON = "route guessing"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRELLIS_PACKAGE_ROOT = REPO_ROOT / "trellis"

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
            cwd=REPO_ROOT,
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
    target_contract = request_metadata.get("comparison_target_contract") or {}
    metadata = {
        "model": model,
        "attempt": attempt,
        "instrument_type": instrument_type,
        "request_id": getattr(request, "request_id", None),
        "task_id": request_metadata.get("task_id"),
        "comparison_target": request_metadata.get("comparison_target"),
        "preferred_method": request_metadata.get("preferred_method"),
        "comparison_target_contract_id": (
            target_contract.get("contract_id")
            if isinstance(target_contract, Mapping)
            else None
        ),
    }
    return {
        key: value
        for key, value in metadata.items()
        if value not in {None, ""}
    }


def _comparison_target_from_build_metadata(
    compiled_request,
    request_metadata: Mapping[str, object] | None,
) -> object | None:
    """Return the active comparison target for one task-build attempt.

    Task runtime passes comparison target metadata alongside the compiled
    request. Some compiled requests are reused across targets, so the explicit
    build metadata must win over the request's original metadata.
    """
    metadata: dict[str, object] = {}
    request = getattr(compiled_request, "request", None)
    metadata.update(dict(getattr(request, "metadata", None) or {}))
    metadata.update(dict(request_metadata or {}))
    return metadata.get("comparison_target")


def _comparison_execution_binding_metadata(
    *,
    pricing_plan,
    generation_plan,
    product_ir,
    request_metadata: Mapping[str, object] | None,
    validation_contract=None,
) -> dict[str, object]:
    """Project actual planner/compiler choices for comparison coherence checks."""
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    route_id = str(
        getattr(generation_plan, "lowering_route_id", "")
        or getattr(primitive_plan, "route", "")
        or getattr(validation_contract, "route_id", "")
        or ""
    ).strip()
    route_family = str(
        getattr(generation_plan, "backend_route_family", "")
        or getattr(primitive_plan, "route_family", "")
        or getattr(validation_contract, "route_family", "")
        or ""
    ).strip()
    backend_binding_id = str(
        getattr(generation_plan, "backend_binding_id", "")
        or getattr(primitive_plan, "backend_binding_id", "")
        or getattr(validation_contract, "backend_binding_id", "")
        or ""
    ).strip()

    exercise_style = str(getattr(product_ir, "exercise_style", "") or "").strip()
    schedule_dependence = bool(
        getattr(product_ir, "schedule_dependence", False)
    )
    state_dependence = str(
        getattr(product_ir, "state_dependence", "") or ""
    ).strip()
    if exercise_style in {"american", "bermudan", "issuer_call", "holder_put"}:
        observation_style = "exercise_schedule"
    elif schedule_dependence:
        observation_style = "fixed_schedule"
    elif state_dependence and state_dependence != "terminal_markov":
        observation_style = "path_dependent"
    else:
        observation_style = "terminal"

    semantic_axes = {
        key: value
        for key, value in {
            "derivative_family": getattr(product_ir, "derivative_family", ""),
            "payoff_family": getattr(product_ir, "payoff_family", ""),
            "exercise_style": exercise_style,
            "model_family": getattr(product_ir, "model_family", ""),
            "observation_style": observation_style,
            "underlying_asset_class": getattr(
                product_ir, "underlying_asset_class", ""
            ),
            "option_type": getattr(product_ir, "option_type", ""),
        }.items()
        if value not in {None, ""}
    }
    raw_target_contract = (request_metadata or {}).get(
        "comparison_target_contract"
    )
    target_contract = (
        dict(raw_target_contract)
        if isinstance(raw_target_contract, Mapping)
        else {}
    )
    return {
        "comparison_target_contract": {},
        "requested_comparison_target_contract": target_contract,
        "selected_method": str(getattr(pricing_plan, "method", "") or "").strip(),
        "selected_route_id": route_id,
        "selected_route_family": route_family,
        "selected_backend_binding_id": backend_binding_id,
        "selected_validation_bundle_id": str(
            getattr(generation_plan, "validation_bundle_id", "")
            or getattr(validation_contract, "bundle_id", "")
            or ""
        ).strip(),
        "selected_semantic_axes": semantic_axes,
    }


def _record_comparison_execution_binding(
    build_meta: dict | None,
    *,
    compiled_request=None,
    pricing_plan,
    generation_plan,
    product_ir,
    request_metadata: Mapping[str, object] | None,
    emit_event: bool = True,
) -> None:
    """Persist comparison binding evidence into the build result metadata."""
    binding = _comparison_execution_binding_metadata(
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        product_ir=product_ir,
        validation_contract=getattr(
            compiled_request,
            "validation_contract",
            None,
        ),
        request_metadata=request_metadata,
    )
    if binding.get("requested_comparison_target_contract"):
        binding["comparison_request_source"] = (
            "compiled_generation_binding"
        )
    if build_meta is not None:
        build_meta.update(binding)
        build_meta["execution_binding"] = dict(binding)
    if emit_event and binding.get("requested_comparison_target_contract"):
        _record_platform_event(
            compiled_request,
            "comparison_target_planned",
            status="ok",
            details=binding,
        )


def _record_artifact_comparison_binding(
    build_meta: dict | None,
    payoff_cls: type,
    *,
    requested_contract: Mapping[str, object] | None,
    artifact_kind: str,
    compiled_request=None,
) -> None:
    """Use only comparison-contract evidence declared by an executable artifact."""
    if build_meta is None or not isinstance(requested_contract, Mapping):
        return
    from trellis.agent.comparison_target_contracts import (
        declared_comparison_target_contract,
    )

    requested_payload = dict(requested_contract)
    target_id = str(requested_payload.get("target_id") or "").strip()
    declarations = getattr(payoff_cls, "__trellis_comparison_bindings__", None)
    declared_contract = declared_comparison_target_contract(declarations, target_id)
    artifact_contract: dict[str, object] = (
        declared_contract.to_payload() if declared_contract is not None else {}
    )

    source = f"{artifact_kind}_artifact_declaration"
    if not artifact_contract:
        source += "_missing"
    build_meta["comparison_target_contract"] = artifact_contract
    build_meta["comparison_binding_evidence_source"] = source
    execution_binding = dict(build_meta.get("execution_binding") or {})
    execution_binding["comparison_target_contract"] = artifact_contract
    execution_binding["comparison_binding_evidence_source"] = source
    build_meta["execution_binding"] = execution_binding

    details = {
        **execution_binding,
        "requested_comparison_target_contract": requested_payload,
    }
    _record_platform_event(
        compiled_request,
        (
            "comparison_target_bound"
            if artifact_contract
            else "comparison_target_binding_unproven"
        ),
        status="ok" if artifact_contract else "error",
        details=details,
    )


def _record_reused_comparison_binding(
    build_meta: dict | None,
    payoff_cls: type,
    *,
    requested_contract: Mapping[str, object] | None,
    compiled_request=None,
) -> None:
    """Record the target contract declared by a cached artifact."""
    _record_artifact_comparison_binding(
        build_meta,
        payoff_cls,
        requested_contract=requested_contract,
        artifact_kind="cached",
        compiled_request=compiled_request,
    )


def _cached_artifact_declares_requested_comparison_target(
    payoff_cls: type,
    request_metadata: Mapping[str, object] | None,
) -> bool:
    """Return whether cached source proves the requested target identity."""
    from trellis.agent.comparison_target_contracts import (
        ComparisonTargetContract,
        comparison_target_contracts_compatible,
        declared_comparison_target_contract,
    )

    raw_contract = (request_metadata or {}).get("comparison_target_contract")
    if not isinstance(raw_contract, Mapping):
        return True
    try:
        expected = ComparisonTargetContract.from_payload(raw_contract)
    except (TypeError, ValueError):
        return False
    declared = declared_comparison_target_contract(
        getattr(payoff_cls, "__trellis_comparison_bindings__", None),
        expected.target_id,
    )
    return declared is not None and comparison_target_contracts_compatible(
        expected,
        declared,
    )


def _record_fresh_comparison_binding(
    build_meta: dict | None,
    payoff_cls: type,
    *,
    requested_contract: Mapping[str, object] | None,
    compiled_request=None,
) -> None:
    """Record the target contract declared by a newly generated artifact."""
    _record_artifact_comparison_binding(
        build_meta,
        payoff_cls,
        requested_contract=requested_contract,
        artifact_kind="fresh",
        compiled_request=compiled_request,
    )


def _artifact_source_text(payoff_cls: type) -> str:
    """Read the source module backing an imported payoff class."""
    module_name = str(getattr(payoff_cls, "__module__", "") or "").strip()
    module = sys.modules.get(module_name) if module_name else None
    module_file = getattr(module, "__file__", None) if module is not None else None
    if module_file:
        try:
            return Path(module_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass
    try:
        return inspect.getsource(payoff_cls)
    except (OSError, TypeError):
        return ""


def _validate_reused_comparison_artifact(
    payoff_cls: type,
    *,
    request_metadata: Mapping[str, object] | None,
    validation: str,
    description: str,
    spec_schema,
    model: str | None,
    compiled_request,
    pricing_plan,
    product_ir,
    build_meta: dict | None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None,
) -> list[str]:
    """Run required deterministic validation before returning a cached target."""
    raw_contract = (request_metadata or {}).get("comparison_target_contract")
    if not isinstance(raw_contract, Mapping):
        return []
    required_bundle = str(
        raw_contract.get("validation_bundle_id") or ""
    ).strip()
    if not required_bundle or validation == "fast":
        return []

    runtime_spec_schema = _resolve_runtime_spec_schema(payoff_cls, spec_schema)
    return _validate_build(
        payoff_cls,
        _artifact_source_text(payoff_cls),
        description,
        runtime_spec_schema,
        validation=validation,
        model=model,
        compiled_request=compiled_request,
        pricing_plan=pricing_plan,
        product_ir=product_ir,
        build_meta=build_meta,
        knowledge_retriever=knowledge_retriever,
    )


def _record_reused_execution_artifact(
    build_meta: dict | None,
    payoff_cls: type,
    *,
    admission_module_path: str,
) -> None:
    """Record the class and file actually returned by cached reuse."""
    if build_meta is None:
        return
    module_name = str(getattr(payoff_cls, "__module__", "") or "").strip()
    module = sys.modules.get(module_name) if module_name else None
    module_file = getattr(module, "__file__", None) if module is not None else None
    file_path = str(Path(module_file).resolve()) if module_file else ""
    try:
        module_path = str(Path(file_path).relative_to(REPO_ROOT)).replace("\\", "/")
    except (TypeError, ValueError):
        module_path = file_path
    build_meta["execution_module_name"] = module_name
    build_meta["execution_module_path"] = module_path
    build_meta["execution_file_path"] = file_path
    normalized_admission_path = str(admission_module_path or "").replace("\\", "/")
    build_meta["admission_target_module_path"] = normalized_admission_path
    build_meta["admission_target_module_name"] = (
        f"trellis.{normalized_admission_path.replace('/', '.').removesuffix('.py')}"
        if normalized_admission_path
        else None
    )
    build_meta["admission_target_file_path"] = (
        str((TRELLIS_PACKAGE_ROOT / normalized_admission_path).resolve())
        if normalized_admission_path
        else None
    )


def _record_comparison_validation_binding(
    build_meta: dict | None,
    *,
    compiled_request=None,
    bundle_id: str,
) -> None:
    """Attach the validation bundle that actually executed for one target."""
    normalized_bundle_id = str(bundle_id or "").strip()
    execution_binding: dict[str, object] = {}
    target_contract: dict[str, object] = {}
    if build_meta is not None:
        raw_binding = build_meta.get("execution_binding")
        if isinstance(raw_binding, Mapping):
            execution_binding = dict(raw_binding)
        raw_target_contract = (
            execution_binding.get("comparison_target_contract")
            or build_meta.get("comparison_target_contract")
        )
        if isinstance(raw_target_contract, Mapping):
            target_contract = dict(raw_target_contract)
        build_meta["selected_validation_bundle_id"] = normalized_bundle_id
        validation_source = (
            "executed_validation_bundle" if normalized_bundle_id else ""
        )
        build_meta["validation_binding_evidence_source"] = validation_source
        execution_binding["selected_validation_bundle_id"] = normalized_bundle_id
        execution_binding["validation_binding_evidence_source"] = validation_source
        build_meta["execution_binding"] = execution_binding

    request = getattr(compiled_request, "request", None)
    request_metadata = getattr(request, "metadata", None) or {}
    raw_requested_contract = request_metadata.get("comparison_target_contract")
    requested_contract = (
        dict(raw_requested_contract)
        if isinstance(raw_requested_contract, Mapping)
        else {}
    )

    if target_contract or requested_contract:
        _record_platform_event(
            compiled_request,
            "comparison_target_validation_bound",
            status="ok",
            details={
                "comparison_target_contract": target_contract,
                "requested_comparison_target_contract": requested_contract,
                "selected_validation_bundle_id": normalized_bundle_id,
                "validation_binding_evidence_source": (
                    "executed_validation_bundle" if normalized_bundle_id else ""
                ),
            },
        )


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
        try:
            query_keys = (
                "instrument_type",
                "payoff_family",
                "method",
                "model_family",
                "features",
                "route_ids",
                "route_families",
                "description",
                "families",
            )
            query = None
            if any(input_data.get(key) for key in query_keys):
                query = ApiMapQuery(
                    instrument_type=str(input_data.get("instrument_type") or ""),
                    payoff_family=str(input_data.get("payoff_family") or ""),
                    method=str(input_data.get("method") or ""),
                    model_family=str(input_data.get("model_family") or ""),
                    features=tuple(input_data.get("features") or ()),
                    route_ids=tuple(input_data.get("route_ids") or ()),
                    route_families=tuple(input_data.get("route_families") or ()),
                    description=str(input_data.get("description") or ""),
                    requested_families=tuple(input_data.get("families") or ()),
                )
            return format_api_map_for_prompt(
                compact=True,
                query=query,
                max_chars=4000,
            )
        except Exception as e:
            return f"Error inspecting API map: {e}"

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
            return json.dumps(payload.to_payload(), indent=2)
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
            "comparison_target_contract": dict(
                (build_meta or {}).get("comparison_target_contract") or {}
            ),
            "execution_binding": dict(
                (build_meta or {}).get("execution_binding") or {}
            ),
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
    semantic_contract=None,
    compiled_request=None,
    build_meta: dict | None = None,
    gap_report=None,
    knowledge_retriever: Callable[[KnowledgeRetrievalRequest], str | None] | None = None,
    knowledge_overlays: Sequence[Mapping[str, object]] | None = None,
    generation_policy: str | GenerationPolicy = GenerationPolicy.DETERMINISTIC_ALLOWED,
) -> type:
    """Build a Payoff class via the multi-agent pipeline.

    Pipeline:
    1. **Quant agent** selects the pricing method and data requirements
    2. **Data check** verifies required market data is available
    3. **Planner** determines spec schema and module path
    4. **Builder agent** generates the code using the prescribed method
    5. **Critic agent** reviews the code
    6. **Arbiter** validates with invariants

    ``fresh_build`` controls path isolation; ``generation_policy`` separately
    controls whether deterministic source construction may satisfy the build.
    """
    from trellis.agent.planner import plan_build
    from trellis.agent.quant import (
        check_data_availability,
        quant_challenger_packet_summary,
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
    from trellis.agent.semantic_contracts import UnsupportedSemanticMethodError
    from trellis.core.market_state import MissingCapabilityError

    model = model or get_default_model()
    generation_policy_value = normalize_generation_policy(generation_policy)
    record_generation_evidence(
        build_meta,
        policy=generation_policy_value,
    )
    if (
        generation_policy_value is GenerationPolicy.BUILDER_SYNTHESIS_REQUIRED
        and not fresh_build
    ):
        raise GenerationPolicyError(
            "Builder synthesis requires fresh_build=True so model source is isolated from admitted adapters.",
            reason="fresh_build_required",
        )
    validate_builder_synthesis_context(
        policy=generation_policy_value,
        request_metadata=request_metadata,
    )
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
                semantic_contract=semantic_contract,
                knowledge_overlays=knowledge_overlays,
            )
            product_ir = compiled_request.product_ir
        except UnsupportedSemanticMethodError as exc:
            blocker_details = exc.to_blocker_details()
            if build_meta is not None:
                build_meta["blocker_details"] = blocker_details
            raise
        except Exception:
            if semantic_contract is not None:
                raise
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
        compiled_metadata = dict(getattr(getattr(compiled_request, "request", None), "metadata", {}) or {})
        overlay_consumption = compiled_metadata.get(
            "intra_run_learning_overlay_consumption"
        )
        if isinstance(overlay_consumption, Mapping):
            build_meta.setdefault("intra_run_learning", {})[
                "deterministic_overlay_consumption"
            ] = dict(overlay_consumption)

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
    quant_challenger_packet = quant_challenger_packet_summary(pricing_plan)
    # Keep identity queryable without decoding the nested challenger packet.
    quant_orientation = role_orientation_summary("quant")
    quant_orientation_resolution = dict(
        getattr(pricing_plan, "orientation_resolution", {}) or {}
    ) or {
        "role": "quant",
        "orientation_identity": (
            f"{quant_orientation['contract_id']}@{quant_orientation['version']}"
        ),
        "prompt_injected": False,
        "reason": "deterministic_method_selection",
    }
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
            "orientation_contract": quant_orientation,
            "orientation_resolution": quant_orientation_resolution,
            "challenger_packet": quant_challenger_packet,
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
            "orientation_contract": quant_orientation,
            "orientation_resolution": quant_orientation_resolution,
            "method_modules": list(pricing_plan.method_modules),
            "reasoning": pricing_plan.reasoning,
            "selection_reason": pricing_plan.selection_reason,
            "assumption_summary": list(pricing_plan.assumption_summary),
            "challenger_packet": quant_challenger_packet,
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
    plan = _sanitize_build_plan_module_paths(plan, request_metadata=request_metadata)
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
    cached_target_binding_valid = (
        existing is None
        or _cached_artifact_declares_requested_comparison_target(
            existing,
            request_metadata,
        )
    )
    target_bound_rematerialization = bool(
        existing is not None
        and not fresh_build
        and not cached_target_binding_valid
        and isinstance(
            (request_metadata or {}).get("comparison_target_contract"),
            Mapping,
        )
    )
    reuse_reason = None
    if existing is not None and not fresh_build and cached_target_binding_valid:
        if _is_deterministic_supported_route(plan):
            reuse_reason = "deterministic_supported_route"
        elif not force_rebuild:
            reuse_reason = "cached_generated_module"
    if existing is not None and reuse_reason is not None:
        record_generation_evidence(
            build_meta,
            policy=generation_policy_value,
            artifact_origin=ArtifactOrigin.REUSED_ADAPTER,
        )
        _record_platform_event(
            compiled_request,
            "existing_generated_module_reused",
            status="ok",
            details={
                "module_path": plan.steps[0].module_path if plan.steps else "",
                "class_name": getattr(existing, "__name__", type(existing).__name__),
                "reason": reuse_reason,
                "generation_evidence": dict(
                    (build_meta or {}).get("generation_evidence") or {}
                ),
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
        _record_comparison_execution_binding(
            build_meta,
            compiled_request=compiled_request,
            pricing_plan=pricing_plan,
            generation_plan=generation_plan,
            product_ir=product_ir,
            request_metadata=request_metadata,
            emit_event=False,
        )
        _record_reused_execution_artifact(
            build_meta,
            existing,
            admission_module_path=(
                plan.steps[0].module_path if plan.steps else ""
            ),
        )
        _record_reused_comparison_binding(
            build_meta,
            existing,
            requested_contract=(request_metadata or {}).get(
                "comparison_target_contract"
            ),
            compiled_request=compiled_request,
        )
        reuse_validation_failures = _validate_reused_comparison_artifact(
            existing,
            request_metadata=request_metadata,
            validation=validation,
            description=payoff_description,
            spec_schema=getattr(plan, "spec_schema", None),
            model=model,
            compiled_request=compiled_request,
            pricing_plan=pricing_plan,
            product_ir=product_ir,
            build_meta=build_meta,
            knowledge_retriever=knowledge_retriever,
        )
        if reuse_validation_failures:
            _record_platform_event(
                compiled_request,
                "existing_generated_module_validation_failed",
                status="error",
                details={
                    "failure_count": len(reuse_validation_failures),
                    "failures": reuse_validation_failures[:10],
                },
            )
            raise RuntimeError(
                "Cached comparison target validation failed: "
                + "; ".join(reuse_validation_failures[:5])
            )
        _emit_analytical_trace_metadata(
            build_meta=build_meta,
            generation_plan=generation_plan,
            compiled_request=compiled_request,
            spec_schema=getattr(plan, "spec_schema", None),
            market_state=market_state,
        )
        return existing
    if existing is not None and (fresh_build or target_bound_rematerialization):
        _record_platform_event(
            compiled_request,
            "existing_generated_module_bypassed",
            status="info",
            details={
                "module_path": plan.steps[0].module_path if plan.steps else "",
                "class_name": getattr(existing, "__name__", type(existing).__name__),
                "reason": (
                    "fresh_build"
                    if fresh_build
                    else "cached_comparison_target_binding_missing_or_incompatible"
                ),
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
    _record_comparison_execution_binding(
        build_meta,
        compiled_request=compiled_request,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        product_ir=product_ir,
        request_metadata=request_metadata,
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
    output_file_path, output_module_path, module_name = _resolve_output_target(
        step.module_path,
        fresh_build=fresh_build,
        isolate_comparison_target=target_bound_rematerialization,
        request_metadata=request_metadata,
    )
    if build_meta is not None:
        admission_target_module_path = str(step.module_path or "").replace("\\", "/").strip()
        build_meta["execution_module_name"] = module_name
        build_meta["execution_module_path"] = output_module_path
        build_meta["execution_file_path"] = str(output_file_path.resolve())
        build_meta["admission_target_module_path"] = admission_target_module_path
        build_meta["admission_target_module_name"] = (
            f"trellis.{admission_target_module_path.replace('/', '.').replace('.py', '')}"
            if admission_target_module_path
            else None
        )
        build_meta["admission_target_file_path"] = (
            str((TRELLIS_PACKAGE_ROOT / admission_target_module_path).resolve())
            if admission_target_module_path
            else None
        )
    _record_platform_event(
        compiled_request,
        "build_started",
        status="info",
        details={
            "module_name": module_name,
            "validation": validation,
            "max_retries": max_retries,
            "generation_policy": generation_policy_value.value,
        },
    )
    if validation != "thorough":
        _record_platform_event(
            compiled_request,
            "model_validator_skipped",
            status="info",
            details={
                "validation": validation,
                "orientation_contract": role_orientation_summary("model_validator"),
            },
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
                "generation_evidence": dict(
                    (build_meta or {}).get("generation_evidence") or {}
                ),
            },
        )
        stage_model = get_model_for_stage("code_generation", model)
        generated_module: GeneratedModuleResult | None = None
        generation_token_usage: dict[str, object] = {}
        comparison_target = _comparison_target_from_build_metadata(
            compiled_request,
            request_metadata,
        )
        try:
            if generation_policy_value is GenerationPolicy.DETERMINISTIC_ALLOWED:
                generated_module = _materialize_semantic_execution_shim_module(
                    skeleton,
                    generation_plan,
                    request_metadata=request_metadata,
                    comparison_target=comparison_target,
                )
                if generated_module is not None:
                    record_generation_evidence(
                        build_meta,
                        policy=generation_policy_value,
                        artifact_origin=ArtifactOrigin.SEMANTIC_SHIM,
                    )
                if generated_module is None:
                    generated_module = _materialize_deterministic_exact_binding_module(
                        skeleton,
                        generation_plan,
                        semantic_blueprint=(
                            getattr(compiled_request, "semantic_blueprint", None)
                            if compiled_request is not None
                            else None
                        ),
                        request_metadata=request_metadata,
                        comparison_target=comparison_target,
                    )
                    if generated_module is not None:
                        record_generation_evidence(
                            build_meta,
                            policy=generation_policy_value,
                            artifact_origin=ArtifactOrigin.DETERMINISTIC_MATERIALIZATION,
                        )
            if generated_module is None:
                record_generation_evidence(
                    build_meta,
                    policy=generation_policy_value,
                    agent_synthesis_attempted=True,
                )
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
                record_generation_evidence(
                    build_meta,
                    policy=generation_policy_value,
                    artifact_origin=ArtifactOrigin.MODEL_GENERATED_SOURCE,
                    agent_synthesis_attempted=True,
                    agent_synthesis_observed=True,
                )
                generation_token_usage = summarize_llm_usage(usage_records)
                enforce_llm_token_budget(stage="code_generation")
            code = generated_module.code
            if build_meta is not None:
                build_meta["code"] = code
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
                    "generation_evidence": dict(
                        (build_meta or {}).get("generation_evidence") or {}
                    ),
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
                "generation_evidence": dict(
                    (build_meta or {}).get("generation_evidence") or {}
                ),
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
            comparison_target_contract=(request_metadata or {}).get(
                "comparison_target_contract"
            ),
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

        file_path = _write_generated_module(output_file_path, output_module_path, code)
        mod = dynamic_import(file_path, module_name)
        payoff_cls = getattr(mod, spec_schema.class_name)
        spec_schema = _resolve_runtime_spec_schema(payoff_cls, spec_schema)
        _record_fresh_comparison_binding(
            build_meta,
            payoff_cls,
            requested_contract=(request_metadata or {}).get(
                "comparison_target_contract"
            ),
            compiled_request=compiled_request,
        )

        actual_market_failures = _smoke_test_actual_market_state(
            payoff_cls,
            spec_schema,
            market_state,
            spec_overrides=_benchmark_spec_overrides_from_compiled_request(compiled_request),
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
                details={
                    "attempts": attempt_number,
                    "generation_evidence": dict(
                        (build_meta or {}).get("generation_evidence") or {}
                    ),
                },
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
                details={
                    "attempts": attempt_number,
                    "generation_evidence": dict(
                        (build_meta or {}).get("generation_evidence") or {}
                    ),
                },
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


def _align_validation_market_payload_to_spec(
    payload: Mapping[str, object],
    spec,
    *,
    rate: float,
    vol: float,
    corr: float,
) -> dict[str, object]:
    """Project exact spec identities onto a synthetic validation market.

    Validation fixtures must exercise the same named bindings as the runtime
    contract. This function only adds synthetic entries under identifiers that
    the spec explicitly declares; it does not weaken runtime resolver fallbacks.
    """
    aligned = dict(payload)
    if spec is None:
        return aligned

    underlier_id = str(getattr(spec, "underlier_id", None) or "").strip()
    if underlier_id:
        underlier_spots = dict(aligned.get("underlier_spots") or {})
        spot = getattr(spec, "spot", None)
        if spot is None:
            spot = aligned.get("spot", 100.0)
        underlier_spots[underlier_id] = float(spot)
        aligned["underlier_spots"] = underlier_spots

    underlier_currency = str(
        getattr(spec, "underlier_currency", None) or ""
    ).strip()
    if underlier_currency and aligned.get("discount") is not None:
        from trellis.curves.yield_curve import YieldCurve

        forecast_curves = dict(aligned.get("forecast_curves") or {})
        foreign_curve = YieldCurve.flat(max(float(rate) - 0.02, 0.005))
        forecast_curves.setdefault(underlier_currency, foreign_curve)
        forecast_curves.setdefault(f"{underlier_currency}-DISC", foreign_curve)
        aligned["forecast_curves"] = forecast_curves

    underlier_vol_key = str(
        getattr(spec, "underlier_vol_surface_key", None) or ""
    ).strip()
    fx_vol_key = str(getattr(spec, "fx_vol_surface_key", None) or "").strip()
    if underlier_vol_key or fx_vol_key:
        from trellis.models.vol_surface import FlatVol

        vol_surfaces = dict(aligned.get("vol_surfaces") or {})
        default_surface = aligned.get("vol_surface") or FlatVol(float(vol))
        if underlier_vol_key:
            vol_surfaces[underlier_vol_key] = default_surface
        if fx_vol_key:
            vol_surfaces[fx_vol_key] = FlatVol(float(vol))
        aligned["vol_surfaces"] = vol_surfaces

    fx_pair = str(getattr(spec, "fx_pair", None) or "").strip()
    if fx_pair:
        from trellis.instruments.fx import FXRate

        domestic_currency = str(
            getattr(spec, "domestic_currency", None) or ""
        ).strip()
        fx_rates = dict(aligned.get("fx_rates") or {})
        if fx_pair not in fx_rates and domestic_currency and underlier_currency:
            fx_rates[fx_pair] = FXRate(
                spot=1.10,
                domestic=domestic_currency,
                foreign=underlier_currency,
            )
        aligned["fx_rates"] = fx_rates

    correlation_key = str(
        getattr(spec, "quanto_correlation_key", None) or ""
    ).strip()
    if correlation_key:
        model_parameters = dict(aligned.get("model_parameters") or {})
        model_parameters[correlation_key] = float(corr)
        aligned["model_parameters"] = model_parameters

    return aligned


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
    validation_method = (
        getattr(pricing_plan, "method", None)
        or getattr(getattr(compiled_request, "execution_plan", None), "route_method", None)
        or "unknown"
    )
    review_policy = determine_review_policy(
        validation=validation,
        method=validation_method,
        instrument_type=itype,
        product_ir=product_ir,
        validation_contract=validation_contract,
    )
    review_knowledge_text = ""
    review_prompt_surface = "none"
    benchmark_spec_overrides = _benchmark_spec_overrides_from_compiled_request(compiled_request)
    runtime_spec_schema = _resolve_runtime_spec_schema(payoff_cls, spec_schema)
    generation_plan = getattr(compiled_request, "generation_plan", None)
    exact_binding_refs = tuple(getattr(generation_plan, "lane_exact_binding_refs", ()) or ())
    requires_merton_jump_parameters = (
        str(getattr(product_ir, "model_family", "") or "").strip().lower() == "jump_diffusion"
        or any("merton_jump_diffusion_option" in str(ref) for ref in exact_binding_refs)
    )
    requires_sabr_model_parameters = (
        str(getattr(product_ir, "model_family", "") or "").strip().lower() == "sabr"
        or any("sabr_option" in str(ref) for ref in exact_binding_refs)
        or str(getattr(getattr(compiled_request, "comparison_spec", None), "target_name", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
        in {"sabr_mc", "sabr_hagan_analytical"}
    )
    requires_levy_model_parameters = (
        str(getattr(product_ir, "model_family", "") or "").strip().lower()
        in {"variance_gamma", "cgmy", "kou"}
        or any("levy_option" in str(ref) for ref in exact_binding_refs)
        or str(getattr(getattr(compiled_request, "comparison_spec", None), "target_name", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
        in {
            "vg_cos",
            "vg_mc",
            "madan_carr_chang_reference",
            "cgmy_cos",
            "cgmy_mc",
            "cgmy_reference_values",
            "kou_fft",
            "kou_mc",
            "kou_reference_values",
        }
    )
    requires_bates_model_parameters = (
        str(getattr(product_ir, "model_family", "") or "").strip().lower() == "bates"
        or any("bates_option" in str(ref) for ref in exact_binding_refs)
        or str(getattr(getattr(compiled_request, "comparison_spec", None), "target_name", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
        in {"bates_fft", "bates_mc"}
    )

    # Try to instantiate the payoff with default test parameters
    try:
        test_payoff = _make_test_payoff(
            payoff_cls,
            runtime_spec_schema,
            settle,
            spec_overrides=benchmark_spec_overrides,
        )
    except Exception as e:
        failures.append(f"Cannot instantiate payoff for validation: {e}")
        if return_failure_details:
            return failures, tuple()
        return failures

    def payoff_factory():
        """Instantiate a fresh payoff under the generated spec schema."""
        return _make_test_payoff(
            payoff_cls,
            runtime_spec_schema,
            settle,
            spec_overrides=benchmark_spec_overrides,
        )

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
        if (
            "model_parameters" in effective_requirements
            or requires_sabr_model_parameters
            or requires_levy_model_parameters
            or requires_bates_model_parameters
        ):
            product_model_family = str(getattr(product_ir, "model_family", "") or "").lower()
            if itype == "short_rate_bond":
                vasicek_payload = {
                    "family": "vasicek",
                    "r0": rate,
                    "a": 0.10,
                    "b": rate,
                    "sigma": 0.01,
                }
                cir_payload = {
                    "family": "cir",
                    "r0": rate,
                    "kappa": 0.10,
                    "theta": rate,
                    "sigma": 0.01,
                }
                payload["model_parameters"] = {
                    "vasicek": vasicek_payload,
                    "cir": cir_payload,
                }
                payload["model_parameter_sets"] = {
                    "vasicek_validation": vasicek_payload,
                    "cir_validation": cir_payload,
                }
            elif product_model_family == "bates" or requires_bates_model_parameters:
                from trellis.models.processes.heston import build_heston_parameter_payload

                bates_payload = build_heston_parameter_payload(
                    kappa=2.0,
                    theta=0.04,
                    xi=0.30,
                    rho=-0.55,
                    v0=0.04,
                    mu=rate,
                    parameter_set_name="bates_validation",
                    source_kind="validation_fixture",
                    metadata={"model_family": "bates"},
                )
                payload["model_parameters"] = bates_payload
                payload["model_parameter_sets"] = {
                    "bates_validation": bates_payload,
                    "heston_validation": bates_payload,
                }
            elif "heston" in itype or product_model_family == "stochastic_volatility":
                from trellis.models.processes.heston import build_heston_parameter_payload

                heston_payload = build_heston_parameter_payload(
                    kappa=2.0,
                    theta=0.04,
                    xi=vol,
                    rho=-0.7,
                    v0=0.04,
                    mu=rate,
                    parameter_set_name="heston_validation",
                    source_kind="validation_fixture",
                )
                payload["model_parameters"] = heston_payload
                payload["model_parameter_sets"] = {"heston_validation": heston_payload}
            elif product_model_family == "sabr" or requires_sabr_model_parameters:
                sabr_payload = {
                    "family": "sabr",
                    "alpha": vol,
                    "beta": 0.5,
                    "rho": -0.2,
                    "nu": 0.35,
                }
                payload["model_parameters"] = {"sabr": sabr_payload}
                payload["model_parameter_sets"] = {"sabr_validation": sabr_payload}
            elif product_model_family == "variance_gamma" or any(
                "variance_gamma_option" in str(ref) for ref in exact_binding_refs
            ):
                vg_payload = {
                    "family": "variance_gamma",
                    "sigma": vol,
                    "theta": -0.08,
                    "nu": 0.20,
                }
                payload["model_parameters"] = {"variance_gamma": vg_payload}
                payload["model_parameter_sets"] = {
                    "variance_gamma_validation": vg_payload,
                }
            elif product_model_family == "cgmy" or any(
                "cgmy_option" in str(ref) for ref in exact_binding_refs
            ):
                cgmy_payload = {
                    "family": "cgmy",
                    "C": 0.40,
                    "G": 5.0,
                    "M": 6.0,
                    "Y": 0.55,
                }
                payload["model_parameters"] = {"cgmy": cgmy_payload}
                payload["model_parameter_sets"] = {"cgmy_validation": cgmy_payload}
            elif product_model_family == "kou" or any(
                "kou_option" in str(ref) for ref in exact_binding_refs
            ):
                kou_payload = {
                    "family": "kou",
                    "sigma": vol,
                    "jump_intensity": 0.35,
                    "up_probability": 0.35,
                    "eta_up": 8.0,
                    "eta_down": 6.0,
                }
                payload["model_parameters"] = {"kou": kou_payload}
                payload["model_parameter_sets"] = {"kou_validation": kou_payload}
            else:
                payload["model_parameters"] = {"quanto_correlation": corr}
        if (
            "jump_parameters" in effective_requirements
            or requires_merton_jump_parameters
            or requires_bates_model_parameters
        ):
            jump_payload = {
                "mu": rate,
                "sigma": vol,
                "lam": 0.35,
                "jump_mean": -0.08,
                "jump_vol": 0.18,
            }
            payload["jump_parameters"] = jump_payload
            payload["jump_parameter_sets"] = {
                "merton_validation": jump_payload,
                "bates_validation": jump_payload,
            }
        payload = _align_validation_market_payload_to_spec(
            payload,
            getattr(test_payoff, "spec", None),
            rate=rate,
            vol=vol,
            corr=corr,
        )
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
        method=validation_method,
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
    _record_comparison_validation_binding(
        build_meta,
        compiled_request=compiled_request,
        bundle_id=validation_bundle.bundle_id,
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
            method=validation_method,
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
    arbiter_verdicts = ()
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
            from trellis.agent.arbiter import run_critic_check_verdicts
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
            arbiter_verdicts = run_critic_check_verdicts(
                concerns,
                test_payoff,
                allowed_check_ids={check.check_id for check in critic_checks},
            )
            critic_failures = [
                verdict.message
                for verdict in arbiter_verdicts
                if verdict.status == "failed" and verdict.message
            ]
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
            "verdicts": [asdict(verdict) for verdict in arbiter_verdicts],
        },
    )

    deterministic_review_blocked = bool(failures)
    deterministic_review_block_reason = (
        "deterministic_validation_failed" if deterministic_review_blocked else None
    )
    model_validation_evidence_packet = {}
    if validation == "thorough":
        from trellis.agent.model_validator import build_model_validation_evidence_packet

        model_validation_evidence_packet = build_model_validation_evidence_packet(
            validation_contract=validation_contract,
            validation_bundle_execution=bundle_execution,
            reference_oracle=oracle_summary if oracle_execution is not None else None,
            arbiter_verdicts=arbiter_verdicts,
        )

    # Thorough: run model validator (MRM-style)
    if validation == "thorough" and not deterministic_review_blocked:
        try:
            from trellis.agent.model_validator import (
                classify_model_validation_findings,
                validate_model,
            )

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
                        method=validation_method,
                        knowledge_context=review_knowledge_text,
                        model=stage_model,
                        product_ir=product_ir,
                        generation_plan=getattr(compiled_request, "generation_plan", None),
                        validation_contract=validation_contract,
                        deterministic_evidence_packet=model_validation_evidence_packet,
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
                        "orientation_contract": role_orientation_summary(
                            "model_validator"
                        ),
                        "orientation_resolution": {
                            "role": "model_validator",
                            "prompt_injected": False,
                            "reason": review_policy.model_validator_reason,
                        },
                    },
                )
                report = validate_model(
                    payoff_factory=payoff_factory,
                    market_state_factory=ms_factory,
                    code=code,
                    instrument_type=itype,
                    method=validation_method,
                    knowledge_context="",
                    model=model,
                    product_ir=product_ir,
                    generation_plan=getattr(compiled_request, "generation_plan", None),
                    validation_contract=validation_contract,
                    deterministic_evidence_packet=model_validation_evidence_packet,
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
            finding_classification = classify_model_validation_findings(report.findings)
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
                    "orientation_contract": role_orientation_summary(
                        "model_validator"
                    ),
                    "orientation_resolution": dict(
                        getattr(report, "orientation_resolution", {}) or {}
                    ) or {
                        "role": "model_validator",
                        "prompt_injected": False,
                        "reason": review_policy.model_validator_reason,
                    },
                    "blocker_count": len(blocker_findings),
                    "approved": report.approved,
                    "llm_review": review_policy.run_model_validator_llm,
                    "risk_level": review_policy.risk_level,
                    "residual_risks": list(
                        dict(model_validation_evidence_packet.get("validation_contract") or {}).get(
                            "residual_risks",
                            (),
                        )
                    ),
                    "deterministic_evidence": model_validation_evidence_packet,
                    "findings": [asdict(finding) for finding in report.findings],
                    "finding_classification": finding_classification,
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
                "reason": deterministic_review_block_reason,
                "orientation_contract": role_orientation_summary(
                    "model_validator"
                ),
                "orientation_resolution": {
                    "role": "model_validator",
                    "prompt_injected": False,
                    "reason": deterministic_review_block_reason,
                },
                "failure_count": len(failures),
                "deterministic_evidence": model_validation_evidence_packet,
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


def _benchmark_spec_overrides_from_compiled_request(compiled_request) -> dict[str, object]:
    """Return normalized benchmark spec overrides carried on the request metadata."""
    request = getattr(compiled_request, "request", None)
    request_metadata = dict(getattr(request, "metadata", None) or {})
    overrides = request_metadata.get("benchmark_spec_overrides")
    if isinstance(overrides, Mapping):
        return {
            str(key): value
            for key, value in overrides.items()
            if str(key).strip() and value is not None
        }
    return {}


def _make_test_payoff(
    payoff_cls,
    spec_schema,
    settle: date,
    market_state=None,
    spec_overrides: Mapping[str, object] | None = None,
):
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

    if is_dataclass(spec_cls):
        actual_fields = tuple(fields(spec_cls))
        actual_field_names = {field.name for field in actual_fields}
        schema_field_names = {
            str(getattr(field, "name", "") or "")
            for field in getattr(spec_schema, "fields", ())
            if str(getattr(field, "name", "") or "")
        }
        required_actual_field_names = {
            field.name
            for field in actual_fields
            if field.default is MISSING and field.default_factory is MISSING
        }
        if schema_field_names and (
            not schema_field_names.issubset(actual_field_names)
            or not required_actual_field_names.issubset(schema_field_names)
        ):
            spec_schema = SimpleNamespace(
                spec_name=spec_cls.__name__,
                fields=[
                    SimpleNamespace(
                        name=field.name,
                        type=_runtime_annotation_to_field_type(field.type),
                        default=(
                            None
                            if field.default is MISSING and field.default_factory is MISSING
                            else repr(field.default if field.default is not MISSING else None)
                        ),
                    )
                    for field in actual_fields
                ],
            )

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
        "tuple[float, ...]": (0.25, 0.5, 0.75, 1.0),
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
        "autocall_barrier": 1.0,
        "protection_barrier": 0.7,
        "coupon_rate": 0.08,
        "initial_spot": 100.0,
        "observation_times": (0.25, 0.5, 0.75, 1.0),
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
            getattr(payoff_cls, "__name__", "") or "",
            getattr(spec_schema, "class_name", "") or "",
            getattr(spec_schema, "spec_name", "") or "",
            getattr(spec_cls, "__doc__", "") or "",
            getattr(payoff_cls, "__doc__", "") or "",
            getattr(module, "__doc__", "") or "",
        )
        if part and str(part).strip()
    ).lower()
    has_spot_field = any(
        field.name in {"spot", "s0", "underlier_spot"} for field in spec_schema.fields
    )
    has_strike_field = any(field.name == "strike" for field in spec_schema.fields)
    spot_option_context = any(
        token in basket_context
        for token in (
            "heston",
            "equity option",
            "european option",
            "vanilla option",
            "underlier spot",
        )
    )
    heston_option_context = "heston" in basket_context
    if spec_schema.spec_name == "AsianOptionSpec":
        # Sparse task rows declare an observation count, not an invented date schedule.
        name_defaults["observation_dates"] = ()
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
        name_defaults["underlier_id"] = None
        name_defaults["underlier_vol_surface_key"] = None
        name_defaults["fx_vol_surface_key"] = None
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
    elif has_strike_field and (has_spot_field or spot_option_context):
        # Spot-based option specs should be instantiated near-the-money so the
        # smoke tests exercise a representative valuation instead of a deeply
        # in-the-money payoff created by the generic rate-like strike default.
        spot_default = name_defaults["spot"]
        market_spot = getattr(market_state, "spot", None)
        if market_spot is not None:
            try:
                spot_default = float(market_spot)
            except (TypeError, ValueError):
                spot_default = name_defaults["spot"]
        name_defaults["notional"] = 1.0 if heston_option_context else 10.0
        name_defaults["spot"] = spot_default
        name_defaults["strike"] = spot_default

    description = getattr(payoff_cls, "__doc__", "") or getattr(module, "__doc__", "") or ""
    description_defaults = _description_spec_defaults(
        spec_schema,
        description=description,
    )
    if description_defaults:
        name_defaults.update(description_defaults)

    if spec_overrides:
        valid_fields = {field.name for field in spec_schema.fields}
        for key, value in spec_overrides.items():
            if key in valid_fields and value is not None:
                name_defaults[key] = value

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


def _resolve_runtime_spec_schema(payoff_cls, spec_schema):
    """Prefer the actual payoff-module dataclass schema when it is richer."""
    if payoff_cls is None:
        return spec_schema

    try:
        module = import_module(payoff_cls.__module__)
    except Exception:
        return spec_schema

    inferred = _runtime_infer_spec_schema_from_module(module, payoff_cls)
    if inferred is None:
        return spec_schema
    if spec_schema is None:
        return inferred

    planned_name = str(getattr(spec_schema, "spec_name", "") or "")
    inferred_name = str(getattr(inferred, "spec_name", "") or "")
    if planned_name and inferred_name and planned_name != inferred_name:
        return spec_schema

    planned_fields = {
        str(getattr(field, "name", "") or "")
        for field in getattr(spec_schema, "fields", ())
        if str(getattr(field, "name", "") or "")
    }
    inferred_fields = {
        str(getattr(field, "name", "") or "")
        for field in getattr(inferred, "fields", ())
        if str(getattr(field, "name", "") or "")
    }
    if not inferred_fields.issuperset(planned_fields):
        return spec_schema

    planned_requirements = {
        str(item) for item in getattr(spec_schema, "requirements", ()) if str(item).strip()
    }
    inferred_requirements = {
        str(item) for item in getattr(inferred, "requirements", ()) if str(item).strip()
    }
    if inferred_fields == planned_fields and inferred_requirements == planned_requirements:
        return spec_schema
    return inferred


def _runtime_infer_spec_schema_from_module(module, payoff_cls: type):
    """Infer a planner-compatible schema directly from the payoff module dataclass."""
    from trellis.agent.planner import FieldDef, SpecSchema

    spec_cls = None
    for value in module.__dict__.values():
        if isinstance(value, type) and is_dataclass(value) and value.__name__.endswith("Spec"):
            spec_cls = value
            break
    if spec_cls is None:
        return None

    field_defs = []
    for field in fields(spec_cls):
        field_defs.append(
            FieldDef(
                name=field.name,
                type=_runtime_annotation_to_field_type(field.type),
                description="",
                default=None
                if field.default is MISSING and field.default_factory is MISSING
                else repr(field.default if field.default is not MISSING else None),
            )
        )

    return SpecSchema(
        class_name=payoff_cls.__name__,
        spec_name=spec_cls.__name__,
        requirements=[],
        fields=field_defs,
    )


def _runtime_annotation_to_field_type(annotation) -> str:
    """Map runtime annotations to the planner field-type strings."""
    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        if normalized == "tuple[date,...]":
            return "tuple[date, ...]"
        if normalized in {"tuple[date,...]|None", "None|tuple[date,...]"}:
            return "tuple[date, ...] | None"
        if annotation in {
            "float",
            "int",
            "str",
            "bool",
            "date",
            "str | None",
            "float | None",
            "int | None",
            "tuple[date, ...]",
            "tuple[date, ...] | None",
            "Frequency",
            "DayCountConvention",
        }:
            return annotation
        return "str"

    if annotation is float:
        return "float"
    if annotation is int:
        return "int"
    if annotation is str:
        return "str"
    if annotation is bool:
        return "bool"
    if annotation is date:
        return "date"

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple and args == (date, Ellipsis):
        return "tuple[date, ...]"
    if args and type(None) in args:
        non_none = tuple(arg for arg in args if arg is not type(None))
        if len(non_none) == 1:
            inner = _runtime_annotation_to_field_type(non_none[0])
            if inner in {"float", "int", "str", "tuple[date, ...]"}:
                return f"{inner} | None"

    annotation_name = getattr(annotation, "__name__", None)
    if annotation_name in {"Frequency", "DayCountConvention"}:
        return annotation_name
    return "str"


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
    if spec_name not in {"AgentCapSpec", "AgentFloorSpec", "BasketOptionSpec"}:
        return {}

    from trellis.core.types import DayCountConvention, Frequency

    overrides: dict[str, object] = {}

    def _match(pattern: str) -> str | None:
        matched = re.search(pattern, text, re.IGNORECASE)
        if matched is None:
            return None
        return matched.group(1).strip()

    def _match_sentence_value(label: str) -> str | None:
        return _match(
            rf"{re.escape(label)}:\s*(.+?)(?=(?:\s+[A-Z][A-Za-z ]+:)|\n|$)"
        )

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

    if spec_name in {"AgentCapSpec", "AgentFloorSpec"}:
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

        model = _match(r"Pricing model:\s*([A-Za-z_]+)")
        if model is not None:
            overrides["model"] = model.rstrip(".,;:").lower()

        shift = _parse_number(
            _match(r"Shift:\s*([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)")
        )
        if shift is not None:
            overrides["shift"] = shift

        sabr_text = _match_sentence_value("SABR parameters")
        if sabr_text is not None:
            sabr_params: dict[str, float] = {}
            for name in ("alpha", "beta", "nu", "rho"):
                matched = re.search(
                    rf"{name}\s*=\s*([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)",
                    sabr_text,
                    re.IGNORECASE,
                )
                if matched is None:
                    sabr_params = {}
                    break
                sabr_params[name] = float(matched.group(1))
            if sabr_params:
                overrides["sabr"] = sabr_params

    if spec_name == "BasketOptionSpec":
        notional = _parse_number(_match_sentence_value("Notional"))
        if notional is not None:
            overrides["notional"] = notional

        strike = _parse_number(_match_sentence_value("Strike"))
        if strike is not None:
            overrides["strike"] = strike

        expiry_date = _parse_date(_match_sentence_value("Expiry date"))
        if expiry_date is not None:
            overrides["expiry_date"] = expiry_date

        for label, field_name in (
            ("Underliers", "underliers"),
            ("Spots", "spots"),
            ("Correlation", "correlation"),
            ("Weights", "weights"),
            ("Vols", "vols"),
            ("Dividend yields", "dividend_yields"),
            ("Basket style", "basket_style"),
            ("Option type", "option_type"),
        ):
            value = _match_sentence_value(label)
            if value is not None:
                normalized = value.rstrip(".,;:")
                if field_name in {"basket_style", "option_type"}:
                    normalized = normalized.strip().lower()
                overrides[field_name] = normalized

    return overrides


def _smoke_test_actual_market_state(
    payoff_cls,
    spec_schema,
    market_state,
    spec_overrides: Mapping[str, object] | None = None,
) -> list[str]:
    """Run a lightweight pricing smoke test against the actual task market state."""
    if market_state is None:
        return []
    from trellis.engine.payoff_pricer import price_payoff

    settle = getattr(market_state, "settlement", date(2024, 11, 15))
    runtime_spec_schema = _resolve_runtime_spec_schema(payoff_cls, spec_schema)
    try:
        payoff = _make_test_payoff(
            payoff_cls,
            runtime_spec_schema,
            settle,
            spec_overrides=spec_overrides,
        )
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
    import_lines = ["from trellis.core.payoff import PricingValue"]
    import_lines.extend(_skeleton_type_import_lines(spec_schema))
    import_lines.extend(_skeleton_exact_binding_import_lines(generation_plan))
    evaluate_preamble_lines = ["        spec = self._spec"]
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

    def evaluate(self, market_state: MarketState) -> PricingValue:
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


def _generation_plan_field(generation_plan, name: str, default=None):
    if generation_plan is None:
        return default
    if isinstance(generation_plan, Mapping):
        return generation_plan.get(name, default)
    return getattr(generation_plan, name, default)


def _skeleton_exact_binding_import_lines(generation_plan) -> tuple[str, ...]:
    """Return import lines for compiler-selected exact bindings."""
    if generation_plan is None:
        return ()

    refs: list[str] = list(_exact_binding_refs(generation_plan))
    primitive_plan = _generation_plan_field(generation_plan, "primitive_plan")
    if primitive_plan is not None:
        refs.extend(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in getattr(primitive_plan, "primitives", ()) or ()
            if (
                getattr(primitive, "required", False)
                or getattr(primitive, "role", "") in {"route_helper", "pricing_kernel", "schedule_builder"}
            )
            and not getattr(primitive, "excluded", False)
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
    refs: list[str] = []

    def add_ref(value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text and text not in refs:
                refs.append(text)
            return
        if isinstance(value, Mapping):
            module = value.get("module")
            symbol = value.get("symbol")
            if module and symbol:
                add_ref(f"{module}.{symbol}")
            for key in (
                "binding_id",
                "backend_binding_id",
                "exact_target_refs",
                "backend_exact_target_refs",
                "helper_refs",
                "backend_helper_refs",
                "primitive_ref",
                "primitive_refs",
                "pricing_kernel_refs",
                "schedule_builder_refs",
                "cashflow_engine_refs",
                "dsl_target_refs",
                "dsl_helper_refs",
                "dsl_target_bindings",
                "target_bindings",
                "reusable_bindings",
                "lowering",
                "construction_identity",
                "route_binding_authority",
            ):
                add_ref(value.get(key))
            add_ref(value.get("backend_binding"))
            add_ref(value.get("primitive_plan"))
            return
        if isinstance(value, Sequence):
            for item in value:
                add_ref(item)

    add_ref(generation_plan)

    for attr in (
        "lane_exact_binding_refs",
        "lane_reusable_primitives",
        "backend_binding_id",
        "backend_exact_target_refs",
        "backend_helper_refs",
        "lowering_target_refs",
        "lowering_helper_refs",
    ):
        add_ref(_generation_plan_field(generation_plan, attr))

    primitive_plan = _generation_plan_field(generation_plan, "primitive_plan")
    if primitive_plan is not None:
        for attr in (
            "backend_binding_id",
            "backend_exact_target_refs",
            "backend_helper_refs",
        ):
            add_ref(_generation_plan_field(primitive_plan, attr))
        add_ref(tuple(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in getattr(primitive_plan, "primitives", ()) or ()
            if getattr(primitive, "role", "") == "route_helper"
            and getattr(primitive, "module", "")
            and getattr(primitive, "symbol", "")
        ))

    authority = _generation_plan_field(generation_plan, "route_binding_authority")
    if authority is not None:
        if isinstance(authority, Mapping):
            add_ref(authority)
        else:
            add_ref(getattr(authority, "backend_binding", None))

    normalized: list[str] = []
    for ref in refs:
        text = str(ref or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


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


def _swaption_tree_model_name(semantic_blueprint, comparison_target: str | None) -> str:
    """Return the admitted short-rate lattice model for a swaption target."""
    normalized_target = str(comparison_target or "").strip().lower().replace("-", "_")
    if normalized_target in {"bdt", "bdt_tree", "black_derman_toy"}:
        return "bdt"
    valuation_context = getattr(semantic_blueprint, "valuation_context", None)
    engine_model_spec = getattr(valuation_context, "engine_model_spec", None)
    model_name = str(getattr(engine_model_spec, "model_name", "") or "").strip().lower()
    if model_name in {"bdt", "black_derman_toy"}:
        return "bdt"
    return "hull_white"


def _bermudan_swaption_lattice_evaluate_body(
    *,
    model_name: str,
    comparison_kwargs: str,
) -> str:
    """Return explicit generic-lattice construction for a Bermudan swaption."""
    return textwrap.dedent(
        f"""\
        tree_steps = getattr(spec, "tree_steps", getattr(spec, "n_steps", None))
        resolved = resolve_bermudan_swaption_tree_inputs(
            market_state,
            spec,
            model="{model_name}"{comparison_kwargs},
            n_steps=tree_steps,
        )
        exercise_dates = tuple(
            exercise_date
            for exercise_date in normalize_explicit_dates(spec.exercise_dates)
            if resolved.settlement < exercise_date < spec.swap_end
        )
        if not exercise_dates:
            return 0.0

        lattice = build_lattice(
            BINOMIAL_1F_TOPOLOGY,
            UNIFORM_ADDITIVE_MESH,
            "{model_name}",
            calibration_target=TERM_STRUCTURE_TARGET(market_state.discount),
            r0=resolved.r0,
            sigma=resolved.sigma,
            a=resolved.mean_reversion,
            T=resolved.tree_horizon,
            n_steps=resolved.n_steps,
        )
        exercise_steps = []
        for exercise_date in exercise_dates:
            exercise_time = float(
                year_fraction(resolved.settlement, exercise_date, spec.day_count)
            )
            quantized_time = (
                round(exercise_time * resolved.exercise_frequency)
                / resolved.exercise_frequency
            )
            exercise_step = lattice_step_from_time(
                quantized_time,
                dt=lattice.dt,
                n_steps=lattice.n_steps,
                allow_terminal_step=False,
            )
            if exercise_step is not None:
                exercise_steps.append(exercise_step)
        valid_exercise_steps = tuple(sorted(set(exercise_steps)))
        if not valid_exercise_steps:
            return 0.0

        swap_start = min(exercise_dates)
        payment_timeline = build_payment_timeline(
            swap_start,
            spec.swap_end,
            spec.swap_frequency,
            day_count=spec.day_count,
            time_origin=resolved.settlement,
            label="bermudan_swaption_fixed_leg",
        )
        frequency_per_year = max(
            int(getattr(spec.swap_frequency, "value", spec.swap_frequency)),
            1,
        )
        first_exercise_step = min(valid_exercise_steps)
        swap_tenor = float(
            year_fraction(swap_start, spec.swap_end, spec.day_count)
        )
        quantized_tenor = (
            round(swap_tenor * frequency_per_year) / frequency_per_year
        )
        swap_end_time = (
            first_exercise_step * lattice.dt + quantized_tenor
        )
        swap_end_step = lattice_step_from_time(
            swap_end_time,
            dt=lattice.dt,
            n_steps=lattice.n_steps,
            allow_terminal_step=True,
        )
        if swap_end_step is None or swap_end_step <= first_exercise_step:
            return 0.0
        swap_start_time = float(
            year_fraction(resolved.settlement, swap_start, spec.day_count)
        )
        coupon = float(spec.notional) * float(spec.strike) / frequency_per_year
        coupon_by_step = {{}}
        for period in payment_timeline:
            absolute_payment_time = (
                float(period.t_payment)
                if period.t_payment is not None
                else float(
                    year_fraction(
                        resolved.settlement,
                        period.payment_date,
                        spec.day_count,
                    )
                )
            )
            payment_tenor = max(absolute_payment_time - swap_start_time, 0.0)
            quantized_payment_tenor = (
                round(payment_tenor * frequency_per_year) / frequency_per_year
            )
            payment_time = (
                first_exercise_step * lattice.dt + quantized_payment_tenor
            )
            payment_step = lattice_step_from_time(
                payment_time,
                dt=lattice.dt,
                n_steps=lattice.n_steps,
                allow_terminal_step=True,
            )
            if (
                payment_step is None
                or payment_step <= first_exercise_step
                or payment_step > swap_end_step
            ):
                continue
            coupon_by_step[payment_step] = (
                coupon_by_step.get(payment_step, 0.0) + coupon
            )

        notional = float(spec.notional)

        def fixed_leg_terminal(step, node, lattice_, obs):
            del node, lattice_, obs
            if step != swap_end_step:
                return 0.0
            return notional + float(coupon_by_step.get(step, 0.0))

        def fixed_leg_cashflow(step, node, lattice_, obs):
            del node, obs
            if step == lattice_.n_steps:
                return 0.0
            amount = float(coupon_by_step.get(step, 0.0))
            if step == swap_end_step:
                amount += notional
            return amount

        fixed_leg_claim = LatticeLinearClaimSpec(
            terminal_payoff=fixed_leg_terminal,
            node_cashflow_fn=fixed_leg_cashflow,
            observable_requirements=("rate",),
        )
        fixed_leg_contract = LatticeContractSpec(claim=fixed_leg_claim)
        fixed_leg_result = value_on_lattice(
            lattice,
            fixed_leg_contract,
            observation_steps=valid_exercise_steps,
        )
        signed_swap_values = {{}}
        for exercise_step in valid_exercise_steps:
            if exercise_step >= swap_end_step:
                continue
            fixed_leg_values = fixed_leg_result.observation_at(
                exercise_step
            ).continuation_values
            payer_values = tuple(
                notional - fixed_leg_value
                for fixed_leg_value in fixed_leg_values
            )
            signed_swap_values[exercise_step] = (
                payer_values
                if bool(spec.is_payer)
                else tuple(-value for value in payer_values)
            )
        if not signed_swap_values:
            return 0.0

        def exercise_value(step, node, lattice_, obs):
            del lattice_, obs
            return max(float(signed_swap_values[step][node]), 0.0)

        option_claim = LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: 0.0,
            observable_requirements=("rate",),
        )
        option_control = LatticeControlSpec(
            objective="holder_max",
            exercise_steps=tuple(sorted(signed_swap_values)),
            exercise_value_fn=exercise_value,
        )
        option_contract = LatticeContractSpec(
            claim=option_claim,
            control=option_control,
        )
        return float(price_on_lattice(lattice, option_contract))
        """
    ).rstrip()


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


def _vanilla_equity_pde_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for vanilla-equity PDE comparison targets."""
    target = str(comparison_target or "").strip().lower().replace("-", "_")
    if target in {"theta_0.5", "theta_0_5", "crank_nicolson"}:
        return ", theta=0.5"
    if target in {"theta_1.0", "theta_1_0", "implicit"}:
        return ", theta=1.0"
    return ""


def _heston_monte_carlo_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for Heston Monte Carlo comparison targets."""
    target = str(comparison_target or "").strip().lower().replace("-", "_")
    if target in {"euler_heston", "heston_euler"}:
        return ', scheme="euler"'
    if target in {
        "heston_mc",
        "qe",
        "heston_qe",
        "qe_heston",
        "andersen_qe",
        "quadratic_exponential",
    }:
        return ', scheme="heston_qe"'
    return ""


def _vanilla_equity_transform_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for transform comparison targets."""
    target = str(comparison_target or "").strip().lower().replace("-", "_")
    if target in {"fft", "digital_fft"}:
        return ', method="fft"'
    if target in {"cos", "digital_cos", "cos_adaptive", "cos_fixed"}:
        return ', method="cos"'
    return ""


def _heston_transform_helper_kwargs(comparison_target: str | None) -> str:
    """Return deterministic helper kwargs for Heston transform comparison targets."""
    target = str(comparison_target or "").strip().lower().replace("-", "_")
    if target in {
        "fft",
        "heston_fft",
        "fft_heston",
        "carr_madan",
        "heston_analytical",
        "analytical_heston",
        "semi_analytical_heston",
    }:
        return ', method="fft"'
    if target in {"cos", "heston_cos", "cos_heston", "fang_oosterlee"}:
        return ', method="cos"'
    if target in {
        "laguerre",
        "gauss_laguerre",
        "laguerre_heston",
        "heston_laguerre",
        "gauss_laguerre_heston",
        "heston_gauss_laguerre",
    }:
        return ', method="gauss_laguerre"'
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


def _american_equity_tree_primitive_body(
    comparison_target: str,
    *,
    include_bermudan_schedule: bool,
) -> str:
    """Return explicit lattice-algebra composition for an early-exercise equity option."""
    steps_expression = (
        "2000"
        if comparison_target == "high_step_tree_2000"
        else 'max(int(getattr(spec, "tree_steps", 800)), 1)'
    )
    prefix = textwrap.dedent(
        f"""\
        resolved = resolve_single_state_diffusion_inputs(market_state, spec)
        if resolved.maturity <= 0.0:
            return float(resolved.notional * terminal_intrinsic_from_resolved(
                resolved.spot, resolved
            ))
        tree_steps = {steps_expression}
        recipe = equity_tree(
            model_family="crr",
            strike=resolved.strike,
            option_type=resolved.option_type,
        )
        exercise_style = str(getattr(spec, "exercise_style", "american")).strip().lower()
        if exercise_style == "american":
            recipe = with_control(recipe, "american")
        """
    ).rstrip()
    control_branches: list[str] = []
    if include_bermudan_schedule:
        control_branches.append(textwrap.dedent(
            """\
            elif exercise_style == "bermudan":
                if not spec.exercise_dates:
                    raise ValueError("Bermudan tree pricing requires spec.exercise_dates")
                settlement = market_state.settlement or market_state.as_of
                if settlement is None:
                    raise ValueError("Bermudan tree pricing requires market settlement or as_of")
                event_times = []
                for exercise_date in spec.exercise_dates:
                    exercise_time = float(year_fraction(settlement, exercise_date, spec.day_count))
                    if 0.0 <= exercise_time <= resolved.maturity:
                        event_times.append(exercise_time)
                if not event_times:
                    raise ValueError(
                        "Bermudan tree pricing requires an exercise date within the pricing horizon"
                    )
                exercise_steps = event_step_indices(
                    event_times, resolved.maturity, tree_steps
                )
                recipe = with_control(
                    recipe, "bermudan", exercise_steps=exercise_steps
                )
            """
        ).rstrip())
    suffix = textwrap.dedent(
        """\
        elif exercise_style != "european":
            raise ValueError(f"Unsupported exercise_style {exercise_style!r}")
        topology, mesh, model, contract = compile_lattice_recipe(recipe)
        lattice = build_lattice(
            topology,
            mesh,
            model,
            spot=resolved.spot,
            rate=resolved.rate,
            dividend_yield=resolved.dividend_yield,
            sigma=resolved.sigma,
            maturity=resolved.maturity,
            n_steps=tree_steps,
        )
        return float(resolved.notional * price_on_lattice(lattice, contract))
        """
    ).rstrip()
    return "\n".join((prefix, *control_branches, suffix))


def _credit_default_swap_helper_body(refs: set[str]) -> str | None:
    """Return a thin deterministic CDS wrapper for analytical or MC exact bindings."""
    analytical_ref = "trellis.models.credit_default_swap.price_cds_analytical"
    monte_carlo_ref = "trellis.models.credit_default_swap.price_cds_monte_carlo"
    if monte_carlo_ref in refs:
        helper = "price_cds_monte_carlo"
        helper_extra = ',\n        n_paths=getattr(spec, "n_paths", 250000) or 250000,\n        seed=42'
    elif analytical_ref in refs:
        helper = "price_cds_analytical"
        helper_extra = ""
    else:
        return None

    return textwrap.dedent(
        f"""\
spec = self._spec
if market_state.credit_curve is None:
    raise ValueError("market_state.credit_curve is required for CDS pricing")
if market_state.discount is None:
    raise ValueError("market_state.discount is required for CDS pricing")
schedule = build_cds_schedule(
    spec.start_date,
    spec.end_date,
    spec.frequency,
    spec.day_count,
    time_origin=getattr(spec, "valuation_date", None) or spec.start_date,
)
return {helper}(
    notional=spec.notional,
    spread_quote=spec.spread,
    recovery=spec.recovery,
    schedule=schedule,
    credit_curve=market_state.credit_curve,
    discount_curve=market_state.discount{helper_extra},
)
"""
    ).rstrip()


def _ranked_observation_basket_primitive_body(refs: set[str]) -> str | None:
    """Return explicit primitive composition for ranked-observation basket MC."""
    required_refs = {
        "trellis.models.resolution.basket_semantics.resolve_basket_semantics",
        "trellis.models.analytical.support.implied_zero_rate",
        "trellis.models.processes.correlated_gbm.CorrelatedGBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        (
            "trellis.models.monte_carlo.ranked_observation_payoffs."
            "build_ranked_observation_basket_state_payoff"
        ),
        (
            "trellis.models.monte_carlo.ranked_observation_payoffs."
            "terminal_ranked_observation_basket_payoff"
        ),
    }
    if not required_refs.issubset(refs):
        return None

    return textwrap.dedent(
        """\
        spec = self._spec
        resolved = resolve_basket_semantics(
            market_state,
            spec,
            constituents=getattr(spec, "constituents", None) or spec.underliers,
            observation_dates=getattr(spec, "observation_dates", None),
            selection_rule=getattr(spec, "selection_rule", None) or "best_of_remaining",
            lock_rule=getattr(spec, "lock_rule", None) or "remove_selected",
            aggregation_rule=getattr(spec, "aggregation_rule", None) or "average_locked_returns",
            option_type=getattr(spec, "option_type", "call"),
            selection_count=getattr(spec, "selection_count", None) or 1,
            day_count=spec.day_count,
            correlation_matrix_key=(
                getattr(spec, "correlation_matrix_key", None)
                or getattr(spec, "correlation_key", None)
            ),
        )
        if resolved.T <= 0.0:
            intrinsic = terminal_ranked_observation_basket_payoff(
                spec,
                [[[float(value) for value in resolved.constituent_spots]]],
                resolved,
            )[0]
            return float(spec.notional) * float(intrinsic)

        domestic_rate = float(implied_zero_rate(resolved.domestic_df, resolved.T))
        process = CorrelatedGBM(
            mu=[domestic_rate - float(carry) for carry in resolved.constituent_carry],
            sigma=[float(value) for value in resolved.constituent_vols],
            corr=[
                [float(cell) for cell in row]
                for row in resolved.correlation_matrix
            ],
        )
        engine_steps = max(
            int(getattr(spec, "n_steps", 252) or 252),
            64,
            int(float(resolved.T) * 252.0 + 0.999999),
            len(resolved.observation_times) * 16 if resolved.observation_times else 64,
        )
        payoff = build_ranked_observation_basket_state_payoff(
            spec,
            resolved,
            n_steps=engine_steps,
        )
        engine = MonteCarloEngine(
            process,
            n_paths=max(int(getattr(spec, "n_paths", 50000) or 50000), 4096),
            n_steps=engine_steps,
            seed=int(getattr(spec, "seed", 42) or 42),
            method=getattr(spec, "mc_method", None) or "exact",
        )
        price_result = engine.price(
            tuple(float(value) for value in resolved.constituent_spots),
            float(resolved.T),
            payoff,
            discount_rate=0.0,
            storage_policy=payoff.path_requirement,
            return_paths=False,
        )
        return (
            float(spec.notional) * float(resolved.domestic_df)
            * float(price_result["price"])
        )
        """
    ).rstrip()


def _terminal_basket_primitive_body(
    refs: set[str],
    *,
    normalized_target: str,
    generation_method: str,
) -> str | None:
    """Return explicit primitive composition for a two-asset terminal basket."""
    analytical_kernels = {
        "trellis.models.analytical.terminal_basket.two_asset_extremum_option_stulz",
        "trellis.models.analytical.terminal_basket.two_asset_spread_option_kirk",
        (
            "trellis.models.analytical.terminal_basket."
            "two_asset_terminal_basket_gauss_hermite"
        ),
    }
    monte_carlo_kernels = {
        "trellis.models.processes.correlated_gbm.CorrelatedGBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.payoffs.terminal_basket_option_payoff",
    }
    transform_kernels = {
        (
            "trellis.models.transforms.spread_option."
            "correlated_gbm_log_return_characteristic_function"
        ),
        (
            "trellis.models.transforms.spread_option."
            "hurd_zhou_spread_option_2d_fft"
        ),
    }
    target = str(normalized_target or "").strip()
    use_transform = generation_method in {"fft_pricing", "transform_fft"}
    use_monte_carlo = generation_method in {"monte_carlo", "qmc"}
    if use_transform and transform_kernels.issubset(refs):
        return textwrap.dedent(
            """\
            spec = self._spec
            resolved = resolve_terminal_basket_inputs(
                market_state, spec, comparison_target=__COMPARISON_TARGET__
            )
            semantics = resolved.semantics
            if len(semantics.constituent_names) != 2:
                raise ValueError("Hurd-Zhou pricing requires exactly two underliers")
            if resolved.basket_style != "spread":
                raise ValueError("Hurd-Zhou pricing requires basket_style='spread'")
            if semantics.T <= 0.0:
                intrinsic = terminal_basket_option_payoff(
                    get_numpy().asarray(
                        [semantics.constituent_spots], dtype=float
                    ),
                    weights=resolved.weights,
                    basket_style=resolved.basket_style,
                    strike=resolved.strike,
                    option_type=resolved.option_type,
                )[0]
                return float(spec.notional) * float(intrinsic)
            domestic_rate = float(
                implied_zero_rate(semantics.domestic_df, semantics.T)
            )

            def characteristic_function(u1, u2):
                return correlated_gbm_log_return_characteristic_function(
                    u1,
                    u2,
                    T=semantics.T,
                    rate=domestic_rate,
                    dividend_yields=resolved.carry,
                    volatilities=resolved.vols,
                    correlation=resolved.correlation_matrix[0][1],
                )

            unit_price = hurd_zhou_spread_option_2d_fft(
                characteristic_function,
                spots=resolved.notional_spots,
                weights=resolved.weights,
                strike=resolved.strike,
                discount_factor=semantics.domestic_df,
                option_type=resolved.option_type,
            )
            return float(spec.notional) * float(unit_price)
            """
        ).rstrip().replace("__COMPARISON_TARGET__", repr(target or None))
    if use_monte_carlo and monte_carlo_kernels.issubset(refs):
        return textwrap.dedent(
            """\
            spec = self._spec
            resolved = resolve_terminal_basket_inputs(
                market_state, spec, comparison_target=__COMPARISON_TARGET__
            )
            semantics = resolved.semantics
            if semantics.T <= 0.0:
                intrinsic = terminal_basket_option_payoff(
                    get_numpy().asarray(
                        [semantics.constituent_spots], dtype=float
                    ),
                    weights=resolved.weights,
                    basket_style=resolved.basket_style,
                    strike=resolved.strike,
                    option_type=resolved.option_type,
                )[0]
                return float(spec.notional) * float(intrinsic)
            domestic_rate = float(
                implied_zero_rate(semantics.domestic_df, semantics.T)
            )
            process = CorrelatedGBM(
                mu=[
                    domestic_rate
                    for _ in semantics.constituent_names
                ],
                sigma=list(resolved.vols),
                corr=[
                    list(row) for row in resolved.correlation_matrix
                ],
                dividend_yield=list(resolved.carry),
            )

            def payoff_fn(simulated_paths):
                return terminal_basket_option_payoff(
                    simulated_paths[:, -1, :],
                    weights=resolved.weights,
                    basket_style=resolved.basket_style,
                    strike=resolved.strike,
                    option_type=resolved.option_type,
                )

            engine = MonteCarloEngine(
                process,
                n_paths=max(
                    int(getattr(spec, "n_paths", 40000) or 40000),
                    8192,
                ),
                n_steps=1,
                seed=int(getattr(spec, "seed", 42) or 42),
                method="exact",
            )
            result = engine.price(
                get_numpy().asarray(
                    semantics.constituent_spots, dtype=float
                ),
                float(semantics.T),
                payoff_fn,
                discount_rate=domestic_rate,
                return_paths=False,
            )
            return float(spec.notional) * float(result["price"])
            """
        ).rstrip().replace("__COMPARISON_TARGET__", repr(target or None))
    if analytical_kernels.intersection(refs):
        return textwrap.dedent(
            """\
            spec = self._spec
            resolved = resolve_terminal_basket_inputs(
                market_state, spec, comparison_target=__COMPARISON_TARGET__
            )
            semantics = resolved.semantics
            if len(semantics.constituent_names) != 2:
                raise ValueError(
                    "terminal basket analytical pricing requires exactly two underliers"
                )
            if semantics.T <= 0.0:
                intrinsic = terminal_basket_option_payoff(
                    get_numpy().asarray(
                        [semantics.constituent_spots], dtype=float
                    ),
                    weights=resolved.weights,
                    basket_style=resolved.basket_style,
                    strike=resolved.strike,
                    option_type=resolved.option_type,
                )[0]
                return float(spec.notional) * float(intrinsic)
            if resolved.basket_style in {"best_of", "worst_of"}:
                unit_price = two_asset_extremum_option_stulz(
                    spots=resolved.notional_spots,
                    strike=resolved.strike,
                    T=semantics.T,
                    discount_factor=semantics.domestic_df,
                    dividend_yields=resolved.carry,
                    volatilities=resolved.vols,
                    correlation=resolved.correlation_matrix[0][1],
                    basket_style=resolved.basket_style,
                    option_type=resolved.option_type,
                )
            elif resolved.basket_style == "spread":
                domestic_rate = float(
                    implied_zero_rate(semantics.domestic_df, semantics.T)
                )
                forwards = tuple(
                    float(spot)
                    * exp(
                        (
                            domestic_rate - float(dividend_yield)
                        )
                        * float(semantics.T)
                    )
                    for spot, dividend_yield in zip(
                        semantics.constituent_spots,
                        resolved.carry,
                        strict=True,
                    )
                )
                unit_price = two_asset_spread_option_kirk(
                    forwards=forwards,
                    strike=resolved.strike,
                    T=semantics.T,
                    discount_factor=semantics.domestic_df,
                    volatilities=resolved.vols,
                    correlation=resolved.correlation_matrix[0][1],
                    weights=resolved.weights,
                    option_type=resolved.option_type,
                )
            else:
                unit_price = two_asset_terminal_basket_gauss_hermite(
                    spots=resolved.notional_spots,
                    weights=resolved.weights,
                    strike=resolved.strike,
                    T=semantics.T,
                    discount_factor=semantics.domestic_df,
                    dividend_yields=resolved.carry,
                    volatilities=resolved.vols,
                    correlation=resolved.correlation_matrix[0][1],
                    basket_style=resolved.basket_style,
                    option_type=resolved.option_type,
                )
            return float(spec.notional) * float(unit_price)
            """
        ).rstrip().replace("__COMPARISON_TARGET__", repr(target or None))
    return None


def _nth_to_default_helper_body(refs: set[str]) -> str | None:
    """Return a thin deterministic wrapper for nth-to-default exact bindings."""
    helper_ref = "trellis.instruments.nth_to_default.price_nth_to_default_basket"
    if helper_ref not in refs:
        return None

    return textwrap.dedent(
        """\
        spec = self._spec
        if market_state.credit_curve is None:
            raise ValueError("market_state.credit_curve is required for nth-to-default pricing")
        if market_state.discount is None:
            raise ValueError("market_state.discount is required for nth-to-default pricing")
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        return price_nth_to_default_basket(
            notional=spec.notional,
            n_names=spec.n_names,
            n_th=spec.n_th,
            horizon=T,
            correlation=spec.correlation,
            recovery=spec.recovery,
            credit_curve=market_state.credit_curve,
            discount_curve=market_state.discount,
        )
        """
    ).rstrip()


def _declared_primitive_refs(generation_plan) -> set[str]:
    """Return every concrete primitive declared by the selected route."""
    primitive_plan = _generation_plan_field(generation_plan, "primitive_plan")
    return {
        f"{primitive.module}.{primitive.symbol}"
        for primitive in getattr(primitive_plan, "primitives", ()) or ()
        if not getattr(primitive, "excluded", False)
        and getattr(primitive, "module", "")
        and getattr(primitive, "symbol", "")
    }


def _scheduled_observation_return_evaluate_body(
    refs: set[str],
    *,
    normalized_target: str,
    generation_method: str,
) -> str | None:
    """Compose scheduled-return valuation from product-neutral primitives."""
    analytical_refs = {
        "trellis.models.observation_returns.ObservationReturnContract",
        "trellis.models.observation_returns.bounded_observation_return_sum",
        "trellis.models.analytical.support.expectations.gauss_hermite_product_expectation",
        "trellis.models.black.black76_call",
        "trellis.models.black.black76_put",
    }
    monte_carlo_refs = {
        "trellis.models.observation_returns.ObservationReturnContract",
        "trellis.models.observation_returns.observation_return_payoff",
        "trellis.models.processes.gbm.PiecewiseConstantGBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
    }
    use_monte_carlo = normalized_target == "monte_carlo" or (
        not normalized_target and generation_method == "monte_carlo"
    )
    if use_monte_carlo and monte_carlo_refs.issubset(refs):
        return textwrap.dedent(
            """\
            settlement = market_state.settlement or market_state.as_of
            if settlement is None:
                raise ValueError("scheduled observation returns require settlement or as_of")
            if market_state.discount is None:
                raise ValueError("scheduled observation returns require market_state.discount")
            if market_state.vol_surface is None:
                raise ValueError("scheduled observation returns require market_state.vol_surface")

            raw_times = getattr(spec, "observation_times", None)
            observation_dates = tuple(sorted(getattr(spec, "observation_dates", ()) or ()))
            time_day_count = (
                getattr(spec, "time_day_count", None)
                or getattr(spec, "day_count", DayCountConvention.ACT_365)
            )
            if raw_times:
                observation_times = tuple(sorted(float(time) for time in raw_times))
            else:
                observation_times = tuple(
                    float(year_fraction(settlement, observation_date, time_day_count))
                    for observation_date in observation_dates
                )
            if not observation_times:
                raise ValueError("scheduled observation returns require observation dates or times")

            option_type = str(getattr(spec, "option_type", "call") or "call").strip().lower()
            if option_type not in {"call", "put"}:
                raise ValueError(f"unsupported observation-return direction {option_type!r}")
            local_floor = getattr(spec, "local_floor", None)
            local_cap = getattr(spec, "local_cap", None)
            global_floor = getattr(spec, "global_floor", None)
            global_cap = getattr(spec, "global_cap", None)
            contract = ObservationReturnContract(
                observation_times=observation_times,
                direction="up" if option_type == "call" else "down",
                local_floor=0.0 if local_floor is None else float(local_floor),
                local_cap=float("inf") if local_cap is None else float(local_cap),
                global_floor=float("-inf") if global_floor is None else float(global_floor),
                global_cap=float("inf") if global_cap is None else float(global_cap),
                payoff_scale=float(spec.notional) * float(spec.spot),
            )
            maturity = observation_times[-1]
            if observation_dates:
                default_steps = max((observation_dates[-1] - settlement).days, 1)
            else:
                default_steps = max(int(round(maturity * 365.0)), 1)
            n_steps = max(int(getattr(spec, "n_steps", default_steps) or default_steps), 1)

            carry = getattr(spec, "dividend_yield", None)
            if carry is None:
                carry = getattr(spec, "dividend_rate", None)
            if carry is None:
                params = dict(getattr(market_state, "model_parameters", None) or {})
                carry_rates = dict(params.get("underlier_carry_rates") or {})
                carry = next(iter(carry_rates.values()), 0.0)
            carry = float(carry)
            interval_mus = []
            interval_sigmas = []
            previous_time = 0.0
            for observation_time in observation_times:
                tau = observation_time - previous_time
                interval_rate = float(
                    market_state.discount.zero_rate(max(observation_time, 1e-6))
                )
                interval_sigma = float(
                    market_state.vol_surface.black_vol(max(tau, 1e-6), float(spec.spot))
                )
                interval_mus.append(interval_rate - carry)
                interval_sigmas.append(interval_sigma)
                previous_time = observation_time
            rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
            process = PiecewiseConstantGBM(
                interval_ends=observation_times,
                mus=tuple(interval_mus),
                sigmas=tuple(interval_sigmas),
            )
            engine = MonteCarloEngine(
                process,
                n_paths=max(int(getattr(spec, "n_paths", 120000) or 120000), 2),
                n_steps=n_steps,
                seed=getattr(spec, "seed", 42),
                method="exact",
            )
            payoff = observation_return_payoff(
                contract,
                maturity=maturity,
                n_steps=n_steps,
            )
            result = engine.price(
                float(spec.spot),
                maturity,
                payoff,
                discount_rate=rate,
                return_paths=False,
            )
            return float(result["price"])
            """
        ).rstrip()

    use_analytical = normalized_target in {"", "analytical"} and generation_method in {
        "",
        "analytical",
    }
    if not use_analytical or not analytical_refs.issubset(refs):
        return None
    return textwrap.dedent(
        """\
        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError("scheduled observation returns require settlement or as_of")
        if market_state.discount is None:
            raise ValueError("scheduled observation returns require market_state.discount")
        if market_state.vol_surface is None:
            raise ValueError("scheduled observation returns require market_state.vol_surface")

        raw_times = getattr(spec, "observation_times", None)
        observation_dates = tuple(sorted(getattr(spec, "observation_dates", ()) or ()))
        time_day_count = (
            getattr(spec, "time_day_count", None)
            or getattr(spec, "day_count", DayCountConvention.ACT_365)
        )
        if raw_times:
            observation_times = tuple(sorted(float(time) for time in raw_times))
        else:
            observation_times = tuple(
                float(year_fraction(settlement, observation_date, time_day_count))
                for observation_date in observation_dates
            )
        if not observation_times:
            raise ValueError("scheduled observation returns require observation dates or times")

        option_type = str(getattr(spec, "option_type", "call") or "call").strip().lower()
        if option_type not in {"call", "put"}:
            raise ValueError(f"unsupported observation-return direction {option_type!r}")
        local_floor = getattr(spec, "local_floor", None)
        local_cap = getattr(spec, "local_cap", None)
        global_floor = getattr(spec, "global_floor", None)
        global_cap = getattr(spec, "global_cap", None)
        contract = ObservationReturnContract(
            observation_times=observation_times,
            direction="up" if option_type == "call" else "down",
            local_floor=0.0 if local_floor is None else float(local_floor),
            local_cap=float("inf") if local_cap is None else float(local_cap),
            global_floor=float("-inf") if global_floor is None else float(global_floor),
            global_cap=float("inf") if global_cap is None else float(global_cap),
        )
        carry = getattr(spec, "dividend_yield", None)
        if carry is None:
            carry = getattr(spec, "dividend_rate", None)
        if carry is None:
            params = dict(getattr(market_state, "model_parameters", None) or {})
            carry_rates = dict(params.get("underlier_carry_rates") or {})
            carry = next(iter(carry_rates.values()), 0.0)
        carry = float(carry)
        notional = float(spec.notional)
        spot = float(spec.spot)

        has_explicit_bounds = any(
            value is not None
            for value in (local_floor, local_cap, global_floor, global_cap)
        )
        if not has_explicit_bounds:
            total = 0.0
            previous_time = 0.0
            for observation_time in observation_times:
                tau = observation_time - previous_time
                rate = float(market_state.discount.zero_rate(max(observation_time, 1e-6)))
                sigma = float(market_state.vol_surface.black_vol(max(tau, 1e-6), spot))
                forward_ratio = exp((rate - carry) * tau)
                if option_type == "call":
                    optionlet = black76_call(forward_ratio, 1.0, sigma, tau)
                else:
                    optionlet = black76_put(forward_ratio, 1.0, sigma, tau)
                total += (
                    spot
                    * exp(-carry * previous_time)
                    * exp(-rate * tau)
                    * optionlet
                )
                previous_time = observation_time
            return float(notional * total)

        period_inputs = []
        previous_time = 0.0
        for observation_time in observation_times:
            tau = observation_time - previous_time
            rate = float(market_state.discount.zero_rate(max(observation_time, 1e-6)))
            sigma = float(market_state.vol_surface.black_vol(max(tau, 1e-6), spot))
            period_inputs.append((tau, rate, sigma))
            previous_time = observation_time

        def interval_payoff(normals):
            gross_returns = [
                exp(
                    (rate - carry - 0.5 * sigma * sigma) * tau
                    + sigma * sqrt(tau) * float(normal)
                )
                for normal, (tau, rate, sigma) in zip(normals, period_inputs)
            ]
            return bounded_observation_return_sum(gross_returns, contract)

        expected_return = gauss_hermite_product_expectation(
            interval_payoff,
            dimension=len(period_inputs),
            order=max(int(getattr(spec, "quadrature_order", 21) or 21), 3),
            max_nodes=max(
                int(getattr(spec, "max_quadrature_nodes", 2000000) or 2000000),
                1,
            ),
        )
        discount_factor = float(market_state.discount.discount(observation_times[-1]))
        return float(notional * spot * discount_factor * expected_return)
        """
    ).rstrip()


def _arithmetic_asian_evaluate_body(
    refs: set[str],
    *,
    normalized_target: str,
    generation_method: str,
) -> str | None:
    """Compose bounded arithmetic-Asian routes from product-neutral primitives."""
    common_refs = {
        "trellis.models.resolution.single_state_diffusion.resolve_single_state_diffusion_inputs",
        "trellis.core.date_utils.year_fraction",
    }
    monte_carlo_refs = common_refs | {
        "trellis.models.observation_aggregation.WeightedObservationContract",
        "trellis.models.observation_aggregation.weighted_observation_payoff",
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.path_state.StateAwarePayoff",
        "trellis.core.differentiable.get_numpy",
    }
    analytical_refs = common_refs | {
        "trellis.models.analytical.support.lognormal_moments.single_factor_lognormal_sum_contract",
        "trellis.models.analytical.support.lognormal_moments.weighted_lognormal_sum_moments",
        "trellis.models.analytical.support.lognormal_moments.match_lognormal_moments",
        "trellis.models.black.black76_call",
        "trellis.models.black.black76_put",
    }

    use_monte_carlo = normalized_target == "mc_asian" or (
        normalized_target in {"", "monte_carlo"}
        and generation_method == "monte_carlo"
    )
    if use_monte_carlo and monte_carlo_refs.issubset(refs):
        return textwrap.dedent(
            """\
            averaging_type = str(
                getattr(spec, "averaging_type", "arithmetic") or "arithmetic"
            ).strip().lower()
            if averaging_type != "arithmetic":
                raise ValueError(
                    "arithmetic Asian composition requires arithmetic averaging"
                )
            resolved = resolve_single_state_diffusion_inputs(market_state, spec)
            if resolved.maturity <= 0.0:
                intrinsic = (
                    max(resolved.strike - resolved.spot, 0.0)
                    if resolved.option_type == "put"
                    else max(resolved.spot - resolved.strike, 0.0)
                )
                return float(resolved.notional * intrinsic)

            settlement = market_state.settlement or market_state.as_of
            if settlement is None:
                raise ValueError(
                    "arithmetic Asian composition requires settlement or as_of"
                )
            observation_dates = tuple(
                getattr(spec, "observation_dates", ()) or ()
            )
            if observation_dates:
                if tuple(sorted(observation_dates)) != observation_dates:
                    raise ValueError(
                        "arithmetic Asian observation dates must be increasing"
                    )
                if observation_dates[-1] != spec.expiry_date:
                    raise ValueError(
                        "arithmetic Asian final observation must equal expiry_date"
                    )
                observation_times = tuple(
                    float(year_fraction(settlement, observation_date, spec.day_count))
                    for observation_date in observation_dates
                )
                if any(time <= 0.0 for time in observation_times):
                    raise ValueError(
                        "arithmetic Asian observations must be after settlement"
                    )
            else:
                observation_count = int(
                    getattr(spec, "n_observations", 0) or 0
                )
                if observation_count <= 0:
                    raise ValueError(
                        "arithmetic Asian composition requires observation_dates "
                        "or positive n_observations"
                    )
                if observation_count == 1:
                    observation_times = (0.0,)
                else:
                    observation_times = tuple(
                        resolved.maturity * index / (observation_count - 1)
                        for index in range(observation_count)
                    )

            observation_count = len(observation_times)
            observation_contract = WeightedObservationContract(
                observation_times=observation_times,
                weights=(1.0 / observation_count,) * observation_count,
            )
            configured_steps = getattr(spec, "n_steps", None)
            if configured_steps is None and not observation_dates:
                configured_steps = max(observation_count - 1, 1)
            n_steps = observation_contract.resolve_uniform_grid_steps(
                maturity=resolved.maturity,
                n_steps=(
                    None
                    if configured_steps is None
                    else max(int(configured_steps), 1)
                ),
                min_steps=max(observation_count - 1, 1),
                max_steps=int(
                    getattr(spec, "max_grid_steps", 4096) or 4096
                ),
            )

            np = get_numpy()
            if resolved.option_type == "put":
                settlement_fn = lambda average: np.maximum(
                    resolved.strike - average, 0.0
                )
            else:
                settlement_fn = lambda average: np.maximum(
                    average - resolved.strike, 0.0
                )
            payoff: StateAwarePayoff = weighted_observation_payoff(
                observation_contract,
                maturity=resolved.maturity,
                n_steps=n_steps,
                settlement_fn=settlement_fn,
                reducer_name="arithmetic_average",
                name="arithmetic_asian_payoff",
            )
            process = GBM(
                mu=resolved.rate - resolved.dividend_yield,
                sigma=resolved.sigma,
            )
            engine = MonteCarloEngine(
                process,
                n_paths=max(int(getattr(spec, "n_paths", 50000) or 50000), 2),
                n_steps=n_steps,
                seed=getattr(spec, "seed", 42),
                method="exact",
            )
            result = engine.price(
                resolved.spot,
                resolved.maturity,
                payoff,
                discount_rate=resolved.rate,
                return_paths=False,
            )
            return float(resolved.notional * result["price"])
            """
        ).rstrip()

    use_analytical = normalized_target == "turnbull_wakeman_approx" or (
        normalized_target in {"", "analytical"}
        and generation_method == "analytical"
    )
    if not use_analytical or not analytical_refs.issubset(refs):
        return None
    return textwrap.dedent(
        """\
        averaging_type = str(
            getattr(spec, "averaging_type", "arithmetic") or "arithmetic"
        ).strip().lower()
        if averaging_type != "arithmetic":
            raise ValueError(
                "arithmetic Asian composition requires arithmetic averaging"
            )
        resolved = resolve_single_state_diffusion_inputs(market_state, spec)
        if resolved.maturity <= 0.0:
            intrinsic = (
                max(resolved.strike - resolved.spot, 0.0)
                if resolved.option_type == "put"
                else max(resolved.spot - resolved.strike, 0.0)
            )
            return float(resolved.notional * intrinsic)

        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError(
                "arithmetic Asian composition requires settlement or as_of"
            )
        observation_dates = tuple(
            getattr(spec, "observation_dates", ()) or ()
        )
        if observation_dates:
            if tuple(sorted(observation_dates)) != observation_dates:
                raise ValueError(
                    "arithmetic Asian observation dates must be increasing"
                )
            if observation_dates[-1] != spec.expiry_date:
                raise ValueError(
                    "arithmetic Asian final observation must equal expiry_date"
                )
            observation_times = tuple(
                float(year_fraction(settlement, observation_date, spec.day_count))
                for observation_date in observation_dates
            )
            if any(time <= 0.0 for time in observation_times):
                raise ValueError(
                    "arithmetic Asian observations must be after settlement"
                )
        else:
            observation_count = int(
                getattr(spec, "n_observations", 0) or 0
            )
            if observation_count <= 0:
                raise ValueError(
                    "arithmetic Asian composition requires observation_dates "
                    "or positive n_observations"
                )
            if observation_count == 1:
                observation_times = (0.0,)
            else:
                observation_times = tuple(
                    resolved.maturity * index / (observation_count - 1)
                    for index in range(observation_count)
                )

        observation_count = len(observation_times)
        moments = weighted_lognormal_sum_moments(
            single_factor_lognormal_sum_contract(
                spot=resolved.spot,
                observation_times=observation_times,
                weights=(1.0 / observation_count,) * observation_count,
                carry=resolved.rate - resolved.dividend_yield,
                volatility=resolved.sigma,
            )
        )
        matched = match_lognormal_moments(moments)
        if resolved.strike <= 0.0:
            undiscounted = (
                0.0
                if resolved.option_type == "put"
                else float(matched.mean - resolved.strike)
            )
        elif resolved.option_type == "put":
            undiscounted = black76_put(
                matched.mean,
                resolved.strike,
                matched.effective_volatility(maturity=resolved.maturity),
                resolved.maturity,
            )
        else:
            undiscounted = black76_call(
                matched.mean,
                resolved.strike,
                matched.effective_volatility(maturity=resolved.maturity),
                resolved.maturity,
            )
        discount_factor = float(
            market_state.discount.discount(resolved.maturity)
        )
        return float(resolved.notional * discount_factor * undiscounted)
        """
    ).rstrip()


def _variance_swap_monte_carlo_evaluate_body(
    refs: set[str],
    *,
    normalized_target: str,
    generation_method: str,
) -> str | None:
    """Compose variance-swap Monte Carlo from product-neutral primitives."""
    required_refs = {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.models.monte_carlo.path_statistics.SquaredLogReturnContract",
        "trellis.models.monte_carlo.path_statistics.annualized_squared_log_return_sum",
        "trellis.models.monte_carlo.path_statistics.build_squared_log_return_reducer",
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.path_state.MonteCarloPathRequirement",
        "trellis.models.monte_carlo.path_state.StateAwarePayoff",
        "trellis.core.differentiable.get_numpy",
    }
    use_monte_carlo = normalized_target == "mc_variance_swap" or (
        normalized_target in {"", "monte_carlo"}
        and generation_method == "monte_carlo"
    )
    if not use_monte_carlo or not required_refs.issubset(refs):
        return None

    return textwrap.dedent(
        """\
        convention = str(
            getattr(spec, "annualization_convention", "per_year")
            or "per_year"
        ).strip().lower()
        if convention != "per_year":
            raise ValueError(
                "variance swap composition supports annualization_convention='per_year'"
            )

        resolved = resolve_scalar_diffusion_market_inputs(
            market_state,
            spec,
            volatility_coordinate=float(spec.spot),
        )
        notional = float(spec.notional)
        strike_variance = float(spec.strike_variance)
        realized_variance = float(
            getattr(spec, "realized_variance", 0.0) or 0.0
        )
        if resolved.spot <= 0.0:
            raise ValueError("variance swap Monte Carlo requires positive spot")
        if resolved.maturity <= 0.0:
            return float(
                notional * (realized_variance - strike_variance)
            )

        n_paths_value = getattr(spec, "n_paths", 60000)
        n_steps_value = getattr(spec, "n_steps", 252)
        seed_value = getattr(spec, "seed", 42)
        n_paths = 60000 if n_paths_value is None else int(n_paths_value)
        n_steps = 252 if n_steps_value is None else int(n_steps_value)
        seed = 42 if seed_value is None else int(seed_value)
        if n_paths < 2:
            raise ValueError("variance swap Monte Carlo requires at least two paths")
        if n_steps <= 0:
            raise ValueError("variance swap Monte Carlo requires positive n_steps")

        contract = SquaredLogReturnContract(
            n_steps=n_steps,
            observation_steps=tuple(range(n_steps + 1)),
            annualization_factor=1.0 / resolved.maturity,
        )
        reducer_name = "annualized_squared_log_returns"
        reducer = build_squared_log_return_reducer(
            contract,
            name=reducer_name,
        )
        np = get_numpy()
        engine_discount_rate = (
            -float(np.log(resolved.discount_factor))
            / resolved.maturity
        )

        def settle(future_variance):
            return (
                notional
                * (
                    realized_variance
                    + future_variance
                    - strike_variance
                )
            )

        payoff = StateAwarePayoff(
            path_requirement=MonteCarloPathRequirement(
                reducers=(reducer,)
            ),
            evaluate_paths_fn=lambda paths: settle(
                annualized_squared_log_return_sum(paths, contract)
            ),
            evaluate_state_fn=lambda state: settle(
                state.reduced_value(reducer_name)
            ),
            name="variance_swap_composed_payoff",
        )
        engine = MonteCarloEngine(
            GBM(
                mu=resolved.rate - resolved.dividend_yield,
                sigma=resolved.sigma,
            ),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            method="exact",
        )
        result = engine.price(
            resolved.spot,
            resolved.maturity,
            payoff,
            discount_rate=engine_discount_rate,
            return_paths=False,
        )
        return float(np.asarray(result["price"]))
        """
    ).rstrip()


def _fixed_lookback_monte_carlo_evaluate_body(
    refs: set[str],
    *,
    normalized_target: str,
    generation_method: str,
) -> str | None:
    """Compose fixed-lookback Monte Carlo from transition-state primitives."""
    required_refs = {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.models.analytical.support.normalized_option_type",
        "trellis.models.monte_carlo.transition_state.ConditionalBridgeExtremumContract",
        "trellis.models.monte_carlo.transition_state.build_conditional_bridge_extremum_reducer",
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.path_state.MonteCarloPathRequirement",
        "trellis.models.monte_carlo.path_state.StateAwarePayoff",
        "trellis.core.differentiable.get_numpy",
    }
    use_monte_carlo = normalized_target == "mc_lookback" or (
        normalized_target in {"", "monte_carlo"}
        and generation_method == "monte_carlo"
    )
    if not use_monte_carlo or not required_refs.issubset(refs):
        return None

    return textwrap.dedent(
        """\
        lookback_type = str(
            getattr(spec, "lookback_type", "fixed_strike")
            or "fixed_strike"
        ).strip().lower()
        if lookback_type != "fixed_strike":
            raise ValueError(
                "lookback Monte Carlo composition supports lookback_type='fixed_strike'"
            )
        monitoring_style = str(
            getattr(spec, "monitoring_style", "continuous")
            or "continuous"
        ).strip().lower()
        if monitoring_style != "continuous":
            raise ValueError(
                "lookback Monte Carlo composition requires monitoring_style='continuous'"
            )

        option_type = normalized_option_type(
            getattr(spec, "option_type", "call")
        )
        strike = float(spec.strike)
        resolved = resolve_scalar_diffusion_market_inputs(
            market_state,
            spec,
            volatility_coordinate=strike,
        )
        np = get_numpy()
        notional = float(spec.notional)
        if not np.isfinite(notional) or not np.isfinite(strike):
            raise ValueError("lookback notional and strike must be finite")
        if resolved.spot <= 0.0:
            raise ValueError("lookback Monte Carlo requires positive spot")

        running_value = getattr(spec, "running_extreme", None)
        running_extreme = (
            resolved.spot
            if running_value is None
            else float(running_value)
        )
        if not np.isfinite(running_extreme) or running_extreme <= 0.0:
            raise ValueError(
                "lookback Monte Carlo requires positive finite running_extreme"
            )

        def settle(extreme):
            if option_type == "put":
                intrinsic = np.maximum(strike - extreme, 0.0)
            else:
                intrinsic = np.maximum(extreme - strike, 0.0)
            return notional * intrinsic

        if resolved.maturity <= 0.0:
            expiry_extreme = (
                min(running_extreme, resolved.spot)
                if option_type == "put"
                else max(running_extreme, resolved.spot)
            )
            return float(np.asarray(settle(expiry_extreme)))

        n_paths_value = getattr(spec, "n_paths", 80000)
        n_steps_value = getattr(spec, "n_steps", 96)
        seed_value = getattr(spec, "seed", 42)
        n_paths = 80000 if n_paths_value is None else int(n_paths_value)
        n_steps = 96 if n_steps_value is None else int(n_steps_value)
        seed = None if seed_value is None else int(seed_value)
        if n_paths < 2:
            raise ValueError("lookback Monte Carlo requires at least two paths")
        if n_steps <= 0:
            raise ValueError("lookback Monte Carlo requires positive n_steps")

        direction = "minimum" if option_type == "put" else "maximum"
        reducer_name = "continuous_running_extreme"
        reducer = build_conditional_bridge_extremum_reducer(
            ConditionalBridgeExtremumContract(
                n_steps=n_steps,
                transition_steps=tuple(range(1, n_steps + 1)),
                direction=direction,
                initial_extremum=running_extreme,
            ),
            name=reducer_name,
        )

        def reject_discrete_paths(_paths):
            raise ValueError(
                "continuous fixed-lookback monitoring requires transition state"
            )

        payoff = StateAwarePayoff(
            path_requirement=MonteCarloPathRequirement(
                transition_reducers=(reducer,)
            ),
            evaluate_paths_fn=reject_discrete_paths,
            evaluate_state_fn=lambda state: settle(
                state.reduced_value(reducer_name)
            ),
            name="fixed_lookback_composed_payoff",
        )
        engine_discount_rate = (
            -float(np.log(resolved.discount_factor))
            / resolved.maturity
        )
        result = MonteCarloEngine(
            GBM(
                mu=resolved.rate - resolved.dividend_yield,
                sigma=resolved.sigma,
            ),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            method="exact",
        ).price(
            resolved.spot,
            resolved.maturity,
            payoff,
            discount_rate=engine_discount_rate,
            return_paths=False,
        )
        price = float(np.asarray(result["price"]))
        standard_error = float(np.asarray(result["std_error"]))
        if (
            not np.isfinite(price)
            or not np.isfinite(standard_error)
            or standard_error < 0.0
        ):
            raise ValueError("lookback Monte Carlo produced invalid estimator diagnostics")
        return price
        """
    ).rstrip()


def _deterministic_exact_binding_evaluate_body(
    generation_plan,
    *,
    semantic_blueprint=None,
    request_metadata: Mapping[str, object] | None = None,
    comparison_target: str | None = None,
) -> str | None:
    """Return a deterministic evaluate body for supported exact-bound routes."""
    metadata = dict(request_metadata or {})
    refs = set(_exact_binding_refs(generation_plan))
    refs.update(_declared_primitive_refs(generation_plan))
    swaption_comparison_kwargs = _swaption_comparison_helper_kwargs(semantic_blueprint)
    swaption_tree_model = _swaption_tree_model_name(semantic_blueprint, comparison_target)
    vanilla_equity_mc_kwargs = _vanilla_equity_monte_carlo_helper_kwargs(comparison_target)
    vanilla_equity_mc_call_kwargs = (
        vanilla_equity_mc_kwargs.lstrip(", ")
        or 'scheme="exact", variance_reduction="none"'
    )
    vanilla_equity_mc_control_kwargs = ""
    if 'variance_reduction="control_variate"' in vanilla_equity_mc_call_kwargs:
        vanilla_equity_mc_control_kwargs = (
            "    control_variate_values=lambda terminal, _resolved: terminal,\n"
            "    control_variate_expected=lambda resolved: float(\n"
            "        resolved.spot * exp(\n"
            "            (resolved.rate - resolved.dividend_yield) * resolved.maturity\n"
            "        )\n"
            "    ),\n"
        )
    vanilla_equity_pde_kwargs = _vanilla_equity_pde_helper_kwargs(comparison_target)
    vanilla_equity_pde_theta = (
        "1.0"
        if str(comparison_target or "").strip().lower().replace("-", "_")
        in {"theta_1.0", "theta_1_0", "implicit"}
        else "0.5"
    )
    heston_mc_kwargs = _heston_monte_carlo_helper_kwargs(comparison_target)
    vanilla_equity_transform_kwargs = _vanilla_equity_transform_helper_kwargs(comparison_target)
    heston_transform_kwargs = _heston_transform_helper_kwargs(comparison_target)
    zcb_option_tree_kwargs = _zcb_option_tree_helper_kwargs(comparison_target)
    credit_basket_tranche_kwargs = _credit_basket_tranche_helper_kwargs(comparison_target)
    runtime_contract = metadata.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        runtime_contract = {}
    instrument_type = str(
        _generation_plan_field(generation_plan, "instrument_type", "")
        or metadata.get("instrument_type")
        or runtime_contract.get("instrument_type")
        or runtime_contract.get("product_family")
        or ""
    ).strip().lower()
    route_free_exact_binding = _generation_plan_field(generation_plan, "primitive_plan") is None
    normalized_target = str(
        comparison_target or metadata.get("comparison_target") or ""
    ).strip().lower().replace("-", "_")
    raw_target_contract = metadata.get("comparison_target_contract")
    target_variants = (
        dict(raw_target_contract.get("variant_parameters") or {})
        if isinstance(raw_target_contract, Mapping)
        else {}
    )
    callable_bond_binding_refs = {
        "trellis.models.callable_bond_pde.price_callable_bond_pde",
    }
    callable_primitive_plan = _generation_plan_field(
        generation_plan,
        "primitive_plan",
    )
    callable_primitive_route = str(
        getattr(callable_primitive_plan, "route", "") or ""
    ).strip()
    has_callable_bond_lattice_composition = (
        instrument_type in {"callable_bond", "puttable_bond"}
        and callable_primitive_route == "exercise_lattice"
        and "trellis.models.trees.algebra.price_on_lattice" in refs
    )
    has_callable_bond_exact_binding = (
        bool(refs & callable_bond_binding_refs)
        or has_callable_bond_lattice_composition
    )
    callable_bond_calibration_kwargs: list[str] = []
    if has_callable_bond_exact_binding:
        for variant_name in ("mean_reversion", "sigma"):
            if variant_name not in target_variants:
                continue
            try:
                variant_value = float(target_variants[variant_name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Callable-bond {variant_name} must be a finite number"
                ) from exc
            if not isfinite(variant_value):
                raise ValueError(
                    f"Callable-bond {variant_name} must be a finite number"
                )
            callable_bond_calibration_kwargs.append(
                f"{variant_name}={variant_value!r}"
            )
    declared_callable_bond_parameter_set = str(
        target_variants.get("model_parameter_set") or ""
    ).strip()
    if (
        has_callable_bond_exact_binding
        and declared_callable_bond_parameter_set
        and not {"mean_reversion", "sigma"}.issubset(target_variants)
    ):
        raise ValueError(
            "Callable-bond model_parameter_set requires explicit "
            "mean_reversion and sigma variants"
        )
    raw_callable_bond_lattice_model = target_variants.get("lattice_model")
    callable_bond_lattice_model = str(
        raw_callable_bond_lattice_model
        if raw_callable_bond_lattice_model is not None
        else "hull_white"
    ).strip().lower()
    if (
        has_callable_bond_lattice_composition
        and callable_bond_lattice_model not in {"bdt", "hull_white"}
    ):
        raise ValueError(
            "Callable-bond lattice_model must be one of: bdt, hull_white"
        )
    callable_bond_pde_theta = 0.5
    if "trellis.models.callable_bond_pde.price_callable_bond_pde" in refs:
        try:
            callable_bond_pde_theta = float(target_variants.get("theta", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Callable-bond PDE theta must be a number between 0 and 1"
            ) from exc
        if not 0.0 <= callable_bond_pde_theta <= 1.0:
            raise ValueError(
                "Callable-bond PDE theta must be a number between 0 and 1"
            )
    is_american_equity_option = instrument_type in {"american_put", "american_option"}
    black_scholes_vanilla_exact_binding = _is_black_scholes_vanilla_exact_binding(
        generation_plan,
        refs,
        instrument_type=instrument_type,
        normalized_target=normalized_target,
        route_free_exact_binding=route_free_exact_binding,
    )
    primitive_plan = _generation_plan_field(generation_plan, "primitive_plan")
    primitive_route = str(getattr(primitive_plan, "route", "") or "").strip()
    generation_method = str(
        _generation_plan_field(generation_plan, "method", "") or ""
    ).strip().lower().replace("-", "_")
    scheduled_return_body = _scheduled_observation_return_evaluate_body(
        refs,
        normalized_target=normalized_target,
        generation_method=generation_method,
    )
    if scheduled_return_body is not None:
        return scheduled_return_body
    if instrument_type == "asian_option":
        asian_body = _arithmetic_asian_evaluate_body(
            refs,
            normalized_target=normalized_target,
            generation_method=generation_method,
        )
        if asian_body is not None:
            return asian_body
    if instrument_type == "lookback_option":
        lookback_body = _fixed_lookback_monte_carlo_evaluate_body(
            refs,
            normalized_target=normalized_target,
            generation_method=generation_method,
        )
        if lookback_body is not None:
            return lookback_body
    if instrument_type == "variance_swap":
        variance_swap_body = _variance_swap_monte_carlo_evaluate_body(
            refs,
            normalized_target=normalized_target,
            generation_method=generation_method,
        )
        if variance_swap_body is not None:
            return variance_swap_body
    if (
        instrument_type in {"swaption", "bermudan_swaption"}
        and primitive_route == "exercise_lattice"
        and {
            "trellis.models.bermudan_swaption_tree.resolve_bermudan_swaption_tree_inputs",
            "trellis.models.trees.algebra.build_lattice",
            "trellis.models.trees.algebra.value_on_lattice",
            "trellis.models.trees.algebra.price_on_lattice",
        }.issubset(refs)
    ):
        return _bermudan_swaption_lattice_evaluate_body(
            model_name=swaption_tree_model,
            comparison_kwargs=swaption_comparison_kwargs,
        )
    if (
        instrument_type == "swaption"
        and primitive_route == "rate_tree_backward_induction"
        and "trellis.models.trees.algebra.price_on_lattice" in refs
    ):
        return textwrap.dedent(
            f"""\
            if spec.swap_start != spec.expiry_date:
                raise ValueError(
                    "Rate-tree swaption composition requires swap_start == expiry_date "
                    "for the single-exercise forward-start contract."
                )
            curve_basis_spread = resolve_swaption_curve_basis_spread(market_state, spec)
            tree_spec = BermudanSwaptionTreeSpec(
                notional=float(spec.notional),
                strike=float(spec.strike) - float(curve_basis_spread),
                exercise_dates=(spec.expiry_date,),
                swap_end=spec.swap_end,
                swap_frequency=spec.swap_frequency,
                day_count=spec.day_count,
                rate_index=spec.rate_index,
                is_payer=bool(spec.is_payer),
            )
            tree_steps = getattr(spec, "tree_steps", getattr(spec, "n_steps", None))
            resolved = resolve_bermudan_swaption_tree_inputs(
                market_state,
                tree_spec,
                model="{swaption_tree_model}"{swaption_comparison_kwargs},
                n_steps=tree_steps,
            )
            lattice = build_lattice(
                BINOMIAL_1F_TOPOLOGY,
                UNIFORM_ADDITIVE_MESH,
                "{swaption_tree_model}",
                calibration_target=TERM_STRUCTURE_TARGET(market_state.discount),
                r0=resolved.r0,
                sigma=resolved.sigma,
                a=resolved.mean_reversion,
                T=resolved.tree_horizon,
                n_steps=resolved.n_steps,
            )
            lattice_contract = compile_bermudan_swaption_contract_spec(
                lattice,
                spec=tree_spec,
                settlement=resolved.settlement,
            )
            return float(price_on_lattice(lattice, lattice_contract))
            """
        ).rstrip()
    if (
        primitive_route == "equity_quanto"
        and {
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
        }.issubset(refs)
    ):
        return textwrap.dedent(
            """\
            resolved = resolve_quanto_inputs(market_state, self._spec)
            option_type = normalized_option_type(self._spec.option_type)
            if resolved.T <= 0.0:
                return float(self._spec.notional) * float(
                    terminal_intrinsic(
                        option_type,
                        spot=resolved.spot,
                        strike=self._spec.strike,
                    )
                )

            forward = quanto_adjusted_forward(
                spot=resolved.spot,
                domestic_df=resolved.domestic_df,
                foreign_df=resolved.foreign_df,
                corr=resolved.corr,
                sigma_underlier=resolved.sigma_underlier,
                sigma_fx=resolved.sigma_fx,
                T=resolved.T,
            )
            if option_type == "call":
                undiscounted = black76_call(
                    forward,
                    self._spec.strike,
                    resolved.sigma_underlier,
                    resolved.T,
                )
            else:
                undiscounted = black76_put(
                    forward,
                    self._spec.strike,
                    resolved.sigma_underlier,
                    resolved.T,
                )
            return float(
                discounted_value(
                    undiscounted,
                    resolved.domestic_df,
                    scale=self._spec.notional,
                )
            )
            """
        ).rstrip()
    if (
        primitive_route == "equity_quanto"
        and "trellis.models.monte_carlo.engine.MonteCarloEngine" in refs
    ):
        sampling_setup = ""
        shocks_argument = ""
        if generation_method == "qmc":
            sampling_setup = textwrap.dedent(
                """\
                requested_paths = max(int(getattr(self._spec, "n_paths", 50000)), 1)
                n_paths = 1 << (requested_paths - 1).bit_length()
                seed = int(getattr(self._spec, "seed", 42))
                shocks = sobol_normals(
                    n_paths,
                    n_steps,
                    n_factors=2,
                    seed=seed,
                )
                """
            ).rstrip()
            shocks_argument = "    shocks=shocks,"
        else:
            sampling_setup = textwrap.dedent(
                """\
                n_paths = max(int(getattr(self._spec, "n_paths", 50000)), 1)
                seed = int(getattr(self._spec, "seed", 42))
                """
            ).rstrip()
        body = textwrap.dedent(
            """\
            resolved = resolve_quanto_inputs(market_state, self._spec)
            option_type = normalized_option_type(self._spec.option_type)
            if resolved.T <= 0.0:
                return float(self._spec.notional) * float(
                    terminal_intrinsic(
                        option_type,
                        spot=resolved.spot,
                        strike=self._spec.strike,
                    )
                )

            domestic_rate = float(implied_zero_rate(resolved.domestic_df, resolved.T))
            foreign_rate = float(implied_zero_rate(resolved.foreign_df, resolved.T))
            underlier_drift = (
                domestic_rate
                - foreign_rate
                - resolved.corr * resolved.sigma_underlier * resolved.sigma_fx
            )
            process = CorrelatedGBM(
                mu=[underlier_drift, domestic_rate - foreign_rate],
                sigma=[resolved.sigma_underlier, resolved.sigma_fx],
                corr=[
                    [1.0, resolved.corr],
                    [resolved.corr, 1.0],
                ],
            )
            payoff = terminal_value_payoff(
                lambda terminal: float(self._spec.notional) * terminal_intrinsic(
                    option_type,
                    spot=terminal[..., 0],
                    strike=self._spec.strike,
                ),
                name="quanto_terminal",
            )
            n_steps = max(int(getattr(self._spec, "n_steps", 252)), 1)
            __SAMPLING_SETUP__
            engine = MonteCarloEngine(
                process,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                method="exact",
            )
            result = engine.price(
                get_numpy().array([resolved.spot, resolved.fx_spot], dtype=float),
                float(resolved.T),
                payoff,
                discount_rate=domestic_rate,
                return_paths=False,
            __SHOCKS_ARGUMENT__
            )
            return float(result["price"])
            """
        ).rstrip()
        return body.replace("__SAMPLING_SETUP__", sampling_setup).replace(
            "__SHOCKS_ARGUMENT__",
            shocks_argument,
        )
    if (
        primitive_route == "analytical_garman_kohlhagen"
        and "trellis.models.analytical.fx.garman_kohlhagen_price_raw" in refs
    ):
        return textwrap.dedent(
            """\
            resolved = resolve_fx_vanilla_inputs(market_state, self._spec)
            return float(resolved.notional) * float(
                garman_kohlhagen_price_raw(
                    resolved.option_type,
                    resolved.garman_kohlhagen,
                )
            )
            """
        ).rstrip()
    if (
        primitive_route == "monte_carlo_fx_vanilla"
        and "trellis.models.monte_carlo.engine.MonteCarloEngine" in refs
    ):
        return textwrap.dedent(
            """\
            resolved = resolve_fx_vanilla_inputs(market_state, self._spec)
            gk = resolved.garman_kohlhagen
            if gk.T <= 0.0:
                return float(resolved.notional) * float(
                    terminal_intrinsic(
                        resolved.option_type,
                        spot=gk.spot,
                        strike=gk.strike,
                    )
                )

            payoff = terminal_value_payoff(
                lambda terminal: resolved.notional * terminal_intrinsic(
                    resolved.option_type,
                    spot=terminal,
                    strike=gk.strike,
                ),
                name="fx_vanilla_terminal",
            )
            process = GBM(
                mu=resolved.domestic_rate - resolved.foreign_rate,
                sigma=float(gk.sigma),
            )
            engine = MonteCarloEngine(
                process,
                n_paths=max(int(getattr(self._spec, "n_paths", 50000)), 1),
                n_steps=max(int(getattr(self._spec, "n_steps", 252)), 1),
                seed=int(getattr(self._spec, "seed", 42)),
                method="exact",
            )
            result = engine.price(
                float(gk.spot),
                float(gk.T),
                payoff,
                discount_rate=float(resolved.domestic_rate),
                return_paths=False,
            )
            return float(result["price"])
            """
        ).rstrip()
    if (
        primitive_route == "analytical_fx_barrier"
        and "trellis.models.analytical.barrier.barrier_option_price" in refs
    ):
        return textwrap.dedent(
            """\
            resolved = resolve_fx_barrier_inputs(market_state, self._spec)
            unit_price = barrier_option_price(
                resolved.spot,
                resolved.strike,
                resolved.barrier,
                resolved.domestic_rate,
                resolved.sigma,
                resolved.maturity,
                barrier_type=resolved.barrier_type,
                option_type=resolved.option_type,
                rebate=resolved.rebate,
                q=resolved.foreign_rate,
                observations_per_year=resolved.observations_per_year,
            )
            return float(resolved.notional) * float(unit_price)
            """
        ).rstrip()
    if (
        primitive_route == "monte_carlo_fx_barrier"
        and "trellis.models.monte_carlo.engine.MonteCarloEngine" in refs
    ):
        return textwrap.dedent(
            """\
            resolved = resolve_fx_barrier_inputs(market_state, self._spec)
            direction = "down" if resolved.barrier_type.startswith("down") else "up"
            knock = "in" if resolved.barrier_type.endswith("_in") else "out"
            initial_touched = (
                resolved.spot <= resolved.barrier
                if direction == "down"
                else resolved.spot >= resolved.barrier
            )
            if resolved.maturity <= 0.0:
                active = initial_touched if knock == "in" else not initial_touched
                intrinsic = terminal_intrinsic(
                    resolved.option_type,
                    spot=resolved.spot,
                    strike=resolved.strike,
                )
                payoff_at_expiry = intrinsic if active else resolved.rebate
                return float(resolved.notional * payoff_at_expiry)

            observation_steps = ()
            if resolved.observations_per_year is not None:
                observation_count = max(
                    int(round(resolved.maturity * resolved.observations_per_year)),
                    1,
                )
                observation_steps = (
                    0,
                    *tuple(
                        sorted(
                            {
                                max(
                                    1,
                                    min(
                                        resolved.n_steps,
                                        int(round(index * resolved.n_steps / observation_count)),
                                    ),
                                )
                                for index in range(1, observation_count + 1)
                            }
                        )
                    ),
                )
            monitor = BarrierMonitor(
                name="barrier",
                level=resolved.barrier,
                direction=direction,
                observation_steps=observation_steps,
            )
            requirement = MonteCarloPathRequirement(barrier_monitors=(monitor,))

            def apply_barrier(terminal, touched):
                intrinsic = terminal_intrinsic(
                    resolved.option_type,
                    spot=terminal,
                    strike=resolved.strike,
                )
                active = touched if knock == "in" else ~touched
                return resolved.notional * raw_np.where(active, intrinsic, resolved.rebate)

            def evaluate_paths(paths):
                observed = paths[:, observation_steps] if observation_steps else paths
                touched = (
                    raw_np.any(observed <= resolved.barrier, axis=1)
                    if direction == "down"
                    else raw_np.any(observed >= resolved.barrier, axis=1)
                )
                return apply_barrier(paths[:, -1], touched)

            def evaluate_state(state):
                return apply_barrier(
                    state.terminal_values,
                    state.barrier_hit(monitor.name),
                )

            payoff = StateAwarePayoff(
                path_requirement=requirement,
                evaluate_paths_fn=evaluate_paths,
                evaluate_state_fn=evaluate_state,
                name="fx_single_barrier",
            )
            process = GBM(
                mu=resolved.domestic_rate - resolved.foreign_rate,
                sigma=resolved.sigma,
            )
            engine = MonteCarloEngine(
                process,
                n_paths=resolved.n_paths,
                n_steps=resolved.n_steps,
                seed=resolved.seed,
                method="exact",
            )
            result = engine.price(
                resolved.spot,
                resolved.maturity,
                payoff,
                discount_rate=resolved.domestic_rate,
                return_paths=False,
            )
            return float(result["price"])
            """
        ).rstrip()

    if instrument_type in {"credit_default_swap", "cds"} and normalized_target in {"mc_cds", "cds_mc"}:
        cds_body = _credit_default_swap_helper_body(
            {"trellis.models.credit_default_swap.price_cds_monte_carlo"}
        )
        if cds_body is not None:
            return cds_body
    if instrument_type in {"credit_default_swap", "cds"} and normalized_target in {
        "analytical_cds",
        "cds_analytical",
    }:
        cds_body = _credit_default_swap_helper_body(
            {"trellis.models.credit_default_swap.price_cds_analytical"}
        )
        if cds_body is not None:
            return cds_body

    target_helper_bodies = {
        "vasicek_tree": (
            "return price_short_rate_zero_coupon_bond_tree("
            'market_state, spec, model="vasicek", '
            'n_steps=getattr(spec, "tree_steps", 360), '
            "allow_benchmark_defaults=True)"
        ),
        "cir_tree": (
            "return price_short_rate_zero_coupon_bond_tree("
            'market_state, spec, model="cir", '
            'n_steps=getattr(spec, "tree_steps", 360), '
            "allow_benchmark_defaults=True)"
        ),
        "vasicek_analytical": (
            "return price_short_rate_zero_coupon_bond_analytical("
            'market_state, spec, model="vasicek", allow_benchmark_defaults=True)'
        ),
        "cir_analytical": (
            "return price_short_rate_zero_coupon_bond_analytical("
            'market_state, spec, model="cir", allow_benchmark_defaults=True)'
        ),
        "heston_mc": (
            "return price_heston_option_monte_carlo("
            f"market_state, spec{heston_mc_kwargs})"
        ),
        "merton_mc": (
            "return price_merton_jump_diffusion_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 100000), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "merton_fft": (
            "return price_merton_jump_diffusion_option_transform("
            'market_state, spec, method="fft")'
        ),
        "merton_cos": (
            "return price_merton_jump_diffusion_option_transform("
            'market_state, spec, method="cos")'
        ),
        "sabr_mc": (
            "return price_sabr_forward_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 120000), '
            'n_steps=getattr(spec, "n_steps", 96), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "sabr_hagan_analytical": (
            "return price_sabr_forward_option_hagan(market_state, spec)"
        ),
        "vg_cos": (
            "return price_variance_gamma_option_transform("
            'market_state, spec, method="cos")'
        ),
        "vg_mc": (
            "return price_variance_gamma_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 120000), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "madan_carr_chang_reference": (
            "return price_variance_gamma_option_reference(market_state, spec)"
        ),
        "cgmy_cos": (
            "return price_cgmy_option_transform("
            'market_state, spec, method="cos")'
        ),
        "cgmy_mc": (
            "return price_cgmy_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 120000), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "cgmy_reference_values": (
            "return price_cgmy_option_reference(market_state, spec)"
        ),
        "kou_fft": (
            "return price_kou_option_transform("
            'market_state, spec, method="fft")'
        ),
        "kou_mc": (
            "return price_kou_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 120000), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "kou_reference_values": (
            "return price_kou_option_reference(market_state, spec)"
        ),
        "bates_fft": (
            "return price_bates_option_transform("
            'market_state, spec, method="fft")'
        ),
        "bates_mc": (
            "return price_bates_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 80000), '
            'n_steps=getattr(spec, "n_steps", 96), '
            'seed=getattr(spec, "seed", 42))'
        ),
    }
    if normalized_target in target_helper_bodies:
        return target_helper_bodies[normalized_target]

    if is_american_equity_option and normalized_target == "psor_pde":
        return (
            "return price_event_aware_equity_option_pde("
            "market_state, spec, theta=1.0, "
            'n_x=getattr(spec, "n_x", 301), '
            'n_t=getattr(spec, "n_t", 400))'
        )
    if is_american_equity_option and normalized_target == "crr_tree":
        control_obligations = tuple(
            _generation_plan_field(generation_plan, "lane_control_obligations", ()) or ()
        )
        return _american_equity_tree_primitive_body(
            normalized_target,
            include_bermudan_schedule="exercise_style:bermudan" in control_obligations,
        )
    if is_american_equity_option and normalized_target == "high_step_tree_2000":
        control_obligations = tuple(
            _generation_plan_field(generation_plan, "lane_control_obligations", ()) or ()
        )
        return _american_equity_tree_primitive_body(
            normalized_target,
            include_bermudan_schedule="exercise_style:bermudan" in control_obligations,
        )
    if is_american_equity_option and normalized_target == "lsm_mc":
        return (
            "resolved = resolve_single_state_monte_carlo_inputs(\n"
            "    market_state, spec, scheme=\"exact\", variance_reduction=\"none\",\n"
            '    n_paths=getattr(spec, "n_paths", 50000),\n'
            '    n_steps=getattr(spec, "n_steps", 96),\n'
            '    seed=getattr(spec, "seed", 42),\n'
            ")\n"
            "if resolved.maturity <= 0.0:\n"
            "    return float(resolved.notional * terminal_intrinsic_from_resolved(\n"
            "        resolved.spot, resolved\n"
            "    ))\n"
            "process = GBM(\n"
            "    mu=resolved.rate - resolved.dividend_yield, sigma=resolved.sigma\n"
            ")\n"
            "engine = MonteCarloEngine(\n"
            "    process, n_paths=resolved.n_paths, n_steps=resolved.n_steps,\n"
            "    seed=resolved.seed, method=\"exact\",\n"
            ")\n"
            "paths = engine.simulate(resolved.spot, resolved.maturity)\n"
            'exercise_style = str(getattr(spec, "exercise_style", "american")).strip().lower()\n'
            'if exercise_style == "american":\n'
            "    exercise_steps = list(range(1, resolved.n_steps + 1))\n"
            'elif exercise_style == "bermudan":\n'
            "    if not spec.exercise_dates:\n"
            '        raise ValueError("Bermudan LSM requires spec.exercise_dates")\n'
            "    settlement = market_state.settlement or market_state.as_of\n"
            "    if settlement is None:\n"
            '        raise ValueError("Bermudan LSM requires market settlement or as_of")\n'
            "    event_times = []\n"
            "    for exercise_date in spec.exercise_dates:\n"
            "        exercise_time = float(year_fraction(settlement, exercise_date, spec.day_count))\n"
            "        if 0.0 <= exercise_time <= resolved.maturity:\n"
            "            event_times.append(exercise_time)\n"
            "    if not event_times:\n"
            '        raise ValueError("Bermudan LSM resolved no valid exercise dates")\n'
            "    exercise_steps = list(event_step_indices(\n"
            "        tuple(event_times), resolved.maturity, resolved.n_steps\n"
            "    ))\n"
            "    exercise_steps = sorted({max(int(step), 1) for step in exercise_steps})\n"
            'elif exercise_style == "european":\n'
            "    exercise_steps = [resolved.n_steps]\n"
            "else:\n"
            '    raise ValueError(f"Unsupported exercise_style {exercise_style!r}")\n'
            "basis = LaguerreBasis(degree=2)\n"
            "payoff_fn = lambda spots: terminal_intrinsic_from_resolved(spots, resolved)\n"
            "price = longstaff_schwartz(\n"
            "    paths, exercise_steps, payoff_fn, discount_rate=resolved.rate,\n"
            "    dt=resolved.maturity / resolved.n_steps,\n"
            "    basis_fn=lambda spots: basis(spots / max(resolved.strike, 1e-12)),\n"
            ")\n"
            "return float(resolved.notional * price)"
        )
    if normalized_target == "cev_pde":
        return (
            "return price_cev_option_pde("
            "market_state, spec, "
            'n_x=getattr(spec, "n_x", 401), '
            'n_t=getattr(spec, "n_t", 501))'
        )
    if normalized_target == "cev_tree":
        return (
            "return price_cev_option_tree("
            "market_state, spec, "
            'n_steps=getattr(spec, "tree_steps", 2000), '
            'n_x=getattr(spec, "tree_grid_size", 301))'
        )

    if (
        normalized_target in {"cn_rannacher", "cn_standard"}
        and "trellis.models.equity_option_pde.price_equity_digital_option_pde" in refs
    ):
        rannacher_default = 2 if normalized_target == "cn_rannacher" else 0
        return (
            "return price_equity_digital_option_pde("
            "market_state, spec, "
            'theta=getattr(spec, "theta", 0.5), '
            f'rannacher_timesteps=getattr(spec, "rannacher_timesteps", {rannacher_default}), '
            'n_x=getattr(spec, "n_x", 401), '
            'n_t=getattr(spec, "n_t", 401))'
        )
    if (
        normalized_target in {"digital_fft", "digital_cos"}
        and "trellis.models.equity_option_transforms.price_equity_digital_option_transform" in refs
    ):
        return (
            "return price_equity_digital_option_transform("
            f"market_state, spec{vanilla_equity_transform_kwargs})"
        )

    cds_body = _credit_default_swap_helper_body(refs)
    if cds_body is not None:
        return cds_body

    if instrument_type == "basket_option":
        terminal_basket_body = _terminal_basket_primitive_body(
            refs,
            normalized_target=normalized_target,
            generation_method=generation_method,
        )
        if terminal_basket_body is not None:
            return terminal_basket_body

    ranked_basket_body = _ranked_observation_basket_primitive_body(refs)
    if ranked_basket_body is not None:
        return ranked_basket_body

    nth_to_default_body = _nth_to_default_helper_body(refs)
    if nth_to_default_body is not None:
        return nth_to_default_body

    if (
        comparison_target == "analytical"
        and instrument_type in {"cap", "floor", "period_rate_option_strip"}
        and "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical" in refs
    ):
        default_kind = "floor" if instrument_type == "floor" else "cap"
        return textwrap.dedent(
            f"""\
            spec = self._spec
            return price_rate_cap_floor_strip_analytical(
                market_state=market_state,
                instrument_class=getattr(spec, "instrument_class", None) or "{default_kind}",
                coupon_dates=getattr(spec, "coupon_dates", None) or getattr(spec, "payment_dates", None),
                accrual_dates=getattr(spec, "accrual_dates", None),
                cap_strike=getattr(spec, "cap_strike", None),
                floor_strike=getattr(spec, "floor_strike", None),
                call_price=getattr(spec, "call_price", None) or getattr(spec, "call_strike", None),
                exercise_dates=getattr(spec, "exercise_dates", None) or getattr(spec, "call_dates", None),
                is_payer=getattr(spec, "is_payer", None),
                notional=getattr(spec, "notional", None),
                strike=(
                    getattr(spec, "strike", None)
                    or getattr(spec, "cap_strike", None)
                    or getattr(spec, "floor_strike", None)
                ),
                start_date=getattr(spec, "start_date", None),
                end_date=getattr(spec, "end_date", None),
                frequency=getattr(spec, "frequency", None),
                day_count=getattr(spec, "day_count", None),
                rate_index=getattr(spec, "rate_index", None),
                calendar_name=getattr(spec, "calendar_name", None),
                business_day_adjustment=getattr(spec, "business_day_adjustment", None),
                model=getattr(spec, "model", None),
                shift=getattr(spec, "shift", None),
                sabr=getattr(spec, "sabr", None),
            )
            """
        ).rstrip()
    if (
        comparison_target == "monte_carlo"
        and instrument_type in {"cap", "floor", "period_rate_option_strip"}
        and "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo" in refs
    ):
        default_kind = "floor" if instrument_type == "floor" else "cap"
        return textwrap.dedent(
            f"""\
            spec = self._spec
            return price_rate_cap_floor_strip_monte_carlo(
                market_state=market_state,
                instrument_class=getattr(spec, "instrument_class", None) or "{default_kind}",
                coupon_dates=getattr(spec, "coupon_dates", None) or getattr(spec, "payment_dates", None),
                accrual_dates=getattr(spec, "accrual_dates", None),
                cap_strike=getattr(spec, "cap_strike", None),
                floor_strike=getattr(spec, "floor_strike", None),
                call_price=getattr(spec, "call_price", None) or getattr(spec, "call_strike", None),
                exercise_dates=getattr(spec, "exercise_dates", None) or getattr(spec, "call_dates", None),
                is_payer=getattr(spec, "is_payer", None),
                n_paths=20000,
                seed=42,
                notional=getattr(spec, "notional", None),
                strike=(
                    getattr(spec, "strike", None)
                    or getattr(spec, "cap_strike", None)
                    or getattr(spec, "floor_strike", None)
                ),
                start_date=getattr(spec, "start_date", None),
                end_date=getattr(spec, "end_date", None),
                frequency=getattr(spec, "frequency", None),
                day_count=getattr(spec, "day_count", None),
                rate_index=getattr(spec, "rate_index", None),
                calendar_name=getattr(spec, "calendar_name", None),
                business_day_adjustment=getattr(spec, "business_day_adjustment", None),
            )
            """
        ).rstrip()
    if (
        comparison_target is None
        and instrument_type in {"cap", "floor"}
        and "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical" in refs
    ):
        return textwrap.dedent(
            f"""\
            spec = self._spec
            return price_rate_cap_floor_strip_analytical(
                market_state,
                spec=spec,
                instrument_class="{instrument_type}",
                notional=spec.notional,
                strike=spec.strike,
                start_date=spec.start_date,
                end_date=spec.end_date,
                frequency=spec.frequency,
                day_count=spec.day_count,
                rate_index=spec.rate_index,
                model=getattr(spec, "model", None),
                shift=getattr(spec, "shift", None),
                sabr=getattr(spec, "sabr", None),
            )
            """
        ).rstrip()
    if (
        comparison_target is None
        and instrument_type in {"cap", "floor"}
        and "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo" in refs
    ):
        return textwrap.dedent(
            f"""\
            spec = self._spec
            return price_rate_cap_floor_strip_monte_carlo(
                market_state,
                spec=spec,
                instrument_class="{instrument_type}",
                notional=spec.notional,
                strike=spec.strike,
                start_date=spec.start_date,
                end_date=spec.end_date,
                frequency=spec.frequency,
                day_count=spec.day_count,
                rate_index=spec.rate_index,
                n_paths=20000,
                seed=42,
            )
            """
        ).rstrip()
    if black_scholes_vanilla_exact_binding:
        return textwrap.dedent(
            """\
            spec = self._spec
            if market_state.discount is None:
                raise ValueError("market_state.discount is required for exact Black-76 vanilla pricing")
            if market_state.vol_surface is None:
                raise ValueError("market_state.vol_surface is required for exact Black-76 vanilla pricing")
            T = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)
            spot = spec.spot
            strike = spec.strike
            option_type = str(spec.option_type or "call").strip().lower()
            if T <= 0.0:
                intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
                return spec.notional * intrinsic
            df = market_state.discount.discount(T)
            sigma = market_state.vol_surface.black_vol(max(T, 1e-6), strike)
            dividend_yield = float(getattr(spec, "dividend_yield", 0.0) or 0.0)
            forward = spot * exp(-dividend_yield * T) / max(df, 1e-12)
            if option_type == "call":
                undiscounted = black76_call(forward, strike, sigma, T)
            elif option_type == "put":
                undiscounted = black76_put(forward, strike, sigma, T)
            else:
                raise ValueError(f"Unsupported option_type {spec.option_type!r}")
            return spec.notional * df * undiscounted
            """
        ).rstrip()
    if comparison_target is None and route_free_exact_binding and instrument_type == "digital_option":
        if (
            "trellis.models.black.black76_cash_or_nothing_call" in refs
            and "trellis.models.black.black76_cash_or_nothing_put" in refs
        ) or (
            "trellis.models.black.black76_asset_or_nothing_call" in refs
            and "trellis.models.black.black76_asset_or_nothing_put" in refs
        ):
            return textwrap.dedent(
                """\
                spec = self._spec
                if market_state.discount is None:
                    raise ValueError("market_state.discount is required for exact Black-76 digital pricing")
                if market_state.vol_surface is None:
                    raise ValueError("market_state.vol_surface is required for exact Black-76 digital pricing")
                T = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)
                spot = spec.spot
                strike = spec.strike
                option_type = str(spec.option_type or "call").strip().lower()
                payout_type = str(getattr(spec, "payout_type", "cash_or_nothing") or "cash_or_nothing").strip().lower()
                cash_payoff = float(getattr(spec, "cash_payoff", 1.0) or 1.0)
                if option_type not in {"call", "put"}:
                    raise ValueError(f"Unsupported option_type {spec.option_type!r}")
                if T <= 0.0:
                    in_the_money = spot > strike if option_type == "call" else spot < strike
                    if payout_type == "cash_or_nothing":
                        return spec.notional * cash_payoff * (1.0 if in_the_money else 0.0)
                    if payout_type == "asset_or_nothing":
                        return spec.notional * (spot if in_the_money else 0.0)
                    raise ValueError(f"Unsupported payout_type {getattr(spec, 'payout_type', None)!r}")
                df = market_state.discount.discount(T)
                sigma = market_state.vol_surface.black_vol(max(T, 1e-6), strike)
                forward = spot / max(df, 1e-12)
                if payout_type == "cash_or_nothing":
                    if option_type == "call":
                        undiscounted = black76_cash_or_nothing_call(forward, strike, sigma, T)
                    else:
                        undiscounted = black76_cash_or_nothing_put(forward, strike, sigma, T)
                    return spec.notional * cash_payoff * df * undiscounted
                if payout_type == "asset_or_nothing":
                    if option_type == "call":
                        undiscounted = black76_asset_or_nothing_call(forward, strike, sigma, T)
                    else:
                        undiscounted = black76_asset_or_nothing_put(forward, strike, sigma, T)
                    return spec.notional * df * undiscounted
                raise ValueError(f"Unsupported payout_type {getattr(spec, 'payout_type', None)!r}")
                """
            ).rstrip()
    if (
        instrument_type == "bermudan_swaption"
        and "trellis.models.rate_style_swaption.price_swaption_black76_raw" in refs
    ):
        return textwrap.dedent(
            """\
            spec = self._spec
            exercise_dates = normalize_explicit_dates(spec.exercise_dates)
            valid_exercise_dates = tuple(
                exercise_date
                for exercise_date in exercise_dates
                if market_state.settlement < exercise_date < spec.swap_end
            )
            if not valid_exercise_dates:
                return 0.0
            final_exercise_date = valid_exercise_dates[-1]
            resolved = resolve_swaption_black76_inputs(
                market_state,
                spec,
                expiry_date=final_exercise_date,
            )
            return price_swaption_black76_raw(resolved)
            """
        ).rstrip()
    swaption_monte_carlo_refs = {
        "trellis.models.rate_style_swaption.resolve_swaption_black76_inputs",
        "trellis.core.date_utils.build_payment_timeline",
        "trellis.models.monte_carlo.event_aware.resolve_hull_white_monte_carlo_process_inputs",
        "trellis.models.monte_carlo.event_aware.build_discounted_swap_pv_payload",
        "trellis.models.monte_carlo.event_aware.build_short_rate_discount_reducer",
        "trellis.models.monte_carlo.event_aware.EventAwareMonteCarloEvent",
        "trellis.models.monte_carlo.event_aware.EventAwareMonteCarloProblemSpec",
        "trellis.models.monte_carlo.event_aware.build_event_aware_monte_carlo_problem",
        "trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo",
    }
    if instrument_type == "swaption" and swaption_monte_carlo_refs.issubset(refs):
        return textwrap.dedent(
            f"""\
            if market_state.discount is None:
                raise ValueError("Rate-style swaption Monte Carlo pricing requires market_state.discount")
            if market_state.vol_surface is None:
                raise ValueError("Rate-style swaption Monte Carlo pricing requires market_state.vol_surface")

            resolved = resolve_swaption_black76_inputs(
                market_state,
                spec,
            )
            settlement = getattr(market_state, "settlement", None) or market_state.as_of
            swap_start = getattr(spec, "swap_start", None) or resolved.expiry_date
            payment_timeline = tuple(
                period
                for period in build_payment_timeline(
                    swap_start,
                    spec.swap_end,
                    spec.swap_frequency,
                    day_count=spec.day_count,
                    time_origin=settlement,
                    label="rate_style_swaption_monte_carlo",
                )
                if period.end_date > settlement
            )
            if not payment_timeline:
                raise ValueError("Rate-style swaption Monte Carlo pricing requires future payments after settlement")

            process_spec, initial_state = resolve_hull_white_monte_carlo_process_inputs(
                market_state,
                option_horizon=max(float(resolved.expiry_years), 1e-6),
                strike=float(spec.strike){swaption_comparison_kwargs},
            )
            forward_curve = None
            rate_index = getattr(spec, "rate_index", None)
            if rate_index and hasattr(market_state, "forecast_forward_curve"):
                forward_curve = market_state.forecast_forward_curve(rate_index)
            if forward_curve is None:
                forward_curve = getattr(market_state, "forward_curve", None)

            settlement_payload = build_discounted_swap_pv_payload(
                payment_timeline=payment_timeline,
                discount_curve=market_state.discount,
                forward_curve=forward_curve,
                exercise_time=float(resolved.expiry_years),
                discount_reducer_name="discount_to_expiry",
                mean_reversion=float(process_spec.mean_reversion or 0.0),
                strike=float(spec.strike),
                notional=float(spec.notional),
                is_payer=bool(spec.is_payer),
                anchor_short_rate=float(initial_state),
            )
            n_paths = max(int(getattr(spec, "n_paths", 20000)), 2)
            n_steps = max(int(getattr(spec, "n_steps", 64)), 1)
            seed = spec.seed if hasattr(spec, "seed") else 42
            problem = build_event_aware_monte_carlo_problem(
                EventAwareMonteCarloProblemSpec(
                    process_spec=process_spec,
                    initial_state=float(initial_state),
                    maturity=float(resolved.expiry_years),
                    n_steps=n_steps,
                    path_requirement_kind="event_replay",
                    reducer_kind="compiled_schedule_payoff",
                    path_reducers=(
                        build_short_rate_discount_reducer(
                            name="discount_to_expiry",
                            maturity=float(resolved.expiry_years),
                        ),
                    ),
                    settlement_event="swaption_settlement",
                    event_specs=(
                        EventAwareMonteCarloEvent(
                            time=float(resolved.expiry_years),
                            name="swaption_observation",
                            kind="observation",
                        ),
                        EventAwareMonteCarloEvent(
                            time=float(resolved.expiry_years),
                            name="swaption_settlement",
                            kind="settlement",
                            priority=1,
                            payload=settlement_payload,
                        ),
                    ),
                )
            )
            result = price_event_aware_monte_carlo(
                problem,
                n_paths=n_paths,
                seed=seed,
                return_paths=False,
            )
            return float(result["price"])
            """
        ).rstrip()
    if has_callable_bond_lattice_composition:
        expected_control_style = (
            "holder_max" if instrument_type == "puttable_bond" else "issuer_min"
        )
        expected_reference_bound = (
            "lower" if instrument_type == "puttable_bond" else "upper"
        )
        reference_bound_operator = (
            "max" if instrument_type == "puttable_bond" else "min"
        )
        calibration_argument_lines = "".join(
            f"                {argument},\n"
            for argument in callable_bond_calibration_kwargs
        )
        return textwrap.dedent(
            f"""\
            if market_state.discount is None:
                raise ValueError("Embedded fixed-income lattice pricing requires market_state.discount")

            settlement = settlement_date_for_fixed_income_claim(market_state, spec)
            horizon = year_fraction(settlement, spec.end_date, spec.day_count)
            if horizon <= 0.0:
                raise ValueError("Embedded fixed-income maturity must be after settlement")

            resolved = resolve_short_rate_lattice_inputs(
                market_state,
                horizon=horizon,
                model="{callable_bond_lattice_model}",
{calibration_argument_lines}                minimum_steps=50,
                maximum_steps=200,
                steps_per_year=50.0,
            )
            tree_model = MODEL_REGISTRY[resolved.model_name]
            lattice = build_lattice(
                BINOMIAL_1F_TOPOLOGY,
                UNIFORM_ADDITIVE_MESH,
                tree_model.as_lattice_model_spec(),
                calibration_target=TERM_STRUCTURE_TARGET(market_state.discount),
                r0=resolved.r0,
                sigma=resolved.sigma,
                a=resolved.mean_reversion,
                T=resolved.horizon,
                n_steps=resolved.n_steps,
            )
            event_timeline = build_embedded_fixed_income_event_timeline(
                spec,
                settlement=settlement,
            )
            if event_timeline.exercise.reference_bound != "{expected_reference_bound}":
                raise ValueError("Embedded fixed-income reference bound does not match product control")
            contract = compile_embedded_fixed_income_lattice_contract_spec(
                spec,
                event_timeline=event_timeline,
                expected_control_style="{expected_control_style}",
                dt=lattice.dt,
                n_steps=lattice.n_steps,
            )
            tree_price = float(price_on_lattice(lattice, contract))
            straight_price = present_value_fixed_coupon_bond(
                market_state,
                spec,
                settlement=settlement,
            )
            return {reference_bound_operator}(tree_price, straight_price)
            """
        ).rstrip()

    helper_bodies = {
        "trellis.models.fx_vanilla.price_fx_vanilla_analytical": (
            "return price_fx_vanilla_analytical(market_state, spec)"
        ),
        "trellis.models.fx_barrier_option.price_fx_barrier_option_analytical": (
            "return price_fx_barrier_option_analytical(market_state, spec)"
        ),
        "trellis.models.fx_barrier_option.price_fx_barrier_option_monte_carlo": (
            "return price_fx_barrier_option_monte_carlo(market_state, spec)"
        ),
        "trellis.models.callable_bond_pde.price_callable_bond_pde": (
            "return price_callable_bond_pde("
            "market_state, spec, "
            + ", ".join(
                (
                    *callable_bond_calibration_kwargs,
                    f"theta={callable_bond_pde_theta!r}",
                )
            )
            + ")"
        ),
        "trellis.models.rate_style_swaption.price_swaption_black76_raw": (
            "resolved = resolve_swaption_black76_inputs(market_state, spec"
            f"{swaption_comparison_kwargs})\n"
            "return price_swaption_black76_raw(resolved)"
        ),
        "trellis.models.monte_carlo.single_state_diffusion.price_single_state_terminal_claim_monte_carlo_result": (
            "result = price_single_state_terminal_claim_monte_carlo_result(\n"
            "    market_state,\n"
            "    spec,\n"
            "    terminal_payoff=lambda terminal, resolved: "
            "terminal_intrinsic_from_resolved(terminal, resolved),\n"
            f"    {vanilla_equity_mc_call_kwargs},\n"
            f"{vanilla_equity_mc_control_kwargs}"
            ")\n"
            "return float(result.price)"
        ),
        "trellis.models.equity_option_pde.price_vanilla_equity_option_pde": (
            "return price_vanilla_equity_option_pde("
            f"market_state, spec{vanilla_equity_pde_kwargs})"
        ),
        "trellis.models.pde.event_aware.solve_event_aware_pde": textwrap.dedent(
            f"""\
            resolved = resolve_single_state_diffusion_inputs(market_state, spec)
            if resolved.maturity <= 0.0:
                return float(
                    resolved.notional
                    * terminal_intrinsic_from_resolved(resolved.spot, resolved)
                )

            s_max = max(
                float(getattr(spec, "s_max", 0.0) or 0.0),
                4.0 * max(resolved.spot, 1e-12),
                2.0 * max(resolved.strike, 1e-12),
                1e-6,
            )
            n_x = max(int(getattr(spec, "n_x", 201)), 5)
            n_t = max(
                int(getattr(spec, "n_t", max(200, round(resolved.maturity * 252)))),
                1,
            )

            def remaining_time(t):
                return max(resolved.maturity - float(t), 0.0)

            if resolved.option_type == "call":
                lower_boundary = 0.0
                upper_boundary = lambda t: (
                    s_max * exp(-resolved.dividend_yield * remaining_time(t))
                    - resolved.strike * exp(-resolved.rate * remaining_time(t))
                )
            else:
                lower_boundary = lambda t: (
                    resolved.strike * exp(-resolved.rate * remaining_time(t))
                )
                upper_boundary = 0.0

            problem = build_event_aware_pde_problem(
                EventAwarePDEProblemSpec(
                    grid_spec=EventAwarePDEGridSpec(
                        x_min=0.0,
                        x_max=s_max,
                        n_x=n_x,
                        maturity=resolved.maturity,
                        n_t=n_t,
                    ),
                    operator_spec=EventAwarePDEOperatorSpec(
                        family="black_scholes_1d",
                        sigma=resolved.sigma,
                        r=resolved.rate,
                        dividend_yield=resolved.dividend_yield,
                    ),
                    terminal_condition=lambda terminal: (
                        terminal_intrinsic_from_resolved(terminal, resolved)
                    ),
                    boundary_spec=EventAwarePDEBoundarySpec(
                        lower=lower_boundary,
                        upper=upper_boundary,
                    ),
                    theta={vanilla_equity_pde_theta},
                )
            )
            values = solve_event_aware_pde(problem)
            return float(
                resolved.notional
                * interpolate_pde_values(values, problem.grid.x, resolved.spot)
            )
            """
        ).rstrip(),
        "trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo": (
            "return price_heston_option_monte_carlo("
            f"market_state, spec{heston_mc_kwargs})"
        ),
        "trellis.models.equity_option_transforms.price_vanilla_equity_option_transform": (
            "return price_vanilla_equity_option_transform("
            f"market_state, spec{vanilla_equity_transform_kwargs})"
        ),
        "trellis.models.equity_option_transforms.price_equity_digital_option_transform": (
            "return price_equity_digital_option_transform("
            f"market_state, spec{vanilla_equity_transform_kwargs})"
        ),
        "trellis.models.transforms.heston.price_heston_option_transform": (
            "return price_heston_option_transform("
            f"market_state, spec{heston_transform_kwargs})"
        ),
        "trellis.models.bates_option.price_bates_option_transform": (
            'return price_bates_option_transform(market_state, spec, method="fft")'
        ),
        "trellis.models.bates_option.price_bates_option_monte_carlo": (
            "return price_bates_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 80000), '
            'n_steps=getattr(spec, "n_steps", 96), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "trellis.models.levy_option.price_kou_option_transform": (
            'return price_kou_option_transform(market_state, spec, method="fft")'
        ),
        "trellis.models.levy_option.price_kou_option_monte_carlo": (
            "return price_kou_option_monte_carlo("
            "market_state, spec, "
            'n_paths=getattr(spec, "n_paths", 120000), '
            'seed=getattr(spec, "seed", 42))'
        ),
        "trellis.models.levy_option.price_kou_option_reference": (
            "return price_kou_option_reference(market_state, spec)"
        ),
        "trellis.models.zcb_option_tree.price_zcb_option_tree": (
            "return price_zcb_option_tree("
            f"market_state, spec{zcb_option_tree_kwargs})"
        ),
        "trellis.models.short_rate_bond.price_short_rate_zero_coupon_bond_analytical": (
            "return price_short_rate_zero_coupon_bond_analytical("
            "market_state, spec, allow_benchmark_defaults=True)"
        ),
        "trellis.models.short_rate_bond.price_short_rate_zero_coupon_bond_tree": (
            "return price_short_rate_zero_coupon_bond_tree("
            'market_state, spec, n_steps=getattr(spec, "tree_steps", 360), '
            "allow_benchmark_defaults=True)"
        ),
        "trellis.models.credit_basket_copula.price_credit_basket_tranche": (
            "return price_credit_basket_tranche("
            f"market_state, spec{credit_basket_tranche_kwargs})"
        ),
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_recursive": (
            "return price_credit_portfolio_loss_distribution_recursive("
            'market_state, spec, copula_family="gaussian")'
        ),
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_transform_proxy": (
            "return price_credit_portfolio_loss_distribution_transform_proxy("
            'market_state, spec, copula_family="gaussian")'
        ),
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_monte_carlo": (
            "return price_credit_portfolio_loss_distribution_monte_carlo("
            'market_state, spec, copula_family="gaussian", n_paths=40000, seed=42)'
        ),
        "trellis.models.analytical.equity_exotics.price_equity_digital_option_analytical": (
            "return price_equity_digital_option_analytical(market_state, spec)"
        ),
        "trellis.models.equity_option_pde.price_equity_digital_option_pde": (
            "return price_equity_digital_option_pde(market_state, spec)"
        ),
        "trellis.models.analytical.barrier.barrier_option_price": (
            "if market_state.discount is None:\n"
            '    raise ValueError("market_state.discount is required for analytical barrier pricing")\n'
            "if market_state.vol_surface is None:\n"
            '    raise ValueError("market_state.vol_surface is required for analytical barrier pricing")\n'
            "T = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)\n"
            "if T <= 0.0:\n"
            "    return 0.0\n"
            "sigma = float(market_state.vol_surface.black_vol(max(T, 1e-6), spec.strike))\n"
            "rate = float(market_state.discount.zero_rate(T))\n"
            "carry_rates = dict(getattr(market_state, 'model_parameters', None) or {}).get('underlier_carry_rates') or {}\n"
            "carry_rate = 0.0\n"
            "if isinstance(carry_rates, dict) and len(carry_rates) == 1:\n"
            "    carry_rate = float(next(iter(carry_rates.values())))\n"
            "price = barrier_option_price(\n"
            "    float(spec.spot),\n"
            "    float(spec.strike),\n"
            "    float(spec.barrier),\n"
            "    rate,\n"
            "    sigma,\n"
            "    T,\n"
            '    barrier_type=str(spec.barrier_type),\n'
            '    option_type=str(spec.option_type),\n'
            "    rebate=float(getattr(spec, 'rebate', 0.0) or 0.0),\n"
            "    q=carry_rate,\n"
            "    observations_per_year=getattr(spec, 'observations_per_year', None),\n"
            ")\n"
            "return spec.notional * price"
        ),
        "trellis.models.single_barrier_option.price_single_barrier_option_pde_result": (
            "return price_single_barrier_option_pde_result(market_state, spec).price"
        ),
        "trellis.models.single_barrier_option.price_single_barrier_option_monte_carlo_result": (
            "return price_single_barrier_option_monte_carlo_result(market_state, spec).price"
        ),
        "trellis.models.double_barrier_option.price_double_barrier_option_pde_result": (
            "return price_double_barrier_option_pde_result(market_state, spec).price"
        ),
        "trellis.models.double_barrier_option.price_double_barrier_option_monte_carlo_result": (
            "return price_double_barrier_option_monte_carlo_result(market_state, spec).price"
        ),
        "trellis.models.pde.heston_adi.price_heston_option_adi_pde_result": (
            "return price_heston_option_adi_pde_result(market_state, spec).price"
        ),
        "trellis.models.autocallable.price_autocallable_monte_carlo_result": (
            "return price_autocallable_monte_carlo_result("
            'market_state, spec, sampling="sobol").price'
            if normalized_target == "mc_autocall_qmc"
            else "return price_autocallable_monte_carlo_result("
            'market_state, spec, sampling="pseudo").price'
        ),
    }
    for ref, body in helper_bodies.items():
        if ref in refs:
            return body
    return None


def _deterministic_exact_binding_benchmark_outputs_block(
    generation_plan,
    *,
    comparison_target: str | None = None,
) -> str | None:
    """Return a complete ``benchmark_outputs`` method for supported routes.

    Returned text is un-indented; the caller must indent it to class scope
    (four spaces) before appending.  Returns ``None`` when the deterministic
    route does not have a native Greek helper available (QUA-862).
    """
    refs = set(_exact_binding_refs(generation_plan))
    normalized_target = str(comparison_target or "").strip().lower().replace("-", "_")
    route_free_exact_binding = _generation_plan_field(generation_plan, "primitive_plan") is None
    instrument_type = str(
        _generation_plan_field(generation_plan, "instrument_type", "") or ""
    ).strip().lower()
    if _is_black_scholes_vanilla_exact_binding(
        generation_plan,
        refs,
        instrument_type=instrument_type,
        normalized_target=normalized_target,
        route_free_exact_binding=route_free_exact_binding,
    ):
        return textwrap.dedent(
            """\
            def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
                return dict(equity_vanilla_bs_outputs(market_state, self._spec))
            """
        )
    # QUA-878: FX vanilla Garman-Kohlhagen analytical route emits native
    # price + delta (and gamma/vega/theta) via the shared helper, so parity
    # scorecards no longer rely on the bump-and-reprice fallback (QUA-863)
    # for F002-style FX vanilla tasks.
    if (
        "trellis.models.fx_vanilla.price_fx_vanilla_analytical" in refs
        or "trellis.models.analytical.fx.garman_kohlhagen_price_raw" in refs
    ):
        return textwrap.dedent(
            """\
            def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
                return dict(fx_vanilla_gk_outputs(market_state, self._spec))
            """
        )
    return None


def _is_black_scholes_vanilla_exact_binding(
    generation_plan,
    refs: set[str],
    *,
    instrument_type: str,
    normalized_target: str,
    route_free_exact_binding: bool,
) -> bool:
    """Return whether an exact Black-76 equity vanilla adapter can be materialized."""
    if "trellis.models.black.black76_call" not in refs:
        return False
    if "trellis.models.black.black76_put" not in refs:
        return False

    primitive_plan = _generation_plan_field(generation_plan, "primitive_plan")
    route = ""
    if isinstance(primitive_plan, Mapping):
        route = str(primitive_plan.get("route") or "").strip().lower()
    elif primitive_plan is not None:
        route = str(getattr(primitive_plan, "route", "") or "").strip().lower()

    if normalized_target == "black_scholes":
        return True

    if normalized_target not in {"", "analytical"}:
        return False

    return instrument_type == "european_option" and (
        route_free_exact_binding or route == "analytical_black76"
    )


def _materialize_semantic_execution_shim_module(
    skeleton: str,
    generation_plan,
    *,
    request_metadata: Mapping[str, object] | None,
    comparison_target: str | None = None,
) -> GeneratedModuleResult | None:
    """Build a thin generated adapter that delegates P001 to execution IR visitors."""
    method = _semantic_execution_shim_method(
        generation_plan,
        request_metadata=request_metadata,
        comparison_target=comparison_target,
    )
    if method is None:
        return None
    body = textwrap.dedent(
        f"""\
        return price_bermudan_best_of_basket_from_compat_spec(
            market_state,
            spec,
            method="{method}",
        )
        """
    ).rstrip()
    rendered = _inject_top_level_imports(
        skeleton,
        (
            "from trellis.execution import "
            "price_bermudan_best_of_basket_from_compat_spec",
        ),
    ).replace(EVALUATE_SENTINEL, textwrap.indent(body, "        "))
    rendered = _inject_comparison_target_contract_declaration(
        rendered,
        request_metadata=request_metadata,
        comparison_target=comparison_target,
    )
    rendered = "\n".join(
        line for line in rendered.splitlines()
        if not line.startswith("from trellis.models.")
    )
    source_report = sanitize_generated_source(rendered)
    return GeneratedModuleResult(
        raw_code=rendered,
        sanitized_code=source_report.sanitized_source,
        code=source_report.sanitized_source.expandtabs(4),
        source_report=source_report,
    )


def _semantic_execution_shim_method(
    generation_plan,
    *,
    request_metadata: Mapping[str, object] | None,
    comparison_target: str | None,
) -> str | None:
    metadata = dict(request_metadata or {})
    task_id = str(metadata.get("task_id") or "").strip().upper()
    instrument_type = str(
        getattr(generation_plan, "instrument_type", "") or ""
    ).strip().lower().replace(" ", "_")
    if task_id != "P001" or instrument_type != "rainbow_option":
        return None

    target = _normalize_semantic_execution_method(
        comparison_target or metadata.get("preferred_method")
    )
    if target is None:
        target = _normalize_semantic_execution_method(
            getattr(generation_plan, "method", None)
        )
    return target


def _normalize_semantic_execution_method(value: object) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"monte_carlo", "montecarlo", "mc"}:
        return "monte_carlo"
    if text in {"rate_tree", "tree", "lattice", "lattices"}:
        return "lattice"
    return None


def _materialize_deterministic_exact_binding_module(
    skeleton: str,
    generation_plan,
    *,
    semantic_blueprint=None,
    request_metadata: Mapping[str, object] | None = None,
    comparison_target: str | None = None,
) -> GeneratedModuleResult | None:
    """Build an exact-bound adapter module without invoking the LLM."""
    body = _deterministic_exact_binding_evaluate_body(
        generation_plan,
        semantic_blueprint=semantic_blueprint,
        request_metadata=request_metadata,
        comparison_target=comparison_target,
    )
    if body is None:
        return None
    benchmark_outputs_block = _deterministic_exact_binding_benchmark_outputs_block(
        generation_plan,
        comparison_target=comparison_target,
    )
    import_lines = list(_deterministic_exact_binding_import_lines(body))
    if benchmark_outputs_block is not None:
        import_lines.extend(
            _deterministic_exact_binding_benchmark_outputs_import_lines(
                benchmark_outputs_block
            )
        )
    skeleton = _inject_top_level_imports(skeleton, import_lines)
    rendered = skeleton.replace(
        EVALUATE_SENTINEL,
        textwrap.indent(body, "        "),
    )
    rendered = _inject_comparison_target_contract_declaration(
        rendered,
        request_metadata=request_metadata,
        comparison_target=comparison_target,
    )
    if "price_merton_jump_diffusion_option_" in body:
        rendered = _merge_generated_requirements(rendered, ("jump_parameters",))
    if "price_bates_option_" in body:
        rendered = _set_generated_requirements(
            rendered,
            ("discount_curve", "jump_parameters", "model_parameters"),
        )
    if "price_sabr_forward_option_" in body:
        rendered = _set_generated_requirements(
            rendered,
            ("discount_curve", "model_parameters"),
        )
    if (
        "price_variance_gamma_option_" in body
        or "price_cgmy_option_" in body
        or "price_kou_option_" in body
    ):
        rendered = _set_generated_requirements(
            rendered,
            ("discount_curve", "model_parameters"),
        )
    if benchmark_outputs_block is not None:
        rendered = (
            rendered.rstrip("\n")
            + "\n\n"
            + textwrap.indent(benchmark_outputs_block, "    ")
            + "\n"
        )
    source_report = sanitize_generated_source(rendered)
    return GeneratedModuleResult(
        raw_code=rendered,
        sanitized_code=source_report.sanitized_source,
        code=source_report.sanitized_source.expandtabs(4),
        source_report=source_report,
    )


def _inject_comparison_target_contract_declaration(
    source: str,
    *,
    request_metadata: Mapping[str, object] | None,
    comparison_target: str | None,
) -> str:
    """Declare the trusted target contract on a deterministic generated wrapper."""
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract

    raw_contract = (request_metadata or {}).get("comparison_target_contract")
    if not isinstance(raw_contract, Mapping):
        return source
    try:
        contract = ComparisonTargetContract.from_payload(raw_contract)
    except (TypeError, ValueError):
        return source
    normalized_target = str(comparison_target or "").strip()
    if normalized_target and contract.target_id != normalized_target:
        return source
    marker = "\n    def __init__(self,"
    if marker not in source:
        return source
    declaration = {
        contract.target_id: {"target_contract": contract.to_payload()}
    }
    return source.replace(
        marker,
        f"\n    __trellis_comparison_bindings__ = {declaration!r}\n{marker}",
        1,
    )


def _merge_generated_requirements(source: str, extra_requirements: Sequence[str]) -> str:
    """Add route-owned market requirements to a deterministic skeleton."""
    extras = {str(requirement) for requirement in extra_requirements if str(requirement)}
    if not extras:
        return source

    pattern = (
        r"(    def requirements\(self\) -> set\[str\]:\n"
        r"        return \{)([^}]*)"
        r"(\})"
    )

    def _replace(match: re.Match[str]) -> str:
        existing = set(re.findall(r'"([^"]+)"', match.group(2)))
        merged = ", ".join(f'"{item}"' for item in sorted(existing | extras))
        return f"{match.group(1)}{merged}{match.group(3)}"

    return re.sub(pattern, _replace, source, count=1)


def _set_generated_requirements(source: str, requirements: Sequence[str]) -> str:
    """Replace deterministic skeleton market requirements with a route contract."""
    required = {str(requirement) for requirement in requirements if str(requirement)}
    pattern = (
        r"(    def requirements\(self\) -> set\[str\]:\n"
        r"        return \{)([^}]*)"
        r"(\})"
    )
    rendered = ", ".join(f'"{item}"' for item in sorted(required))
    return re.sub(pattern, rf"\1{rendered}\3", source, count=1)


def _deterministic_exact_binding_import_lines(body: str) -> tuple[str, ...]:
    """Return extra imports required by deterministic exact-binding bodies.

    Exact-bound bodies can reference shared support functions that are not part
    of the terminal backend-binding symbol set. Keep those imports explicit so
    isolated proof artifacts exercise the same declared dependency surface as
    ordinary generated modules.
    """
    imports: list[str] = []
    if "exp(" in body:
        imports.append("from math import exp")
    if "sqrt(" in body:
        imports.append("from math import sqrt")
    if "year_fraction(" in body:
        imports.append("from trellis.core.date_utils import year_fraction")
    if "resolve_short_rate_lattice_inputs(" in body:
        imports.append(
            "from trellis.models.short_rate_lattice import "
            "resolve_short_rate_lattice_inputs"
        )
    if "build_embedded_fixed_income_event_timeline(" in body:
        imports.append(
            "from trellis.models.short_rate_fixed_income import (\n"
            "    build_embedded_fixed_income_event_timeline,\n"
            "    compile_embedded_fixed_income_lattice_contract_spec,\n"
            "    present_value_fixed_coupon_bond,\n"
            "    settlement_date_for_fixed_income_claim,\n"
            ")"
        )
    if "MODEL_REGISTRY[" in body:
        imports.append("from trellis.models.trees.models import MODEL_REGISTRY")
    if "normalize_explicit_dates(" in body:
        imports.append("from trellis.core.date_utils import normalize_explicit_dates")
    if "build_payment_timeline(" in body:
        imports.append("from trellis.core.date_utils import build_payment_timeline")
    if "resolve_swaption_black76_inputs(" in body:
        imports.append(
            "from trellis.models.rate_style_swaption import "
            "resolve_swaption_black76_inputs"
        )
    if "resolve_swaption_curve_basis_spread(" in body:
        imports.append(
            "from trellis.models.rate_style_swaption import "
            "resolve_swaption_curve_basis_spread"
        )
    if "BermudanSwaptionTreeSpec(" in body:
        imports.append(
            "from trellis.models.bermudan_swaption_tree import (\n"
            "    BermudanSwaptionTreeSpec,\n"
            "    compile_bermudan_swaption_contract_spec,\n"
            "    resolve_bermudan_swaption_tree_inputs,\n"
            ")"
        )
    elif "resolve_bermudan_swaption_tree_inputs(" in body:
        imports.append(
            "from trellis.models.bermudan_swaption_tree import "
            "resolve_bermudan_swaption_tree_inputs"
        )
    if "BINOMIAL_1F_TOPOLOGY" in body and "price_on_lattice(" in body:
        algebra_symbols = [
            "BINOMIAL_1F_TOPOLOGY",
            "TERM_STRUCTURE_TARGET",
            "UNIFORM_ADDITIVE_MESH",
            "build_lattice",
        ]
        algebra_symbols.extend(
            symbol
            for symbol in (
                "LatticeLinearClaimSpec",
                "LatticeControlSpec",
                "LatticeContractSpec",
                "value_on_lattice",
            )
            if symbol in body
        )
        algebra_symbols.append("price_on_lattice")
        imports.append(
            "from trellis.models.trees.algebra import (\n"
            + "".join(f"    {symbol},\n" for symbol in algebra_symbols)
            + ")"
        )
    if "lattice_step_from_time(" in body:
        imports.append(
            "from trellis.models.trees.control import lattice_step_from_time"
        )
    if "price_heston_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.monte_carlo.stochastic_vol import price_heston_option_monte_carlo"
        )
    if "price_merton_jump_diffusion_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.merton_jump_diffusion_option import "
            "price_merton_jump_diffusion_option_monte_carlo"
        )
    if "price_merton_jump_diffusion_option_transform(" in body:
        imports.append(
            "from trellis.models.merton_jump_diffusion_option import "
            "price_merton_jump_diffusion_option_transform"
        )
    if "price_sabr_forward_option_hagan(" in body:
        imports.append(
            "from trellis.models.sabr_option import price_sabr_forward_option_hagan"
        )
    if "price_sabr_forward_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.sabr_option import "
            "price_sabr_forward_option_monte_carlo"
        )
    if "price_variance_gamma_option_transform(" in body:
        imports.append(
            "from trellis.models.levy_option import price_variance_gamma_option_transform"
        )
    if "price_variance_gamma_option_reference(" in body:
        imports.append(
            "from trellis.models.levy_option import price_variance_gamma_option_reference"
        )
    if "price_variance_gamma_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.levy_option import price_variance_gamma_option_monte_carlo"
        )
    if "price_cgmy_option_transform(" in body:
        imports.append(
            "from trellis.models.levy_option import price_cgmy_option_transform"
        )
    if "price_cgmy_option_reference(" in body:
        imports.append(
            "from trellis.models.levy_option import price_cgmy_option_reference"
        )
    if "price_cgmy_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.levy_option import price_cgmy_option_monte_carlo"
        )
    if "price_kou_option_transform(" in body:
        imports.append(
            "from trellis.models.levy_option import price_kou_option_transform"
        )
    if "price_kou_option_reference(" in body:
        imports.append(
            "from trellis.models.levy_option import price_kou_option_reference"
        )
    if "price_kou_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.levy_option import price_kou_option_monte_carlo"
        )
    if "price_bates_option_transform(" in body:
        imports.append(
            "from trellis.models.bates_option import price_bates_option_transform"
        )
    if "price_bates_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.bates_option import price_bates_option_monte_carlo"
        )
    if "price_short_rate_zero_coupon_bond_analytical(" in body:
        imports.append(
            "from trellis.models.short_rate_bond import "
            "price_short_rate_zero_coupon_bond_analytical"
        )
    if "price_short_rate_zero_coupon_bond_tree(" in body:
        imports.append(
            "from trellis.models.short_rate_bond import "
            "price_short_rate_zero_coupon_bond_tree"
        )
    if "price_cds_analytical(" in body:
        imports.append(
            "from trellis.models.credit_default_swap import build_cds_schedule, price_cds_analytical"
        )
    if "price_cds_monte_carlo(" in body:
        imports.append(
            "from trellis.models.credit_default_swap import build_cds_schedule, price_cds_monte_carlo"
        )
    if "resolve_basket_semantics(" in body:
        imports.append(
            "from trellis.models.resolution.basket_semantics import resolve_basket_semantics"
        )
    if "resolve_terminal_basket_inputs(" in body:
        imports.append(
            "from trellis.models.resolution.terminal_basket import "
            "resolve_terminal_basket_inputs"
        )
    if any(
        symbol in body
        for symbol in (
            "two_asset_extremum_option_stulz(",
            "two_asset_spread_option_kirk(",
            "two_asset_terminal_basket_gauss_hermite(",
        )
    ):
        symbols = [
            symbol
            for symbol in (
                "two_asset_extremum_option_stulz",
                "two_asset_spread_option_kirk",
                "two_asset_terminal_basket_gauss_hermite",
            )
            if f"{symbol}(" in body
        ]
        imports.append(
            "from trellis.models.analytical.terminal_basket import "
            + ", ".join(symbols)
        )
    if "terminal_basket_option_payoff(" in body:
        imports.append(
            "from trellis.models.payoffs import terminal_basket_option_payoff"
        )
    if (
        "correlated_gbm_log_return_characteristic_function(" in body
        or "hurd_zhou_spread_option_2d_fft(" in body
    ):
        imports.append(
            "from trellis.models.transforms.spread_option import "
            "correlated_gbm_log_return_characteristic_function, "
            "hurd_zhou_spread_option_2d_fft"
        )
    if (
        "build_ranked_observation_basket_state_payoff(" in body
        or "terminal_ranked_observation_basket_payoff(" in body
    ):
        imports.append(
            "from trellis.models.monte_carlo.ranked_observation_payoffs import "
            "build_ranked_observation_basket_state_payoff, "
            "terminal_ranked_observation_basket_payoff"
        )
    if "price_nth_to_default_basket(" in body:
        imports.append(
            "from trellis.instruments.nth_to_default import price_nth_to_default_basket"
        )
    if "price_event_aware_equity_option_pde(" in body:
        imports.append(
            "from trellis.models.equity_option_pde import price_event_aware_equity_option_pde"
        )
    if "price_cev_option_pde(" in body:
        imports.append(
            "from trellis.models.equity_option_pde import price_cev_option_pde"
        )
    if "price_equity_digital_option_pde(" in body:
        imports.append(
            "from trellis.models.equity_option_pde import price_equity_digital_option_pde"
        )
    if "price_equity_digital_option_transform(" in body:
        imports.append(
            "from trellis.models.equity_option_transforms import price_equity_digital_option_transform"
        )
    if "price_fx_barrier_option_analytical(" in body:
        imports.append(
            "from trellis.models.fx_barrier_option import price_fx_barrier_option_analytical"
        )
    if "price_fx_barrier_option_monte_carlo(" in body:
        imports.append(
            "from trellis.models.fx_barrier_option import price_fx_barrier_option_monte_carlo"
        )
    if "resolve_fx_vanilla_inputs(" in body:
        imports.append(
            "from trellis.models.fx_vanilla import resolve_fx_vanilla_inputs"
        )
    if "garman_kohlhagen_price_raw(" in body:
        imports.append(
            "from trellis.models.analytical.fx import garman_kohlhagen_price_raw"
        )
    if "terminal_value_payoff(" in body:
        imports.append(
            "from trellis.models.monte_carlo.path_state import terminal_value_payoff"
        )
    if re.search(r"(?<!Correlated)GBM\(", body):
        imports.append("from trellis.models.processes.gbm import GBM")
    if "CorrelatedGBM(" in body:
        imports.append(
            "from trellis.models.processes.correlated_gbm import CorrelatedGBM"
        )
    if "MonteCarloEngine(" in body:
        imports.append(
            "from trellis.models.monte_carlo.engine import MonteCarloEngine"
        )
    if "WeightedObservationContract(" in body:
        imports.append(
            "from trellis.models.observation_aggregation import "
            "WeightedObservationContract, weighted_observation_payoff"
        )
    if "StateAwarePayoff" in body and "BarrierMonitor(" not in body:
        imports.append(
            "from trellis.models.monte_carlo.path_state import StateAwarePayoff"
        )
    if "single_factor_lognormal_sum_contract(" in body:
        imports.append(
            "from trellis.models.analytical.support.lognormal_moments import "
            "match_lognormal_moments, single_factor_lognormal_sum_contract, "
            "weighted_lognormal_sum_moments"
        )
    if "ObservationReturnContract(" in body:
        observation_symbols = ["ObservationReturnContract"]
        if "bounded_observation_return_sum(" in body:
            observation_symbols.append("bounded_observation_return_sum")
        if "observation_return_payoff(" in body:
            observation_symbols.append("observation_return_payoff")
        imports.append(
            "from trellis.models.observation_returns import "
            + ", ".join(observation_symbols)
        )
    if "gauss_hermite_product_expectation(" in body:
        imports.append(
            "from trellis.models.analytical.support.expectations import "
            "gauss_hermite_product_expectation"
        )
    if "resolve_quanto_inputs(" in body:
        imports.append(
            "from trellis.models.resolution.quanto import resolve_quanto_inputs"
        )
    if any(
        symbol in body
        for symbol in (
            "discounted_value(",
            "implied_zero_rate(",
            "normalized_option_type(",
            "quanto_adjusted_forward(",
            "terminal_intrinsic(",
        )
    ):
        support_symbols = [
            symbol
            for symbol in (
                "discounted_value",
                "implied_zero_rate",
                "normalized_option_type",
                "quanto_adjusted_forward",
                "terminal_intrinsic",
            )
            if f"{symbol}(" in body
        ]
        imports.append(
            "from trellis.models.analytical.support import "
            + ", ".join(support_symbols)
        )
    if "black76_call(" in body or "black76_put(" in body:
        imports.append("from trellis.models.black import black76_call, black76_put")
    if "get_numpy(" in body:
        imports.append("from trellis.core.differentiable import get_numpy")
    if "sobol_normals(" in body:
        imports.append(
            "from trellis.models.monte_carlo.variance_reduction import sobol_normals"
        )
    if "resolve_fx_barrier_inputs(" in body:
        imports.append(
            "from trellis.models.fx_barrier_option import resolve_fx_barrier_inputs"
        )
    if "barrier_option_price(" in body:
        imports.append(
            "from trellis.models.analytical.barrier import barrier_option_price"
        )
    if "BarrierMonitor(" in body:
        imports.append(
            "from trellis.models.monte_carlo.path_state import "
            "BarrierMonitor, MonteCarloPathRequirement, StateAwarePayoff"
        )
    if "raw_np." in body:
        imports.append("import numpy as raw_np")
    if "price_vanilla_equity_option_tree(" in body:
        imports.append(
            "from trellis.models.equity_option_tree import price_vanilla_equity_option_tree"
        )
    if "price_cev_option_tree(" in body:
        imports.append(
            "from trellis.models.equity_option_tree import price_cev_option_tree"
        )
    if "resolve_single_state_monte_carlo_inputs(" in body:
        imports.append(
            "from trellis.models.monte_carlo.single_state_diffusion import "
            "resolve_single_state_monte_carlo_inputs"
        )
    if "price_single_state_terminal_claim_monte_carlo_result(" in body:
        imports.append(
            "from trellis.models.monte_carlo.single_state_diffusion import "
            "price_single_state_terminal_claim_monte_carlo_result"
        )
    if "build_event_aware_pde_problem(" in body:
        imports.append(
            "from trellis.models.pde.event_aware import (\n"
            "    EventAwarePDEBoundarySpec,\n"
            "    EventAwarePDEGridSpec,\n"
            "    EventAwarePDEOperatorSpec,\n"
            "    EventAwarePDEProblemSpec,\n"
            "    build_event_aware_pde_problem,\n"
            "    interpolate_pde_values,\n"
            "    solve_event_aware_pde,\n"
            ")"
        )
    if "resolve_single_state_diffusion_inputs(" in body:
        imports.append(
            "from trellis.models.resolution.single_state_diffusion import (\n"
            "    resolve_single_state_diffusion_inputs,\n"
            "    terminal_intrinsic_from_resolved,\n"
            ")"
        )
    elif "terminal_intrinsic_from_resolved(" in body:
        imports.append(
            "from trellis.models.resolution.single_state_diffusion import "
            "terminal_intrinsic_from_resolved"
        )
    if "equity_tree(" in body and "compile_lattice_recipe(" in body:
        imports.append(
            "from trellis.models.trees.algebra import (\n"
            "    build_lattice,\n"
            "    compile_lattice_recipe,\n"
            "    equity_tree,\n"
            "    price_on_lattice,\n"
            "    with_control,\n"
            ")"
        )
    if "longstaff_schwartz(" in body:
        imports.append("from trellis.models.monte_carlo.lsm import longstaff_schwartz")
    if "LaguerreBasis(" in body:
        imports.append("from trellis.models.monte_carlo.schemes import LaguerreBasis")
    if "event_step_indices(" in body:
        imports.append(
            "from trellis.models.monte_carlo.event_state import event_step_indices"
        )
    return tuple(imports)


def _deterministic_exact_binding_benchmark_outputs_import_lines(block: str) -> tuple[str, ...]:
    """Return imports required by the injected ``benchmark_outputs`` block.

    Covers both the Black-Scholes equity vanilla helper (QUA-862) and the
    Garman-Kohlhagen FX vanilla helper (QUA-878).
    """
    imports: list[str] = []
    if "equity_vanilla_bs_outputs(" in block:
        imports.append(
            "from trellis.models.analytical.equity_vanilla_bs import equity_vanilla_bs_outputs"
        )
    if "fx_vanilla_gk_outputs(" in block:
        imports.append(
            "from trellis.models.analytical.fx_vanilla_gk import fx_vanilla_gk_outputs"
        )
    return tuple(imports)


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
        try:
            exists = file_path.exists()
        except OSError:
            return None
        if not exists:
            return None

    last_step = plan.steps[-1]
    file_path = TRELLIS_ROOT / last_step.module_path
    module_name = f"trellis.{last_step.module_path.replace('/', '.').replace('.py', '')}"

    try:
        mod = dynamic_import(file_path, module_name)
        return getattr(mod, plan.payoff_class_name, None)
    except Exception:
        return None


def _sanitize_build_plan_module_paths(plan, *, request_metadata: Mapping[str, object] | None):
    """Sanitize generated `_agent` module paths before reuse or write resolution."""
    steps = list(getattr(plan, "steps", ()) or ())
    if not steps:
        return plan

    sanitized_steps = []
    changed = False
    for step in steps:
        module_path = str(getattr(step, "module_path", "") or "")
        sanitized = _sanitize_agent_output_module_path(
            module_path,
            request_metadata=request_metadata,
        )
        if sanitized != module_path:
            step = replace_dataclass(step, module_path=sanitized)
            changed = True
        sanitized_steps.append(step)

    if not changed:
        return plan
    return replace_dataclass(plan, steps=sanitized_steps)


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


def _resolve_output_target(
    module_path: str,
    *,
    fresh_build: bool,
    isolate_comparison_target: bool = False,
    request_metadata: Mapping[str, object] | None,
) -> tuple[Path, str, str]:
    """Resolve the output file path, module path, and import name for one build.

    When ``fresh_build=True`` the resolved target must never land under
    ``trellis/instruments/_agent/``.  The redirect branches below handle the
    normal case; the final guard defends against a regression in the redirect
    so a pilot run fails loudly at resolve time instead of burning LLM tokens
    and catching the leak post-build (QUA-872, layered under QUA-866).
    """
    normalized_module_path = str(module_path or "").replace("\\", "/").strip()
    normalized_module_path = _sanitize_agent_output_module_path(
        normalized_module_path,
        request_metadata=request_metadata,
    )
    if isolate_comparison_target and _is_agent_module_path(normalized_module_path):
        metadata = dict(request_metadata or {})
        task_id = _normalize_fresh_build_token(metadata.get("task_id")) or "task"
        target = (
            _normalize_fresh_build_token(metadata.get("comparison_target"))
            or "target"
        )
        filename = Path(normalized_module_path).name
        output_file_path = (
            REPO_ROOT
            / "task_runs"
            / "comparison_target_artifacts"
            / task_id
            / target
            / filename
        )
        output_module_path = str(output_file_path.relative_to(REPO_ROOT)).replace(
            "\\", "/"
        )
        stem = _normalize_fresh_build_token(Path(filename).stem) or "module"
        module_name = (
            f"trellis_task_targets.{task_id}.{target}.{stem}"
        )
        return output_file_path, output_module_path, module_name
    if fresh_build and _is_agent_module_path(normalized_module_path):
        benchmark_root = _benchmark_fresh_build_root(request_metadata)
        if benchmark_root is not None:
            output_file_path = benchmark_root / Path(normalized_module_path).name
            output_module_path = str(output_file_path.relative_to(REPO_ROOT)).replace("\\", "/")
            module_name = _benchmark_fresh_build_module_name(
                normalized_module_path,
                request_metadata=request_metadata,
            )
            _assert_fresh_build_target_is_not_admitted(
                output_file_path=output_file_path,
                output_module_path=output_module_path,
                module_name=module_name,
                source_module_path=normalized_module_path,
            )
            return output_file_path, output_module_path, module_name
        output_module_path = _fresh_build_module_path(normalized_module_path)
        output_file_path = TRELLIS_PACKAGE_ROOT / output_module_path
        module_name = f"trellis.{output_module_path.replace('/', '.').replace('.py', '')}"
        _assert_fresh_build_target_is_not_admitted(
            output_file_path=output_file_path,
            output_module_path=output_module_path,
            module_name=module_name,
            source_module_path=normalized_module_path,
        )
        return output_file_path, output_module_path, module_name

    output_file_path = TRELLIS_PACKAGE_ROOT / normalized_module_path
    module_name = f"trellis.{normalized_module_path.replace('/', '.').replace('.py', '')}"
    return output_file_path, normalized_module_path, module_name


def _sanitize_agent_output_module_path(
    module_path: str,
    *,
    request_metadata: Mapping[str, object] | None,
    max_stem_length: int = 72,
) -> str:
    """Bound unsafe `_agent` module filenames derived from free-form prompts."""
    normalized = str(module_path or "").replace("\\", "/").strip()
    if not normalized or not _is_agent_module_path(normalized):
        return normalized

    parent, separator, filename = normalized.rpartition("/")
    raw_stem = filename[:-3] if filename.endswith(".py") else filename
    safe_stem = _normalize_fresh_build_token(raw_stem)
    safe_stem = re.sub(r"^buildapricerfor_?", "", safe_stem).strip("_")
    if not safe_stem:
        metadata = dict(request_metadata or {})
        safe_stem = (
            _normalize_fresh_build_token(metadata.get("comparison_target"))
            or _normalize_fresh_build_token(metadata.get("preferred_method"))
            or _normalize_fresh_build_token(metadata.get("task_id"))
            or "agent_payoff"
        )
    if len(safe_stem) > max_stem_length:
        safe_stem = safe_stem[:max_stem_length].rstrip("_")
    safe_filename = f"{safe_stem or 'agent_payoff'}.py"
    if filename == safe_filename:
        return normalized
    return f"{parent}{separator}{safe_filename}" if separator else safe_filename


_FRESH_ISOLATION_SEGMENT = "/_fresh/"
_FRESH_ISOLATION_DOT_PREFIX = "trellis.instruments._agent._fresh."


def _assert_fresh_build_target_is_not_admitted(
    *,
    output_file_path: Path,
    output_module_path: str,
    module_name: str,
    source_module_path: str,
) -> None:
    """Layered defense: fail closed before any LLM call if the fresh-build target leaked to `_agent`.

    ``_agent/_fresh/`` and ``trellis.instruments._agent._fresh.*`` are the
    scratch isolation namespace used by the deterministic route fallback --
    they are *not* the admitted surface and must still be allowed here.
    """
    from trellis.agent.fresh_generated_boundary import FreshGeneratedBoundaryError

    normalized_path = str(output_module_path or "").replace("\\", "/")
    file_path_text = str(output_file_path).replace("\\", "/")
    module_text = str(module_name or "")

    in_isolation_namespace = (
        _FRESH_ISOLATION_SEGMENT in f"/{normalized_path}"
        or _FRESH_ISOLATION_SEGMENT in file_path_text
        or module_text.startswith(_FRESH_ISOLATION_DOT_PREFIX)
    )
    if in_isolation_namespace:
        return

    if (
        _is_agent_module_path(normalized_path)
        or "/trellis/instruments/_agent/" in file_path_text
        or module_text.startswith("trellis.instruments._agent.")
        or module_text == "trellis.instruments._agent"
    ):
        raise FreshGeneratedBoundaryError(
            "QUA-872: fresh-build resolver produced an admitted `_agent` target "
            f"for source `{source_module_path}` "
            f"(module_name={module_name!r}, module_path={output_module_path!r}); "
            "refusing to dispatch the LLM request."
        )


def _benchmark_fresh_build_root(
    request_metadata: Mapping[str, object] | None,
) -> Path | None:
    """Return the repo-local benchmark artifact root for fresh FinancePy pilot builds."""
    metadata = dict(request_metadata or {})
    if str(metadata.get("task_corpus") or "").strip().lower() != "benchmark_financepy":
        return None
    task_id = _normalize_fresh_build_token(metadata.get("task_id")) or "unknown_task"
    target = (
        _normalize_fresh_build_token(metadata.get("comparison_target"))
        or _normalize_fresh_build_token(metadata.get("preferred_method"))
        or "default"
    )
    return REPO_ROOT / "task_runs" / "financepy_benchmarks" / "generated" / task_id / target


def _benchmark_fresh_build_module_name(
    module_path: str,
    *,
    request_metadata: Mapping[str, object] | None,
) -> str:
    """Return a stable import name for benchmark-only fresh-generated modules."""
    metadata = dict(request_metadata or {})
    task_id = _normalize_fresh_build_token(metadata.get("task_id")) or "unknown_task"
    target = (
        _normalize_fresh_build_token(metadata.get("comparison_target"))
        or _normalize_fresh_build_token(metadata.get("preferred_method"))
        or "default"
    )
    stem = _normalize_fresh_build_token(Path(module_path).stem) or "module"
    return f"trellis_benchmarks._fresh.{task_id}.{target}.{stem}"


def _normalize_fresh_build_token(value: object | None) -> str:
    """Normalize path/module-name tokens for fresh benchmark artifacts."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")


def _write_generated_module(
    output_file_path: Path,
    output_module_path: str,
    code: str,
) -> Path:
    """Write generated source either into the package tree or a benchmark artifact root.

    Both paths are guarded: the benchmark artifact path must resolve under
    ``REPO_ROOT`` and the package-tree path is guarded by
    ``write_module``'s own ``_validate_write_target``.  Refs: QUA-382.
    """
    from trellis.agent.builder import validate_write_target

    normalized_module_path = str(output_module_path or "").replace("\\", "/")
    if normalized_module_path.startswith("task_runs/"):
        validate_write_target(output_file_path, REPO_ROOT, "_write_generated_module")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(code)
        return output_file_path
    return write_module(output_module_path, code)


def _is_agent_module_path(module_path: str) -> bool:
    """Return whether a planner step resolves under the generated adapter package."""
    normalized = module_path.replace("\\", "/").strip()
    return normalized.startswith("instruments/_agent/") or normalized.startswith(
        "trellis/instruments/_agent/"
    )


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
            modules.append(("trellis.models.transforms.heston", "Heston transform helper"))
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
        record_route_tags = {
            tag for tag in record_tags if tag.startswith("route:")
        }
        if route_id_tags and record_route_tags:
            return bool(route_id_tags & record_route_tags)
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

    if comparison_target == "heston_adi_pde":
        return (
            "For `heston_adi_pde`, bind canonical Heston parameters through the runtime binding boundary before assembling the PDE calculation.",
            "Use explicit `kappa`, `theta`, `xi`, `rho`, and `v0`; do not read Black vol surface nodes as live Heston parameters.",
            "Legacy aliases such as `theta_var`, `sigma_v`, and `initial_variance` are normalized only at the binding boundary.",
        )
    if comparison_target in {"pde_double_barrier", "mc_double_barrier"}:
        return (
            "For double-barrier targets, prefer `trellis.models.double_barrier_option` pricing-facing helpers before hand assembly.",
            "PDE adapters can call `price_double_barrier_option_pde_result(market_state, spec).price`; manual assembly must use the bounded `[lower_barrier, upper_barrier]` grid and absorbing boundaries.",
            "MC adapters can call `price_double_barrier_option_monte_carlo_result(market_state, spec).price`; manual assembly must use two barrier monitors or the shared state payoff primitive.",
            "Do not adapt the single-barrier Reiner-Rubinstein formula to lower/upper barriers by hand.",
        )
    if comparison_target in {"mc_autocall", "mc_autocall_qmc"}:
        sampling = "sobol" if comparison_target.endswith("_qmc") else "pseudo"
        return (
            "For autocallable targets, prefer `trellis.models.autocallable.price_autocallable_monte_carlo_result` before writing event branching by hand.",
            f"Call the helper with `sampling=\"{sampling}\"` for this comparison target; pseudo-MC targets must not require Sobol primitives.",
            "The helper owns GBM exact-path simulation, observation-step mapping, first-trigger redemption, coupon accrual, terminal protection, and deterministic discounting.",
            "Do not instantiate `GBM(spot=...)`; pass spot to the simulation engine and keep event branching tied to the task observation schedule.",
        )

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
            "Callable rate-tree routes compose the public short-rate, event-contract, and generic lattice primitives directly.",
            "Resolve settlement and maturity, then call `resolve_short_rate_lattice_inputs(...)`; do not reconstruct curve/volatility fallback rules inside the adapter.",
            "Select the resolved model through `MODEL_REGISTRY`, then compose `BINOMIAL_1F_TOPOLOGY`, `UNIFORM_ADDITIVE_MESH`, `TERM_STRUCTURE_TARGET(...)`, and `build_lattice(...)`.",
            "Build one `build_embedded_fixed_income_event_timeline(...)` result and pass it to `compile_embedded_fixed_income_lattice_contract_spec(..., expected_control_style=\"issuer_min\")`.",
            "Call `price_on_lattice(...)`, compare with `present_value_fixed_coupon_bond(...)`, and return the upper-bound-safe `min(tree_price, straight_price)`.",
            "Do not call `price_callable_bond_tree`; it is an excluded compatibility reference for this route.",
        )
    if (
        instrument == "puttable_bond"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Puttable bond rate-tree routes use the public short-rate resolver, embedded event compiler, and generic lattice rollback directly.",
            "Compose the same `MODEL_REGISTRY`, topology, mesh, calibration target, and `build_lattice(...)` surface as the callable route.",
            "Compile the shared event timeline with `expected_control_style=\"holder_max\"`; issuer-min semantics must fail closed.",
            "Call `price_on_lattice(...)`, compare with `present_value_fixed_coupon_bond(...)`, and return the lower-bound-safe `max(tree_price, straight_price)`.",
            "Do not call `price_callable_bond_tree`; it is an excluded compatibility reference for this route.",
        )
    if (
        instrument == "bermudan_swaption"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Bermudan swaption rate-tree routes compose the admitted generic lattice and contract surfaces directly.",
            "Use `normalize_explicit_dates(...)`, `year_fraction(...)`, `build_payment_timeline(...)`, and `lattice_step_from_time(...)` to bind exercise and fixed-leg schedules.",
            "Resolve market and Hull-White/BDT controls once with `resolve_bermudan_swaption_tree_inputs(...)`, then compose `BINOMIAL_1F_TOPOLOGY`, `UNIFORM_ADDITIVE_MESH`, `TERM_STRUCTURE_TARGET(...)`, and `build_lattice(...)`.",
            "Represent coupons and principal with `LatticeLinearClaimSpec` plus `LatticeContractSpec`, and obtain fixed-leg node values with `value_on_lattice(..., observation_steps=exercise_steps)`.",
            "Use each rollback observation's `continuation_values` for payer/receiver swap algebra so an exercise-time fixed coupon is not double counted.",
            "Attach `LatticeControlSpec(objective=\"holder_max\", ...)` to the option contract and return `price_on_lattice(...)`.",
            "Do not delegate to a product-level Bermudan helper, a product contract compiler, or a callable-bond route; they are compatibility/reference or different-product surfaces, not this route's construction authority.",
        )
    if (
        instrument == "bermudan_swaption"
        and pricing_method == "analytical"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "Bermudan swaption analytical comparators should compose the final-exercise European lower bound from public schedule, binding, and kernel surfaces.",
            "Import `normalize_explicit_dates` from `trellis.core.date_utils` plus `resolve_swaption_black76_inputs` and `price_swaption_black76_raw` from `trellis.models.rate_style_swaption`.",
            "Interpret `black76_european_lower_bound` as the European swaption exercisable only on the final Bermudan date.",
            "Normalize the exercise schedule, keep dates strictly after `market_state.settlement` and before `spec.swap_end`, return zero when none remain, then resolve once with `expiry_date=valid_exercise_dates[-1]` and call the raw kernel.",
            "Do not sum one European Black76 price per exercise date, do not rebuild forward-swap-rate or annuity loops inline, and do not use the product-level lower-bound helper as construction authority.",
        )
    if (
        instrument == "swaption"
        and pricing_method == "monte_carlo"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "European rate-style swaption Monte Carlo routes should compose the checked public event-aware primitives directly.",
            "Use `resolve_swaption_black76_inputs(...)` for the typed expiry basis and `build_payment_timeline(...)` from explicit `swap_start` through `swap_end`.",
            "Bind `resolve_hull_white_monte_carlo_process_inputs(...)`, then assemble `build_discounted_swap_pv_payload(...)`, `build_short_rate_discount_reducer(...)`, `EventAwareMonteCarloEvent`, and `EventAwareMonteCarloProblemSpec`.",
            "Compile with `build_event_aware_monte_carlo_problem(...)` and evaluate with `price_event_aware_monte_carlo(...)`, preserving day count, swap frequency, rate index, path/step/seed controls, and explicit comparison parameters.",
            "do not hardcode `sigma = 0.01` unless the semantic comparison contract supplies it, and do not synthesize a GBM equity path. Resolve Hull-White parameters through the bounded market/model path.",
            "`price_swaption_monte_carlo(...)` and `resolve_swaption_monte_carlo_problem(...)` are compatibility/reference surfaces, not generated construction authority.",
        )
    if (
        instrument == "swaption"
        and pricing_method == "rate_tree"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "European rate-style swaption tree routes should compose the checked public lattice primitives directly.",
            "Require the single-exercise forward-start boundary `swap_start == expiry_date`, then construct `BermudanSwaptionTreeSpec` with that one exercise date.",
            "Apply `resolve_swaption_curve_basis_spread(...)` to the tree strike, then call `resolve_bermudan_swaption_tree_inputs(...)` so schedule, volatility, mean reversion, horizon, and step controls stay on one resolved basis.",
            "Build the calibrated lattice with `build_lattice(BINOMIAL_1F_TOPOLOGY, UNIFORM_ADDITIVE_MESH, model, calibration_target=TERM_STRUCTURE_TARGET(market_state.discount), ...)`.",
            "Compile the claim with `compile_bermudan_swaption_contract_spec(...)` and evaluate it with generic `price_on_lattice(...)`.",
            "Preserve model, explicit comparison parameters, tree steps, day count, swap frequency, rate index, and payer/receiver direction.",
            "`price_swaption_tree(...)` and `build_swaption_tree_spec(...)` remain compatibility/reference APIs, not generated construction authority.",
            "Keep cap/floor-style period loops separate. This route is for a single-exercise European swaption comparison target, not for caplet or floorlet strips.",
        )
    if (
        instrument == "swaption"
        and pricing_method == "analytical"
        and stage in {"code_generation_failed", "semantic_validation_failed", "validation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "European rate-style swaption analytical routes should preserve the resolved-input Black76 composition.",
            "Import `resolve_swaption_black76_inputs` and `price_swaption_black76_raw` from `trellis.models.rate_style_swaption`; resolve once, then pass the typed result to the raw kernel.",
            "When the request supplies explicit Hull-White comparison parameters, pass `mean_reversion=` and `sigma=` to `resolve_swaption_black76_inputs(...)` so the analytical lane uses a Hull-White-implied Black vol instead of drifting to an unrelated market vol surface.",
            "Do not rebuild annuity, forward-swap-rate, expiry year-fraction, payment-count loops, or swaption-vol normalization inline, and do not use the product-level compatibility wrapper as construction authority.",
            "Keep cap/floor-style period loops separate. This composition is for single-exercise European swaptions, not for caplet or floorlet strips.",
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
            "Vanilla European PDE routes should compose the admitted single-state resolver, terminal payoff primitive, and event-aware PDE substrate.",
            "Use `resolve_single_state_diffusion_inputs` and `terminal_intrinsic_from_resolved`, then build an `EventAwarePDEProblemSpec` with explicit grid, operator, boundary, and terminal contracts.",
            "Map `implementation_target=theta_0.5` to `theta=0.5` and `implementation_target=theta_1.0` to `theta=1.0`.",
            "Call `build_event_aware_pde_problem`, `solve_event_aware_pde`, and `interpolate_pde_values`; do not route through `price_vanilla_equity_option_pde`.",
            "Keep dividend carry explicit in `EventAwarePDEOperatorSpec` and in the far-field boundary functions.",
        )
    if (
        instrument == "european_option"
        and pricing_method == "analytical"
        and comparison_target == "black_scholes"
        and stage in {"code_generation_failed", "actual_market_smoke_failed", "import_validation_failed"}
    ):
        return (
            "This retry is only for the plain Black-Scholes / Black76 comparator lane, not a general analytical decomposition route.",
            "Keep the module minimal: compute `T`, `df`, `sigma`, `dividend_yield`, and `forward = spec.spot * exp(-dividend_yield * T) / max(df, 1e-12)`, then call `black76_call` or `black76_put`.",
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
