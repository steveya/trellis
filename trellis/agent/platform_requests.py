"""Canonical platform request/compiler layer.

This module unifies the front-door request surfaces:

- natural-language ask requests
- direct Session price/greeks/analytics requests
- Pipeline book requests
- structured user-defined product requests

All of them compile to the same internal shape before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from trellis.agent.codegen_guardrails import build_generation_plan, enrich_generation_plan
from trellis.agent.family_contract_templates import (
    family_template_as_semantic_contract,
)
from trellis.agent.knowledge import (
    build_shared_knowledge_payload,
    retrieve_for_product_ir,
)
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.semantic_escalation import semantic_role_ownership_summary
from trellis.agent.market_binding import (
    market_binding_spec_summary,
    required_data_spec_summary,
)
from trellis.agent.quant import (
    PricingPlan,
    select_pricing_method_for_product_ir,
)
from trellis.agent.sensitivity_support import normalize_requested_outputs
from trellis.agent.valuation_context import (
    build_valuation_context,
    valuation_context_summary,
)

if TYPE_CHECKING:
    from trellis.agent.knowledge.schema import ProductIR


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Convert a mutable dict (or None) into a read-only MappingProxyType for use in frozen dataclasses."""
    return MappingProxyType(dict(mapping or {}))


def _freeze_tuple(values: list | tuple | None) -> tuple:
    """Convert a list or None to a tuple so it can be stored in a frozen dataclass."""
    return tuple(values or ())


def _normalize_measures(measures: list | tuple | None) -> tuple[str, ...]:
    """Backward-compatible shim for canonical requested-output normalization."""
    return normalize_requested_outputs(measures)


def _new_request_id(entry_point: str, request_type: str) -> str:
    """Generate a sortable request identifier tagged by entry point and type."""
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{entry_point}_{request_type}_{stamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class PlatformRequest:
    """Canonical request object for the full platform loop."""

    request_id: str
    request_type: str
    entry_point: str
    settlement: date | None = None
    market_snapshot: Any | None = None
    description: str | None = None
    instrument_type: str | None = None
    measures: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()
    measure_specs: tuple[Any, ...] = ()
    measure_context: Mapping[str, object] = field(default_factory=dict)
    model: str | None = None
    term_sheet: Any | None = None
    product_spec: Any | None = None
    comparison_spec: Any | None = None
    instrument: Any | None = None
    book: Any | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Freeze mutable metadata so requests remain hash- and trace-stable."""
        normalized_outputs = _normalize_measures(self.requested_outputs or self.measures)
        normalized_measures = _normalize_measures(self.measures or normalized_outputs)
        object.__setattr__(self, "requested_outputs", normalized_outputs)
        object.__setattr__(self, "measures", normalized_measures)
        object.__setattr__(self, "measure_specs", _freeze_tuple(self.measure_specs))
        object.__setattr__(self, "measure_context", _freeze_mapping(self.measure_context))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ExecutionPlan:
    """Execution decision after request compilation."""

    action: str
    reason: str
    measures: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()
    route_method: str | None = None
    requires_build: bool = False
    should_trace: bool = True

    def __post_init__(self):
        """Keep the legacy ``measures`` field aligned with canonical requested outputs."""
        normalized_outputs = _normalize_measures(self.requested_outputs or self.measures)
        normalized_measures = _normalize_measures(self.measures or normalized_outputs)
        object.__setattr__(self, "requested_outputs", normalized_outputs)
        object.__setattr__(self, "measures", normalized_measures)


@dataclass(frozen=True)
class ComparisonSpec:
    """Request-level comparison intent for multi-method tasks."""

    method_families: tuple[str, ...]
    reference_method: str | None = None
    validation_targets: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize methods and validation targets into immutable containers."""
        object.__setattr__(self, "method_families", _freeze_tuple(self.method_families))
        object.__setattr__(self, "validation_targets", _freeze_mapping(self.validation_targets))


@dataclass(frozen=True)
class ComparisonMethodPlan:
    """One method-specific compiled plan inside a comparison request."""

    preferred_method: str
    pricing_plan: PricingPlan
    generation_plan: Any | None = None
    blocker_report: Any | None = None
    new_primitive_workflow: Any | None = None
    knowledge: dict[str, Any] | None = None
    knowledge_text: str = ""
    review_knowledge_text: str = ""
    routing_knowledge_text: str = ""
    knowledge_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Freeze summarized knowledge metadata for safe downstream reuse."""
        object.__setattr__(self, "knowledge_summary", _freeze_mapping(self.knowledge_summary))


@dataclass(frozen=True)
class CompiledPlatformRequest:
    """Canonical compiled request ready for execution or blocking."""

    request: PlatformRequest
    market_snapshot: Any | None
    execution_plan: ExecutionPlan
    semantic_contract: Any | None = None
    semantic_blueprint: Any | None = None
    product_ir: ProductIR | None = None
    pricing_plan: PricingPlan | None = None
    generation_plan: Any | None = None
    validation_contract: Any | None = None
    blocker_report: Any | None = None
    new_primitive_workflow: Any | None = None
    knowledge: dict[str, Any] | None = None
    knowledge_text: str = ""
    review_knowledge_text: str = ""
    routing_knowledge_text: str = ""
    knowledge_summary: Mapping[str, object] = field(default_factory=dict)
    comparison_spec: ComparisonSpec | None = None
    comparison_method_plans: tuple[ComparisonMethodPlan, ...] = ()

    def __post_init__(self):
        """Freeze knowledge summaries carried on the compiled request envelope."""
        object.__setattr__(self, "knowledge_summary", _freeze_mapping(self.knowledge_summary))


def make_term_sheet_request(
    description: str,
    term_sheet,
    session,
    *,
    measures: list | None = None,
    model: str | None = None,
    request_type: str = "price",
) -> PlatformRequest:
    """Create a canonical platform request from the ask/term-sheet surface."""
    return PlatformRequest(
        request_id=_new_request_id("ask", request_type),
        request_type=request_type,
        entry_point="ask",
        settlement=session.settlement,
        market_snapshot=session.market_snapshot,
        description=description,
        instrument_type=term_sheet.instrument_type,
        measures=_normalize_measures(measures),
        measure_specs=_freeze_tuple(measures),
        model=model,
        term_sheet=term_sheet,
    )


def make_user_defined_request(
    spec,
    *,
    request_type: str = "build",
    measures: list | None = None,
    model: str | None = None,
) -> PlatformRequest:
    """Create a canonical request for a structured user-defined product spec."""
    return PlatformRequest(
        request_id=_new_request_id("user_defined", request_type),
        request_type=request_type,
        entry_point="user_defined",
        measures=_normalize_measures(measures),
        measure_specs=_freeze_tuple(measures),
        model=model,
        product_spec=spec,
    )


def make_comparison_request(
    *,
    description: str,
    instrument_type: str | None,
    methods: list[str] | tuple[str, ...],
    reference_method: str | None = None,
    validation_targets: Mapping[str, object] | None = None,
    market_snapshot=None,
    settlement: date | None = None,
    model: str | None = None,
    request_type: str = "build",
) -> PlatformRequest:
    """Create a request that asks the platform to compare multiple method families."""
    return PlatformRequest(
        request_id=_new_request_id("comparison", request_type),
        request_type=request_type,
        entry_point="comparison",
        settlement=settlement,
        market_snapshot=market_snapshot,
        description=description,
        instrument_type=instrument_type,
        model=model,
        comparison_spec=ComparisonSpec(
            method_families=tuple(normalize_method(method) for method in methods),
            reference_method=normalize_method(reference_method) if reference_method else None,
            validation_targets=validation_targets or {},
        ),
    )


def make_session_request(
    session,
    *,
    instrument=None,
    book=None,
    request_type: str = "price",
    measures: list | None = None,
    measure_context: Mapping[str, object] | None = None,
    description: str | None = None,
    model: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PlatformRequest:
    """Create the canonical request shape for direct ``Session`` entry points."""
    return PlatformRequest(
        request_id=_new_request_id("session", request_type),
        request_type=request_type,
        entry_point="session",
        settlement=session.settlement,
        market_snapshot=session.market_snapshot,
        description=description,
        measures=_normalize_measures(measures),
        measure_specs=_freeze_tuple(measures),
        measure_context=measure_context or {},
        model=model,
        instrument=instrument,
        book=book,
        metadata={
            "agent_enabled": bool(getattr(session, "agent_enabled", False)),
            "discount_curve_name": getattr(session, "discount_curve_name", None),
            "vol_surface_name": getattr(session, "vol_surface_name", None),
            **dict(metadata or {}),
        },
    )


def make_pipeline_request(
    *,
    book,
    market_snapshot=None,
    settlement: date | None = None,
    measures: list | None = None,
    measure_context: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PlatformRequest:
    """Create the canonical request shape for batch ``Pipeline`` execution."""
    return PlatformRequest(
        request_id=_new_request_id("pipeline", "price"),
        request_type="price",
        entry_point="pipeline",
        settlement=settlement,
        market_snapshot=market_snapshot,
        measures=_normalize_measures(measures),
        measure_specs=_freeze_tuple(measures),
        measure_context=measure_context or {},
        book=book,
        metadata=metadata or {},
    )


def compile_platform_request(request: PlatformRequest) -> CompiledPlatformRequest:
    """Compile any front-door request into semantic execution artifacts."""
    if request.comparison_spec is not None:
        return _compile_comparison_request(request)
    if request.product_spec is not None:
        return _compile_user_defined_request(request)
    if request.term_sheet is not None:
        return _compile_term_sheet_request(request)
    if request.book is not None:
        return _finalize_compiled_request(
            request=request,
            market_snapshot=request.market_snapshot,
            execution_plan=_execution_plan_for_request(
                request,
                action="price_book",
                reason="direct_book_request",
                route_method="direct_existing",
            ),
        )
    if request.instrument is not None:
        return _finalize_compiled_request(
            request=request,
            market_snapshot=request.market_snapshot,
            execution_plan=_direct_instrument_execution_plan(request),
        )
    raise ValueError("PlatformRequest has no compilable payload")


def _direct_instrument_execution_plan(request: PlatformRequest) -> ExecutionPlan:
    """Map direct session requests onto existing-instrument execution actions."""
    if request.request_type == "greeks":
        action = "compute_greeks"
    elif request.request_type == "analytics":
        action = "analyze_existing_instrument"
    else:
        action = "price_existing_instrument"
    return ExecutionPlan(
        action=action,
        reason="direct_session_request",
        requested_outputs=request.requested_outputs,
        route_method="direct_existing",
    )


def _execution_plan_for_request(
    request: PlatformRequest,
    *,
    action: str,
    reason: str,
    route_method: str | None = None,
    requires_build: bool = False,
) -> ExecutionPlan:
    """Create a request-scoped execution plan with the standard measure wiring."""
    return ExecutionPlan(
        action=action,
        reason=reason,
        requested_outputs=request.requested_outputs,
        route_method=route_method,
        requires_build=requires_build,
    )


def _generation_plan_should_block(generation_plan) -> bool:
    """Return whether a generation plan should block execution."""
    blocker_report = getattr(generation_plan, "blocker_report", None)
    return bool(blocker_report and blocker_report.should_block)


def _compile_user_defined_request(request: PlatformRequest) -> CompiledPlatformRequest:
    """Compile a structured user-defined product into build/price execution artifacts."""
    from trellis.agent.user_defined_products import compile_user_defined_product

    compiled = compile_user_defined_product(
        request.product_spec,
        requested_measures=request.measures,
    )
    should_block = _generation_plan_should_block(compiled.generation_plan)
    action = "block" if should_block else (
        "compile_only" if request.request_type == "build" else "build_then_price"
    )
    return _finalize_compiled_request(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=_execution_plan_for_request(
            request,
            action=action,
            reason="structured_user_defined_product",
            route_method=compiled.pricing_plan.method,
            requires_build=not should_block,
        ),
        product_ir=compiled.product_ir,
        pricing_plan=compiled.pricing_plan,
        generation_plan=compiled.generation_plan,
        blocker_report=compiled.generation_plan.blocker_report,
        new_primitive_workflow=compiled.generation_plan.new_primitive_workflow,
        knowledge_bundle={
            "knowledge": compiled.knowledge,
            "builder_text_distilled": compiled.knowledge_text,
            "review_text_distilled": compiled.review_knowledge_text,
            "routing_text_distilled": compiled.routing_knowledge_text,
            "summary": compiled.knowledge_summary,
        },
    )


def _known_family_id(
    *,
    description: str | None,
    instrument_type: str | None,
    term_sheet=None,
) -> str | None:
    """Resolve a known checked-in family contract id from request context."""
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized_instrument == "quanto_option":
        return "quanto_option"
    lower_description = (description or "").lower()
    if "quanto" in lower_description:
        return "quanto_option"
    term_sheet_type = getattr(term_sheet, "instrument_type", "")
    normalized_term_sheet = str(term_sheet_type).strip().lower().replace(" ", "_")
    if normalized_term_sheet == "quanto_option":
        return "quanto_option"
    return None


def _compile_known_family_request(
    *,
    family_id: str,
    request: PlatformRequest,
    reason: str,
    description: str,
    preferred_method: str | None = None,
) -> CompiledPlatformRequest:
    """Compile a request through a checked-in family contract template.

    When the family template can be converted to a SemanticContract the
    request is routed through the unified semantic compilation pipeline.
    Known-family request routing must not fall back to the deprecated
    family-contract compiler.
    """
    semantic_contract = family_template_as_semantic_contract(family_id)
    if semantic_contract is None:
        raise ValueError(
            "Known checked-in family template "
            f"{family_id!r} has no semantic bridge. "
            "Retire the registration or add a semantic contract bridge before "
            "routing it through the platform request compiler."
        )
    return _compile_semantic_request(
        request=request,
        semantic_contract=semantic_contract,
        reason=reason or "family_template_request",
        preferred_method=preferred_method,
    )


def _draft_semantic_contract(
    description: str,
    *,
    instrument_type: str | None = None,
    term_sheet=None,
):
    """Draft the canonical semantic contract from a front-door request."""
    from trellis.agent.semantic_contracts import (
        _looks_like_ranked_observation_basket_request,
        draft_semantic_contract,
    )

    try:
        return draft_semantic_contract(
            description,
            instrument_type=instrument_type,
            term_sheet=term_sheet,
        )
    except ValueError:
        semantic_text = "\n".join(
            part
            for part in (
                description,
                instrument_type,
                getattr(term_sheet, "raw_description", None),
                getattr(term_sheet, "instrument_type", None),
            )
            if part
        )
        if _looks_like_ranked_observation_basket_request(semantic_text):
            raise
        return None


def _request_with_semantic_metadata(
    request: PlatformRequest,
    semantic_contract,
    *,
    semantic_blueprint=None,
    semantic_role_ownership: Mapping[str, object] | None = None,
) -> PlatformRequest:
    """Attach a YAML-safe semantic summary to request metadata."""
    from trellis.agent.semantic_contracts import semantic_contract_summary

    metadata = dict(request.metadata or {})
    metadata["semantic_contract"] = semantic_contract_summary(semantic_contract)
    if semantic_blueprint is not None:
        metadata["semantic_blueprint"] = _semantic_blueprint_summary(semantic_blueprint)
    if semantic_role_ownership is not None:
        metadata["semantic_role_ownership"] = dict(semantic_role_ownership)
    return replace(request, metadata=metadata)


def _semantic_blueprint_summary(semantic_blueprint) -> dict[str, object]:
    """Return a compact YAML-safe summary of lowering-relevant blueprint state."""
    lowering = getattr(semantic_blueprint, "dsl_lowering", None)
    lane_plan = getattr(semantic_blueprint, "lane_plan", None)
    control_styles = tuple(
        getattr(style, "value", str(style))
        for style in getattr(lowering, "control_styles", ())
    )
    return {
        "preferred_method": semantic_blueprint.preferred_method,
        "primitive_routes": list(getattr(semantic_blueprint, "primitive_routes", ()) or ()),
        "route_modules": list(getattr(semantic_blueprint, "route_modules", ()) or ()),
        "dsl_route": getattr(lowering, "route_id", None),
        "dsl_route_family": getattr(lowering, "route_family", None),
        "dsl_expr_kind": None if lowering is None or getattr(lowering, "normalized_expr", None) is None else type(lowering.normalized_expr).__name__,
        "dsl_helper_refs": list(getattr(lowering, "helper_refs", ()) or ()),
        "dsl_control_styles": list(control_styles),
        "dsl_family_ir_type": None if lowering is None or getattr(lowering, "family_ir", None) is None else type(lowering.family_ir).__name__,
        "dsl_family_ir": _yaml_safe_value(getattr(lowering, "family_ir", None)),
        "dsl_target_bindings": [
            {
                "module": binding.module,
                "symbol": binding.symbol,
                "role": binding.role,
                "required": binding.required,
            }
            for binding in getattr(lowering, "target_bindings", ()) or ()
        ],
        "dsl_lowering_errors": [
            {
                "route_id": item.route_id,
                "stage": item.stage,
                "code": item.code,
                "message": item.message,
            }
            for item in getattr(lowering, "errors", ()) or ()
        ],
        "lane_plan": _yaml_safe_value(lane_plan),
        "requested_outputs": list(getattr(semantic_blueprint, "requested_outputs", ()) or ()),
        "valuation_context": valuation_context_summary(semantic_blueprint.valuation_context)
        if getattr(semantic_blueprint, "valuation_context", None) is not None
        else None,
        "required_data_spec": required_data_spec_summary(semantic_blueprint.required_data_spec)
        if getattr(semantic_blueprint, "required_data_spec", None) is not None
        else None,
        "market_binding_spec": market_binding_spec_summary(semantic_blueprint.market_binding_spec)
        if getattr(semantic_blueprint, "market_binding_spec", None) is not None
        else None,
    }


def _yaml_safe_value(value):
    """Project dataclass-heavy lowering metadata onto YAML-safe primitives."""
    if value is None:
        return None
    if is_dataclass(value):
        return {
            field.name: _yaml_safe_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _yaml_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_yaml_safe_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and type(value).__module__ != "builtins":
        return enum_value
    return value


def _request_with_semantic_gap_metadata(
    request: PlatformRequest,
    semantic_gap,
    *,
    semantic_role_ownership: Mapping[str, object] | None = None,
) -> PlatformRequest:
    """Attach a YAML-safe semantic-gap summary to request metadata."""
    from trellis.agent.semantic_contract_validation import semantic_gap_summary

    metadata = dict(request.metadata or {})
    metadata["semantic_gap"] = semantic_gap_summary(semantic_gap)
    if semantic_role_ownership is not None:
        metadata["semantic_role_ownership"] = dict(semantic_role_ownership)
    return replace(request, metadata=metadata)


def _request_with_semantic_extension_metadata(
    request: PlatformRequest,
    semantic_gap,
    semantic_extension,
    *,
    semantic_role_ownership: Mapping[str, object] | None = None,
) -> PlatformRequest:
    """Attach a YAML-safe semantic-extension summary to request metadata."""
    from trellis.agent.semantic_contract_validation import (
        semantic_extension_summary,
        semantic_gap_summary,
    )

    metadata = dict(request.metadata or {})
    metadata["semantic_gap"] = semantic_gap_summary(semantic_gap)
    metadata["semantic_extension"] = semantic_extension_summary(semantic_extension)
    if semantic_role_ownership is not None:
        metadata["semantic_role_ownership"] = dict(semantic_role_ownership)
    return replace(request, metadata=metadata)


def _record_semantic_extension_artifact(
    request: PlatformRequest,
    semantic_gap,
    semantic_extension,
    *,
    route_method: str | None = None,
    semantic_role_ownership: Mapping[str, object] | None = None,
) -> str | None:
    """Persist the semantic-extension trace and any reusable lesson artifact."""
    from trellis.agent.knowledge.promotion import record_semantic_extension_trace
    from trellis.agent.semantic_contract_validation import (
        semantic_extension_summary,
        semantic_gap_summary,
    )
    from trellis.agent.semantic_escalation import semantic_role_ownership_summary as _semantic_role_ownership_summary

    if not isinstance(semantic_gap, dict):
        semantic_gap = semantic_gap_summary(semantic_gap)
    if not isinstance(semantic_extension, dict):
        semantic_extension = semantic_extension_summary(semantic_extension)
    if semantic_role_ownership is None:
        semantic_role_ownership = _semantic_role_ownership_summary(
            stage="trace_handoff",
            semantic_gap=semantic_gap,
            semantic_extension=semantic_extension,
        )

    try:
        return record_semantic_extension_trace(
            request_id=request.request_id,
            request_text=request.description or "",
            instrument_type=request.instrument_type,
            semantic_gap=semantic_gap,
            semantic_extension=semantic_extension,
            route_method=route_method,
            semantic_role_ownership=semantic_role_ownership,
        )
    except Exception:
        return None


def _finalize_compiled_request(
    *,
    request: PlatformRequest,
    market_snapshot,
    execution_plan: ExecutionPlan,
    semantic_contract=None,
    semantic_blueprint=None,
    product_ir=None,
    pricing_plan=None,
    generation_plan=None,
    blocker_report=None,
    new_primitive_workflow=None,
    knowledge_bundle: dict[str, Any] | None = None,
    comparison_spec: ComparisonSpec | None = None,
    comparison_method_plans: tuple[ComparisonMethodPlan, ...] = (),
) -> CompiledPlatformRequest:
    """Construct a compiled request with the standard knowledge and plan fields."""
    bundle = knowledge_bundle or {}
    request_metadata = dict(request.metadata or {})
    if (
        "semantic_role_ownership" not in request_metadata
        and pricing_plan is not None
    ):
        route_stage = "route_assembly"
        route_artifact = "GenerationPlan" if generation_plan is not None else "PricingPlan"
        request_metadata["semantic_role_ownership"] = semantic_role_ownership_summary(
            stage=route_stage,
            trigger_condition=getattr(pricing_plan, "selection_reason", None) or "pricing_plan_selection",
            artifact_kind=route_artifact,
            semantic_contract=semantic_contract is not None,
        )
    validation_contract = None
    if any(
        item is not None
        for item in (pricing_plan, product_ir, semantic_blueprint)
    ):
        from trellis.agent.validation_contract import (
            compile_validation_contract,
            validation_contract_summary,
        )

        validation_contract = compile_validation_contract(
            request=request,
            product_ir=product_ir,
            pricing_plan=pricing_plan,
            generation_plan=generation_plan,
            semantic_blueprint=semantic_blueprint,
            comparison_spec=comparison_spec,
            instrument_type=request.instrument_type,
        )
        validation_summary = validation_contract_summary(validation_contract)
        if validation_summary is not None:
            request_metadata["validation_contract"] = validation_summary
    if generation_plan is not None and semantic_blueprint is not None:
        generation_plan = enrich_generation_plan(
            generation_plan,
            request=request,
            semantic_blueprint=semantic_blueprint,
            validation_contract=validation_contract,
        )
    if generation_plan is not None:
        from trellis.agent.route_registry import (
            compile_route_binding_authority,
            route_binding_authority_summary,
        )

        if getattr(generation_plan, "route_binding_authority", None) is None:
            generation_plan = replace(
                generation_plan,
                route_binding_authority=compile_route_binding_authority(
                    generation_plan=generation_plan,
                    validation_contract=validation_contract,
                    semantic_blueprint=semantic_blueprint,
                    product_ir=product_ir,
                    request=request,
                ),
            )
        authority_summary = route_binding_authority_summary(
            getattr(generation_plan, "route_binding_authority", None)
        )
        if authority_summary is not None:
            request_metadata["route_binding_authority"] = authority_summary
    if request_metadata != dict(request.metadata or {}):
        request = replace(request, metadata=request_metadata)
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=market_snapshot,
        execution_plan=execution_plan,
        semantic_contract=semantic_contract,
        semantic_blueprint=semantic_blueprint,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        validation_contract=validation_contract,
        blocker_report=blocker_report,
        new_primitive_workflow=new_primitive_workflow,
        knowledge=bundle.get("knowledge"),
        knowledge_text=bundle.get("builder_text_distilled", ""),
        review_knowledge_text=bundle.get("review_text_distilled", ""),
        routing_knowledge_text=bundle.get("routing_text_distilled", ""),
        knowledge_summary=bundle.get("summary", {}),
        comparison_spec=comparison_spec,
        comparison_method_plans=comparison_method_plans,
    )


def _compile_semantic_request(
    *,
    request: PlatformRequest,
    semantic_contract,
    reason: str,
    preferred_method: str | None = None,
) -> CompiledPlatformRequest:
    """Compile a semantic contract into execution artifacts."""
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract

    valuation_context = _valuation_context_for_request(
        request,
        semantic_contract=semantic_contract,
    )
    semantic_blueprint = compile_semantic_contract(
        semantic_contract,
        valuation_context=valuation_context,
        requested_measures=request.measures,
        preferred_method=preferred_method,
    )
    request = _request_with_semantic_metadata(
        request,
        semantic_blueprint.contract,
        semantic_blueprint=semantic_blueprint,
    )
    pricing_plan = select_pricing_method_for_product_ir(
        semantic_blueprint.product_ir,
        preferred_method=semantic_blueprint.preferred_method,
        requested_measures=semantic_blueprint.requested_outputs,
        context_description=request.description,
    )
    inspected_modules = semantic_blueprint.route_modules
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=getattr(semantic_blueprint.product_ir, "instrument", None),
        inspected_modules=inspected_modules,
        product_ir=semantic_blueprint.product_ir,
    )
    if not getattr(semantic_blueprint, "primitive_routes", ()):
        uncertainty_flags = tuple(
            dict.fromkeys(
                (
                    *getattr(generation_plan, "uncertainty_flags", ()),
                    "primitive_plan_not_available",
                )
            )
        )
        generation_plan = replace(
            generation_plan,
            primitive_plan=None,
            blocker_report=None,
            new_primitive_workflow=None,
            uncertainty_flags=uncertainty_flags,
        )
    knowledge_bundle = _shared_knowledge_bundle(
        semantic_blueprint.product_ir,
        preferred_method=pricing_plan.method,
        generation_plan=generation_plan,
        knowledge_profile=_knowledge_profile_for_request(request),
    )
    should_block = bool(
        generation_plan.blocker_report
        and generation_plan.blocker_report.should_block
    )
    action = "block" if should_block else (
        "compile_only" if request.request_type == "build" else "build_then_price"
    )
    return _finalize_compiled_request(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=ExecutionPlan(
            action=action,
            reason=reason,
            requested_outputs=request.requested_outputs,
            route_method=pricing_plan.method,
            requires_build=not should_block,
        ),
        semantic_contract=semantic_blueprint.contract,
        semantic_blueprint=semantic_blueprint,
        product_ir=semantic_blueprint.product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        blocker_report=generation_plan.blocker_report,
        new_primitive_workflow=generation_plan.new_primitive_workflow,
        knowledge_bundle=knowledge_bundle,
    )


def _compile_term_sheet_request(request: PlatformRequest) -> CompiledPlatformRequest:
    """Compile a natural-language/term-sheet request into existing or buildable paths."""
    from trellis.agent.ask import match_payoff

    term_sheet = request.term_sheet
    description = request.description or getattr(term_sheet, "raw_description", term_sheet.instrument_type)
    semantic_contract = _draft_semantic_contract(
        description,
        instrument_type=request.instrument_type,
        term_sheet=term_sheet,
    )
    if semantic_contract is not None:
        return _compile_semantic_request(
            request=request,
            semantic_contract=semantic_contract,
            reason="semantic_contract_request",
        )
    known_family = _known_family_id(
        description=description,
        instrument_type=request.instrument_type,
        term_sheet=term_sheet,
    )
    if known_family is not None:
        return _compile_known_family_request(
            family_id=known_family,
            request=request,
            reason="known_family_term_sheet_request",
            description=description,
        )
    product_ir = decompose_to_ir(
        description,
        instrument_type=term_sheet.instrument_type,
    )

    match = match_payoff(term_sheet, request.settlement or date.today())
    if match is not None:
        knowledge_bundle = _shared_knowledge_bundle(
            product_ir,
            preferred_method="direct_existing",
            knowledge_profile=_knowledge_profile_for_request(request),
        )
        return _finalize_compiled_request(
            request=request,
            market_snapshot=request.market_snapshot,
            execution_plan=_execution_plan_for_request(
                request,
                action="price_existing_payoff",
                reason="term_sheet_matched_existing_payoff",
                route_method="direct_existing",
            ),
            product_ir=product_ir,
            knowledge_bundle=knowledge_bundle,
        )

    from trellis.agent.semantic_contract_validation import classify_semantic_gap
    from trellis.agent.semantic_contract_validation import propose_semantic_extension
    from trellis.agent.semantic_contract_validation import semantic_extension_summary
    from trellis.agent.semantic_contract_validation import semantic_gap_summary

    semantic_gap = classify_semantic_gap(
        description,
        instrument_type=term_sheet.instrument_type,
        term_sheet=term_sheet,
    )
    semantic_extension = propose_semantic_extension(semantic_gap)
    semantic_gap_data = semantic_gap_summary(semantic_gap)
    semantic_extension_data = semantic_extension_summary(semantic_extension)
    request = _request_with_semantic_extension_metadata(
        request,
        semantic_gap,
        semantic_extension,
        semantic_role_ownership=semantic_role_ownership_summary(
            stage="primitive_proposal",
            semantic_gap=semantic_gap_data,
            semantic_extension=semantic_extension_data,
        ),
    )
    semantic_extension_trace = _record_semantic_extension_artifact(
        request,
        semantic_gap,
        semantic_extension,
        semantic_role_ownership=semantic_role_ownership_summary(
            stage="trace_handoff",
            semantic_gap=semantic_gap_data,
            semantic_extension=semantic_extension_data,
        ),
    )
    if semantic_extension_trace:
        metadata = dict(request.metadata or {})
        metadata["semantic_extension_trace"] = semantic_extension_trace
        request = replace(request, metadata=metadata)

    pricing_plan = select_pricing_method_for_product_ir(
        product_ir,
        preferred_method=None,
        requested_measures=request.measures,
        context_description=description,
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=term_sheet.instrument_type,
        inspected_modules=tuple(pricing_plan.method_modules),
        product_ir=product_ir,
    )
    knowledge_bundle = _shared_knowledge_bundle(
        product_ir,
        preferred_method=pricing_plan.method,
        generation_plan=generation_plan,
        knowledge_profile=_knowledge_profile_for_request(request),
    )
    should_block = _generation_plan_should_block(generation_plan)
    return _finalize_compiled_request(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=_execution_plan_for_request(
            request,
            action="block" if should_block else "build_then_price",
            reason="term_sheet_requires_build",
            route_method=pricing_plan.method,
            requires_build=not should_block,
        ),
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        blocker_report=generation_plan.blocker_report,
        new_primitive_workflow=generation_plan.new_primitive_workflow,
        knowledge_bundle=knowledge_bundle,
    )


def compile_build_request(
    description: str,
    *,
    instrument_type: str | None = None,
    market_snapshot=None,
    settlement: date | None = None,
    model: str | None = None,
    preferred_method: str | None = None,
    measures: list | None = None,
    knowledge_profile: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CompiledPlatformRequest:
    """Compile a free-form build request through the canonical path."""
    request = PlatformRequest(
        request_id=_new_request_id("executor", "build"),
        request_type="build",
        entry_point="executor",
        settlement=settlement,
        market_snapshot=market_snapshot,
        description=description,
        instrument_type=instrument_type,
        measures=_normalize_measures(measures),
        model=model,
        metadata={
            **(metadata or {}),
            **(
                {"knowledge_profile": knowledge_profile}
                if knowledge_profile is not None
                else {}
            ),
        },
    )
    semantic_contract = _draft_semantic_contract(
        description,
        instrument_type=instrument_type,
    )
    if semantic_contract is not None:
        return _compile_semantic_request(
            request=request,
            semantic_contract=semantic_contract,
            reason="semantic_contract_request",
            preferred_method=preferred_method,
        )
    known_family = _known_family_id(
        description=description,
        instrument_type=instrument_type,
    )
    if known_family is not None:
        return _compile_known_family_request(
            family_id=known_family,
            request=request,
            reason="known_family_build_request",
            description=description,
            preferred_method=preferred_method,
        )
    from trellis.agent.semantic_contract_validation import classify_semantic_gap
    from trellis.agent.semantic_contract_validation import propose_semantic_extension
    from trellis.agent.semantic_contract_validation import semantic_extension_summary
    from trellis.agent.semantic_contract_validation import semantic_gap_summary

    semantic_gap = classify_semantic_gap(
        description,
        instrument_type=instrument_type,
    )
    semantic_extension = propose_semantic_extension(semantic_gap)
    semantic_gap_data = semantic_gap_summary(semantic_gap)
    semantic_extension_data = semantic_extension_summary(semantic_extension)
    request = _request_with_semantic_extension_metadata(
        request,
        semantic_gap,
        semantic_extension,
        semantic_role_ownership=semantic_role_ownership_summary(
            stage="primitive_proposal",
            semantic_gap=semantic_gap_data,
            semantic_extension=semantic_extension_data,
        ),
    )
    semantic_extension_trace = _record_semantic_extension_artifact(
        request,
        semantic_gap,
        semantic_extension,
        semantic_role_ownership=semantic_role_ownership_summary(
            stage="trace_handoff",
            semantic_gap=semantic_gap_data,
            semantic_extension=semantic_extension_data,
        ),
    )
    if semantic_extension_trace:
        metadata = dict(request.metadata or {})
        metadata["semantic_extension_trace"] = semantic_extension_trace
        request = replace(request, metadata=metadata)
    product_ir = decompose_to_ir(description, instrument_type=instrument_type)
    pricing_plan = select_pricing_method_for_product_ir(
        product_ir,
        preferred_method=preferred_method,
        requested_measures=_normalize_measures(measures),
        context_description=description,
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=tuple(pricing_plan.method_modules),
        product_ir=product_ir,
    )
    knowledge_bundle = _shared_knowledge_bundle(
        product_ir,
        preferred_method=pricing_plan.method,
        generation_plan=generation_plan,
        knowledge_profile=_knowledge_profile_for_request(request),
    )
    should_block = _generation_plan_should_block(generation_plan)
    return _finalize_compiled_request(
        request=request,
        market_snapshot=market_snapshot,
        execution_plan=_execution_plan_for_request(
            request,
            action="block" if should_block else "compile_only",
            reason="free_form_build_request",
            route_method=pricing_plan.method,
            requires_build=not should_block,
        ),
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        blocker_report=generation_plan.blocker_report,
        new_primitive_workflow=generation_plan.new_primitive_workflow,
        knowledge_bundle=knowledge_bundle,
    )


def _compile_comparison_request(request: PlatformRequest) -> CompiledPlatformRequest:
    """Compile a multi-method comparison request into per-method plans."""
    comparison_spec = request.comparison_spec
    if comparison_spec is None:
        raise ValueError("Comparison request missing comparison spec")
    if not request.description:
        raise ValueError("Comparison request requires a description")

    product_ir = decompose_to_ir(
        request.description,
        instrument_type=request.instrument_type,
    )
    method_plans = tuple(
        _compile_comparison_method_plan(
            product_ir=product_ir,
            instrument_type=request.instrument_type,
            preferred_method=method,
            knowledge_profile=_knowledge_profile_for_request(request),
            measures=request.measures,
            description=request.description,
        )
        for method in comparison_spec.method_families
    )
    should_block = any(
        plan.blocker_report is not None and plan.blocker_report.should_block
        for plan in method_plans
    )
    knowledge_summary = _aggregate_comparison_knowledge(method_plans, product_ir)
    return _finalize_compiled_request(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=_execution_plan_for_request(
            request,
            action="block" if should_block else "compare_methods",
            reason="comparison_request",
            route_method="comparison",
            requires_build=not should_block,
        ),
        product_ir=product_ir,
        knowledge_bundle={
            "summary": knowledge_summary,
            "routing_text_distilled": method_plans[0].routing_knowledge_text if method_plans else "",
        },
        comparison_spec=comparison_spec,
        comparison_method_plans=method_plans,
    )


def _compile_comparison_method_plan(
    *,
    product_ir,
    instrument_type: str | None,
    preferred_method: str,
    knowledge_profile: str = "default",
    measures: tuple[str, ...] = (),
    description: str | None = None,
) -> ComparisonMethodPlan:
    """Compile one candidate method route inside a multi-method comparison request."""
    pricing_plan = select_pricing_method_for_product_ir(
        product_ir,
        preferred_method=preferred_method,
        requested_measures=measures,
        context_description=description,
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=tuple(pricing_plan.method_modules),
        product_ir=product_ir,
    )
    knowledge_bundle = _shared_knowledge_bundle(
        product_ir,
        preferred_method=pricing_plan.method,
        generation_plan=generation_plan,
        knowledge_profile=knowledge_profile,
    )
    return ComparisonMethodPlan(
        preferred_method=preferred_method,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        blocker_report=generation_plan.blocker_report,
        new_primitive_workflow=generation_plan.new_primitive_workflow,
        knowledge=knowledge_bundle["knowledge"],
        knowledge_text=knowledge_bundle["builder_text_distilled"],
        review_knowledge_text=knowledge_bundle["review_text_distilled"],
        routing_knowledge_text=knowledge_bundle["routing_text_distilled"],
        knowledge_summary=knowledge_bundle["summary"],
    )


def pricing_plan_for_request(request: PlatformRequest) -> PricingPlan | None:
    """Compatibility helper for callers that only need the pricing plan."""
    compiled = compile_platform_request(request)
    return compiled.pricing_plan


def _knowledge_profile_for_request(request: PlatformRequest | None) -> str:
    """Return the requested knowledge profile for compiler and build-loop prompts."""
    metadata = getattr(request, "metadata", None) or {}
    profile = str(metadata.get("knowledge_profile") or "").strip().lower().replace(" ", "_")
    return profile or "default"


def _knowledge_light_bundle() -> dict[str, Any]:
    """Return a compiler-first knowledge bundle for tranche-2 proving runs."""
    builder_text = (
        "## Knowledge-Light Mode\n"
        "- Use the semantic contract, lane obligations, DSL lowering, validation contract, and approved imports as the primary contract.\n"
        "- Treat route-specific lessons, cookbook examples, and prompt-local helper lore as intentionally unavailable.\n"
        "- If the compiler emitted an exact backend binding, use it directly. Otherwise build only the smallest lane-consistent kernel required by the construction steps."
    )
    review_text = (
        "## Knowledge-Light Review Mode\n"
        "- Review against the semantic contract, lane obligations, lowering boundary, and validation contract first.\n"
        "- Do not assume missing cookbook or lesson guidance implies the build is invalid; focus on whether the generated code satisfied the compiled lane contract."
    )
    routing_text = (
        "## Knowledge-Light Routing Mode\n"
        "- Prefer compiler-emitted lane obligations over route-local prompt heuristics.\n"
        "- Treat exact backend bindings as optional only when the compiler did not emit one."
    )
    summary = {
        "knowledge_profile": "knowledge_light",
        "retrieval_mode": "compiler_first_minimal_prompt_surface",
    }
    return {
        "knowledge": {"knowledge_profile": "knowledge_light"},
        "builder_text_distilled": builder_text,
        "builder_text": builder_text,
        "builder_text_expanded": builder_text,
        "review_text_distilled": review_text,
        "review_text": review_text,
        "review_text_expanded": review_text,
        "routing_text_distilled": routing_text,
        "routing_text": routing_text,
        "routing_text_expanded": routing_text,
        "summary": summary,
    }


def _shared_knowledge_bundle(
    product_ir,
    *,
    preferred_method: str | None = None,
    generation_plan=None,
    knowledge_profile: str = "default",
) -> dict[str, Any]:
    """Build all prompt/trace views for one ProductIR retrieval result."""
    if knowledge_profile == "knowledge_light":
        return _knowledge_light_bundle()
    knowledge = retrieve_for_product_ir(
        product_ir,
        preferred_method=preferred_method,
    )
    route_ids: tuple[str, ...] = ()
    if generation_plan is not None:
        primitive_plan = getattr(generation_plan, "primitive_plan", None)
        route_ids = tuple(
            dict.fromkeys(
                item
                for item in (
                    getattr(generation_plan, "lowering_route_id", None),
                    getattr(primitive_plan, "route", None),
                )
                if isinstance(item, str) and item.strip()
            )
        )
    return build_shared_knowledge_payload(
        knowledge,
        pricing_method=preferred_method,
        route_ids=route_ids,
        route_families=tuple(getattr(product_ir, "route_families", ()) or ()),
    )


def _aggregate_comparison_knowledge(
    method_plans: tuple[ComparisonMethodPlan, ...],
    product_ir,
) -> Mapping[str, object]:
    """Merge compact shared-knowledge summaries across comparison routes."""
    summary = {
        "principle_ids": [],
        "lesson_ids": [],
        "lesson_titles": [],
        "cookbook_methods": [],
        "data_contracts": [],
        "unresolved_primitives": [],
        "selected_artifact_ids": [],
        "selected_artifact_titles": [],
    }
    for plan in method_plans:
        plan_summary = dict(plan.knowledge_summary or {})
        summary["principle_ids"].extend(plan_summary.get("principle_ids", ()))
        summary["lesson_ids"].extend(plan_summary.get("lesson_ids", ()))
        summary["lesson_titles"].extend(plan_summary.get("lesson_titles", ()))
        if plan_summary.get("cookbook_method"):
            summary["cookbook_methods"].append(plan_summary["cookbook_method"])
        summary["data_contracts"].extend(plan_summary.get("data_contracts", ()))
        summary["unresolved_primitives"].extend(plan_summary.get("unresolved_primitives", ()))
        summary["selected_artifact_ids"].extend(plan_summary.get("selected_artifact_ids", ()))
        summary["selected_artifact_titles"].extend(plan_summary.get("selected_artifact_titles", ()))

    merged = {
        key: tuple(sorted(set(values)))
        for key, values in summary.items()
    }
    merged["instrument"] = getattr(product_ir, "instrument", None)
    merged["payoff_family"] = getattr(product_ir, "payoff_family", None)
    merged["exercise_style"] = getattr(product_ir, "exercise_style", None)
    merged["model_family"] = getattr(product_ir, "model_family", None)
    return merged


def _request_reporting_currency(
    request: PlatformRequest,
    *,
    semantic_contract=None,
) -> str:
    """Infer the best available reporting currency from request and semantic context."""
    term_sheet_currency = getattr(getattr(request, "term_sheet", None), "currency", None)
    if term_sheet_currency:
        return str(term_sheet_currency).strip()
    if semantic_contract is not None:
        conventions = getattr(getattr(semantic_contract, "product", None), "conventions", None)
        for attr in ("reporting_currency", "payment_currency"):
            value = getattr(conventions, attr, None)
            if value:
                return str(value).strip()
    return ""


def _valuation_context_for_request(
    request: PlatformRequest,
    *,
    semantic_contract=None,
):
    """Build the tranche-1 valuation context for one semantic request."""
    return build_valuation_context(
        market_snapshot=request.market_snapshot,
        model_spec=request.model,
        reporting_currency=_request_reporting_currency(
            request,
            semantic_contract=semantic_contract,
        ),
        requested_outputs=request.requested_outputs,
    )
