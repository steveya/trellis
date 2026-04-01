"""Canonical platform request/compiler layer.

This module unifies the front-door request surfaces:

- natural-language ask requests
- direct Session price/greeks/analytics requests
- Pipeline book requests
- structured user-defined product requests

All of them compile to the same internal shape before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from trellis.agent.codegen_guardrails import build_generation_plan
from trellis.agent.family_contract_compiler import compile_family_contract
from trellis.agent.family_contract_templates import (
    family_template_as_semantic_contract,
    get_family_contract_template,
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
    select_pricing_method_for_family_blueprint,
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
    family_blueprint: Any | None = None
    semantic_contract: Any | None = None
    semantic_blueprint: Any | None = None
    product_ir: ProductIR | None = None
    pricing_plan: PricingPlan | None = None
    generation_plan: Any | None = None
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
    description: str | None = None,
    model: str | None = None,
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
        model=model,
        instrument=instrument,
        book=book,
        metadata={"agent_enabled": bool(getattr(session, "agent_enabled", False))},
    )


def make_pipeline_request(
    *,
    book,
    market_snapshot=None,
    settlement: date | None = None,
    measures: list | None = None,
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
    The legacy family-contract compiler is used only as a fallback.
    """
    semantic_contract = family_template_as_semantic_contract(family_id)
    if semantic_contract is not None:
        return _compile_semantic_request(
            request=request,
            semantic_contract=semantic_contract,
            reason=reason or "family_template_request",
            preferred_method=preferred_method,
        )
    # Fallback: family has no semantic conversion yet.
    blueprint = compile_family_contract(
        get_family_contract_template(family_id),
        requested_measures=request.measures,
    )
    pricing_plan = select_pricing_method_for_family_blueprint(
        blueprint,
        preferred_method=preferred_method,
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=blueprint.product_ir.instrument,
        inspected_modules=tuple(pricing_plan.method_modules),
        product_ir=blueprint.product_ir,
    )
    knowledge_bundle = _shared_knowledge_bundle(
        blueprint.product_ir,
        preferred_method=pricing_plan.method,
    )
    should_block = _generation_plan_should_block(generation_plan)
    action = "block" if should_block else (
        "compile_only" if request.request_type == "build" else "build_then_price"
    )
    return _finalize_compiled_request(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=_execution_plan_for_request(
            request,
            action=action,
            reason=reason,
            route_method=pricing_plan.method,
            requires_build=not should_block,
        ),
        family_blueprint=blueprint,
        product_ir=blueprint.product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        blocker_report=generation_plan.blocker_report,
        new_primitive_workflow=generation_plan.new_primitive_workflow,
        knowledge_bundle=knowledge_bundle,
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
        "dsl_helper_refs": list(getattr(lowering, "helper_refs", ()) or ()),
        "dsl_control_styles": list(control_styles),
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
    family_blueprint=None,
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
        request = replace(request, metadata=request_metadata)
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=market_snapshot,
        execution_plan=execution_plan,
        family_blueprint=family_blueprint,
        semantic_contract=semantic_contract,
        semantic_blueprint=semantic_blueprint,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
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
    knowledge_bundle = _shared_knowledge_bundle(
        semantic_blueprint.product_ir,
        preferred_method=pricing_plan.method,
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
        knowledge_bundle = _shared_knowledge_bundle(product_ir)
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
        metadata=metadata or {},
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


def _shared_knowledge_bundle(product_ir, *, preferred_method: str | None = None) -> dict[str, Any]:
    """Build all prompt/trace views for one ProductIR retrieval result."""
    knowledge = retrieve_for_product_ir(
        product_ir,
        preferred_method=preferred_method,
    )
    return build_shared_knowledge_payload(knowledge)


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
