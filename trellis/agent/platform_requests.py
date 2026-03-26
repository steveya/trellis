"""Canonical platform request/compiler layer.

This module unifies the front-door request surfaces:

- natural-language ask requests
- direct Session price/greeks/analytics requests
- Pipeline book requests
- structured user-defined product requests

All of them compile to the same internal shape before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from trellis.agent.codegen_guardrails import build_generation_plan
from trellis.agent.knowledge import (
    build_shared_knowledge_payload,
    retrieve_for_product_ir,
)
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.quant import (
    PricingPlan,
    select_pricing_method_for_product_ir,
)

if TYPE_CHECKING:
    from trellis.agent.knowledge.schema import ProductIR


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable snapshot of optional mapping metadata."""
    return MappingProxyType(dict(mapping or {}))


def _freeze_tuple(values: list | tuple | None) -> tuple:
    """Return tuple-normalized request fields for frozen dataclasses."""
    return tuple(values or ())


def _normalize_measures(measures: list | tuple | None) -> tuple[str, ...]:
    """Normalize heterogeneous measure inputs to canonical measure-name strings."""
    if not measures:
        return ()
    normalized: list[str] = []
    for measure in measures:
        if isinstance(measure, str):
            normalized.append(measure)
        elif isinstance(measure, dict) and measure:
            normalized.append(str(next(iter(measure))))
        else:
            normalized.append(getattr(measure, "name", type(measure).__name__.lower()))
    return tuple(normalized)


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
    model: str | None = None
    term_sheet: Any | None = None
    product_spec: Any | None = None
    comparison_spec: Any | None = None
    instrument: Any | None = None
    book: Any | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Freeze mutable metadata so requests remain hash- and trace-stable."""
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ExecutionPlan:
    """Execution decision after request compilation."""

    action: str
    reason: str
    measures: tuple[str, ...] = ()
    route_method: str | None = None
    requires_build: bool = False
    should_trace: bool = True


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
        return CompiledPlatformRequest(
            request=request,
            market_snapshot=request.market_snapshot,
            execution_plan=ExecutionPlan(
                action="price_book",
                reason="direct_book_request",
                measures=request.measures,
                route_method="direct_existing",
            ),
        )
    if request.instrument is not None:
        return CompiledPlatformRequest(
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
        measures=request.measures,
        route_method="direct_existing",
    )


def _compile_user_defined_request(request: PlatformRequest) -> CompiledPlatformRequest:
    """Compile a structured user-defined product into build/price execution artifacts."""
    from trellis.agent.user_defined_products import compile_user_defined_product

    compiled = compile_user_defined_product(
        request.product_spec,
        requested_measures=request.measures,
    )
    should_block = bool(
        compiled.generation_plan.blocker_report
        and compiled.generation_plan.blocker_report.should_block
    )
    action = "block" if should_block else (
        "compile_only" if request.request_type == "build" else "build_then_price"
    )
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=ExecutionPlan(
            action=action,
            reason="structured_user_defined_product",
            measures=request.measures,
            route_method=compiled.pricing_plan.method,
            requires_build=not should_block,
        ),
        product_ir=compiled.product_ir,
        pricing_plan=compiled.pricing_plan,
        generation_plan=compiled.generation_plan,
        blocker_report=compiled.generation_plan.blocker_report,
        new_primitive_workflow=compiled.generation_plan.new_primitive_workflow,
        knowledge=compiled.knowledge,
        knowledge_text=compiled.knowledge_text,
        review_knowledge_text=compiled.review_knowledge_text,
        routing_knowledge_text=compiled.routing_knowledge_text,
        knowledge_summary=compiled.knowledge_summary,
    )


def _compile_term_sheet_request(request: PlatformRequest) -> CompiledPlatformRequest:
    """Compile a natural-language/term-sheet request into existing or buildable paths."""
    from trellis.agent.ask import match_payoff

    term_sheet = request.term_sheet
    description = request.description or getattr(term_sheet, "raw_description", term_sheet.instrument_type)
    product_ir = decompose_to_ir(
        description,
        instrument_type=term_sheet.instrument_type,
    )

    match = match_payoff(term_sheet, request.settlement or date.today())
    if match is not None:
        knowledge_bundle = _shared_knowledge_bundle(product_ir)
        return CompiledPlatformRequest(
            request=request,
            market_snapshot=request.market_snapshot,
            execution_plan=ExecutionPlan(
                action="price_existing_payoff",
                reason="term_sheet_matched_existing_payoff",
                measures=request.measures,
                route_method="direct_existing",
            ),
            product_ir=product_ir,
            knowledge=knowledge_bundle["knowledge"],
            knowledge_text=knowledge_bundle["builder_text_distilled"],
            review_knowledge_text=knowledge_bundle["review_text_distilled"],
            routing_knowledge_text=knowledge_bundle["routing_text_distilled"],
            knowledge_summary=knowledge_bundle["summary"],
        )

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
    should_block = bool(
        generation_plan.blocker_report
        and generation_plan.blocker_report.should_block
    )
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=ExecutionPlan(
            action="block" if should_block else "build_then_price",
            reason="term_sheet_requires_build",
            measures=request.measures,
            route_method=pricing_plan.method,
            requires_build=not should_block,
        ),
        product_ir=product_ir,
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
    should_block = bool(
        generation_plan.blocker_report
        and generation_plan.blocker_report.should_block
    )
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=market_snapshot,
        execution_plan=ExecutionPlan(
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
        knowledge=knowledge_bundle["knowledge"],
        knowledge_text=knowledge_bundle["builder_text_distilled"],
        review_knowledge_text=knowledge_bundle["review_text_distilled"],
        routing_knowledge_text=knowledge_bundle["routing_text_distilled"],
        knowledge_summary=knowledge_bundle["summary"],
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
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=request.market_snapshot,
        execution_plan=ExecutionPlan(
            action="block" if should_block else "compare_methods",
            reason="comparison_request",
            measures=request.measures,
            route_method="comparison",
            requires_build=not should_block,
        ),
        product_ir=product_ir,
        knowledge_summary=knowledge_summary,
        routing_knowledge_text=method_plans[0].routing_knowledge_text if method_plans else "",
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
