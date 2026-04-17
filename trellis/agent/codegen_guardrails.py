"""Structured generation context and import validation for agent-built modules."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from dataclasses import dataclass, replace

from trellis.agent.blocker_planning import BlockerReport, plan_blockers, render_blocker_report
from trellis.agent.lane_obligations import compile_fallback_lane_construction_plan
from trellis.agent.knowledge.instructions import resolve_instruction_records
from trellis.agent.knowledge.schema import (
    InstructionRecord,
    PackageMap,
    ProductIR,
    ResolvedInstructionSet,
    SymbolMap,
    TestMap,
)
from trellis.agent.new_primitive_workflow import (
    NewPrimitiveWorkflow,
    plan_new_primitive_workflow,
    render_new_primitive_workflow,
)
from trellis.agent.knowledge.import_registry import (
    get_package_map,
    get_repo_revision,
    get_symbol_map,
    get_test_map,
    get_registry_snapshot,
    is_valid_import,
    list_module_exports,
    module_exists,
    suggest_tests_for_symbol,
)
from trellis.agent.knowledge.methods import normalize_method


COMMON_APPROVED_MODULES = (
    "trellis.core.date_utils",
    "trellis.core.differentiable",
    "trellis.core.market_state",
    "trellis.core.payoff",
    "trellis.core.types",
    "trellis.models.black",
)

METHOD_FAMILY_PREFIXES = {
    "analytical": (
        "trellis.models.black",
        "trellis.models.analytical",
    ),
    "rate_tree": (
        "trellis.models.trees",
    ),
    "monte_carlo": (
        "trellis.models.monte_carlo",
    ),
    "qmc": (
        "trellis.models.qmc",
        "trellis.models.monte_carlo",
        "trellis.models.processes",
    ),
    "pde_solver": (
        "trellis.models.pde",
    ),
    "fft_pricing": (
        "trellis.models.transforms",
        "trellis.models.processes",
    ),
    "copula": (
        "trellis.models.copulas",
    ),
    "waterfall": (
        "trellis.models.cashflow_engine",
    ),
}

METHOD_TEST_TARGETS = {
    "analytical": ("tests/test_agent/test_build_loop.py",),
    "rate_tree": (
        "tests/test_agent/test_build_loop.py",
        "tests/test_agent/test_callable_bond.py",
    ),
    "monte_carlo": ("tests/test_agent/test_build_loop.py",),
    "qmc": (
        "tests/test_agent/test_build_loop.py",
        "tests/test_models/test_monte_carlo/test_mc.py",
    ),
    "pde_solver": ("tests/test_agent/test_build_loop.py",),
    "fft_pricing": ("tests/test_agent/test_build_loop.py",),
    "copula": ("tests/test_agent/test_build_loop.py",),
    "waterfall": ("tests/test_agent/test_build_loop.py",),
}

FAMILY_SUPPORT_MODULES = {
    "callable_bond": (
        "trellis.models.callable_bond_tree",
    ),
    "puttable_bond": (
        "trellis.models.callable_bond_tree",
    ),
    "bermudan_swaption": (
        "trellis.models.bermudan_swaption_tree",
    ),
    "quanto_option": (
        "trellis.models.resolution.quanto",
        "trellis.models.analytical.quanto",
        "trellis.models.monte_carlo.quanto",
    ),
    "nth_to_default": (
        "trellis.models.copulas",
        "trellis.models.copulas.gaussian",
        "trellis.models.copulas.student_t",
        "trellis.models.copulas.factor",
    ),
    "credit_default_swap": (),
}

INSTRUMENT_TEST_TARGETS = {
    "american_option": (
        "tests/test_tasks/test_t07_american_put_3way.py",
        "tests/test_models/test_trees/test_trees.py",
    ),
    "barrier_option": (
        "tests/test_tasks/test_t09_barrier.py",
        "tests/test_models/test_barrier.py",
    ),
    "swaption": ("tests/test_agent/test_swaption_demo.py",),
    "bermudan_swaption": ("tests/test_tasks/test_t04_bermudan_swaption.py",),
    "callable_bond": ("tests/test_agent/test_callable_bond.py",),
    "puttable_bond": ("tests/test_tasks/test_t05_puttable_bond.py",),
}

_GENERATION_PLAN_CACHE: dict[tuple[object, ...], "GenerationPlan"] = {}
_GENERATION_PLAN_CACHE_HITS = 0
_GENERATION_PLAN_CACHE_MISSES = 0


@dataclass(frozen=True)
class PrimitiveRef:
    """A single reusable primitive selected for assembly-first generation."""

    module: str
    symbol: str
    role: str
    required: bool = True
    excluded: bool = False  # if True, generated code must NOT call this symbol


@dataclass(frozen=True)
class PrimitivePlan:
    """Deterministic route + primitive selection for a product/method pair."""

    route: str
    engine_family: str
    primitives: tuple[PrimitiveRef, ...]
    adapters: tuple[str, ...]
    blockers: tuple[str, ...]
    route_family: str = ""
    backend_binding_id: str = ""
    backend_binding_aliases: tuple[str, ...] = ()
    backend_exact_target_refs: tuple[str, ...] = ()
    backend_helper_refs: tuple[str, ...] = ()
    backend_pricing_kernel_refs: tuple[str, ...] = ()
    backend_schedule_builder_refs: tuple[str, ...] = ()
    backend_cashflow_engine_refs: tuple[str, ...] = ()
    backend_market_binding_refs: tuple[str, ...] = ()
    backend_compatibility_alias_policy: str = "operator_visible"
    notes: tuple[str, ...] = ()
    score: float = 0.0


@dataclass(frozen=True)
class GenerationPlan:
    """Structured context passed into module generation and validation."""

    method: str
    instrument_type: str | None
    inspected_modules: tuple[str, ...]
    approved_modules: tuple[str, ...]
    symbols_to_reuse: tuple[str, ...]
    proposed_tests: tuple[str, ...]
    uncertainty_flags: tuple[str, ...] = ()
    primitive_plan: PrimitivePlan | None = None
    blocker_report: BlockerReport | None = None
    new_primitive_workflow: NewPrimitiveWorkflow | None = None
    repo_revision: str = ""
    symbol_map: SymbolMap | None = None
    package_map: PackageMap | None = None
    test_map: TestMap | None = None
    resolved_instructions: ResolvedInstructionSet | None = None
    semantic_contract_id: str = ""
    semantic_requested_instrument_type: str = ""
    semantic_compatibility_bridge_status: str = ""
    semantic_matched_wrapper: str = ""
    semantic_instrument_class: str = ""
    semantic_payoff_family: str = ""
    semantic_underlier_structure: str = ""
    valuation_market_source: str = ""
    valuation_reporting_currency: str = ""
    valuation_requested_outputs: tuple[str, ...] = ()
    lane_family: str = ""
    lane_plan_kind: str = ""
    lane_timeline_roles: tuple[str, ...] = ()
    lane_market_requirements: tuple[str, ...] = ()
    lane_state_obligations: tuple[str, ...] = ()
    lane_control_obligations: tuple[str, ...] = ()
    lane_construction_steps: tuple[str, ...] = ()
    lane_reusable_primitives: tuple[str, ...] = ()
    lane_exact_binding_refs: tuple[str, ...] = ()
    lane_unresolved_primitives: tuple[str, ...] = ()
    backend_binding_id: str = ""
    backend_binding_aliases: tuple[str, ...] = ()
    backend_exact_target_refs: tuple[str, ...] = ()
    backend_helper_refs: tuple[str, ...] = ()
    backend_pricing_kernel_refs: tuple[str, ...] = ()
    backend_schedule_builder_refs: tuple[str, ...] = ()
    backend_cashflow_engine_refs: tuple[str, ...] = ()
    backend_market_binding_refs: tuple[str, ...] = ()
    backend_engine_family: str = ""
    backend_route_family: str = ""
    backend_compatibility_alias_policy: str = "operator_visible"
    lowering_route_id: str = ""
    lowering_expr_kind: str = ""
    lowering_family_ir_type: str = ""
    lowering_helper_refs: tuple[str, ...] = ()
    validation_bundle_id: str = ""
    validation_check_ids: tuple[str, ...] = ()
    validation_residual_risks: tuple[str, ...] = ()
    route_binding_authority: object | None = None


@dataclass(frozen=True)
class ImportReference:
    """A single Trellis import extracted from generated code."""

    module: str
    symbol: str | None
    kind: str
    lineno: int


@dataclass(frozen=True)
class ImportValidationReport:
    """Validation result for Trellis imports inside generated code."""

    imports: tuple[ImportReference, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return ``True`` when every Trellis import is real and approved."""
        return not self.errors


@dataclass(frozen=True)
class GeneratedSourceSanitizationReport:
    """Deterministic sanitation report for raw LLM-generated source text."""

    raw_source: str
    sanitized_source: str
    source_status: str
    fence_removed: bool
    fence_language: str | None
    fence_count: int
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return ``True`` when the source was safely sanitized."""
        return not self.errors

    @property
    def raw_line_count(self) -> int:
        """Return the number of raw source lines."""
        return len(self.raw_source.splitlines())

    @property
    def sanitized_line_count(self) -> int:
        """Return the number of sanitized source lines."""
        return len(self.sanitized_source.splitlines())


_OPENING_FENCE_RE = re.compile(r"^\s*```(?P<language>[A-Za-z0-9_+\-.]*)\s*$")
_CLOSING_FENCE_RE = re.compile(r"^\s*```\s*$")


def build_generation_plan(
    *,
    pricing_plan,
    instrument_type: str | None,
    inspected_modules: tuple[str, ...],
    product_ir: ProductIR | None = None,
) -> GenerationPlan:
    """Build a deterministic generation plan from quant + reference context."""
    global _GENERATION_PLAN_CACHE_HITS, _GENERATION_PLAN_CACHE_MISSES
    repo_revision = get_repo_revision()

    cache_key = _generation_plan_cache_key(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=inspected_modules,
        product_ir=product_ir,
        repo_revision=repo_revision,
    )
    cached = _GENERATION_PLAN_CACHE.get(cache_key)
    if cached is not None:
        _GENERATION_PLAN_CACHE_HITS += 1
        return cached

    _GENERATION_PLAN_CACHE_MISSES += 1
    method = normalize_method(pricing_plan.method) if pricing_plan else "analytical"
    approved = set(COMMON_APPROVED_MODULES)
    approved.update(inspected_modules)
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    approved.update(FAMILY_SUPPORT_MODULES.get(normalized_instrument, ()))
    route_selection_ir = _augment_product_ir_for_requested_method(
        product_ir,
        preferred_method=method,
    )
    primitive_plan = build_primitive_plan(
        pricing_plan=pricing_plan,
        product_ir=route_selection_ir,
    )
    symbol_map = get_symbol_map()
    package_map = get_package_map()
    test_map = get_test_map()

    for module_path in getattr(pricing_plan, "method_modules", ()) or ():
        approved.add(module_path)

    for prefix in METHOD_FAMILY_PREFIXES.get(method, ()):
        approved.update(_modules_with_prefix(prefix))

    if primitive_plan is not None:
        for primitive in primitive_plan.primitives:
            approved.add(primitive.module)

    symbols = set()
    for module_path in approved:
        symbols.update(list_module_exports(module_path))

    proposed_tests = list(METHOD_TEST_TARGETS.get(method, ("tests/test_agent",)))
    if instrument_type:
        proposed_tests.extend(INSTRUMENT_TEST_TARGETS.get(instrument_type, ()))

    uncertainty_flags: list[str] = []
    blocker_report = None
    new_primitive_workflow = None
    if not getattr(pricing_plan, "method_modules", None):
        uncertainty_flags.append("quant_plan_has_no_explicit_method_modules")
    if not instrument_type:
        uncertainty_flags.append("instrument_type_not_provided")
    if primitive_plan is None:
        uncertainty_flags.append("primitive_plan_not_available")
    elif primitive_plan.blockers:
        uncertainty_flags.append("primitive_plan_has_blockers")
        blocker_report = plan_blockers(
            primitive_plan.blockers,
            product_ir=product_ir,
        )
        new_primitive_workflow = plan_new_primitive_workflow(
            blocker_report,
            product_ir=product_ir,
        )
    lane_plan = compile_fallback_lane_construction_plan(
        preferred_method=method,
        required_market_data=tuple(sorted(getattr(pricing_plan, "required_market_data", ()) or ())),
        primitive_plan=primitive_plan,
        product_ir=route_selection_ir,
        instrument_type=instrument_type,
    )
    lane_plan_kind = str(getattr(lane_plan, "plan_kind", "") or "")
    backend_exact_target_refs = tuple(getattr(primitive_plan, "backend_exact_target_refs", ()) or ())
    if not backend_exact_target_refs and lane_plan_kind == "exact_target_binding":
        backend_exact_target_refs = tuple(getattr(lane_plan, "exact_target_refs", ()) or ())
    backend_binding_id = (
        str(getattr(primitive_plan, "backend_binding_id", "") or "").strip()
        or (str(backend_exact_target_refs[0]).strip() if backend_exact_target_refs else "")
    )

    plan = GenerationPlan(
        method=method,
        instrument_type=instrument_type,
        inspected_modules=tuple(sorted(set(inspected_modules))),
        approved_modules=tuple(sorted(approved)),
        symbols_to_reuse=tuple(sorted(symbols)),
        proposed_tests=tuple(sorted(set(proposed_tests))),
        uncertainty_flags=tuple(sorted(set(uncertainty_flags))),
        primitive_plan=primitive_plan,
        blocker_report=blocker_report,
        new_primitive_workflow=new_primitive_workflow,
        repo_revision=repo_revision,
        symbol_map=symbol_map,
        package_map=package_map,
        test_map=test_map,
        lane_family=str(getattr(lane_plan, "lane_family", "") or ""),
        lane_plan_kind=lane_plan_kind,
        lane_timeline_roles=tuple(getattr(lane_plan, "timeline_roles", ()) or ()),
        lane_market_requirements=tuple(getattr(lane_plan, "market_requirements", ()) or ()),
        lane_state_obligations=tuple(getattr(lane_plan, "state_obligations", ()) or ()),
        lane_control_obligations=tuple(getattr(lane_plan, "control_obligations", ()) or ()),
        lane_construction_steps=tuple(getattr(lane_plan, "construction_steps", ()) or ()),
        lane_reusable_primitives=tuple(
            str(getattr(binding, "primitive_ref", "") or "")
            for binding in (getattr(lane_plan, "reusable_bindings", ()) or ())
            if str(getattr(binding, "primitive_ref", "") or "").strip()
        ),
        lane_exact_binding_refs=tuple(getattr(lane_plan, "exact_target_refs", ()) or ()),
        lane_unresolved_primitives=tuple(getattr(lane_plan, "unresolved_primitives", ()) or ()),
        backend_binding_id=backend_binding_id,
        backend_binding_aliases=tuple(getattr(primitive_plan, "backend_binding_aliases", ()) or ()),
        backend_exact_target_refs=backend_exact_target_refs,
        backend_helper_refs=tuple(getattr(primitive_plan, "backend_helper_refs", ()) or ()),
        backend_pricing_kernel_refs=tuple(
            getattr(primitive_plan, "backend_pricing_kernel_refs", ()) or ()
        ),
        backend_schedule_builder_refs=tuple(
            getattr(primitive_plan, "backend_schedule_builder_refs", ()) or ()
        ),
        backend_cashflow_engine_refs=tuple(
            getattr(primitive_plan, "backend_cashflow_engine_refs", ()) or ()
        ),
        backend_market_binding_refs=tuple(
            getattr(primitive_plan, "backend_market_binding_refs", ()) or ()
        ),
        backend_engine_family=str(getattr(primitive_plan, "engine_family", "") or ""),
        backend_route_family=str(getattr(primitive_plan, "route_family", "") or ""),
        backend_compatibility_alias_policy=str(
            getattr(primitive_plan, "backend_compatibility_alias_policy", None)
            or "operator_visible"
        ),
    )
    plan = replace(
        plan,
        resolved_instructions=_resolve_generation_instructions(plan),
    )
    _GENERATION_PLAN_CACHE[cache_key] = plan
    return plan


def generation_plan_cache_stats() -> dict[str, int]:
    """Return generation-plan cache statistics."""
    return {
        "hits": _GENERATION_PLAN_CACHE_HITS,
        "misses": _GENERATION_PLAN_CACHE_MISSES,
        "size": len(_GENERATION_PLAN_CACHE),
    }


def clear_generation_plan_cache() -> None:
    """Clear the deterministic generation-plan cache."""
    global _GENERATION_PLAN_CACHE_HITS, _GENERATION_PLAN_CACHE_MISSES
    _GENERATION_PLAN_CACHE.clear()
    _GENERATION_PLAN_CACHE_HITS = 0
    _GENERATION_PLAN_CACHE_MISSES = 0


def _generation_plan_cache_key(
    *,
    pricing_plan,
    instrument_type: str | None,
    inspected_modules: tuple[str, ...],
    product_ir: ProductIR | None,
    repo_revision: str,
) -> tuple[object, ...]:
    """Build a stable cache key for deterministic generation planning."""
    pricing_plan_key = None
    if pricing_plan is not None:
        pricing_plan_key = (
            normalize_method(pricing_plan.method),
            tuple(pricing_plan.method_modules),
            tuple(sorted(pricing_plan.required_market_data)),
            pricing_plan.model_to_build,
            pricing_plan.reasoning,
            tuple(pricing_plan.modeling_requirements),
        )
    return (
        pricing_plan_key,
        instrument_type,
        tuple(sorted(set(inspected_modules))),
        product_ir,
        repo_revision,
    )


def _render_lane_obligation_lines(plan: GenerationPlan, *, compact: bool) -> list[str]:
    """Render compiler-emitted lane obligations ahead of route-local details."""
    if not plan.lane_family:
        return []

    lines = [
        "- Lane obligations:",
        f"  - Lane family: `{plan.lane_family}`",
    ]
    if plan.lane_plan_kind:
        lines.append(f"  - Plan kind: `{plan.lane_plan_kind}`")
    if plan.lane_timeline_roles:
        roles = ", ".join(f"`{role}`" for role in plan.lane_timeline_roles[: (4 if compact else 8)])
        lines.append(f"  - Timeline roles: {roles}")
    if plan.lane_market_requirements:
        reqs = ", ".join(
            f"`{item}`" for item in plan.lane_market_requirements[: (4 if compact else 8)]
        )
        lines.append(f"  - Market bindings: {reqs}")
    if plan.lane_control_obligations:
        # QUA-880 Codex P1 round 1: exercise-MC routes append 3
        # exercise-policy obligations (approved / implemented / name
        # invariant) after the usual Monte Carlo control fields (which
        # already include base control/role/measure/numeraire + semantic
        # mirrors + possibly-many event_kinds + calibration fields).  The
        # baseline total can be 10+ obligations, so caps of 3 / 6 clipped
        # the exercise-policy guidance off the end.  Raise caps to 20
        # (both compact and non-compact) so migrated obligations stay
        # visible; obligations are short tagged strings so the cost is
        # negligible compared to dropping guidance.
        cap = 20
        controls = ", ".join(
            f"`{item}`" for item in plan.lane_control_obligations[:cap]
        )
        lines.append(f"  - Control semantics: {controls}")
    if plan.lane_state_obligations:
        states = ", ".join(
            f"`{item}`" for item in plan.lane_state_obligations[: (4 if compact else 8)]
        )
        lines.append(f"  - State obligations: {states}")
    if plan.lane_construction_steps:
        lines.append("  - Construction steps:")
        # QUA-880: PDE lanes emit 3 base + up to 5 kernel-contract steps
        # (previously route-card notes).  Raise both caps so rendered
        # cards keep the kernel-contract invariants visible:
        # - compact (used by ``render_generation_route_card``) -> 8 so
        #   PDE base + 5 kernel contracts fit in the short card
        # - non-compact -> 10 for extended variants
        lines.extend(
            f"    - {step}"
            for step in plan.lane_construction_steps[: (8 if compact else 10)]
        )
    if plan.lane_reusable_primitives:
        label = "Exact backend bindings" if plan.lane_exact_binding_refs else "Reusable primitives"
        lines.append(f"  - {label}:")
        lines.extend(
            f"    - `{primitive}`"
            for primitive in plan.lane_reusable_primitives[: (6 if compact else 10)]
        )
        signature_lines = _exact_binding_signature_lines(plan, compact=compact)
        if signature_lines:
            lines.append("  - Exact binding signatures:")
            lines.extend(signature_lines)
    if plan.lane_unresolved_primitives:
        lines.append("  - Unresolved primitives:")
        lines.extend(
            f"    - `{item}`"
            for item in plan.lane_unresolved_primitives[: (4 if compact else 8)]
        )
    return lines


def _exact_binding_signature_lines(plan: GenerationPlan, *, compact: bool) -> list[str]:
    """Render callable signatures for compiler-selected exact bindings."""
    refs = tuple(
        ref
        for ref in plan.lane_exact_binding_refs[: (2 if compact else 4)]
        if isinstance(ref, str) and ref.strip()
    )
    if not refs:
        return []

    lines: list[str] = []
    for ref in refs:
        module_name, _, symbol_name = ref.rpartition(".")
        if not module_name or not symbol_name:
            continue
        if not module_exists(module_name) or not is_valid_import(module_name, symbol_name):
            continue
        try:
            obj = getattr(importlib.import_module(module_name), symbol_name)
            signature = inspect.signature(obj)
        except Exception:
            continue
        lines.append(f"    - `{symbol_name}{signature}`")
    return lines


def _render_backend_binding_lines(plan: GenerationPlan, *, compact: bool) -> list[str]:
    """Render route/helper bindings as a secondary backend-binding section."""
    if plan.primitive_plan is None:
        return []

    lines = ["- Backend binding:"]
    lines.append(f"  - Route: `{plan.primitive_plan.route}`")
    lines.append(f"  - Engine family: `{plan.primitive_plan.engine_family}`")
    if plan.primitive_plan.route_family:
        lines.append(f"  - Route family: `{plan.primitive_plan.route_family}`")
    if not compact:
        lines.append(f"  - Route score: `{plan.primitive_plan.score:.2f}`")
    if plan.primitive_plan.primitives:
        lines.append("  - Selected primitives:")
        lines.extend(
            f"    - `{primitive.module}.{primitive.symbol}` ({primitive.role})"
            for primitive in plan.primitive_plan.primitives[: (8 if compact else 12)]
        )
    resolved_instructions = _resolve_generation_instructions(plan)
    if resolved_instructions.effective_instructions:
        lines.append("  - Resolved instructions:")
        lines.extend(
            f"    - [{instruction.instruction_type}] {instruction.statement}"
            for instruction in resolved_instructions.effective_instructions[: (8 if compact else 12)]
        )
        schedule_instructions = _schedule_related_instructions(resolved_instructions)
        if schedule_instructions:
            lines.append("  - Schedule construction:")
            lines.extend(
                f"    - [{instruction.instruction_type}] {instruction.statement}"
                for instruction in schedule_instructions[: (8 if compact else 12)]
            )
    if resolved_instructions.conflicts:
        lines.append("  - Instruction conflicts:")
        lines.extend(
            f"    - {conflict.reason}"
            for conflict in resolved_instructions.conflicts[: (4 if compact else 8)]
        )
    if plan.primitive_plan.adapters:
        lines.append("  - Required adapters:")
        lines.extend(
            f"    - `{adapter}`"
            for adapter in plan.primitive_plan.adapters[: (6 if compact else 10)]
        )
    if plan.primitive_plan.notes and compact:
        lines.append("  - Backend notes:")
        lines.extend(f"    - {note}" for note in plan.primitive_plan.notes[:4])
    return lines


def _render_route_authority_lines(plan: GenerationPlan, *, compact: bool) -> list[str]:
    """Render the structured route-binding authority packet."""
    if plan.route_binding_authority is None:
        return []

    from trellis.agent.route_registry import route_binding_authority_summary, should_surface_route_alias

    authority = route_binding_authority_summary(plan.route_binding_authority)
    if not authority:
        return []

    lines = ["- Route authority:"]
    authority_bits = []
    backend_binding = dict(authority.get("backend_binding") or {})
    if backend_binding.get("binding_id"):
        authority_bits.append(f"binding=`{backend_binding['binding_id']}`")
    if backend_binding.get("engine_family"):
        authority_bits.append(f"engine=`{backend_binding['engine_family']}`")
    if authority.get("authority_kind"):
        authority_bits.append(f"authority=`{authority['authority_kind']}`")
    if authority_bits:
        lines.append(f"  - {', '.join(authority_bits)}")
    if should_surface_route_alias(authority):
        lines.append(f"  - Route alias: `{authority['route_id']}`")
    if authority.get("validation_bundle_id"):
        lines.append(f"  - Validation bundle: `{authority['validation_bundle_id']}`")
    check_ids = tuple(authority.get("validation_check_ids") or ())
    if check_ids:
        selected = check_ids[: (4 if compact else 6)]
        lines.append(
            "  - Validation checks: "
            + ", ".join(f"`{check_id}`" for check_id in selected)
        )
    canary_task_ids = tuple(authority.get("canary_task_ids") or ())
    if canary_task_ids:
        selected = ", ".join(f"`{task_id}`" for task_id in canary_task_ids[: (2 if compact else 4)])
        lines.append(f"  - Canary coverage: canaries={selected}")
    helper_refs = tuple(backend_binding.get("helper_refs") or ())
    if helper_refs:
        selected = helper_refs[: (2 if compact else 4)]
        lines.append(
            "  - Helper authority: "
            + ", ".join(f"`{helper}`" for helper in selected)
        )
    exact_refs = tuple(backend_binding.get("exact_target_refs") or ())
    if exact_refs:
        selected = exact_refs[: (2 if compact else 4)]
        lines.append(
            "  - Exact target bindings: "
            + ", ".join(f"`{ref}`" for ref in selected)
        )
    admissibility_failures = tuple(backend_binding.get("admissibility_failures") or ())
    if admissibility_failures:
        lines.append(
            "  - Admissibility failures: "
            + ", ".join(f"`{failure}`" for failure in admissibility_failures[: (3 if compact else 6)])
        )
    return lines


def render_generation_plan(plan: GenerationPlan) -> str:
    """Format the structured generation plan for prompt injection."""
    lines = [
        "## Structured Generation Plan",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
    ]
    lines.extend(_render_compiled_boundary_lines(plan, compact=False))
    lines.extend(_render_lane_obligation_lines(plan, compact=False))
    lines.extend(_render_route_authority_lines(plan, compact=False))
    if plan.repo_revision:
        lines.append(f"- Repo revision: `{plan.repo_revision}`")
    lines.append("- Inspected modules:")
    lines.extend(f"  - `{module}`" for module in plan.inspected_modules)
    lines.append("- Approved Trellis modules for imports:")
    lines.extend(f"  - `{module}`" for module in plan.approved_modules)
    if plan.symbols_to_reuse:
        symbol_limit = 24 if plan.lane_family else 80
        lines.append("- Public symbols available from the approved modules:")
        lines.extend(f"  - `{symbol}`" for symbol in plan.symbols_to_reuse[:symbol_limit])
    if plan.proposed_tests:
        lines.append("- Tests to run after generation:")
        lines.extend(f"  - `{target}`" for target in plan.proposed_tests)
    if plan.test_map is not None and plan.symbols_to_reuse:
        likely_test_lines = []
        for symbol in plan.symbols_to_reuse[:8]:
            likely = suggest_tests_for_symbol(symbol)
            if likely:
                likely_test_lines.append(f"  - `{symbol}` -> {', '.join(likely[:4])}")
        if likely_test_lines:
            lines.append("- Likely tests for reused symbols:")
            lines.extend(likely_test_lines)
    if plan.primitive_plan is not None:
        lines.extend(_render_backend_binding_lines(plan, compact=False))
        lines.append(
            "  - Instruction precedence: follow the compiler-emitted lane obligations first, "
            "then satisfy the exact backend binding and approved imports. If older guidance "
            "conflicts, treat it as stale and obey this plan."
        )
        if plan.blocker_report is not None:
            lines.append("  - Blockers:")
            lines.extend(
                f"    - `{blocker}`"
                for blocker in plan.primitive_plan.blockers
            )
            lines.append("")
            lines.append(render_blocker_report(plan.blocker_report))
        if plan.new_primitive_workflow is not None:
            lines.append("")
            lines.append(render_new_primitive_workflow(plan.new_primitive_workflow))
    if plan.uncertainty_flags:
        lines.append("- Uncertainty flags:")
        lines.extend(f"  - `{flag}`" for flag in plan.uncertainty_flags)
    lines.append(
        "- Every `trellis.*` import in the generated code MUST come from the approved module list above."
    )
    lines.append(
        "- If you need functionality outside the approved list, say so explicitly instead of inventing an import."
    )
    return "\n".join(lines)


def render_generation_route_card(plan: GenerationPlan) -> str:
    """Render a compact route card for token-efficient first-pass prompting."""
    lines = [
        "## Structured Lane Card",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
    ]
    lines.extend(_render_compiled_boundary_lines(plan, compact=True))
    lines.extend(_render_lane_obligation_lines(plan, compact=True))
    lines.extend(_render_route_authority_lines(plan, compact=True))
    if plan.primitive_plan is not None:
        lines.extend(_render_backend_binding_lines(plan, compact=True))
    if plan.inspected_modules:
        lines.append("- Primary modules to inspect/reuse:")
        lines.extend(f"  - `{module}`" for module in plan.inspected_modules[:6])
    if plan.proposed_tests:
        lines.append("- Post-build test targets:")
        lines.extend(f"  - `{target}`" for target in plan.proposed_tests[:4])
    lines.append(
        "- Instruction precedence: follow the lane obligations in this card first. "
        "Treat backend route/helper details as exact-fit bindings, not as permission "
        "to invent a different numerical path."
    )
    lines.append(
        "- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan."
    )
    lines.append(
        "- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires."
    )
    return "\n".join(lines)


def render_review_contract_card(plan: GenerationPlan) -> str:
    """Render the compact compiled route contract for reviewer prompts."""
    lines = [
        "## Compiled Route Contract",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
    ]
    lines.extend(_render_compiled_boundary_lines(plan, compact=True))
    lines.extend(_render_lane_obligation_lines(plan, compact=True))
    lines.extend(_render_route_authority_lines(plan, compact=True))
    if plan.primitive_plan is not None:
        lines.extend(_render_backend_binding_lines(plan, compact=True))
    if plan.approved_modules:
        lines.append("- Approved modules in scope:")
        lines.extend(f"  - `{module}`" for module in plan.approved_modules[:8])
    lines.append(
        "- Review the generated code against this compiled boundary. Treat route, helper, or validation drift as real findings."
    )
    return "\n".join(lines)


def _resolve_generation_instructions(plan: GenerationPlan) -> ResolvedInstructionSet:
    """Build and resolve route instructions from the generation plan."""
    if plan.resolved_instructions is not None:
        return plan.resolved_instructions
    return resolve_instruction_records(
        _build_instruction_records(plan),
        method=plan.method,
        instrument_type=plan.instrument_type,
        route=plan.primitive_plan.route if plan.primitive_plan is not None else "",
    )


def _build_instruction_records(plan: GenerationPlan) -> tuple[InstructionRecord, ...]:
    """Translate route-helper notes into structured instruction records."""
    primitive_plan = plan.primitive_plan
    if primitive_plan is None:
        return ()

    records: list[InstructionRecord] = []
    route = primitive_plan.route
    route_scope_methods = (plan.method,)
    route_scope_instruments = (plan.instrument_type,) if plan.instrument_type else ()
    route_scope_routes = (route,)

    route_helpers = [
        primitive
        for primitive in primitive_plan.primitives
        if primitive.role == "route_helper"
    ]
    if route_helpers:
        helper = route_helpers[0]
        records.append(
            InstructionRecord(
                id=f"{route}:route-helper",
                title="Use the selected route helper directly",
                instruction_type="hard_constraint",
                source_kind="route_card",
                source_id=f"{route}:route-helper",
                scope_methods=route_scope_methods,
                scope_instruments=route_scope_instruments,
                scope_routes=route_scope_routes,
                scope_modules=(helper.module,),
                precedence_rank=100,
                statement=(
                    "Use the route helper directly inside `evaluate()`; do not "
                    "rebuild the process, engine, or discount glue manually."
                ),
                rationale="The helper already owns the route-specific engine and payoff mapping.",
            )
        )

    schedule_builder = next(
        (
            primitive
            for primitive in primitive_plan.primitives
            if primitive.role == "schedule_builder" or primitive.symbol == "generate_schedule"
        ),
        None,
    )
    if schedule_builder is not None:
        records.append(
            InstructionRecord(
                id=f"{route}:schedule-builder",
                title="Use the shared schedule builder",
                instruction_type="route_hint",
                source_kind="route_card",
                source_id=f"{route}:schedule-builder",
                scope_methods=route_scope_methods,
                scope_instruments=route_scope_instruments,
                scope_routes=route_scope_routes,
                scope_modules=(schedule_builder.module,),
                precedence_rank=90,
                statement=(
                    f"Use `{schedule_builder.module}.{schedule_builder.symbol}` "
                    "to build the route schedule before pricing."
                ),
                rationale="Schedule construction is a shared route capability, not payoff-body glue.",
            )
        )
        records.append(
            InstructionRecord(
                id=f"{route}:schedule-body",
                title="Avoid hard-coded schedule grids",
                instruction_type="route_hint",
                source_kind="route_card",
                source_id=f"{route}:schedule-body",
                scope_methods=route_scope_methods,
                scope_instruments=route_scope_instruments,
                scope_routes=route_scope_routes,
                precedence_rank=85,
                statement="Do not hard-code observation or payment grids inside the payoff body.",
                rationale="Route plans should delegate date construction to the shared primitive.",
            )
        )

    if primitive_plan.notes:
        for index, note in enumerate(primitive_plan.notes, 1):
            records.append(
                InstructionRecord(
                    id=f"{route}:note:{index}",
                    title=f"Route note {index}",
                    instruction_type="historical_note" if "do not" not in note.lower() else "route_hint",
                    source_kind="route_card",
                    source_id=f"{route}:note:{index}",
                    scope_methods=route_scope_methods,
                    scope_instruments=route_scope_instruments,
                    scope_routes=route_scope_routes,
                    precedence_rank=50 - index,
                    statement=note,
                    rationale="Route notes are retained as structured guidance and resolved by precedence.",
                )
            )

    return tuple(records)


def _schedule_related_instructions(
    resolved_instructions: ResolvedInstructionSet,
) -> tuple[InstructionRecord, ...]:
    """Return resolved instructions that mention schedule construction."""
    instructions = []
    for instruction in resolved_instructions.effective_instructions:
        text = f"{instruction.title} {instruction.statement}".lower()
        if "generate_schedule" in text or "schedule" in text:
            instructions.append(instruction)
    return tuple(instructions)


def render_import_repair_card(plan: GenerationPlan) -> str:
    """Render a compact import-repair card for retry prompts."""
    lines = [
        "## Import Repair Card",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
        "- Approved Trellis modules:",
    ]
    lines.extend(f"  - `{module}`" for module in plan.approved_modules[:18])
    if plan.symbols_to_reuse:
        lines.append("- Approved public symbols to reuse:")
        lines.extend(f"  - `{symbol}`" for symbol in plan.symbols_to_reuse[:40])
    lines.append("- Use only modules and symbols listed above.")
    lines.append("- Do not invent imports, aliases, or wildcard imports.")
    return "\n".join(lines)


def render_semantic_repair_card(plan: GenerationPlan) -> str:
    """Render a semantic-repair card for retry prompts."""
    lines = [
        "## Semantic Repair Card",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
    ]
    lines.extend(_render_compiled_boundary_lines(plan, compact=True))
    if plan.primitive_plan is not None:
        lines.append(f"- Route: `{plan.primitive_plan.route}`")
        lines.append(f"- Engine family: `{plan.primitive_plan.engine_family}`")
        if plan.primitive_plan.route_family:
            lines.append(f"- Route family: `{plan.primitive_plan.route_family}`")
        if plan.primitive_plan.primitives:
            lines.append("- Required primitives:")
            lines.extend(
                f"  - `{primitive.module}.{primitive.symbol}` ({primitive.role})"
                for primitive in plan.primitive_plan.primitives[:10]
            )
        if plan.primitive_plan.adapters:
            lines.append("- Required adapters:")
            lines.extend(f"  - `{adapter}`" for adapter in plan.primitive_plan.adapters[:8])
        if plan.primitive_plan.notes:
            lines.append("- Route notes:")
            lines.extend(f"  - {note}" for note in plan.primitive_plan.notes[:4])
    if plan.uncertainty_flags:
        lines.append("- Active uncertainty flags:")
        lines.extend(f"  - `{flag}`" for flag in plan.uncertainty_flags[:6])
    lines.append("- Match the product semantics, required primitives, and approved imports exactly.")
    lines.append("- Return the complete module again, not only an evaluate-body fragment, patch, or diff.")
    return "\n".join(lines)


def enrich_generation_plan(
    plan: GenerationPlan,
    *,
    request=None,
    semantic_blueprint=None,
    validation_contract=None,
) -> GenerationPlan:
    """Attach semantic, lowering, and validation summaries for prompt rendering."""
    from trellis.agent.route_registry import compile_route_binding_authority

    semantic_contract = getattr(semantic_blueprint, "contract", None)
    product = getattr(semantic_contract, "product", None)
    valuation_context = getattr(semantic_blueprint, "valuation_context", None)
    lowering = getattr(semantic_blueprint, "dsl_lowering", None)
    lane_plan = getattr(semantic_blueprint, "lane_plan", None)
    semantic_id = str(getattr(semantic_blueprint, "semantic_id", "") or "")
    requested_instrument_type = str(getattr(request, "instrument_type", "") or "")
    matched_wrapper, bridge_status = _semantic_bridge_metadata(
        semantic_id=semantic_id,
        requested_instrument_type=requested_instrument_type,
    )
    lane_plan_kind = str(getattr(lane_plan, "plan_kind", "") or "")
    preserve_backend_binding = bool(
        plan.primitive_plan is not None or lane_plan_kind == "exact_target_binding"
    )
    backend_exact_target_refs = tuple(getattr(plan, "backend_exact_target_refs", ()) or ())
    if not plan.primitive_plan and lane_plan_kind == "exact_target_binding":
        backend_exact_target_refs = tuple(getattr(lane_plan, "exact_target_refs", ()) or ())
    elif not preserve_backend_binding:
        backend_exact_target_refs = ()
    backend_binding_id = (
        str(getattr(plan.primitive_plan, "backend_binding_id", "") or "").strip()
        or (str(backend_exact_target_refs[0]).strip() if backend_exact_target_refs else "")
        or (
            str(getattr(plan, "backend_binding_id", "") or "").strip()
            if preserve_backend_binding
            else ""
        )
    )

    enriched_plan = replace(
        plan,
        semantic_contract_id=semantic_id,
        semantic_requested_instrument_type=requested_instrument_type,
        semantic_compatibility_bridge_status=bridge_status,
        semantic_matched_wrapper=matched_wrapper,
        semantic_instrument_class=str(getattr(product, "instrument_class", "") or ""),
        semantic_payoff_family=str(getattr(product, "payoff_family", "") or ""),
        semantic_underlier_structure=str(getattr(product, "underlier_structure", "") or ""),
        valuation_market_source=str(getattr(valuation_context, "market_source", "") or ""),
        valuation_reporting_currency=str(
            getattr(getattr(valuation_context, "reporting_policy", None), "reporting_currency", "") or ""
        ),
        valuation_requested_outputs=tuple(getattr(semantic_blueprint, "requested_outputs", ()) or ()),
        lane_family=str(getattr(lane_plan, "lane_family", "") or ""),
        lane_plan_kind=lane_plan_kind,
        lane_timeline_roles=tuple(getattr(lane_plan, "timeline_roles", ()) or ()),
        lane_market_requirements=tuple(getattr(lane_plan, "market_requirements", ()) or ()),
        lane_state_obligations=tuple(getattr(lane_plan, "state_obligations", ()) or ()),
        lane_control_obligations=tuple(getattr(lane_plan, "control_obligations", ()) or ()),
        lane_construction_steps=tuple(getattr(lane_plan, "construction_steps", ()) or ()),
        lane_reusable_primitives=tuple(
            str(getattr(binding, "primitive_ref", "") or "")
            for binding in (getattr(lane_plan, "reusable_bindings", ()) or ())
            if str(getattr(binding, "primitive_ref", "") or "").strip()
        ),
        lane_exact_binding_refs=tuple(getattr(lane_plan, "exact_target_refs", ()) or ()),
        lane_unresolved_primitives=tuple(getattr(lane_plan, "unresolved_primitives", ()) or ()),
        backend_binding_id=backend_binding_id,
        backend_binding_aliases=(
            tuple(getattr(plan, "backend_binding_aliases", ()) or ())
            if preserve_backend_binding
            else ()
        ),
        backend_exact_target_refs=backend_exact_target_refs,
        backend_helper_refs=(
            tuple(getattr(plan, "backend_helper_refs", ()) or ())
            if preserve_backend_binding
            else ()
        ),
        backend_pricing_kernel_refs=(
            tuple(getattr(plan, "backend_pricing_kernel_refs", ()) or ())
            if preserve_backend_binding
            else ()
        ),
        backend_schedule_builder_refs=(
            tuple(getattr(plan, "backend_schedule_builder_refs", ()) or ())
            if preserve_backend_binding
            else ()
        ),
        backend_cashflow_engine_refs=(
            tuple(getattr(plan, "backend_cashflow_engine_refs", ()) or ())
            if preserve_backend_binding
            else ()
        ),
        backend_market_binding_refs=(
            tuple(getattr(plan, "backend_market_binding_refs", ()) or ())
            if preserve_backend_binding
            else ()
        ),
        backend_engine_family=(
            str(getattr(plan, "backend_engine_family", "") or "")
            if preserve_backend_binding
            else ""
        ),
        backend_route_family=(
            str(getattr(plan, "backend_route_family", "") or "")
            if preserve_backend_binding
            else ""
        ),
        backend_compatibility_alias_policy=(
            str(getattr(plan, "backend_compatibility_alias_policy", "") or "operator_visible")
            if preserve_backend_binding
            else "operator_visible"
        ),
        lowering_route_id=str(getattr(lowering, "route_id", "") or ""),
        lowering_expr_kind=(
            "" if lowering is None or getattr(lowering, "normalized_expr", None) is None
            else type(lowering.normalized_expr).__name__
        ),
        lowering_family_ir_type=(
            "" if lowering is None or getattr(lowering, "family_ir", None) is None
            else type(lowering.family_ir).__name__
        ),
        lowering_helper_refs=tuple(getattr(lowering, "helper_refs", ()) or ()),
        validation_bundle_id=str(getattr(validation_contract, "bundle_id", "") or ""),
        validation_check_ids=tuple(
            str(check.check_id)
            for check in (getattr(validation_contract, "deterministic_checks", ()) or ())
        ),
        validation_residual_risks=tuple(
            str(risk) for risk in (getattr(validation_contract, "residual_risks", ()) or ())
        ),
    )
    return replace(
        enriched_plan,
        route_binding_authority=compile_route_binding_authority(
            generation_plan=enriched_plan,
            validation_contract=validation_contract,
            semantic_blueprint=semantic_blueprint,
            product_ir=getattr(semantic_blueprint, "product_ir", None),
            request=request,
        ),
    )


def _semantic_bridge_metadata(
    *,
    semantic_id: str,
    requested_instrument_type: str,
) -> tuple[str, str]:
    """Return matched wrapper and compatibility bridge status for prompt rendering."""
    normalized_request = _normalize_semantic_label(requested_instrument_type)
    normalized_semantic = _normalize_semantic_label(semantic_id)
    if not normalized_request:
        return "", "implicit_semantic_request"
    if normalized_request == normalized_semantic:
        return "", "canonical_semantic"

    wrappers = set()
    if normalized_semantic:
        try:
            from trellis.agent.semantic_concepts import get_semantic_concept_definition

            concept = get_semantic_concept_definition(normalized_semantic)
            wrappers = {
                _normalize_semantic_label(item)
                for item in getattr(concept, "compatibility_wrappers", ()) or ()
                if _normalize_semantic_label(item)
            }
        except Exception:
            wrappers = set()

    if normalized_request in wrappers:
        return requested_instrument_type, "thin_compatibility_wrapper"
    return "", "request_alias"


def _normalize_semantic_label(value: object) -> str:
    """Normalize semantic labels for wrapper-status comparisons."""
    return str(value or "").strip().lower().replace(" ", "_")


def _render_compiled_boundary_lines(plan: GenerationPlan, *, compact: bool) -> list[str]:
    """Render semantic/valuation/lowering/validation summaries for prompts."""
    from trellis.agent.route_registry import route_binding_authority_summary, should_surface_route_alias

    lines: list[str] = []
    if plan.semantic_contract_id:
        semantic_bits = [
            f"`{plan.semantic_contract_id}`",
        ]
        if plan.semantic_requested_instrument_type:
            semantic_bits.append(f"request=`{plan.semantic_requested_instrument_type}`")
        if plan.semantic_compatibility_bridge_status:
            semantic_bits.append(
                f"bridge=`{plan.semantic_compatibility_bridge_status}`"
            )
        if plan.semantic_matched_wrapper:
            semantic_bits.append(f"wrapper=`{plan.semantic_matched_wrapper}`")
        if plan.semantic_instrument_class:
            semantic_bits.append(f"instrument=`{plan.semantic_instrument_class}`")
        if plan.semantic_payoff_family:
            semantic_bits.append(f"payoff=`{plan.semantic_payoff_family}`")
        if plan.semantic_underlier_structure:
            semantic_bits.append(f"structure=`{plan.semantic_underlier_structure}`")
        lines.append(f"- Semantic contract: {', '.join(semantic_bits)}")

    valuation_bits: list[str] = []
    if plan.valuation_market_source:
        valuation_bits.append(f"market_source=`{plan.valuation_market_source}`")
    if plan.valuation_reporting_currency:
        valuation_bits.append(f"reporting=`{plan.valuation_reporting_currency}`")
    if plan.valuation_requested_outputs:
        valuation_bits.append(
            "outputs=" + ", ".join(f"`{output}`" for output in plan.valuation_requested_outputs[:6])
        )
    if valuation_bits:
        lines.append(f"- Valuation context: {', '.join(valuation_bits)}")

    lane_bits: list[str] = []
    if plan.lane_family:
        lane_bits.append(f"family=`{plan.lane_family}`")
    if plan.lane_plan_kind:
        lane_bits.append(f"kind=`{plan.lane_plan_kind}`")
    if plan.lane_timeline_roles:
        lane_bits.append(
            "timeline_roles=" + ", ".join(f"`{role}`" for role in plan.lane_timeline_roles[: (3 if compact else 6)])
        )
    if plan.lane_exact_binding_refs:
        refs = ", ".join(
            f"`{ref}`" for ref in plan.lane_exact_binding_refs[: (2 if compact else 4)]
        )
        lane_bits.append(f"exact_bindings={refs}")
    elif plan.lane_reusable_primitives:
        refs = ", ".join(
            f"`{ref}`" for ref in plan.lane_reusable_primitives[: (2 if compact else 4)]
        )
        lane_bits.append(f"primitives={refs}")
    if plan.lane_unresolved_primitives:
        unresolved = ", ".join(
            f"`{item}`" for item in plan.lane_unresolved_primitives[: (2 if compact else 4)]
        )
        lane_bits.append(f"unresolved={unresolved}")
    if lane_bits:
        lines.append(f"- Lane boundary: {', '.join(lane_bits)}")

    lowering_bits: list[str] = []
    if plan.lowering_family_ir_type:
        lowering_bits.append(f"family_ir=`{plan.lowering_family_ir_type}`")
    if plan.lowering_expr_kind:
        lowering_bits.append(f"expr=`{plan.lowering_expr_kind}`")
    if plan.lowering_helper_refs:
        helper_limit = 2 if compact else 4
        helper_refs = ", ".join(
            f"`{helper}`" for helper in plan.lowering_helper_refs[:helper_limit]
        )
        lowering_bits.append(f"helpers={helper_refs}")
    authority = route_binding_authority_summary(plan.route_binding_authority)
    if plan.lowering_route_id and should_surface_route_alias(authority):
        lowering_bits.append(f"route_alias=`{plan.lowering_route_id}`")
    if lowering_bits:
        lines.append(f"- Lowering boundary: {', '.join(lowering_bits)}")

    validation_bits: list[str] = []
    if plan.validation_bundle_id:
        validation_bits.append(f"bundle=`{plan.validation_bundle_id}`")
    if plan.validation_check_ids:
        if compact and len(plan.validation_check_ids) > 4:
            selected_checks = tuple(
                dict.fromkeys(
                    (
                        *plan.validation_check_ids[:2],
                        *plan.validation_check_ids[-2:],
                    )
                )
            )
        else:
            selected_checks = plan.validation_check_ids[: (4 if compact else 6)]
        checks = ", ".join(f"`{check}`" for check in selected_checks)
        validation_bits.append(f"checks={checks}")
    if plan.validation_residual_risks:
        risk_limit = 2 if compact else 4
        risks = ", ".join(f"`{risk}`" for risk in plan.validation_residual_risks[:risk_limit])
        validation_bits.append(f"residual_risks={risks}")
    if validation_bits:
        lines.append(f"- Validation contract: {', '.join(validation_bits)}")
    return lines


def extract_trellis_imports(source: str) -> tuple[ImportReference, ...]:
    """Extract absolute Trellis imports from Python source."""
    tree = ast.parse(source)
    imports: list[ImportReference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("trellis."):
                    imports.append(
                        ImportReference(
                            module=alias.name,
                            symbol=None,
                            kind="import",
                            lineno=node.lineno,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module or not node.module.startswith("trellis."):
                continue
            for alias in node.names:
                imports.append(
                    ImportReference(
                        module=node.module,
                        symbol=alias.name,
                        kind="from",
                        lineno=node.lineno,
                    )
                )
    return tuple(imports)


_ADMITTED_AGENT_IMPORT_PREFIX = "trellis.instruments._agent"


def _is_admitted_agent_import(module_name: str) -> bool:
    """Return whether the import targets the admitted generated adapter tree."""
    text = str(module_name or "").strip()
    if not text:
        return False
    return text == _ADMITTED_AGENT_IMPORT_PREFIX or text.startswith(
        _ADMITTED_AGENT_IMPORT_PREFIX + "."
    )


def validate_generated_imports(source: str, plan: GenerationPlan) -> ImportValidationReport:
    """Validate that generated Trellis imports are real and approved.

    ``trellis.instruments._agent`` is always rejected because that package is
    the admitted/generated adapter surface: it must never be a dependency of
    newly generated code, and blocking it keeps the fresh-generated FinancePy
    pilot path honest (QUA-866).
    """
    try:
        imports = extract_trellis_imports(source)
    except SyntaxError as exc:
        return ImportValidationReport(
            imports=(),
            errors=(f"Cannot validate imports because generated code is not valid Python: {exc}",),
        )

    errors: list[str] = []
    approved = set(plan.approved_modules)
    for ref in imports:
        if ref.symbol == "*":
            errors.append(
                f"Line {ref.lineno}: wildcard imports from `{ref.module}` are not allowed."
            )
            continue

        if _is_admitted_agent_import(ref.module):
            errors.append(
                f"Line {ref.lineno}: imports from `{ref.module}` are not allowed. "
                "`trellis.instruments._agent` is the admitted generated adapter "
                "tree and must not be a dependency of freshly generated code."
            )
            continue

        if ref.kind == "import":
            if not module_exists(ref.module):
                errors.append(
                    f"Line {ref.lineno}: unknown Trellis module `{ref.module}`."
                )
            elif ref.module not in approved:
                errors.append(
                    f"Line {ref.lineno}: unapproved Trellis module `{ref.module}`. "
                    "Use an inspected/approved module from the generation plan."
                )
            continue

        if not module_exists(ref.module):
            errors.append(
                f"Line {ref.lineno}: unknown Trellis module `{ref.module}`."
            )
            continue
        if ref.module not in approved:
            errors.append(
                f"Line {ref.lineno}: unapproved Trellis module `{ref.module}`. "
                "Use an inspected/approved module from the generation plan."
            )
            continue
        if ref.symbol is None:
            continue
        if not is_valid_import(ref.module, ref.symbol):
            errors.append(
                f"Line {ref.lineno}: `{ref.symbol}` is not exported by `{ref.module}`."
            )

    return ImportValidationReport(
        imports=imports,
        errors=tuple(errors),
    )


def sanitize_generated_source(source: str) -> GeneratedSourceSanitizationReport:
    """Strip one unambiguous outer markdown fence block from generated source."""
    raw_source = source or ""
    stripped = raw_source.strip()
    if not stripped:
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source="",
            source_status="rejected",
            fence_removed=False,
            fence_language=None,
            fence_count=0,
            errors=("generated source is empty after trimming whitespace",),
        )

    lines = raw_source.splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source="",
            source_status="rejected",
            fence_removed=False,
            fence_language=None,
            fence_count=0,
            errors=("generated source is empty after trimming whitespace",),
        )

    fence_indices = [
        index for index, line in enumerate(lines) if _OPENING_FENCE_RE.match(line) or _CLOSING_FENCE_RE.match(line)
    ]
    if not fence_indices:
        sanitized_source = textwrap.dedent(raw_source).strip("\n")
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source=sanitized_source,
            source_status="accepted",
            fence_removed=False,
            fence_language=None,
            fence_count=0,
        )

    if len(fence_indices) != 2:
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source=raw_source,
            source_status="rejected",
            fence_removed=False,
            fence_language=None,
            fence_count=len(fence_indices),
            errors=(
                "generated source contains ambiguous markdown fences; emit raw Python "
                "or one outer fenced block only",
            ),
        )

    first_index, last_index = fence_indices
    if first_index != nonempty_indices[0] or last_index != nonempty_indices[-1]:
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source=raw_source,
            source_status="rejected",
            fence_removed=False,
            fence_language=None,
            fence_count=len(fence_indices),
            errors=(
                "generated source contains markdown fences outside a single outer code "
                "block; emit raw Python only",
            ),
        )

    opening_line = lines[first_index]
    closing_line = lines[last_index]
    opening_match = _OPENING_FENCE_RE.match(opening_line)
    if opening_match is None or not _CLOSING_FENCE_RE.match(closing_line):
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source=raw_source,
            source_status="rejected",
            fence_removed=False,
            fence_language=None,
            fence_count=len(fence_indices),
            errors=(
                "generated source has an incomplete markdown fence wrapper; emit raw "
                "Python only",
            ),
        )

    inner_lines = lines[first_index + 1 : last_index]
    if any(_OPENING_FENCE_RE.match(line) or _CLOSING_FENCE_RE.match(line) for line in inner_lines):
        return GeneratedSourceSanitizationReport(
            raw_source=raw_source,
            sanitized_source=raw_source,
            source_status="rejected",
            fence_removed=False,
            fence_language=None,
            fence_count=len(fence_indices),
            errors=(
                "generated source contains embedded markdown fences inside the code "
                "body; emit raw Python only",
            ),
        )

    sanitized_source = textwrap.dedent("\n".join(inner_lines)).strip("\n")
    return GeneratedSourceSanitizationReport(
        raw_source=raw_source,
        sanitized_source=sanitized_source,
        source_status="sanitized",
        fence_removed=True,
        fence_language=opening_match.group("language") or None,
        fence_count=len(fence_indices),
    )


def _modules_with_prefix(prefix: str) -> tuple[str, ...]:
    """Return all registry modules equal to or nested under ``prefix``."""
    return tuple(
        module_path
        for module_path in get_registry_snapshot()
        if module_path == prefix or module_path.startswith(prefix + ".")
    )


def build_primitive_plan(
    *,
    pricing_plan,
    product_ir: ProductIR | None,
) -> PrimitivePlan | None:
    """Build a deterministic primitive plan from method + ProductIR."""
    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )
    return ranked[0] if ranked else None


def rank_primitive_routes(
    *,
    pricing_plan,
    product_ir: ProductIR | None,
) -> tuple[PrimitivePlan, ...]:
    """Return candidate primitive routes ranked by deterministic heuristic score.

    Delegates to the declarative route registry for candidate enumeration,
    primitive resolution, and family mapping.  Scoring still uses the legacy
    ``_route_score`` heuristic (Phase 3 replaces it with a learned scorer).
    """
    if pricing_plan is None:
        return ()

    product_ir = _augment_product_ir_for_requested_method(
        product_ir,
        preferred_method=getattr(pricing_plan, "method", None),
    )

    from trellis.agent.route_registry import (
        load_route_registry,
        match_candidate_routes,
        resolve_route_adapters,
        resolve_route_family,
        resolve_route_notes,
        resolve_route_primitives,
    )
    from trellis.agent.backend_bindings import (
        load_backend_binding_catalog,
        resolve_backend_binding_by_route_id,
    )
    from trellis.agent.route_scorer import RouteScorer, ScoringContext

    method = normalize_method(pricing_plan.method)
    registry = load_route_registry()
    binding_catalog = load_backend_binding_catalog(registry=registry)
    candidates = match_candidate_routes(
        registry, method, product_ir, pricing_plan=pricing_plan,
    )

    scorer = RouteScorer(registry)

    ranked: list[PrimitivePlan] = []
    for spec in candidates:
        binding_spec = resolve_backend_binding_by_route_id(
            spec.id,
            product_ir=product_ir,
            catalog=binding_catalog,
        )
        route_primitives = list(
            resolve_route_primitives(
                spec,
                product_ir,
                binding_spec=binding_spec,
            )
        )
        adapters = resolve_route_adapters(spec, product_ir)
        notes = resolve_route_notes(spec, product_ir)
        route = spec.id
        primitives = list(getattr(binding_spec, "primitives", ()) or route_primitives)
        blockers = list(product_ir.unresolved_primitives if product_ir is not None else ())
        blockers.extend(_verify_primitives(primitives))
        engine_family = str(getattr(binding_spec, "engine_family", "") or spec.engine_family)
        route_family = str(
            getattr(binding_spec, "route_family", "")
            or resolve_route_family(
                spec,
                product_ir,
                binding_spec=binding_spec,
            )
        )
        if route == "exercise_lattice" and route_family == "equity_tree":
            engine_family = "tree"

        ctx = ScoringContext(
            product_ir=product_ir,
            route_spec=spec,
            pricing_plan=pricing_plan,
            blockers=blockers,
            route_family=route_family,
        )
        route_score = scorer.score_route(ctx)

        plan = PrimitivePlan(
            route=route,
            engine_family=engine_family,
            route_family=route_family,
            primitives=tuple(primitives),
            adapters=tuple(adapters),
            blockers=tuple(dict.fromkeys(blockers)),
            notes=tuple(notes),
            backend_binding_id=str(getattr(binding_spec, "binding_id", "") or ""),
            backend_binding_aliases=tuple(getattr(binding_spec, "aliases", ()) or ()),
            backend_exact_target_refs=tuple(getattr(binding_spec, "exact_target_refs", ()) or ()),
            backend_helper_refs=tuple(getattr(binding_spec, "helper_refs", ()) or ()),
            backend_pricing_kernel_refs=tuple(getattr(binding_spec, "pricing_kernel_refs", ()) or ()),
            backend_schedule_builder_refs=tuple(getattr(binding_spec, "schedule_builder_refs", ()) or ()),
            backend_cashflow_engine_refs=tuple(getattr(binding_spec, "cashflow_engine_refs", ()) or ()),
            backend_market_binding_refs=tuple(getattr(binding_spec, "market_binding_refs", ()) or ()),
            backend_compatibility_alias_policy=str(
                getattr(binding_spec, "compatibility_alias_policy", None) or "operator_visible"
            ),
            score=route_score.final_score,
        )
        ranked.append(plan)

    ranked.sort(
        key=lambda plan: (
            -plan.score,
            len(plan.blockers),
            plan.route,
        )
    )
    return tuple(ranked)


def _augment_product_ir_for_requested_method(
    product_ir: ProductIR | None,
    *,
    preferred_method: str | None,
) -> ProductIR | None:
    """Augment route-selection IR with explicit method-family intent.

    Static decompositions often predate newer promoted routes, so the raw IR can
    understate the engine families implied by an explicit request such as
    ``preferred_method="fft_pricing"``. Route matching should honor that
    request-level method intent without mutating the base decomposition.
    """
    if product_ir is None:
        return None
    method = normalize_method(preferred_method) if preferred_method else ""
    method_hints = {
        "analytical": {
            "engine_families": ("analytical",),
            "route_families": ("analytical",),
        },
        "rate_tree": {
            "engine_families": ("lattice",),
            "route_families": ("rate_lattice",),
        },
        "monte_carlo": {
            "engine_families": ("monte_carlo",),
            "route_families": ("monte_carlo",),
        },
        "qmc": {
            "engine_families": ("qmc",),
            "route_families": ("qmc",),
        },
        "pde_solver": {
            "engine_families": ("pde",),
            "route_families": ("pde_solver",),
        },
        "fft_pricing": {
            "engine_families": ("transforms",),
            "route_families": ("fft_pricing",),
        },
        "copula": {
            "engine_families": ("copula",),
            "route_families": ("copula",),
        },
        "waterfall": {
            "engine_families": ("cashflow",),
            "route_families": ("waterfall",),
        },
    }.get(method)
    if not method_hints:
        return product_ir

    candidate_engine_families = list(getattr(product_ir, "candidate_engine_families", ()) or ())
    route_families = list(getattr(product_ir, "route_families", ()) or ())
    exercise_style = str(getattr(product_ir, "exercise_style", "") or "").strip().lower()
    if method in {"rate_tree", "monte_carlo", "qmc"} and exercise_style not in {"", "none", "european"}:
        method_hints = {
            "engine_families": tuple(dict.fromkeys((*method_hints["engine_families"], "exercise"))),
            "route_families": ("exercise",),
        }
    changed = False
    for family in method_hints["engine_families"]:
        if family not in candidate_engine_families:
            candidate_engine_families.append(family)
            changed = True
    if route_families:
        for family in method_hints["route_families"]:
            if family not in route_families:
                route_families.append(family)
                changed = True
    if not changed:
        return product_ir
    return replace(
        product_ir,
        candidate_engine_families=tuple(candidate_engine_families),
        route_families=tuple(route_families),
    )




def _verify_primitives(primitives: list[PrimitiveRef]) -> list[str]:
    """Return missing-module or missing-symbol blockers for referenced primitives."""
    blockers: list[str] = []
    for primitive in primitives:
        if not module_exists(primitive.module):
            blockers.append(f"missing_module:{primitive.module}")
            continue
        if primitive.required and not is_valid_import(primitive.module, primitive.symbol):
            blockers.append(f"missing_symbol:{primitive.module}.{primitive.symbol}")
    return blockers


def _route_score(
    spec,
    product_ir: ProductIR | None,
    blockers: list[str],
    *,
    route_family: str = "",
    pricing_plan=None,
) -> float:
    """Score a route using YAML-driven score_hints and generic heuristics.

    ``spec`` is a :class:`RouteSpec` from the route registry.  Route-specific
    bonuses and penalties are declared in ``score_hints`` within routes.yaml
    rather than hard-coded per route name. Fallback scoring should prefer
    family capability and backend-binding facts over route identity.
    """
    from trellis.agent.route_registry import (
        evaluate_route_capability_match,
        resolve_route_primitives,
    )

    hints = spec.score_hints if spec is not None else {}
    engine_family = spec.engine_family if spec is not None else ""
    if not route_family:
        route_family = spec.route_family if spec is not None else ""

    score = 0.0
    if product_ir is not None and spec is not None:
        resolved_primitives = tuple(resolve_route_primitives(spec, product_ir))
        resolved_roles = {primitive.role for primitive in resolved_primitives}
        exact_surface_roles = {"route_helper", "pricing_kernel", "cashflow_engine"}
        if engine_family in product_ir.candidate_engine_families:
            score += 1.5
        if resolved_roles.intersection(exact_surface_roles):
            score += 1.5
        elif "exercise_control" in resolved_roles:
            score += 0.75
        elif "low_discrepancy_sampler" in resolved_roles:
            score += 0.5
        model_support_roles = {
            "credit_copula": {"default_time_sampler", "loss_distribution"},
        }.get(str(getattr(product_ir, "model_family", "") or "").strip(), set())
        if resolved_roles.intersection(model_support_roles):
            score += 1.0

        capability = evaluate_route_capability_match(spec, product_ir)
        if capability.ok:
            score += 1.0 + 0.25 * len(capability.matched_predicates)
        else:
            score -= 2.5 + 0.5 * len(capability.failures)

        match_instruments = set(getattr(spec, "match_instruments", ()) or ())
        if match_instruments and product_ir.instrument in match_instruments:
            score += 1.25

        match_payoff_family = set(getattr(spec, "match_payoff_family", ()) or ())
        if match_payoff_family and product_ir.payoff_family in match_payoff_family:
            score += 1.0

        match_exercise = set(getattr(spec, "match_exercise", ()) or ())
        if match_exercise and product_ir.exercise_style in match_exercise:
            score += 0.5

        # --- YAML-driven hints ---

        # exercise_match_bonus: bonus when exercise matches listed styles
        exercise_bonus = hints.get("exercise_match_bonus", 0.0)
        exercise_styles = set(hints.get("exercise_match_styles", ()))
        if exercise_bonus and product_ir.exercise_style in exercise_styles:
            score += exercise_bonus

        # vanilla_exercise_bonus: bonus when payoff_family and exercise match
        vanilla_bonus = hints.get("vanilla_exercise_bonus", 0.0)
        if vanilla_bonus:
            vanilla_payoff = hints.get("vanilla_exercise_payoff", "")
            vanilla_styles = set(hints.get("vanilla_exercise_styles", ()))
            if (
                product_ir.payoff_family == vanilla_payoff
                and product_ir.exercise_style in vanilla_styles
            ):
                score += vanilla_bonus

        # schedule_dependence_bonus
        sched_bonus = hints.get("schedule_dependence_bonus", 0.0)
        if sched_bonus and product_ir.schedule_dependence:
            score += sched_bonus

        # payoff_family_bonus: dict mapping payoff_family -> bonus
        pf_bonuses = hints.get("payoff_family_bonus", {})
        if isinstance(pf_bonuses, dict) and product_ir.payoff_family in pf_bonuses:
            score += pf_bonuses[product_ir.payoff_family]

        # non_european_penalty: penalty when exercise is not european/none
        non_euro_penalty = hints.get("non_european_penalty", 0.0)
        if non_euro_penalty and product_ir.exercise_style not in {"none", "european"}:
            score += non_euro_penalty

        # bonus_when_market_data: dict mapping market_data_key -> bonus
        bonus_md = hints.get("bonus_when_market_data", {})
        if isinstance(bonus_md, dict) and getattr(pricing_plan, "required_market_data", None):
            required = set(pricing_plan.required_market_data)
            for md_key, md_bonus in bonus_md.items():
                if md_key in required:
                    score += md_bonus

        # penalize_when_market_data: dict mapping market_data_key -> penalty
        penalize_md = hints.get("penalize_when_market_data", {})
        if isinstance(penalize_md, dict) and getattr(pricing_plan, "required_market_data", None):
            required = set(pricing_plan.required_market_data)
            for md_key, md_penalty in penalize_md.items():
                if md_key in required:
                    score += md_penalty

        if blockers and not product_ir.supported:
            score -= 2.0

    score -= len(blockers) * 6.0
    return score
