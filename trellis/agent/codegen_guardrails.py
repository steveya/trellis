"""Structured generation context and import validation for agent-built modules."""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, replace

from trellis.agent.blocker_planning import BlockerReport, plan_blockers, render_blocker_report
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
    "credit_default_swap": (
        "trellis.models.copulas",
        "trellis.models.copulas.gaussian",
    ),
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
    primitive_plan = build_primitive_plan(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
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


def render_generation_plan(plan: GenerationPlan) -> str:
    """Format the structured generation plan for prompt injection."""
    lines = [
        "## Structured Generation Plan",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
    ]
    if plan.repo_revision:
        lines.append(f"- Repo revision: `{plan.repo_revision}`")
    lines.append("- Inspected modules:")
    lines.extend(f"  - `{module}`" for module in plan.inspected_modules)
    lines.append("- Approved Trellis modules for imports:")
    lines.extend(f"  - `{module}`" for module in plan.approved_modules)
    if plan.symbols_to_reuse:
        lines.append("- Public symbols available from the approved modules:")
        lines.extend(f"  - `{symbol}`" for symbol in plan.symbols_to_reuse[:80])
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
        lines.append("- Primitive route:")
        lines.append(f"  - Route: `{plan.primitive_plan.route}`")
        lines.append(f"  - Engine family: `{plan.primitive_plan.engine_family}`")
        if plan.primitive_plan.route_family:
            lines.append(f"  - Route family: `{plan.primitive_plan.route_family}`")
        lines.append(f"  - Route score: `{plan.primitive_plan.score:.2f}`")
        if plan.primitive_plan.primitives:
            lines.append("  - Selected primitives:")
            lines.extend(
                f"    - `{primitive.module}.{primitive.symbol}` ({primitive.role})"
                for primitive in plan.primitive_plan.primitives
            )
        resolved_instructions = _resolve_generation_instructions(plan)
        if resolved_instructions.effective_instructions:
            lines.append("  - Resolved instructions:")
            lines.extend(
                f"    - [{instruction.instruction_type}] {instruction.statement}"
                for instruction in resolved_instructions.effective_instructions
            )
            schedule_instructions = _schedule_related_instructions(resolved_instructions)
            if schedule_instructions:
                lines.append("  - Schedule construction:")
                lines.extend(
                    f"    - [{instruction.instruction_type}] {instruction.statement}"
                    for instruction in schedule_instructions
                )
        if resolved_instructions.conflicts:
            lines.append("  - Instruction conflicts:")
            lines.extend(
                f"    - {conflict.reason}"
                for conflict in resolved_instructions.conflicts
            )
        if plan.primitive_plan.adapters:
            lines.append("  - Required adapters:")
            lines.extend(f"    - `{adapter}`" for adapter in plan.primitive_plan.adapters)
        lines.append(
            "  - Instruction precedence: follow the approved modules, primitives, "
            "and route helper in this plan. If older guidance conflicts, treat it "
            "as stale and obey this plan."
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
        "## Structured Route Card",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
    ]
    if plan.primitive_plan is not None:
        lines.append(f"- Route: `{plan.primitive_plan.route}`")
        lines.append(f"- Engine family: `{plan.primitive_plan.engine_family}`")
        if plan.primitive_plan.route_family:
            lines.append(f"- Route family: `{plan.primitive_plan.route_family}`")
        if plan.primitive_plan.primitives:
            lines.append("- Required primitives:")
            lines.extend(
                f"  - `{primitive.module}.{primitive.symbol}` ({primitive.role})"
                for primitive in plan.primitive_plan.primitives[:8]
            )
        resolved_instructions = _resolve_generation_instructions(plan)
        if resolved_instructions.effective_instructions:
            lines.append("- Resolved instructions:")
            lines.extend(
                f"  - [{instruction.instruction_type}] {instruction.statement}"
                for instruction in resolved_instructions.effective_instructions[:8]
            )
            schedule_instructions = _schedule_related_instructions(resolved_instructions)
            if schedule_instructions:
                lines.append("- Schedule construction:")
                lines.extend(
                    f"  - [{instruction.instruction_type}] {instruction.statement}"
                    for instruction in schedule_instructions[:8]
                )
        if resolved_instructions.conflicts:
            lines.append("- Instruction conflicts:")
            lines.extend(
                f"  - {conflict.reason}"
                for conflict in resolved_instructions.conflicts[:4]
            )
        if plan.primitive_plan.adapters:
            lines.append("- Required adapters:")
            lines.extend(f"  - `{adapter}`" for adapter in plan.primitive_plan.adapters[:6])
    if plan.inspected_modules:
        lines.append("- Primary modules to inspect/reuse:")
        lines.extend(f"  - `{module}`" for module in plan.inspected_modules[:6])
    if plan.proposed_tests:
        lines.append("- Post-build test targets:")
        lines.extend(f"  - `{target}`" for target in plan.proposed_tests[:4])
    lines.append(
        "- Instruction precedence: follow the approved modules, primitives, and "
        "route helper in this card. If older guidance conflicts, treat it as "
        "stale and obey this plan."
    )
    lines.append(
        "- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels."
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

    needs_schedule_builder = any(
        primitive.role == "schedule_builder" or primitive.symbol == "generate_schedule"
        for primitive in primitive_plan.primitives
    )
    if needs_schedule_builder:
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
                scope_modules=("trellis.core.date_utils",),
                precedence_rank=90,
                statement="Use `trellis.core.date_utils.generate_schedule` to build ordered dates before pricing.",
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


def validate_generated_imports(source: str, plan: GenerationPlan) -> ImportValidationReport:
    """Validate that generated Trellis imports are real and approved."""
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

    from trellis.agent.route_registry import (
        load_route_registry,
        match_candidate_routes,
        resolve_route_adapters,
        resolve_route_family,
        resolve_route_notes,
        resolve_route_primitives,
    )
    from trellis.agent.route_scorer import RouteScorer, ScoringContext

    method = normalize_method(pricing_plan.method)
    registry = load_route_registry()
    candidates = match_candidate_routes(
        registry, method, product_ir, pricing_plan=pricing_plan,
    )

    scorer = RouteScorer(registry)

    ranked: list[PrimitivePlan] = []
    for spec in candidates:
        primitives = list(resolve_route_primitives(spec, product_ir))
        adapters = resolve_route_adapters(spec, product_ir)
        notes = resolve_route_notes(spec, product_ir)
        route = spec.id
        blockers = list(product_ir.unresolved_primitives if product_ir is not None else ())
        blockers.extend(_verify_primitives(primitives))
        engine_family = spec.engine_family
        route_family = resolve_route_family(spec, product_ir)
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
    rather than hard-coded per route name.
    """
    hints = spec.score_hints if spec is not None else {}
    route = spec.id if spec is not None else ""
    engine_family = spec.engine_family if spec is not None else ""
    if not route_family:
        route_family = spec.route_family if spec is not None else ""

    score = 0.0
    if route:
        score += 2.0

    if product_ir is not None:
        # Semantic fit: route_family or engine_family matches ProductIR
        ir_route_families = set(getattr(product_ir, "route_families", ()) or ())
        if route_family in ir_route_families:
            score += 2.5
        elif engine_family in product_ir.candidate_engine_families:
            score += 2.5

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


