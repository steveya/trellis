"""Structured generation context and import validation for agent-built modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from trellis.agent.blocker_planning import BlockerReport, plan_blockers, render_blocker_report
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.new_primitive_workflow import (
    NewPrimitiveWorkflow,
    plan_new_primitive_workflow,
    render_new_primitive_workflow,
)
from trellis.agent.knowledge.import_registry import (
    get_registry_snapshot,
    is_valid_import,
    list_module_exports,
    module_exists,
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

INSTRUMENT_TEST_TARGETS = {
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


@dataclass(frozen=True)
class PrimitivePlan:
    """Deterministic route + primitive selection for a product/method pair."""

    route: str
    engine_family: str
    primitives: tuple[PrimitiveRef, ...]
    adapters: tuple[str, ...]
    blockers: tuple[str, ...]
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


def build_generation_plan(
    *,
    pricing_plan,
    instrument_type: str | None,
    inspected_modules: tuple[str, ...],
    product_ir: ProductIR | None = None,
) -> GenerationPlan:
    """Build a deterministic generation plan from quant + reference context."""
    global _GENERATION_PLAN_CACHE_HITS, _GENERATION_PLAN_CACHE_MISSES

    cache_key = _generation_plan_cache_key(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=inspected_modules,
        product_ir=product_ir,
    )
    cached = _GENERATION_PLAN_CACHE.get(cache_key)
    if cached is not None:
        _GENERATION_PLAN_CACHE_HITS += 1
        return cached

    _GENERATION_PLAN_CACHE_MISSES += 1
    method = normalize_method(pricing_plan.method) if pricing_plan else "analytical"
    approved = set(COMMON_APPROVED_MODULES)
    approved.update(inspected_modules)
    primitive_plan = build_primitive_plan(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

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
    )


def render_generation_plan(plan: GenerationPlan) -> str:
    """Format the structured generation plan for prompt injection."""
    lines = [
        "## Structured Generation Plan",
        f"- Method family: `{plan.method}`",
        f"- Instrument type: `{plan.instrument_type or 'unknown'}`",
        "- Inspected modules:",
    ]
    lines.extend(f"  - `{module}`" for module in plan.inspected_modules)
    lines.append("- Approved Trellis modules for imports:")
    lines.extend(f"  - `{module}`" for module in plan.approved_modules)
    if plan.symbols_to_reuse:
        lines.append("- Public symbols available from the approved modules:")
        lines.extend(f"  - `{symbol}`" for symbol in plan.symbols_to_reuse[:80])
    if plan.proposed_tests:
        lines.append("- Tests to run after generation:")
        lines.extend(f"  - `{target}`" for target in plan.proposed_tests)
    if plan.primitive_plan is not None:
        lines.append("- Primitive route:")
        lines.append(f"  - Route: `{plan.primitive_plan.route}`")
        lines.append(f"  - Engine family: `{plan.primitive_plan.engine_family}`")
        lines.append(f"  - Route score: `{plan.primitive_plan.score:.2f}`")
        if plan.primitive_plan.primitives:
            lines.append("  - Selected primitives:")
            lines.extend(
                f"    - `{primitive.module}.{primitive.symbol}` ({primitive.role})"
                for primitive in plan.primitive_plan.primitives
            )
        if plan.primitive_plan.adapters:
            lines.append("  - Required adapters:")
            lines.extend(f"    - `{adapter}`" for adapter in plan.primitive_plan.adapters)
        if plan.primitive_plan.notes:
            lines.append("  - Route notes:")
            lines.extend(f"    - {note}" for note in plan.primitive_plan.notes)
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
        if plan.primitive_plan.primitives:
            lines.append("- Required primitives:")
            lines.extend(
                f"  - `{primitive.module}.{primitive.symbol}` ({primitive.role})"
                for primitive in plan.primitive_plan.primitives[:8]
            )
        if plan.primitive_plan.adapters:
            lines.append("- Required adapters:")
            lines.extend(f"  - `{adapter}`" for adapter in plan.primitive_plan.adapters[:6])
        if plan.primitive_plan.notes:
            lines.append("- Route notes:")
            lines.extend(f"  - {note}" for note in plan.primitive_plan.notes[:4])
    if plan.inspected_modules:
        lines.append("- Primary modules to inspect/reuse:")
        lines.extend(f"  - `{module}`" for module in plan.inspected_modules[:6])
    if plan.proposed_tests:
        lines.append("- Post-build test targets:")
        lines.extend(f"  - `{target}`" for target in plan.proposed_tests[:4])
    lines.append(
        "- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels."
    )
    return "\n".join(lines)


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
    """Return candidate primitive routes ranked by deterministic heuristic score."""
    if pricing_plan is None:
        return ()

    method = normalize_method(pricing_plan.method)
    ranked: list[PrimitivePlan] = []
    for route in _candidate_routes(method, product_ir, pricing_plan=pricing_plan):
        primitives, adapters, notes = _route_components(
            route,
            product_ir=product_ir,
            pricing_plan=pricing_plan,
        )
        blockers = list(product_ir.unresolved_primitives if product_ir is not None else ())
        blockers.extend(_verify_primitives(primitives))
        plan = PrimitivePlan(
            route=route,
            engine_family=_route_engine_family(route),
            primitives=tuple(primitives),
            adapters=tuple(adapters),
            blockers=tuple(dict.fromkeys(blockers)),
            notes=tuple(notes),
            score=_route_score(route, product_ir, blockers, pricing_plan=pricing_plan),
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


def _candidate_routes(
    method: str,
    product_ir: ProductIR | None,
    *,
    pricing_plan=None,
) -> tuple[str, ...]:
    """Return plausible primitive assembly routes for a method/product pair."""
    candidates: list[str] = []

    if method in {"monte_carlo", "qmc"} and product_ir is not None:
        if product_ir.exercise_style in {"american", "bermudan"}:
            candidates.append("exercise_monte_carlo")
        if method == "monte_carlo" and _requires_local_vol(pricing_plan):
            candidates.append("local_vol_monte_carlo")
        if method == "monte_carlo":
            candidates.append("monte_carlo_paths")
        if method == "qmc":
            candidates.append("qmc_sobol_paths")
        return tuple(dict.fromkeys(candidates))

    if method == "rate_tree":
        if product_ir is not None and product_ir.exercise_style in {
            "american",
            "bermudan",
            "issuer_call",
            "holder_put",
        }:
            candidates.append("exercise_lattice")
        candidates.append("rate_tree_backward_induction")
        return tuple(dict.fromkeys(candidates))

    analytical_route = (
        "analytical_garman_kohlhagen"
        if _is_fx_analytical_context(pricing_plan)
        else "analytical_black76"
    )
    fallback = {
        "analytical": (analytical_route,),
        "fft_pricing": ("transform_fft",),
        "pde_solver": ("pde_theta_1d",),
        "copula": ("copula_loss_distribution",),
        "waterfall": ("waterfall_cashflows",),
    }
    return fallback.get(method, ())


def _route_components(
    route: str,
    *,
    product_ir: ProductIR | None = None,
    pricing_plan=None,
) -> tuple[list[PrimitiveRef], tuple[str, ...], tuple[str, ...]]:
    """Return primitives, adapter obligations, and notes for a named route."""
    if route == "exercise_monte_carlo":
        from trellis.agent.early_exercise_policy import (
            render_early_exercise_policy_summary,
            render_implemented_early_exercise_policy_summary,
        )

        return (
            [
                PrimitiveRef("trellis.models.processes.gbm", "GBM", "state_process"),
                PrimitiveRef("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
                PrimitiveRef("trellis.models.monte_carlo.lsm", "longstaff_schwartz", "exercise_control", required=False),
                PrimitiveRef("trellis.models.monte_carlo.tv_regression", "tsitsiklis_van_roy", "exercise_control", required=False),
                PrimitiveRef("trellis.models.monte_carlo.primal_dual", "primal_dual_mc", "exercise_control", required=False),
                PrimitiveRef("trellis.models.monte_carlo.stochastic_mesh", "stochastic_mesh", "exercise_control", required=False),
            ],
            (
                "derive_exercise_dates_from_schedule_or_time_grid",
                "build_spot_to_exercise_payoff_callback",
                "select_continuation_estimator_or_basis",
            ),
            (
                "Use an approved early-exercise control primitive instead of inventing new engine modes.",
                f"Approved policy classes: {render_early_exercise_policy_summary()}.",
                f"Implemented today: {render_implemented_early_exercise_policy_summary()}.",
                "Regression basis choice is a continuation-estimator detail, not a required route primitive.",
            ),
        )
    if route == "analytical_black76":
        if product_ir is not None and product_ir.payoff_family == "vanilla_option":
            return (
                [
                    PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
                    PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
                    PrimitiveRef("trellis.core.date_utils", "year_fraction", "time_measure"),
                ],
                ("map_spot_discount_and_vol_to_forward_black76",),
                (
                    "For European vanilla equity options, derive the forward from spot and discounting before calling Black-style kernels.",
                ),
            )
        return (
            [
                PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
                PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
                PrimitiveRef("trellis.core.date_utils", "generate_schedule", "schedule_builder"),
                PrimitiveRef("trellis.core.date_utils", "year_fraction", "time_measure"),
            ],
            ("extract_forward_and_annuity_from_market_state",),
            ("Prefer thin orchestration around existing analytical kernels.",),
        )
    if route == "analytical_garman_kohlhagen":
        return (
            [
                PrimitiveRef("trellis.models.black", "garman_kohlhagen_call", "pricing_kernel"),
                PrimitiveRef("trellis.models.black", "garman_kohlhagen_put", "pricing_kernel"),
                PrimitiveRef("trellis.core.date_utils", "year_fraction", "time_measure"),
            ],
            ("map_fx_spot_and_curves_to_garman_kohlhagen_inputs",),
            (
                "For European vanilla FX options, use Garman-Kohlhagen with domestic discounting, foreign discounting, spot FX, and Black vol.",
            ),
        )
    if route == "rate_tree_backward_induction":
        return (
            [
                PrimitiveRef("trellis.models.trees.lattice", "build_rate_lattice", "lattice_builder"),
                PrimitiveRef("trellis.models.trees.lattice", "lattice_backward_induction", "backward_induction"),
            ],
            ("map_cashflows_and_exercise_dates_to_tree_steps",),
            ("Use calibrated rate-tree primitives for scheduled early-exercise products.",),
        )
    if route == "exercise_lattice":
        return (
            [
                PrimitiveRef("trellis.models.trees.lattice", "build_rate_lattice", "lattice_builder"),
                PrimitiveRef("trellis.models.trees.lattice", "lattice_backward_induction", "backward_induction"),
            ],
            (
                "map_cashflows_and_exercise_dates_to_tree_steps",
                "select_exercise_fn_for_issuer_or_holder",
            ),
            (
                "Use lattice_backward_induction with schedule-aware exercise steps for callable, puttable, and Bermudan products.",
            ),
        )
    if route == "transform_fft":
        return (
            [
                PrimitiveRef("trellis.models.transforms.fft_pricer", "fft_price", "transform_pricer"),
                PrimitiveRef("trellis.models.transforms.cos_method", "cos_price", "transform_pricer", required=False),
            ],
            ("build_vector_safe_characteristic_function",),
            ("Characteristic functions must be array-safe for transform pricing.",),
        )
    if route == "qmc_sobol_paths":
        return (
            [
                PrimitiveRef("trellis.models.qmc", "sobol_normals", "low_discrepancy_sampler"),
                PrimitiveRef("trellis.models.qmc", "brownian_bridge", "path_construction", required=False),
                PrimitiveRef("trellis.models.processes.gbm", "GBM", "state_process"),
            ],
            ("assemble_mc_style_estimator_on_sobol_paths",),
            ("QMC in Trellis accelerates MC-style estimators rather than replacing them.",),
        )
    if route == "monte_carlo_paths":
        return (
            [
                PrimitiveRef("trellis.models.processes.gbm", "GBM", "state_process"),
                PrimitiveRef("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
            ],
            ("build_payoff_vector_from_paths",),
            ("Prefer existing MC simulation helpers over bespoke path loops.",),
        )
    if route == "local_vol_monte_carlo":
        return (
            [
                PrimitiveRef("trellis.models.processes.local_vol", "LocalVol", "state_process"),
                PrimitiveRef("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
                PrimitiveRef(
                    "trellis.models.monte_carlo.local_vol",
                    "local_vol_european_vanilla_price",
                    "pricing_kernel",
                ),
            ],
            (
                "map_market_state_local_vol_surface_spot_and_discount_into_local_vol_mc_inputs",
                "derive_option_type_strike_and_expiry_for_vanilla_equity_option",
            ),
            (
                "For vanilla equity local-vol requests, prefer the reusable local-vol Monte Carlo helper over bespoke path loops.",
            ),
        )
    if route == "pde_theta_1d":
        return (
            [
                PrimitiveRef("trellis.models.pde.theta_method", "theta_method_1d", "time_stepping"),
            ],
            ("define_operator_boundary_terminal_conditions",),
            (),
        )
    if route == "copula_loss_distribution":
        return (
            [
                PrimitiveRef("trellis.models.copulas.factor", "FactorCopula", "loss_distribution"),
            ],
            ("map_credit_curve_to_marginal_default_probabilities",),
            (),
        )
    if route == "waterfall_cashflows":
        return (
            [
                PrimitiveRef("trellis.models.cashflow_engine.waterfall", "Waterfall", "cashflow_engine"),
                PrimitiveRef("trellis.models.cashflow_engine.waterfall", "Tranche", "cashflow_engine"),
            ],
            ("map_collateral_cashflows_into_structure",),
            (),
        )
    return ([], (), ())


def _route_engine_family(route: str) -> str:
    """Map a primitive route name onto the coarse engine-family taxonomy."""
    if route == "exercise_monte_carlo":
        return "exercise"
    if route == "exercise_lattice":
        return "lattice"
    if route == "rate_tree_backward_induction":
        return "lattice"
    if route == "transform_fft":
        return "fft_pricing"
    if route == "qmc_sobol_paths":
        return "qmc"
    if route == "monte_carlo_paths":
        return "monte_carlo"
    if route == "local_vol_monte_carlo":
        return "monte_carlo"
    if route == "analytical_black76":
        return "analytical"
    if route == "analytical_garman_kohlhagen":
        return "analytical"
    if route == "pde_theta_1d":
        return "pde_solver"
    if route == "copula_loss_distribution":
        return "copula"
    if route == "waterfall_cashflows":
        return "waterfall"
    return "unknown"


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
    route: str,
    product_ir: ProductIR | None,
    blockers: list[str],
    *,
    pricing_plan=None,
) -> float:
    """Score a route using semantic fit bonuses and heavy penalties for blockers."""
    score = 0.0
    if route:
        score += 2.0
    if product_ir is not None:
        if _route_engine_family(route) in product_ir.candidate_engine_families:
            score += 2.5
        if route == "exercise_monte_carlo" and product_ir.exercise_style in {"american", "bermudan"}:
            score += 2.0
        if route == "exercise_lattice" and product_ir.exercise_style in {
            "american",
            "bermudan",
            "issuer_call",
            "holder_put",
        }:
            score += 2.0
        if route == "exercise_lattice" and product_ir.schedule_dependence:
            score += 1.0
        if route == "local_vol_monte_carlo" and getattr(pricing_plan, "required_market_data", None):
            required = set(pricing_plan.required_market_data)
            if "local_vol_surface" in required:
                score += 3.0
            if "spot" in required:
                score += 1.0
        if route == "monte_carlo_paths" and product_ir.exercise_style not in {"none", "european"}:
            score -= 2.0
        if route == "monte_carlo_paths" and getattr(pricing_plan, "required_market_data", None):
            if "local_vol_surface" in set(pricing_plan.required_market_data):
                score -= 3.0
        if route == "rate_tree_backward_induction" and product_ir.exercise_style not in {"none", "european"}:
            score -= 2.0
        if route == "analytical_black76" and product_ir.payoff_family in {"swaption", "vanilla_option"}:
            score += 1.0
        if route == "analytical_black76" and product_ir.exercise_style not in {"none", "european"}:
            score -= 4.0
        if route == "analytical_garman_kohlhagen" and product_ir.payoff_family == "vanilla_option":
            score += 1.0
        if route == "analytical_garman_kohlhagen" and product_ir.exercise_style not in {"none", "european"}:
            score -= 4.0
        if blockers and not product_ir.supported:
            score -= 2.0
    score -= len(blockers) * 6.0
    return score


def _is_fx_analytical_context(pricing_plan) -> bool:
    """Whether the analytical route should use FX vanilla primitives."""
    if pricing_plan is None:
        return False
    required = set(getattr(pricing_plan, "required_market_data", ()) or ())
    return "fx_rates" in required and "forward_curve" in required


def _requires_local_vol(pricing_plan) -> bool:
    """Whether the pricing plan explicitly depends on a local-vol surface."""
    if pricing_plan is None:
        return False
    required = set(getattr(pricing_plan, "required_market_data", ()) or ())
    return "local_vol_surface" in required
