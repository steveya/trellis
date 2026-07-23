"""System prompt templates for the Trellis agent."""

from __future__ import annotations

from pathlib import Path


def load_requirements() -> str:
    """Load the project-level coding rules and constraints from requirements.md."""
    req_path = Path(__file__).parent.parent / "requirements.md"
    if req_path.exists():
        return req_path.read_text()
    return ""


def system_prompt() -> str:
    """Build the full system prompt for the interactive agent."""
    requirements = load_requirements()
    return f"""You are the Trellis pricing agent — an AI that builds, extends, and operates
a quantitative pricing library. You inspect the library, identify gaps, generate code,
run tests, fetch market data, and price instruments on demand.

## Library Rules and Constraints
{requirements}

## Capabilities
- inspect_api_map: Inspect the bounded API catalog or select task-relevant cards from semantic fields before broad tree exploration.
- inspect_library: See all modules, classes, and functions in the trellis package.
- read_module: Read source code of any module.
- find_symbol: Locate a public Trellis symbol before importing it.
- list_exports: Inspect a module's public exports before using it.
- resolve_import_candidates: Resolve candidate modules for one or more symbols.
- lookup_primitive_route: Inspect the deterministic primitive route, required primitives, and adapters.
- build_thin_adapter_plan: Render the thin-adapter obligations for a generated payoff.
- select_invariant_pack: Inspect which deterministic invariant checks the generated payoff should satisfy.
- build_comparison_harness: Expand TASK-style comparison metadata into concrete build/validation targets.
- capture_cookbook_candidate: Extract a deterministic cookbook-candidate payload from successful code.
- search_repo: Search the trellis source tree for existing implementations.
- search_tests: Search tests for existing coverage and examples.
- search_lessons: Search lessons and traces for past failure patterns.
- write_module: Create or update modules (the agent writes code into the library).
- run_tests: Execute the test suite to verify correctness.
- fetch_market_data: Pull Treasury yields from FRED or Treasury.gov.
- execute_pricing: Price an instrument using a yield curve.

## Workflow
1. Understand the user's request.
2. Inspect the API map first if the module family is unclear, then inspect the library, exports, tests, and lessons to see what already exists.
3. Resolve every `trellis.*` import against the live package before writing code. Start from the API map or `find_symbol`, confirm with `list_exports`, and only then call `read_module`.
4. If something is missing, plan what to build, write the code, and test it.
5. Fetch any needed market data.
6. Execute the pricing and return results.

Always test your code before using it for pricing. Max 3 retry attempts on test failures.
Never invent Trellis import paths or symbols. If the repo inspection tools do not confirm
an import, do not use it.
"""


def _ordered_unique_modules(modules) -> tuple[str, ...]:
    """Preserve first-seen module order while removing blanks and duplicates."""
    ordered: list[str] = []
    seen: set[str] = set()
    for module in modules or ():
        module_path = str(module or "").strip()
        if not module_path or module_path in seen:
            continue
        seen.add(module_path)
        ordered.append(module_path)
    return tuple(ordered)


def _route_bound_modules(generation_plan) -> tuple[str, ...]:
    """Return exact route modules when the compiler resolved a primitive binding."""
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is None:
        return ()
    return _ordered_unique_modules(
        getattr(primitive, "module", "")
        for primitive in (getattr(primitive_plan, "primitives", ()) or ())
        if not getattr(primitive, "excluded", False)
    )


def _render_prompt_module_requirements(pricing_plan=None, generation_plan=None) -> str:
    """Render the module requirements block for builder prompts."""
    route_modules = _route_bound_modules(generation_plan)
    if route_modules:
        modules_block = "\n".join(f"- `{module}`" for module in route_modules)
        return (
            "Route-bound modules to import and use:\n"
            f"{modules_block}\n\n"
            "You MUST import and use these route-bound modules in your implementation.\n"
            "Do not import a generic parent package such as `from trellis.models import ...` "
            "just to satisfy the method family.\n\n"
        )

    inspected_modules = _ordered_unique_modules(
        getattr(generation_plan, "inspected_modules", ()) if generation_plan is not None else ()
    )
    if inspected_modules:
        modules_block = "\n".join(f"- `{module}`" for module in inspected_modules)
        return (
            "Modules to import and use:\n"
            f"{modules_block}\n\n"
            "You MUST import and use these compiler-selected modules in your implementation.\n"
            "Do not fall back to a generic family module when the compiler already narrowed the inspected surface.\n\n"
        )

    plan_modules = _ordered_unique_modules(getattr(pricing_plan, "method_modules", ()) or ())
    modules_block = (
        "\n".join(f"- `{module}`" for module in plan_modules)
        if plan_modules
        else "No special modules needed — use standard discounting."
    )
    return (
        "Modules to import and use:\n"
        f"{modules_block}\n\n"
        + (
            "You MUST import and use these modules in your implementation.\n\n"
            if plan_modules
            else "No special modules needed — use standard discounting.\n\n"
        )
    )


# ---------------------------------------------------------------------------
# Two-step structured code generation prompts
# ---------------------------------------------------------------------------

def spec_design_prompt(
    payoff_description: str,
    requirements: set[str],
) -> str:
    """Step 1 prompt: design the spec schema via structured JSON output."""
    from trellis.core.capabilities import capability_summary

    return f"""You are designing the specification dataclass for a payoff in the Trellis pricing library.

## Task
Design the spec fields for: {payoff_description}

## Requirements
This payoff requires these MarketState capabilities: {sorted(requirements)}

{capability_summary(requirements, include_methods=False)}

## Allowed field types
- "float" — numeric value
- "int" — integer value
- "str" — string value
- "bool" — boolean flag
- "date" — from datetime import date
- "str | None" — optional string
- "float | None" — optional numeric value
- "int | None" — optional integer value
- "tuple[date, ...]" — ordered explicit schedule dates
- "tuple[date, ...] | None" — optional ordered explicit schedule dates
- "Frequency" — from trellis.core.types (ANNUAL, SEMI_ANNUAL, QUARTERLY, MONTHLY)
- "DayCountConvention" — from trellis.core.types (ACT_360, ACT_365, THIRTY_360)

## Literal / enum defaults
- String defaults must be quoted Python literals, not bare identifiers.
- If a default needs an enum-like token, use a real imported enum member or a quoted literal that the module can validate later.
- Do not invent standalone names such as `AMERICAN`, `absorbing`, or `dirichlet` just to satisfy a default value.

## Naming conventions
Follow these exact naming conventions:
- notional (not notional_amount, principal)
- strike (not strike_rate, strike_price)
- expiry_date (not expiry, option_expiry, maturity)
- start_date or swap_start (not effective_date)
- end_date or swap_end (not termination_date)
- rate_index (str | None, for multi-curve)
- is_payer (bool, True = pay fixed)
- day_count (DayCountConvention)
- frequency or swap_frequency (Frequency)
- call_dates / put_dates / exercise_dates / observation_dates should use `tuple[date, ...]`, not comma-separated strings

## Output
Return a JSON object with this exact structure:
{{
    "class_name": "ThePayoffClassName",
    "spec_name": "TheSpecClassName",
    "requirements": ["cap1", "cap2"],
    "fields": [
        {{"name": "field_name", "type": "float", "description": "...", "default": null}},
        {{"name": "field_with_default", "type": "Frequency", "description": "...", "default": "Frequency.SEMI_ANNUAL"}}
    ]
}}

Required fields (default: null) must come before optional fields.
Return ONLY the JSON object, no other text."""


def evaluate_prompt(
    skeleton_code: str,
    spec_schema,
    reference_sources: dict[str, str],
    pricing_plan=None,
    knowledge_context: str = "",
    generation_plan=None,
    prompt_surface: str = "expanded",
) -> str:
    """Step 2 prompt: implement only the evaluate() method body.

    Parameters
    ----------
    pricing_plan : PricingPlan or None
        If provided, includes the quant agent's method recommendation.
    knowledge_context : str
        Pre-formatted knowledge from the KnowledgeStore.  When provided,
        replaces the inline method_guidance construction (cookbook, contracts,
        requirements, lessons).
    """
    field_descs = "\n".join(
        f"- `self._spec.{f.name}` ({f.type}): {f.description}"
        for f in spec_schema.fields
    )

    refs = _render_reference_sources(
        reference_sources,
        prompt_surface=prompt_surface,
        pricing_plan=pricing_plan,
    )
    selection_notes = _render_pricing_plan_selection_notes(pricing_plan)

    method_guidance = ""
    if knowledge_context:
        # New path: unified knowledge from KnowledgeStore
        method_guidance = knowledge_context
        # Still add method selection info if available
        if pricing_plan:
            from trellis.agent.knowledge.methods import normalize_method
            method_name = normalize_method(pricing_plan.method)
            selection_notes = _render_pricing_plan_selection_notes(pricing_plan)
            method_guidance = (
                f"\n## Pricing Method (selected by the quant agent — you MUST use this)\n"
                f"Method: **{method_name}**\n"
                f"Reasoning: {pricing_plan.reasoning}\n"
                f"{selection_notes}\n"
                + _render_prompt_module_requirements(
                    pricing_plan=pricing_plan,
                    generation_plan=generation_plan,
                )
                + method_guidance
            )
    elif pricing_plan:
        # Compatibility path: build unified shared knowledge on demand.
        from trellis.agent.knowledge.methods import normalize_method
        method_name = normalize_method(pricing_plan.method)
        shared_knowledge_text = ""
        try:
            from trellis.agent.knowledge import (
                build_shared_knowledge_payload,
                retrieve_for_task,
            )

            shared_knowledge = retrieve_for_task(
                method=method_name,
                instrument=pricing_plan.model_to_build,
            )
            shared_knowledge_text = build_shared_knowledge_payload(
                shared_knowledge,
                pricing_method=method_name,
            )["builder_text_distilled"]
        except Exception:
            shared_knowledge_text = ""

        method_guidance = f"""
## Pricing Method (selected by the quant agent — you MUST use this)
Method: **{method_name}**
Reasoning: {pricing_plan.reasoning}
{selection_notes}
{_render_prompt_module_requirements(
    pricing_plan=pricing_plan,
    generation_plan=generation_plan,
).rstrip()}
        """
        if shared_knowledge_text:
            method_guidance += (
                "\n"
                + shared_knowledge_text
                + "\n\nIMPORTANT: Follow the shared cookbook and lesson guidance above. "
                  "Adapt it for this specific instrument but keep the same structure: "
                  "imports, market data access, method invocation, return type.\n"
            )

    generation_context = _render_generation_context(
        generation_plan,
        prompt_surface=prompt_surface,
    )
    assembly_context = _render_assembly_context(
        spec_schema,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
    )
    family_route_guidance = _render_family_route_guidance(
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
    )
    early_exercise_guidance = _render_early_exercise_guidance()

    return f"""You are implementing the evaluate() method for `{spec_schema.class_name}` in the Trellis pricing library.

## Complete module (skeleton — everything is fixed except evaluate)
```python
{skeleton_code}
```

## Your task
Write ONLY the body of the `evaluate()` method. The signature is already
defined in the skeleton.

## Spec fields available via self._spec
{field_descs}
{method_guidance}
{generation_context}
{assembly_context}
{family_route_guidance}
## Conventions
- evaluate() returns the present-value scalar (PV) of the instrument
- On smooth autodiff-compatible routes that scalar may be traced by the active differentiable backend
- Do not wrap the final present value in `float(...)` solely because `evaluate()` is public
- Prefer the raw-kernel-plus-wrapper pattern when the route has a reusable resolved-input kernel: keep market resolution in `evaluate()`, keep reusable math in the raw helper, and preserve the same traced PV through both surfaces
- If a plain Python `float` is required, convert only at an explicit reporting or solver boundary, not at the payoff adapter boundary
- You MUST handle all discounting internally — use `market_state.discount.discount(t)`
- For forward rates: `market_state.forecast_forward_curve(self._spec.rate_index)`
- For vol: `market_state.vol_surface.black_vol(T, strike)`
- For discount factors: `market_state.discount.discount(t)`
- `market_state.fx_rates[pair]` returns an `FXRate` wrapper; extract `.spot` before scalar arithmetic or process seeding
- Schedule generation: prefer `build_payment_timeline(...)`, `build_observation_timeline(...)`, or `build_period_schedule(...)` for accrual/event routes; use `generate_schedule(start, end, freq)` only for plain date lists with no period semantics
- Year fractions: `year_fraction(date1, date2, day_count)`
- Never use wall-clock dates such as `date.today()` or `datetime.now()` inside `evaluate()`; derive valuation time from `market_state` or shared resolver outputs.
- Black76: `black76_call(F, K, sigma, T)`, `black76_put(F, K, sigma, T)` — undiscounted
- Black76 digital: `black76_cash_or_nothing_call(F, K, sigma, T)`, `black76_cash_or_nothing_put(F, K, sigma, T)` — undiscounted cash-or-nothing digitals
- For CDS / nth-to-default: use `market_state.credit_curve.survival_probability(t)` and `market_state.credit_curve.hazard_rate(t)` on an explicit payment/default schedule; do not route credit-default pricing through Black76 call/put primitives.
- For equity trees: resolve scalar inputs with `resolve_single_state_diffusion_inputs`, declare the claim with `equity_tree`, attach `with_control`, then use `compile_lattice_recipe`, `build_lattice`, and `price_on_lattice`
- For vanilla European PDE routes: compose `resolve_single_state_diffusion_inputs`, `terminal_intrinsic_from_resolved`, `EventAwarePDEProblemSpec`, `build_event_aware_pde_problem`, `solve_event_aware_pde`, and `interpolate_pde_values`
- For rate lattices: follow the selected route card and compose its market resolver, topology, mesh, calibration target, contract compiler, and generic rollback primitive; do not default to a product-specific pricing wrapper
- For schedule-dependent rate lattices: `from trellis.models.trees.control import lattice_steps_from_timeline, resolve_lattice_exercise_policy`
- For MC: `from trellis.models.monte_carlo import MonteCarloEngine`
- For QMC accelerators: `from trellis.models.qmc import sobol_normals, brownian_bridge`
- For copulas: `from trellis.models.copulas import GaussianCopula, FactorCopula`
- Do not invent MonteCarloEngine method strings. Valid `method=` values are `euler`, `milstein`, and `exact`.
- {early_exercise_guidance}
- If you use `LaguerreBasis`, import it from `trellis.models.monte_carlo.schemes`, not from `trellis.models.monte_carlo.lsm`.
- For FFT/COS pricing, characteristic functions must accept vector `u` and use array-safe numerics such as `numpy`, not scalar `math`/`cmath`.
- Black76 basis: `black76_asset_or_nothing_call(F, K, sigma, T)`, `black76_asset_or_nothing_put(F, K, sigma, T)`, `black76_cash_or_nothing_call(F, K, sigma, T)`, `black76_cash_or_nothing_put(F, K, sigma, T)` — exact terminal basis claims.
- For terminal vanilla payoffs, prefer exact basis assembly via `terminal_vanilla_from_basis(...)` from `trellis.models.analytical`.
- For FX vanilla options, treat the route as Garman-Kohlhagen: map spot FX and domestic/foreign discount factors to `ResolvedGarmanKohlhagenInputs`, then prefer `garman_kohlhagen_price_raw(spec.option_type, resolved)` from `trellis.models.analytical.fx`; use explicit basis-claim assembly only when the request explicitly needs the decomposition.
- For cash-or-nothing digital options, use the Black76 digital helpers directly; do not approximate them with vanilla call/put prices or divide by spot.
- You MUST use only real, approved `trellis.*` imports from the structured generation plan and import registry
- Treat the compiler-emitted lane obligations in the structured generation plan as the backbone of the implementation.
- Reuse the listed exact backend bindings when present; otherwise build the smallest lane-consistent kernel that satisfies the construction steps.
- Do not replace selected primitives with bespoke numerical kernels or alternative Trellis routes unless the plan explicitly permits a new lane implementation.
- Do not add wildcard imports
- If the approved modules do not contain what you need, reuse the closest existing implementation and keep the gap explicit instead of inventing a path

## Reference implementations
{refs}

## Output
Return the COMPLETE Python module — copy the skeleton exactly, but replace the
`raise NotImplementedError(...)` line with the actual implementation of `evaluate()`.
Do NOT change imports, class names, spec fields, or the requirements property.
Only implement the evaluate() method body. You may add imports at the top.
No markdown fences, no explanation — just the Python code."""


def _render_generation_context(generation_plan, *, prompt_surface: str) -> str:
    """Render compact or expanded structured generation context."""
    if generation_plan is None:
        return ""
    from trellis.agent.codegen_guardrails import (
        render_import_repair_card,
        render_generation_plan,
        render_generation_route_card,
        render_semantic_repair_card,
    )

    if prompt_surface == "compact":
        return render_generation_route_card(generation_plan)
    if prompt_surface == "import_repair":
        return render_import_repair_card(generation_plan)
    if prompt_surface == "semantic_repair":
        return render_semantic_repair_card(generation_plan)
    return render_generation_plan(generation_plan)


def _render_early_exercise_guidance() -> str:
    """Render compact prompt guidance for Monte Carlo early exercise."""
    from trellis.agent.early_exercise_policy import (
        render_early_exercise_policy_summary,
        render_implemented_early_exercise_policy_summary,
    )

    return (
        "For Monte Carlo early exercise, use an approved control primitive; do "
        "not pretend that `MonteCarloEngine.price(...)` or `method=\"lsm\"` "
        "implements early exercise. Approved policy classes: "
        f"{render_early_exercise_policy_summary()}. Currently implemented in "
        f"Trellis: {render_implemented_early_exercise_policy_summary()}."
    )


def _render_pricing_plan_selection_notes(pricing_plan) -> str:
    """Render the method-choice rationale and defaulted assumptions."""
    if pricing_plan is None:
        return ""

    selection_reason = getattr(pricing_plan, "selection_reason", "") or ""
    assumptions = tuple(getattr(pricing_plan, "assumption_summary", ()) or ())
    if not selection_reason and not assumptions:
        return ""

    lines = ["Selection basis and assumptions:"]
    if selection_reason:
        lines.append(f"- Selection basis: `{selection_reason}`")
    if assumptions:
        lines.append("- Assumptions / defaulted context:")
        lines.extend(f"  - `{assumption}`" for assumption in assumptions)
    else:
        lines.append("- Assumptions / defaulted context: none recorded")
    return "\n".join(lines)


def _render_family_route_guidance(
    *,
    pricing_plan=None,
    generation_plan=None,
) -> str:
    """Render narrow family-specific implementation guidance when the route is known."""
    instrument_type = (
        getattr(generation_plan, "instrument_type", None)
        or getattr(pricing_plan, "model_to_build", None)
        or ""
    ).strip().lower().replace(" ", "_")
    method = (
        getattr(pricing_plan, "method", None)
        or getattr(generation_plan, "method", None)
        or ""
    ).strip().lower().replace(" ", "_")

    lines: list[str] = []

    if instrument_type in {"american_option", "american_put"}:
        lines.append("## Family Route Guidance")
        if method == "rate_tree":
            lines.extend([
                "- For American/Bermudan equity tree routes, compose `resolve_single_state_diffusion_inputs`, `equity_tree`, `with_control`, `compile_lattice_recipe`, `build_lattice`, and `price_on_lattice`.",
                "- Map Bermudan exercise dates to lattice steps before passing `exercise_steps` to `with_control`; American control remains open at every rollback step.",
                "- Do not invent a `crr_tree` symbol, an `AMERICAN` constant, or a `method=\"lsm\"` fallback on the tree route.",
                "- Use the lattice algebra for transition probabilities and backward induction; keep the generated adapter focused on contract and market composition.",
            ])
        elif method == "pde_solver":
            lines.extend([
                "- For American PDE routes, import `Grid` from `trellis.models.pde.grid`, `BlackScholesOperator` from `trellis.models.pde.operator`, and `theta_method_1d` from `trellis.models.pde.theta_method`.",
                "- Use the real `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` constructor; do not invent helper names like `psor_pde`, `log_grid_pde`, or `uniform_grid_pde`.",
                "- Pass `exercise_values` and `exercise_fn=max` for American puts; do not invent a bare `AMERICAN` constant or a `dirichlet`/`absorbing` token.",
                "- For discontinuous terminal payoffs, set `rannacher_timesteps` so the solver can smooth the first backward steps.",
            ])
        elif method == "monte_carlo":
            lines.extend([
                "- For American Monte Carlo, simulate paths with `MonteCarloEngine`, then hand the path tensor to `longstaff_schwartz` from `trellis.models.monte_carlo.lsm`.",
                "- Do not invent `lsm_mc` as a symbol or treat `engine.price(...)` as an early-exercise solver.",
                "- If you need a continuation basis, choose it explicitly; `LaguerreBasis` lives in `trellis.models.monte_carlo.schemes`.",
            ])

    if instrument_type == "european_option":
        if method == "pde_solver":
            lines.append("## Family Route Guidance")
            lines.extend([
                "- For vanilla European PDE routes, resolve market inputs with `resolve_single_state_diffusion_inputs` and declare terminal intrinsic with `terminal_intrinsic_from_resolved`.",
                "- Map `implementation_target=theta_0.5` to `theta=0.5` and `implementation_target=theta_1.0` to `theta=1.0`.",
                "- Build `EventAwarePDEProblemSpec` from grid, Black-Scholes operator, terminal payoff, and boundary specs; then call `build_event_aware_pde_problem`, `solve_event_aware_pde`, and `interpolate_pde_values`.",
                "- Keep product terms and market binding in the adapter while delegating only reusable numerical mechanics to the event-aware PDE substrate.",
            ])

    if instrument_type in {"cds", "credit_default_swap"}:
        lines.append("## Family Route Guidance")
        if method in {"monte_carlo", "qmc"}:
            lines.extend([
                "- For single-name CDS Monte Carlo routes, keep the premium leg and protection leg on an explicit payment/default schedule for one reference entity.",
                "- Prefer `build_cds_schedule` and `price_cds_monte_carlo` from `trellis.models.credit_default_swap` so the adapter delegates to checked-in CDS helpers instead of open-coding the leg loop.",
                "- If the spec exposes `n_paths`, pass `spec.n_paths` through to `price_cds_monte_carlo(...)` instead of hard-coding a smaller path count in the adapter.",
                "- Do not hard-code `n_paths=50000` for a comparison-quality CDS route. Use `spec.n_paths` when available; otherwise pick a comparison-stable path count such as `250000`.",
                "- Keep comparison-task randomness reproducible with `seed=42` unless the spec explicitly carries a different seed input.",
                "- Do not import or instantiate `MonteCarloEngine` for this route. Here Monte Carlo means direct random default-time draws, not a generic diffusion-engine wrapper.",
                "- CDS running spreads are often quoted in basis points in task text. Normalize them at the top of `evaluate()` with `spread = float(spec.spread)` and `if spread > 1.0: spread *= 1e-4`, for example `150 bp -> 0.015`.",
                "- After that normalization step, use only the local `spread` variable in the premium leg. Do not read raw `spec.spread` again later in the body.",
                "- Treat `100` and `0.01` as semantically equivalent CDS running spreads. The route should price them the same up to numerical tolerance.",
                "- Start the body with `from trellis.core.differentiable import get_numpy` and `np = get_numpy()` so the route uses the approved array backend.",
                "- Use `rng = np.random.default_rng(...)` or equivalent direct RNG draws to sample default times from the credit curve hazard structure.",
                "- Use `market_state.credit_curve.hazard_rate(t)` or `market_state.credit_curve.survival_probability(t)` directly on the schedule; do not hide the credit curve behind an alias.",
                "- Use `market_state.discount.discount(t)` directly for each payment-date discount factor.",
                "- Build the explicit schedule with `build_period_schedule(spec.start_date, spec.end_date, spec.frequency, day_count=spec.day_count, time_origin=spec.start_date)` and iterate over `period.payment_date`, `period.accrual_fraction`, and `period.t_payment`.",
                "- This route must price a Monte Carlo expectation over many paths. Use `n_paths = ...`, `alive = np.ones(n_paths, dtype=bool)`, vectorized `default_in_interval`, and return a path average such as `float(np.mean(protection_pv - premium_pv))`.",
                "- Do not collapse the Monte Carlo leg to scalar `alive`, a single `rng.random()` draw per coupon date, or a one-scenario loop with `break` after default.",
                "- Compute interval default probability from `survival_probability(prev_t)` and `survival_probability(t_pay)` as `1.0 - s_pay / s_prev` when `s_prev > 0.0`.",
                "- Use `hazard_rate` only for within-interval default-time interpolation after an interval default is sampled; do not replace the interval default probability with `1.0 - exp(-hazard * dt)` when survival probabilities are available.",
                "- For this comparison route, keep protection-leg discounting aligned with the analytical schedule loop: use the payment-date discount factor `discount(t_pay)` for interval default mass.",
                "- Do not discount protection at sampled default times `tau` or replace interval default mass with sampled settlement-time discounting in the comparison build.",
                "- Use `spec.start_date` as the time origin for Monte Carlo schedule times so the MC and analytical CDS legs share the same `t` convention.",
                "- If you track both accrual dates and survival/default times, keep `prev_date` and `prev_t` as separate variables. Do not compare float year-fractions to `date` objects or pass floats into `year_fraction(...)` date slots.",
                "- Keep a persistent `alive` indicator across the schedule. Sample `default_in_interval` once per accrual interval, add protection only on that interval, then update `alive` before the next payment date.",
                "- Update `alive` immediately after drawing `default_in_interval`, then use the updated `alive` state for premium accrual at the payment date.",
                "- Premium accrual should use the fraction of paths still alive through the payment date, not the start-of-interval alive state. Do not overwrite or reinitialize the default state inside each loop iteration.",
                "- Keep the body as a single explicit schedule loop plus a final `premium_leg` / `protection_leg` PV aggregation; do not invent helper names or route credit-default pricing through Black76.",
                "- Do not import copulas or reinterpret a single-name CDS as nth-to-default, basket CDS, or first-to-default.",
                "- A good shape is: initialize `premium_leg = 0.0`, `protection_leg = 0.0`, loop over the payment dates, update both legs, then `return protection_leg - premium_leg`.",
            ])
        elif method == "analytical":
            lines.extend([
                "- For single-name CDS analytical routes, build the premium leg and protection leg directly from the credit curve on the explicit payment schedule.",
                "- Prefer `build_cds_schedule` and `price_cds_analytical` from `trellis.models.credit_default_swap` so the adapter stays thin and reuses the checked-in leg logic.",
                "- Prefer `build_period_schedule(...)` over raw `generate_schedule(...)` so the route reads explicit `SchedulePeriod` objects instead of rebuilding `prev_date` and coupon boundaries by hand.",
                "- CDS running spreads are often quoted in basis points in task text. Convert them to decimals before accrual, for example `150 bp -> 0.015`.",
                "- After that normalization step, use only the local `spread` variable in the premium leg. Do not read raw `spec.spread` again later in the body.",
                "- Use `market_state.credit_curve.survival_probability(t)` directly; do not route credit-default pricing through Black76 call/put primitives.",
                "- Do not reinterpret the request as nth-to-default or basket credit, and do not import copula helpers for a single-name CDS.",
                "- Use `spec.start_date` as the time origin for schedule year fractions so the analytical and Monte Carlo CDS legs share the same `t` convention.",
                "- Keep discounting explicit with `market_state.discount.discount(pay_t)` and use that payment-date discount factor for both the premium leg and the interval default mass.",
                "- Keep the premium leg to `spread * accrual * df * survival` only. Do not add an accrued-on-default premium adjustment like `0.5 * spread * accrual * df * (prev_survival - survival)`.",
                "- Do not average adjacent discount factors, trapezoid the protection leg, or introduce midpoint expressions like `0.5 * (prev_discount + discount)`.",
                "- A good shape is: initialize `premium_leg = 0.0`, `protection_leg = 0.0`, build `periods = build_period_schedule(...).periods`, loop over the periods, update the running survival probability, then `return protection_leg - premium_leg`.",
            ])

    if instrument_type == "callable_bond" and method == "rate_tree":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- Resolve settlement and maturity, then call `resolve_short_rate_lattice_inputs(...)`; this public resolver owns curve, volatility, calibrated parameter, and step binding.",
            "- Select the resolved model from `MODEL_REGISTRY` and compose `BINOMIAL_1F_TOPOLOGY`, `UNIFORM_ADDITIVE_MESH`, `TERM_STRUCTURE_TARGET(market_state.discount)`, and `build_lattice(...)`.",
            "- Build one timeline with `build_embedded_fixed_income_event_timeline(...)` and pass that same object to `compile_embedded_fixed_income_lattice_contract_spec(..., expected_control_style=\"issuer_min\", dt=lattice.dt, n_steps=lattice.n_steps)`.",
            "- Roll back with `price_on_lattice(...)`, compute the straight-bond reference with `present_value_fixed_coupon_bond(...)`, and enforce the callable holder-value upper bound with `min(tree_price, straight_price)`.",
            "- `price_callable_bond_tree` is only a compatibility reference. Do not call it from new generated routes.",
        ])

    if instrument_type == "puttable_bond" and method == "rate_tree":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- Use the same public short-rate resolver and generic lattice composition as callable bonds; do not delegate to the callable-bond compatibility wrapper.",
            "- Build one embedded event timeline and compile it with `expected_control_style=\"holder_max\"` so a callable-style issuer-min objective fails closed.",
            "- Roll back with `price_on_lattice(...)`, value the straight bond with `present_value_fixed_coupon_bond(...)`, and enforce the puttable holder-value lower bound with `max(tree_price, straight_price)`.",
            "- Keep the typed `put_dates` and `put_price` on the spec; the shared event compiler owns schedule mapping and quoted-price conversion.",
        ])

    if instrument_type == "bermudan_swaption" and method == "rate_tree":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- Normalize live exercise dates with `normalize_explicit_dates(...)`, build the underlying fixed-leg schedule with `build_payment_timeline(...)`, and map measured dates with `year_fraction(...)` plus `lattice_step_from_time(...)`.",
            "- Resolve curve, volatility, Hull-White/BDT parameters, horizon, and step controls once with `resolve_bermudan_swaption_tree_inputs(...)`.",
            "- Compose `BINOMIAL_1F_TOPOLOGY`, `UNIFORM_ADDITIVE_MESH`, `TERM_STRUCTURE_TARGET(market_state.discount)`, and generic `build_lattice(...)`; pass `resolved.mean_reversion` as `a=`.",
            "- Represent fixed coupons and principal with `LatticeLinearClaimSpec` and `LatticeContractSpec`, then call `value_on_lattice(..., observation_steps=exercise_steps)`.",
            "- Form payer/receiver swap values from each observation's `continuation_values`; using post-cashflow values would double count any exercise-time coupon.",
            "- Attach `LatticeControlSpec(objective=\"holder_max\", ...)` to the option `LatticeContractSpec` and finish with `price_on_lattice(...)`.",
            "- Product-level Bermudan pricing wrappers and contract compilers are compatibility/reference APIs, not generated construction authority.",
        ])

    if instrument_type == "bermudan_swaption" and method == "analytical":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- For the analytical comparator lane, import `normalize_explicit_dates` from `trellis.core.date_utils` plus `resolve_swaption_black76_inputs` and `price_swaption_black76_raw` from `trellis.models.rate_style_swaption`.",
            "- Interpret `black76_european_lower_bound` as the European swaption exercisable only on the final Bermudan date.",
            "- Normalize the exercise schedule, keep dates strictly after `market_state.settlement` and before `self._spec.swap_end`, return zero when none remain, then resolve once with `expiry_date=valid_exercise_dates[-1]` and pass the typed result to the raw kernel.",
            "- Do not sum or maximize one European Black76 price per exercise date. Do not rebuild co-terminal swap schedule loops, annuity extraction, or forward-swap-rate assembly inline, and do not use the product-level lower-bound helper as construction authority.",
        ])

    if instrument_type == "swaption" and method == "analytical":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- For European rate-style swaptions, import `resolve_swaption_black76_inputs` and `price_swaption_black76_raw` from `trellis.models.rate_style_swaption`.",
            "- Resolve once with `resolved = resolve_swaption_black76_inputs(market_state, self._spec, ...)`, then return `price_swaption_black76_raw(resolved)`. The resolver owns the market and schedule binding; the raw kernel owns only the Black76 formula and scaling.",
            "- When the request supplies explicit Hull-White comparison parameters, pass `mean_reversion=` and `sigma=` to `resolve_swaption_black76_inputs(...)` so the resolved inputs carry a Hull-White-implied Black vol instead of an unrelated market surface quote.",
            "- Do not rebuild annuity, forward-swap-rate, expiry year-fraction, payment-count loops, or swaption-vol normalization inline. Do not use the product-level `price_swaption_black76(...)` compatibility wrapper as generated construction authority.",
        ])

    if instrument_type == "swaption" and method == "rate_tree":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- For single-exercise European rate-style swaptions, require `spec.swap_start == spec.expiry_date` and construct a one-exercise `BermudanSwaptionTreeSpec` directly.",
            "- Apply `resolve_swaption_curve_basis_spread(...)`, then bind schedule, Hull-White/BDT parameters, horizon, and tree steps with `resolve_bermudan_swaption_tree_inputs(...)`.",
            "- Compose `BINOMIAL_1F_TOPOLOGY`, `UNIFORM_ADDITIVE_MESH`, `TERM_STRUCTURE_TARGET(market_state.discount)`, and generic `build_lattice(...)`; preserve explicit comparison parameters and conventions.",
            "- Compile with `compile_bermudan_swaption_contract_spec(...)` and evaluate with generic `price_on_lattice(...)`.",
            "- `price_swaption_tree(...)` and `build_swaption_tree_spec(...)` are compatibility/reference APIs, not generated construction authority.",
            "- Keep cap/floor-style period loops separate. This route is for a single-exercise European swaption comparison target, not for caplet or floorlet strips.",
        ])

    if instrument_type == "swaption" and method == "monte_carlo":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- For European rate-style swaptions, compose the public market, schedule, process, event, problem, and estimator primitives directly. Use `resolve_swaption_black76_inputs(...)` for the typed European expiry basis and `build_payment_timeline(...)` from the explicit `swap_start` to `swap_end`.",
            "- Bind `resolve_hull_white_monte_carlo_process_inputs(...)`, then build the settlement with `build_discounted_swap_pv_payload(...)` and the path state with `build_short_rate_discount_reducer(...)`.",
            "- Declare `EventAwareMonteCarloEvent` values inside `EventAwareMonteCarloProblemSpec`, compile with `build_event_aware_monte_carlo_problem(...)`, and evaluate with `price_event_aware_monte_carlo(...)`.",
            "- Preserve `day_count`, `swap_frequency`, `rate_index`, explicit `swap_start`, path/step/seed controls, and any explicit Hull-White comparison parameters carried by the task contract.",
            "- Do not hardcode `sigma = 0.01` unless the semantic comparison contract explicitly supplies it, and do not synthesize a GBM equity path. Hull-White process parameters must flow through the bounded market/model resolver.",
            "- `price_swaption_monte_carlo(...)` and `resolve_swaption_monte_carlo_problem(...)` remain compatibility/reference APIs. Do not use either as generated construction authority.",
        ])

    if instrument_type == "zcb_option":
        lines.append("## Family Route Guidance")
        if method == "rate_tree":
            lines.extend([
                "- For zero-coupon bond option tree routes, prefer `price_zcb_option_tree(market_state, self._spec, model=\"ho_lee\"|\"hull_white\")` from `trellis.models.zcb_option_tree`.",
                "- Map `implementation_target=ho_lee_tree` to `model=\"ho_lee\"` and `implementation_target=hull_white_tree` to `model=\"hull_white\"`.",
                "- If you must build the tree directly, import `build_generic_lattice` from `trellis.models.trees.lattice` and `MODEL_REGISTRY` from `trellis.models.trees.models`, then call `build_generic_lattice(MODEL_REGISTRY[...], r0=..., sigma=..., a=..., T=..., n_steps=..., discount_curve=market_state.discount)`.",
                "- Do not call `build_rate_lattice(...)` with invented keyword forms such as `market_state=...`, `maturity=...`, or `steps=...`.",
                "- Build the tree to `spec.bond_maturity_date`, not just to expiry, because the route needs nested backward induction for the underlying bond price at option expiry.",
            ])
        elif method == "analytical":
            lines.extend([
                "- For Jamshidian analytical routes, prefer `price_zcb_option_jamshidian(market_state, self._spec, mean_reversion=0.1)` from `trellis.models.zcb_option`.",
                "- If you need the resolved-input lane under that wrapper, use `resolve_zcb_option_hw_inputs(...)` from `trellis.models.zcb_option`, then pass `resolved.jamshidian` into `zcb_option_hw_raw(...)` from `trellis.models.analytical.jamshidian`.",
                "- Treat `ResolvedJamshidianInputs` as the traced contract for expiry discounting, bond discounting, normalized strike, expiry, bond maturity, volatility, and mean reversion.",
                "- Do not degrade this route to generic Black76 on a forward bond price when the task explicitly asks for Jamshidian / Hull-White.",
                "- Normalize strike quotes to unit face before the closed-form kernel. Treat `63` on `100` face as `0.63`.",
                "- Use `spec.expiry_date` and `spec.bond_maturity_date`; validate that the bond maturity is strictly after expiry.",
            ])

    if instrument_type == "european_option" and method == "analytical":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- For plain European call/put comparators, keep the adapter minimal: compute `T`, `df`, `sigma`, `forward`, then call `black76_call` or `black76_put` directly.",
            "- Use `forward = spec.spot / max(df, 1e-12)` for the equity-spot to forward bridge before the Black-style kernel.",
            "- Do not use `terminal_vanilla_from_basis`, asset-or-nothing helpers, or cash-or-nothing helpers unless the request explicitly asks for digital/binary decomposition.",
            "- Return a complete module with the spec class and payoff class; do not emit only an `evaluate()` fragment or a partial class body.",
        ])

    if instrument_type == "nth_to_default":
        lines.append("## Family Route Guidance")
        if method in {"monte_carlo", "qmc", "copula"}:
            lines.extend([
                "- For nth-to-default and first-to-default basket-credit routes, keep the multi-name contract explicit: reference entities, default order, and correlation model stay visible in the code.",
                "- Use approved copula helpers such as `GaussianCopula` or `FactorCopula` only for multi-name basket credit. Do not collapse the request into a single-name CDS premium/protection loop.",
                "- Treat credit-curve access and default-correlation access as separate concerns: marginal default probabilities come from `market_state.credit_curve`, while dependence comes from the approved copula layer.",
                "- Do not reuse single-name CDS route notes or helpers unless the request explicitly reduces to one reference entity and first-default order is no longer part of the contract.",
            ])

    if instrument_type == "barrier_option":
        lines.append("## Family Route Guidance")
        if method == "pde_solver":
            lines.extend([
                "- For barrier PDE routes, import `Grid` from `trellis.models.pde.grid`, `BlackScholesOperator` from `trellis.models.pde.operator`, and `theta_method_1d` from `trellis.models.pde.theta_method`.",
                "- Use the real `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` constructor; do not invent helper names like `log_grid_pde` or `uniform_grid_pde`.",
                "- Set `lower_bc_fn` / `upper_bc_fn` callables explicitly; do not encode boundary handling with bare tokens such as `absorbing` or `dirichlet`.",
                "- For discontinuous terminal payoffs, pass `rannacher_timesteps` to `theta_method_1d` so the solver can smooth the first backward steps.",
                "- If the barrier sits close to spot, use a non-uniform grid with more resolution near the barrier and keep `x_min` / `x_max` as actual spot bounds.",
            ])
        elif method == "monte_carlo":
            lines.extend([
                "- For barrier Monte Carlo, use the path-state barrier helpers and monitor the barrier at the exact observation steps; do not interpolate between steps.",
                "- Do not invent a bare `absorbing` route token. The barrier logic belongs in the path payoff / monitoring helper, not in a made-up symbol name.",
            ])
        elif method == "analytical":
            lines.extend([
                "- For analytical barrier pricing, import the Reiner-Rubinstein helpers from `trellis.models.analytical.barrier` and delegate to them instead of re-deriving the formula inline.",
            ])

    if instrument_type == "quanto_option":
        lines.append("## Family Route Guidance")
        lines.extend([
            "- For quanto routes, import and call `resolve_quanto_inputs` from `trellis.models.resolution.quanto`.",
            "- Do not reimplement spot / FX / curve / correlation lookup inside `evaluate()`.",
        ])
        if method == "analytical":
            lines.extend([
                "- Compose analytical quanto pricing from `quanto_adjusted_forward`, `black76_call` / `black76_put`, and `discounted_value`; do not delegate to a quanto product-pricing wrapper.",
                "- Use `normalized_option_type` and `terminal_intrinsic` so option direction and expiry behavior remain explicit.",
                "- If you need terminal binary-style subproblems, use `cash_or_nothing_intrinsic` or `asset_or_nothing_intrinsic` from `trellis.models.analytical.support` instead of open-coding indicator branches.",
                "- For analytical quanto, compute the quanto-adjusted forward as `spot * foreign_df / domestic_df * exp(-corr * sigma_underlier * sigma_fx * T)`.",
                "- Then price with `black76_call` or `black76_put` on that forward and multiply by `domestic_df * notional`.",
                "- `trellis.models.quanto_option` and `trellis.models.analytical.quanto` retain compatibility/reference wrappers, but they are not generated-route construction authority.",
            ])
        elif method in {"monte_carlo", "qmc"}:
            lines.extend([
                "- Compose the simulation lane from `resolve_quanto_inputs`, `implied_zero_rate`, `CorrelatedGBM`, `MonteCarloEngine`, `terminal_value_payoff`, and `terminal_intrinsic`; do not delegate to a quanto product-pricing wrapper.",
                "- Use the real Trellis correlated-GBM signature: `CorrelatedGBM(mu=[...], sigma=[...], corr=[[1.0, rho], [rho, 1.0]])`.",
                "- Seed the joint process with `np.array([resolved.spot, resolved.fx_spot], dtype=float)` when calling `engine.simulate(...)` or `engine.price(...)`.",
                "- Do not seed a multi-asset correlated GBM with only `resolved.spot` or other scalar initial states.",
                "- Pass a terminal-only payoff that reads the underlier factor at `terminal[..., 0]`, then discount with the resolved domestic discount factor.",
            ])
            if method == "qmc":
                lines.extend([
                    "- Generate seeded two-factor shocks with `sobol_normals(n_paths, n_steps, n_factors=2, seed=seed)` and pass them through `MonteCarloEngine.price(..., shocks=shocks)`.",
                    "- Round the QMC path count up to a power of two so the Sobol balance property is preserved.",
                ])

    return "\n".join(lines)


def _render_assembly_context(
    spec_schema,
    *,
    pricing_plan=None,
    generation_plan=None,
) -> str:
    """Render structured assembly tools for thin-adapter generation."""
    if pricing_plan is None and generation_plan is None:
        return ""

    from trellis.agent.assembly_tools import (
        lookup_primitive_route_from_context,
        render_invariant_pack,
        render_thin_adapter_plan,
    )

    lookup = lookup_primitive_route_from_context(
        generation_plan=generation_plan,
        pricing_plan=pricing_plan,
    )
    lookup_lines = [
        "## Backend Lookup (Secondary To Lane Obligations)",
        f"- Method family: `{lookup.method}`",
        f"- Route: `{lookup.route or 'unknown'}`",
    ]
    if generation_plan is not None and getattr(generation_plan, "lane_family", ""):
        lookup_lines.insert(1, f"- Lane family: `{generation_plan.lane_family}`")
        if getattr(generation_plan, "lane_plan_kind", ""):
            lookup_lines.insert(2, f"- Lane plan kind: `{generation_plan.lane_plan_kind}`")
    if lookup.engine_family:
        lookup_lines.append(f"- Engine family: `{lookup.engine_family}`")
    if lookup.primitives:
        lookup_lines.append("- Required primitive symbols:")
        lookup_lines.extend(f"  - `{primitive}`" for primitive in lookup.primitives[:8])
    if lookup.adapters:
        lookup_lines.append("- Adapter obligations:")
        lookup_lines.extend(f"  - `{adapter}`" for adapter in lookup.adapters[:8])

    invariant_text = render_invariant_pack(
        instrument_type=(
            generation_plan.instrument_type if generation_plan is not None else None
        ) or getattr(pricing_plan, "model_to_build", None),
        method=pricing_plan.method if pricing_plan is not None else lookup.method,
    )
    thin_adapter_text = render_thin_adapter_plan(
        spec_schema,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
    )
    return "\n".join(lookup_lines) + "\n" + thin_adapter_text + "\n" + invariant_text


def _render_reference_sources(
    reference_sources: dict[str, str],
    *,
    prompt_surface: str,
    pricing_plan=None,
) -> str:
    """Render reference snippets with compact first-pass truncation."""
    if not reference_sources:
        return ""

    blocks: list[str] = []
    items = list(reference_sources.items())
    if prompt_surface == "import_repair":
        return ""
    if prompt_surface == "semantic_repair":
        max_refs = 2
        max_chars = 700
    elif prompt_surface == "compact":
        max_refs = 4
        max_chars = 900
        if pricing_plan is not None:
            required_market_data = set(getattr(pricing_plan, "required_market_data", ()) or ())
            if {"fx_rates", "forward_curve"} <= required_market_data:
                max_refs = 2
                max_chars = 450
    else:
        max_refs = len(items)
        max_chars = None

    for name, source in items[:max_refs]:
        excerpt = _truncate_reference_source(source, max_chars=max_chars)
        blocks.append(f"\n### {name}\n```python\n{excerpt}\n```\n")

    omitted = len(items) - min(len(items), max_refs)
    if omitted > 0:
        blocks.append(f"\n[omitted {omitted} additional reference modules]\n")
    return "".join(blocks)


def _truncate_reference_source(source: str, *, max_chars: int | None) -> str:
    """Trim reference source blocks for compact prompting."""
    if max_chars is None or len(source) <= max_chars:
        return source
    return source[:max_chars].rstrip() + "\n# [truncated reference]"
