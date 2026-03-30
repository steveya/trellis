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
- inspect_api_map: Inspect the compact API navigation map before broad tree exploration.
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
        requirements, experience).
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
            modules_block = (
                "\n".join(f"- `{m}`" for m in pricing_plan.method_modules)
                if pricing_plan.method_modules
                else "No special modules needed — use standard discounting."
            )
            method_guidance = (
                f"\n## Pricing Method (selected by the quant agent — you MUST use this)\n"
                f"Method: **{method_name}**\n"
                f"Reasoning: {pricing_plan.reasoning}\n"
                f"{selection_notes}\n"
                f"Modules to import and use:\n"
                + modules_block
                + "\n\nYou MUST import and use these modules in your implementation.\n\n"
                + method_guidance
            )
    elif pricing_plan:
        # Compatibility path: build unified shared knowledge on demand.
        from trellis.agent.knowledge.methods import normalize_method
        method_name = normalize_method(pricing_plan.method)
        shared_knowledge_text = ""
        try:
            from trellis.agent.knowledge import (
                format_knowledge_for_prompt,
                retrieve_for_task,
            )

            shared_knowledge = retrieve_for_task(
                method=method_name,
                instrument=pricing_plan.model_to_build,
            )
            shared_knowledge_text = format_knowledge_for_prompt(shared_knowledge, compact=True)
        except Exception:
            shared_knowledge_text = ""

        method_guidance = f"""
## Pricing Method (selected by the quant agent — you MUST use this)
Method: **{method_name}**
Reasoning: {pricing_plan.reasoning}
{selection_notes}
Modules to import and use:
{chr(10).join(f'- `{m}`' for m in pricing_plan.method_modules)}

{"You MUST import and use these modules in your implementation." if pricing_plan.method_modules else "No special modules needed — use standard discounting."}
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
Write ONLY the body of the `evaluate()` method. The signature is already defined:

    def evaluate(self, market_state: MarketState) -> float:

## Spec fields available via self._spec
{field_descs}
{method_guidance}
{generation_context}
{assembly_context}
{family_route_guidance}
## Conventions
- evaluate() returns a FLOAT — the present value (PV) of the instrument
- You MUST handle all discounting internally — use `market_state.discount.discount(t)`
- For forward rates: `market_state.forecast_forward_curve(self._spec.rate_index)`
- For vol: `market_state.vol_surface.black_vol(T, strike)`
- For discount factors: `market_state.discount.discount(t)`
- Schedule generation: `generate_schedule(start, end, freq)` returns list[date]
- Year fractions: `year_fraction(date1, date2, day_count)`
- Never use wall-clock dates such as `date.today()` or `datetime.now()` inside `evaluate()`; derive valuation time from `market_state` or shared resolver outputs.
- Black76: `black76_call(F, K, sigma, T)`, `black76_put(F, K, sigma, T)` — undiscounted
- Black76 digital: `black76_cash_or_nothing_call(F, K, sigma, T)`, `black76_cash_or_nothing_put(F, K, sigma, T)` — undiscounted cash-or-nothing digitals
- For CDS / nth-to-default: use `market_state.credit_curve.survival_probability(t)` and `market_state.credit_curve.hazard_rate(t)` on an explicit payment/default schedule; do not route credit-default pricing through Black76 call/put primitives.
- For equity trees: `from trellis.models.trees.binomial import BinomialTree` and `from trellis.models.trees.backward_induction import backward_induction`
- For rate lattices: `from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction`
- For MC: `from trellis.models.monte_carlo import MonteCarloEngine`
- For QMC accelerators: `from trellis.models.qmc import sobol_normals, brownian_bridge`
- For copulas: `from trellis.models.copulas import GaussianCopula, FactorCopula`
- Do not invent MonteCarloEngine method strings. Valid `method=` values are `euler`, `milstein`, and `exact`.
- {early_exercise_guidance}
- If you use `LaguerreBasis`, import it from `trellis.models.monte_carlo.schemes`, not from `trellis.models.monte_carlo.lsm`.
- For FFT/COS pricing, characteristic functions must accept vector `u` and use array-safe numerics such as `numpy`, not scalar `math`/`cmath`.
- Black76 basis: `black76_asset_or_nothing_call(F, K, sigma, T)`, `black76_asset_or_nothing_put(F, K, sigma, T)`, `black76_cash_or_nothing_call(F, K, sigma, T)`, `black76_cash_or_nothing_put(F, K, sigma, T)` — exact terminal basis claims.
- For terminal vanilla payoffs, prefer exact basis assembly via `terminal_vanilla_from_basis(...)` from `trellis.models.analytical`.
- For FX vanilla options, map spot FX and domestic/foreign discount factors to a forward, then assemble the terminal payoff from the same Black76 basis claims via `terminal_vanilla_from_basis(...)`; keep Garman-Kohlhagen as the model identity and compatibility surface, not the place where payoff algebra is duplicated.
- For cash-or-nothing digital options, use the Black76 digital helpers directly; do not approximate them with vanilla call/put prices or divide by spot.
- You MUST use only real, approved `trellis.*` imports from the structured generation plan and import registry
- Treat the selected primitive route in the structured generation plan as the backbone of the implementation.
- Reuse the listed required primitives directly and write only the thinnest adapter/orchestration layer around them.
- Do not replace selected primitives with bespoke numerical kernels or alternative Trellis routes unless the plan explicitly permits it.
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
                "- For American/Bermudan equity tree routes, import `BinomialTree` from `trellis.models.trees.binomial` and `backward_induction` from `trellis.models.trees.backward_induction`.",
                "- Build the tree with `BinomialTree.crr(S0, T, n_steps, r, sigma)`; do not invent a `crr_tree` symbol or helper.",
                "- Call `backward_induction(..., exercise_type=\"american\")` or `backward_induction(..., exercise_type=\"bermudan\")` with a quoted string literal; do not invent an `AMERICAN` constant.",
                "- Keep the early-exercise logic on the tree side. Do not route American exercise through `method=\"lsm\"` or an `lsm_mc` alias.",
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

    if instrument_type in {"cds", "nth_to_default"}:
        lines.append("## Family Route Guidance")
        if method in {"monte_carlo", "qmc"}:
            lines.extend([
                "- For CDS / nth-to-default Monte Carlo routes, keep the premium leg and protection leg on an explicit payment/default schedule.",
                "- Start the body with `from trellis.core.differentiable import get_numpy` and `np = get_numpy()` so the route uses the approved array backend.",
                "- Use `market_state.credit_curve.hazard_rate(t)` or `market_state.credit_curve.survival_probability(t)` directly on the schedule; do not hide the credit curve behind an alias.",
                "- Use `market_state.discount.discount(t)` directly for each payment-date discount factor.",
                "- Keep the body as a single explicit schedule loop plus a final `premium_leg` / `protection_leg` PV aggregation; do not invent helper names or route credit-default pricing through Black76.",
                "- A good shape is: initialize `premium_leg = 0.0`, `protection_leg = 0.0`, loop over the payment dates, update both legs, then `return protection_leg - premium_leg`.",
            ])
        elif method == "analytical":
            lines.extend([
                "- For CDS / nth-to-default analytical routes, build the premium leg and protection leg directly from the credit curve on the explicit payment schedule.",
                "- Use `market_state.credit_curve.survival_probability(t)` directly; do not route credit-default pricing through Black76 call/put primitives.",
                "- Keep discounting explicit with `market_state.discount.discount(t)` and return protection leg minus premium leg.",
                "- A good shape is: initialize `premium_leg = 0.0`, `protection_leg = 0.0`, loop over the payment dates, update the running survival probability, then `return protection_leg - premium_leg`.",
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
                "- For analytical quanto, import and call `price_quanto_option_analytical` from `trellis.models.analytical.quanto`.",
                "- Prefer support helpers from `trellis.models.analytical.support`, especially `normalized_option_type`, `terminal_intrinsic`, `quanto_adjusted_forward`, and `discounted_value`.",
                "- Do not reimplement the quanto-adjusted analytical pricing body inside `evaluate()`.",
                "- If you need terminal binary-style subproblems, use `cash_or_nothing_intrinsic` or `asset_or_nothing_intrinsic` from `trellis.models.analytical.support` instead of open-coding indicator branches.",
                "- For analytical quanto, compute the quanto-adjusted forward as `spot * foreign_df / domestic_df * exp(-corr * sigma_underlier * sigma_fx * T)`.",
                "- Then price with `black76_call` or `black76_put` on that forward and multiply by `domestic_df * notional`.",
            ])
        elif method == "monte_carlo":
            lines.extend([
                "- For Monte Carlo quanto, import and call `price_quanto_option_monte_carlo` from `trellis.models.monte_carlo.quanto`.",
                "- Do not reimplement process / engine / payoff / discount wiring inside `evaluate()`.",
                "- The shared Monte Carlo helper already owns the joint underlier/FX process, engine controls, terminal payoff mapping, and domestic discounting policy.",
                "- For Monte Carlo quanto, reuse `resolve_quanto_inputs`, build a `CorrelatedGBM`, and simulate the underlier/FX pair jointly only when you are reading or validating the helper surface.",
                "- Use the real Trellis correlated-GBM signature: `CorrelatedGBM(mu=[...], sigma=[...], corr=[[1.0, rho], [rho, 1.0]])`.",
                "- Seed the joint process with `np.array([resolved.spot, resolved.fx_spot], dtype=float)` when calling `engine.simulate(...)` or `engine.price(...)`.",
                "- Do not seed a multi-asset correlated GBM with only `resolved.spot` or other scalar initial states.",
                "- Discount the terminal payoff with the domestic discount factor returned by the shared resolver.",
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
        "## Primitive Lookup",
        f"- Method family: `{lookup.method}`",
        f"- Route: `{lookup.route or 'unknown'}`",
    ]
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
