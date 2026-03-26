"""System prompt templates for the Trellis agent."""

from __future__ import annotations

from pathlib import Path


def load_requirements() -> str:
    """Load the agent's constitution from requirements.md."""
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

## Library Constitution
{requirements}

## Capabilities
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
2. Inspect the library, exports, tests, and lessons to see what already exists.
3. Resolve every `trellis.*` import against the live package before writing code.
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

    method_guidance = ""
    if knowledge_context:
        # New path: unified knowledge from KnowledgeStore
        method_guidance = knowledge_context
        # Still add method selection info if available
        if pricing_plan and pricing_plan.method_modules:
            from trellis.agent.knowledge.methods import normalize_method
            method_name = normalize_method(pricing_plan.method)
            method_guidance = (
                f"\n## Pricing Method (selected by the quant agent — you MUST use this)\n"
                f"Method: **{method_name}**\n"
                f"Reasoning: {pricing_plan.reasoning}\n"
                f"Modules to import and use:\n"
                + "\n".join(f"- `{m}`" for m in pricing_plan.method_modules)
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
## Conventions
- evaluate() returns a FLOAT — the present value (PV) of the instrument
- You MUST handle all discounting internally — use `market_state.discount.discount(t)`
- For forward rates: `market_state.forecast_forward_curve(self._spec.rate_index)`
- For vol: `market_state.vol_surface.black_vol(T, strike)`
- For discount factors: `market_state.discount.discount(t)`
- Schedule generation: `generate_schedule(start, end, freq)` returns list[date]
- Year fractions: `year_fraction(date1, date2, day_count)`
- Black76: `black76_call(F, K, sigma, T)`, `black76_put(F, K, sigma, T)` — undiscounted
- For trees: `from trellis.models.trees import BinomialTree, backward_induction`
- For MC: `from trellis.models.monte_carlo import MonteCarloEngine`
- For QMC accelerators: `from trellis.models.qmc import sobol_normals, brownian_bridge`
- For copulas: `from trellis.models.copulas import GaussianCopula, FactorCopula`
- Do not invent MonteCarloEngine method strings. Valid `method=` values are `euler`, `milstein`, and `exact`.
- {early_exercise_guidance}
- If you use `LaguerreBasis`, import it from `trellis.models.monte_carlo.schemes`, not from `trellis.models.monte_carlo.lsm`.
- For FFT/COS pricing, characteristic functions must accept vector `u` and use array-safe numerics such as `numpy`, not scalar `math`/`cmath`.
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
