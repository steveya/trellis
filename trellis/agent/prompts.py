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
- write_module: Create or update modules (the agent writes code into the library).
- run_tests: Execute the test suite to verify correctness.
- fetch_market_data: Pull Treasury yields from FRED or Treasury.gov.
- execute_pricing: Price an instrument using a yield curve.

## Workflow
1. Understand the user's request.
2. Inspect the library to see what already exists.
3. If something is missing, plan what to build, write the code, and test it.
4. Fetch any needed market data.
5. Execute the pricing and return results.

Always test your code before using it for pricing. Max 3 retry attempts on test failures.
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

{capability_summary()}

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
) -> str:
    """Step 2 prompt: implement only the evaluate() method body.

    Parameters
    ----------
    pricing_plan : PricingPlan or None
        If provided, includes the quant agent's method recommendation.
    """
    field_descs = "\n".join(
        f"- `self._spec.{f.name}` ({f.type}): {f.description}"
        for f in spec_schema.fields
    )

    refs = ""
    for name, source in reference_sources.items():
        refs += f"\n### {name}\n```python\n{source}\n```\n"

    method_guidance = ""
    if pricing_plan:
        from trellis.agent.cookbooks import get_cookbook
        cookbook = get_cookbook(pricing_plan.method)

        method_guidance = f"""
## Pricing Method (selected by the quant agent — you MUST use this)
Method: **{pricing_plan.method}**
Reasoning: {pricing_plan.reasoning}
Modules to import and use:
{chr(10).join(f'- `{m}`' for m in pricing_plan.method_modules)}

{"You MUST import and use these modules in your implementation." if pricing_plan.method_modules else "No special modules needed — use standard discounting."}

{cookbook}

IMPORTANT: Follow the cookbook pattern above. Adapt it for this specific instrument
but keep the same structure: imports, market data access, method invocation, return type.
"""

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
- For copulas: `from trellis.models.copulas import GaussianCopula, FactorCopula`

## Reference implementations
{refs}

## Output
Return the COMPLETE Python module — copy the skeleton exactly, but replace the
`raise NotImplementedError(...)` line with the actual implementation of `evaluate()`.
Do NOT change imports, class names, spec fields, or the requirements property.
Only implement the evaluate() method body. You may add imports at the top.
No markdown fences, no explanation — just the Python code."""
