"""Top-level agent executor: plan → build → fetch → price → return."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from trellis.agent.introspection import get_package_tree, read_module_source
from trellis.agent.prompts import system_prompt
from trellis.agent.tools import TOOLS
from trellis.agent.builder import write_module, run_tests


# Sentinel in the skeleton that gets replaced by the LLM-generated body.
EVALUATE_SENTINEL = '        raise NotImplementedError("evaluate not yet implemented")'


def _handle_tool_call(name: str, input_data: dict) -> str:
    """Dispatch a tool call from the LLM agent."""
    if name == "inspect_library":
        tree = get_package_tree()
        return json.dumps(tree, indent=2, default=str)

    elif name == "read_module":
        try:
            src = read_module_source(input_data["module_path"])
            return src
        except Exception as e:
            return f"Error reading module: {e}"

    elif name == "write_module":
        path = write_module(input_data["file_path"], input_data["content"])
        return f"Module written to {path}"

    elif name == "run_tests":
        result = run_tests(input_data.get("test_path"))
        return json.dumps(result, indent=2)

    elif name == "fetch_market_data":
        source = input_data.get("source", "fred")
        as_of_str = input_data.get("as_of")
        as_of = datetime.strptime(as_of_str, "%Y-%m-%d").date() if as_of_str else None

        if source == "fred":
            from trellis.data.fred import FredDataProvider
            data = FredDataProvider().fetch_yields(as_of)
        else:
            from trellis.data.treasury_gov import TreasuryGovDataProvider
            data = TreasuryGovDataProvider().fetch_yields(as_of)
        return json.dumps(data, indent=2)

    elif name == "execute_pricing":
        from trellis.instruments.bond import Bond
        from trellis.curves.yield_curve import YieldCurve
        from trellis.engine.pricer import price_instrument

        params = input_data["params"]
        curve_data = {float(k): float(v) for k, v in input_data["curve_data"].items()}
        curve = YieldCurve.from_treasury_yields(curve_data)

        instrument_type = input_data.get("instrument_type", "Bond")
        if instrument_type == "Bond":
            if "maturity_date" in params and isinstance(params["maturity_date"], str):
                params["maturity_date"] = datetime.strptime(params["maturity_date"], "%Y-%m-%d").date()
            bond = Bond(**params)
        else:
            return f"Unknown instrument type: {instrument_type}"

        result = price_instrument(bond, curve)
        return json.dumps({
            "clean_price": result.clean_price,
            "dirty_price": result.dirty_price,
            "accrued_interest": result.accrued_interest,
            "greeks": result.greeks,
        }, indent=2, default=str)

    return f"Unknown tool: {name}"


def execute(query: str, max_turns: int = 10, model: str = "claude-sonnet-4-6") -> dict:
    """Run the full agent loop for a natural-language pricing request."""
    from trellis.agent.config import get_anthropic_client

    client = get_anthropic_client()
    messages = [{"role": "user", "content": query}]
    sys_prompt = system_prompt()

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=sys_prompt,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [b for b in assistant_content if b.type == "tool_use"]
        if not tool_uses:
            text_parts = [b.text for b in assistant_content if hasattr(b, "text")]
            return {"response": "\n".join(text_parts), "turns": turn + 1}

        tool_results = []
        for tool_use in tool_uses:
            result = _handle_tool_call(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return {"response": "Max turns reached", "turns": max_turns}


# ---------------------------------------------------------------------------
# Structured payoff builder (two-step pipeline)
# ---------------------------------------------------------------------------

def build_payoff(
    payoff_description: str,
    requirements: set[str] | None = None,
    model: str | None = None,
    max_retries: int = 3,
    force_rebuild: bool = False,
    validation: str = "standard",
    market_state=None,
    instrument_type: str | None = None,
) -> type:
    """Build a Payoff class via the multi-agent pipeline.

    Pipeline:
    1. **Quant agent** selects the pricing method and data requirements
    2. **Data check** verifies required market data is available
    3. **Planner** determines spec schema and module path
    4. **Builder agent** generates the code using the prescribed method
    5. **Critic agent** reviews the code
    6. **Arbiter** validates with invariants
    """
    from trellis.agent.planner import plan_build
    from trellis.agent.quant import select_pricing_method, check_data_availability
    from trellis.agent.builder import dynamic_import, ensure_agent_package
    from trellis.agent.config import get_default_model
    from trellis.core.market_state import MissingCapabilityError

    model = model or get_default_model()

    # Step 1: Quant agent selects method + data requirements
    pricing_plan = select_pricing_method(
        payoff_description,
        instrument_type=instrument_type,
        model=model,
    )

    # Step 2: Check market data availability (early — before writing code)
    if market_state is not None:
        data_errors = check_data_availability(pricing_plan, market_state)
        if data_errors:
            raise MissingCapabilityError(
                pricing_plan.required_market_data - market_state.available_capabilities,
                market_state.available_capabilities,
                details=data_errors,
            )

    # Use quant agent's data requirements if caller didn't specify
    if requirements is None:
        requirements = pricing_plan.required_market_data

    # Step 3: Plan (spec schema + module path)
    plan = plan_build(payoff_description, requirements, model=model)

    # Step 3b: Check cache
    if not force_rebuild:
        existing = _try_import_existing(plan)
        if existing is not None:
            return existing

    # Step 4: Design spec
    if plan.spec_schema is not None:
        spec_schema = plan.spec_schema
    else:
        spec_schema = _design_spec(payoff_description, requirements, model)

    # Step 5: Generate skeleton
    skeleton = _generate_skeleton(spec_schema, payoff_description)

    # Step 6-9: Generate code with method guidance, validate, retry
    reference_sources = _gather_references(pricing_plan)
    ensure_agent_package()
    step = plan.steps[0]
    module_name = f"trellis.{step.module_path.replace('/', '.').replace('.py', '')}"

    validation_feedback = ""
    payoff_cls = None
    for attempt in range(max_retries):
        code = _generate_module(
            skeleton, spec_schema, reference_sources, model, 1,
            extra_context=validation_feedback,
            pricing_plan=pricing_plan,
        )

        file_path = write_module(step.module_path, code)
        mod = dynamic_import(file_path, module_name)
        payoff_cls = getattr(mod, spec_schema.class_name)

        if validation == "fast":
            return payoff_cls

        failures = _validate_build(
            payoff_cls, code, payoff_description, spec_schema,
            validation=validation, model=model,
        )

        if not failures:
            return payoff_cls

        validation_feedback = (
            "\n\n## VALIDATION FAILURES (your previous code had these issues):\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\n\nFix ALL of the above issues in your implementation."
        )

    return payoff_cls


def _validate_build(
    payoff_cls,
    code: str,
    description: str,
    spec_schema,
    validation: str = "standard",
    model: str | None = None,
) -> list[str]:
    """Run validation checks on a built payoff. Returns list of failures."""
    from trellis.agent.invariants import check_non_negativity, check_bounded_by_reference
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol
    from trellis.core.payoff import DeterministicCashflowPayoff
    from trellis.instruments.bond import Bond

    settle = date(2024, 11, 15)
    failures = []

    # Try to instantiate the payoff with default test parameters
    try:
        test_payoff = _make_test_payoff(payoff_cls, spec_schema, settle)
    except Exception as e:
        failures.append(f"Cannot instantiate payoff for validation: {e}")
        return failures

    # Basic: non-negativity at a standard MarketState
    ms = MarketState(
        as_of=settle, settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )
    failures.extend(check_non_negativity(test_payoff, ms))

    # Standard: bounding check (callable ≤ straight bond across rates)
    if validation in ("standard", "thorough"):
        def payoff_factory():
            return _make_test_payoff(payoff_cls, spec_schema, settle)

        # Create a reference straight bond
        bond = Bond(
            face=100, coupon=0.05,
            maturity_date=date(2034, 11, 15),
            maturity=10, frequency=2,
        )

        def reference_factory():
            return DeterministicCashflowPayoff(bond)

        def ms_factory(rate=0.05, vol=0.20):
            return MarketState(
                as_of=settle, settlement=settle,
                discount=YieldCurve.flat(rate),
                vol_surface=FlatVol(vol),
            )

        bound_failures = check_bounded_by_reference(
            payoff_factory, reference_factory, ms_factory,
            rate_range=(0.02, 0.05, 0.08),
            relation="<=",
        )
        failures.extend(bound_failures)

    # Standard: run critic
    if validation in ("standard", "thorough"):
        try:
            from trellis.agent.critic import critique
            from trellis.agent.arbiter import run_critic_tests
            concerns = critique(code, description, model=model)
            critic_failures = run_critic_tests(concerns, test_payoff)
            failures.extend(critic_failures)
        except Exception:
            pass  # Critic failure shouldn't block the build

    return failures


def _make_test_payoff(payoff_cls, spec_schema, settle: date):
    """Create a test payoff instance from the spec schema with default values."""
    import sys
    # Get the spec class from the module
    for mod_name, mod in list(sys.modules.items()):
        if mod and hasattr(mod, spec_schema.spec_name):
            spec_cls = getattr(mod, spec_schema.spec_name)
            break
    else:
        raise RuntimeError(f"Cannot find {spec_schema.spec_name} in loaded modules")

    # Build kwargs from field definitions with test defaults
    kwargs = {}
    type_defaults = {
        "float": 100.0,
        "int": 10,
        "str": "test",
        "bool": True,
        "date": date(2034, 11, 15),
        "str | None": None,
        "Frequency": None,  # use dataclass default
        "DayCountConvention": None,  # use dataclass default
    }
    # More specific field-name defaults
    name_defaults = {
        "notional": 100.0,
        "coupon": 0.05,
        "strike": 0.05,
        "expiry_date": date(2025, 11, 15),
        "swap_start": date(2025, 11, 15),
        "swap_end": date(2034, 11, 15),
        "start_date": settle,
        "end_date": date(2034, 11, 15),
        "is_payer": True,
    }

    for field in spec_schema.fields:
        if field.name in name_defaults:
            kwargs[field.name] = name_defaults[field.name]
        elif field.default is not None:
            pass  # let the dataclass default handle it
        elif field.type in type_defaults:
            kwargs[field.name] = type_defaults[field.type]

    try:
        spec = spec_cls(**kwargs)
    except TypeError:
        # If we're missing required fields, try with all defaults
        spec = spec_cls(**{f.name: name_defaults.get(f.name, type_defaults.get(f.type, ""))
                          for f in spec_schema.fields if f.default is None})

    return payoff_cls(spec)


def _design_spec(
    payoff_description: str,
    requirements: set[str],
    model: str,
):
    """LLM call #1: design the spec schema via structured JSON output."""
    from trellis.agent.config import llm_generate_json, ALLOWED_FIELD_TYPES
    from trellis.agent.prompts import spec_design_prompt
    from trellis.agent.planner import SpecSchema, FieldDef

    prompt = spec_design_prompt(payoff_description, requirements)
    data = llm_generate_json(prompt, model=model)

    fields = []
    for f in data["fields"]:
        ftype = f["type"]
        if ftype not in ALLOWED_FIELD_TYPES:
            ftype = "str"  # fallback
        fields.append(FieldDef(
            name=f["name"],
            type=ftype,
            description=f.get("description", ""),
            default=f.get("default"),
        ))

    return SpecSchema(
        class_name=data["class_name"],
        spec_name=data["spec_name"],
        requirements=data["requirements"],
        fields=fields,
    )


def _generate_skeleton(spec_schema, description: str) -> str:
    """Deterministically generate the full module skeleton from the spec schema."""
    required = [f for f in spec_schema.fields if f.default is None]
    optional = [f for f in spec_schema.fields if f.default is not None]
    field_lines = []
    for f in required + optional:
        if f.default is None:
            field_lines.append(f"    {f.name}: {f.type}")
        else:
            field_lines.append(f"    {f.name}: {f.type} = {f.default}")
    fields_block = "\n".join(field_lines)

    requirements_str = ", ".join(f'"{r}"' for r in sorted(spec_schema.requirements))

    return f'''"""Agent-generated payoff: {description}."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class {spec_schema.spec_name}:
    """Specification for {description}."""
{fields_block}


class {spec_schema.class_name}:
    """{description}."""

    def __init__(self, spec: {spec_schema.spec_name}):
        self._spec = spec

    @property
    def spec(self) -> {spec_schema.spec_name}:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {{{requirements_str}}}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
{EVALUATE_SENTINEL}
'''


def _generate_module(
    skeleton: str,
    spec_schema,
    reference_sources: dict[str, str],
    model: str,
    max_retries: int,
    extra_context: str = "",
    pricing_plan=None,
) -> str:
    """LLM call #2: generate the complete module with evaluate() filled in."""
    from trellis.agent.config import llm_generate
    from trellis.agent.prompts import evaluate_prompt

    prompt = evaluate_prompt(skeleton, spec_schema, reference_sources,
                             pricing_plan=pricing_plan)
    if extra_context:
        prompt += extra_context

    last_error = ""
    for attempt in range(max_retries):
        if attempt > 0:
            full_prompt = prompt + f"\n\n## Previous attempt had error:\n{last_error}\nFix the code."
        else:
            full_prompt = prompt

        code = llm_generate(full_prompt, model=model)

        # Strip markdown fences
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        code = code.expandtabs(4)

        try:
            compile(code, "<agent>", "exec")
            return code
        except SyntaxError as e:
            last_error = str(e)
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Agent failed to produce valid module after {max_retries} attempts"
                ) from e

    raise RuntimeError("Unreachable")


def _normalize_indent(code: str, target: int = 8) -> str:
    """Re-indent code so the base level is *target* spaces, preserving relative indent.

    Uses textwrap.dedent to strip common leading whitespace first,
    then prepends *target* spaces to each line.
    """
    import textwrap
    dedented = textwrap.dedent(code)
    lines = dedented.split("\n")
    new_lines = []
    for line in lines:
        if line.strip():
            new_lines.append(" " * target + line)
        else:
            new_lines.append("")
    return "\n".join(new_lines)


def _combine_skeleton_and_body(skeleton: str, evaluate_body: str) -> str:
    """Replace the evaluate sentinel in the skeleton with the generated body."""
    if EVALUATE_SENTINEL not in skeleton:
        raise ValueError("Skeleton does not contain evaluate sentinel")
    return skeleton.replace(EVALUATE_SENTINEL, evaluate_body)


def _try_import_existing(plan) -> type | None:
    """Try to import a previously built payoff class."""
    from trellis.agent.builder import TRELLIS_ROOT, dynamic_import

    for step in plan.steps:
        file_path = TRELLIS_ROOT / step.module_path
        if not file_path.exists():
            return None

    last_step = plan.steps[-1]
    file_path = TRELLIS_ROOT / last_step.module_path
    module_name = f"trellis.{last_step.module_path.replace('/', '.').replace('.py', '')}"

    try:
        mod = dynamic_import(file_path, module_name)
        return getattr(mod, plan.payoff_class_name, None)
    except Exception:
        return None


def _gather_references(pricing_plan=None) -> dict[str, str]:
    """Read reference implementations for the code generation prompt.

    Includes method-specific references based on the quant agent's plan.
    """
    refs = {}
    modules = [
        ("trellis.core.payoff", "Payoff protocol + Cashflows/PresentValue return types"),
    ]

    if pricing_plan:
        method = pricing_plan.method
        if method == "analytical":
            modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))
            modules.append(("trellis.models.black", "Black76 formulas"))
        elif method == "rate_tree":
            modules.append(("trellis.instruments.callable_bond", "CallableBondPayoff (tree reference)"))
        elif method == "monte_carlo":
            modules.append(("trellis.instruments.barrier_option", "BarrierOptionPayoff (MC reference)"))
        elif method == "copula":
            modules.append(("trellis.instruments.nth_to_default", "NthToDefaultPayoff (copula reference)"))
        else:
            modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))
    else:
        modules.append(("trellis.instruments.cap", "CapPayoff (analytical reference)"))

    for mod_path, label in modules:
        try:
            refs[label] = read_module_source(mod_path)
        except Exception:
            pass
    return refs
