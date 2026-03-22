"""Build planning for the agent — gap analysis, spec schemas, step decomposition."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from trellis.core.capabilities import analyze_gap


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldDef:
    """A single field in a spec dataclass."""
    name: str
    type: str  # "float", "int", "str", "bool", "date", "str | None", "Frequency", "DayCountConvention"
    description: str
    default: str | None = None  # None = required; otherwise a Python literal string


@dataclass(frozen=True)
class SpecSchema:
    """Schema for a payoff spec dataclass — deterministic, no LLM ambiguity."""
    class_name: str
    spec_name: str
    requirements: list[str]
    fields: list[FieldDef]


@dataclass(frozen=True)
class BuildStep:
    module_path: str
    description: str
    depends_on: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    capability_provided: str | None = None


@dataclass(frozen=True)
class BuildPlan:
    steps: list[BuildStep]
    description: str
    payoff_class_name: str
    requirements: frozenset[str]
    satisfied: frozenset[str]
    missing: frozenset[str]
    spec_schema: SpecSchema | None = None


# ---------------------------------------------------------------------------
# Static spec schemas for common instruments
# ---------------------------------------------------------------------------

STATIC_SPECS: dict[str, SpecSchema] = {
    "swaption": SpecSchema(
        class_name="SwaptionPayoff",
        spec_name="SwaptionSpec",
        requirements=["discount", "forward_rate", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Notional amount"),
            FieldDef("strike", "float", "Strike rate (fixed rate of underlying swap)"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("swap_start", "date", "Underlying swap start date"),
            FieldDef("swap_end", "date", "Underlying swap end date"),
            FieldDef("swap_frequency", "Frequency", "Swap payment frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
            FieldDef("rate_index", "str | None", "Forecast curve key for multi-curve", "None"),
            FieldDef("is_payer", "bool", "True=payer swaption, False=receiver", "True"),
        ],
    ),
    "cap": SpecSchema(
        class_name="AgentCapPayoff",
        spec_name="AgentCapSpec",
        requirements=["discount", "forward_rate", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Notional amount"),
            FieldDef("strike", "float", "Cap strike rate"),
            FieldDef("start_date", "date", "First accrual period start"),
            FieldDef("end_date", "date", "Final payment date"),
            FieldDef("frequency", "Frequency", "Caplet frequency", "Frequency.QUARTERLY"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
            FieldDef("rate_index", "str | None", "Forecast curve key", "None"),
        ],
    ),
    "floor": SpecSchema(
        class_name="AgentFloorPayoff",
        spec_name="AgentFloorSpec",
        requirements=["discount", "forward_rate", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Notional amount"),
            FieldDef("strike", "float", "Floor strike rate"),
            FieldDef("start_date", "date", "First accrual period start"),
            FieldDef("end_date", "date", "Final payment date"),
            FieldDef("frequency", "Frequency", "Floorlet frequency", "Frequency.QUARTERLY"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
            FieldDef("rate_index", "str | None", "Forecast curve key", "None"),
        ],
    ),
    "callable_bond": SpecSchema(
        class_name="CallableBondPayoff",
        spec_name="CallableBondSpec",
        requirements=["discount", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Face value / notional"),
            FieldDef("coupon", "float", "Annual coupon rate"),
            FieldDef("start_date", "date", "Bond issue / settlement date"),
            FieldDef("end_date", "date", "Maturity date"),
            FieldDef("call_dates", "str", "Comma-separated ISO dates: '2027-11-15,2029-11-15'"),
            FieldDef("call_price", "float", "Call price (typically par)", "100.0"),
            FieldDef("frequency", "Frequency", "Coupon frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "puttable_bond": SpecSchema(
        class_name="PuttableBondPayoff",
        spec_name="PuttableBondSpec",
        requirements=["discount", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Face value / notional"),
            FieldDef("coupon", "float", "Annual coupon rate"),
            FieldDef("start_date", "date", "Bond issue / settlement date"),
            FieldDef("end_date", "date", "Maturity date"),
            FieldDef("put_dates", "str", "Comma-separated ISO dates"),
            FieldDef("put_price", "float", "Put price (typically par)", "100.0"),
            FieldDef("frequency", "Frequency", "Coupon frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "barrier_option": SpecSchema(
        class_name="BarrierOptionPayoff",
        spec_name="BarrierOptionSpec",
        requirements=["discount", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Notional / number of shares"),
            FieldDef("spot", "float", "Current spot price"),
            FieldDef("strike", "float", "Option strike price"),
            FieldDef("barrier", "float", "Barrier level"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("barrier_type", "str", "Type: 'up_and_out', 'down_and_out', 'up_and_in', 'down_and_in'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "asian_option": SpecSchema(
        class_name="AsianOptionPayoff",
        spec_name="AsianOptionSpec",
        requirements=["discount", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Notional / number of shares"),
            FieldDef("spot", "float", "Current spot price"),
            FieldDef("strike", "float", "Option strike price"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("averaging_type", "str", "Averaging: 'arithmetic' or 'geometric'", "'arithmetic'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("n_observations", "int", "Number of averaging observations", "12"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "cdo": SpecSchema(
        class_name="CDOTranchePayoff",
        spec_name="CDOTrancheSpec",
        requirements=["discount", "credit"],
        fields=[
            FieldDef("notional", "float", "Tranche notional"),
            FieldDef("n_names", "int", "Number of names in reference portfolio"),
            FieldDef("attachment", "float", "Attachment point (e.g. 0.03 for 3%)"),
            FieldDef("detachment", "float", "Detachment point (e.g. 0.07 for 7%)"),
            FieldDef("end_date", "date", "Protection end date"),
            FieldDef("correlation", "float", "Default correlation", "0.3"),
            FieldDef("recovery", "float", "Recovery rate", "0.4"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
        ],
    ),
    "nth_to_default": SpecSchema(
        class_name="NthToDefaultPayoff",
        spec_name="NthToDefaultSpec",
        requirements=["discount", "credit"],
        fields=[
            FieldDef("notional", "float", "Protection notional"),
            FieldDef("n_names", "int", "Number of names in basket"),
            FieldDef("n_th", "int", "Which default triggers (1=first, 2=second, etc.)"),
            FieldDef("end_date", "date", "Protection end date"),
            FieldDef("correlation", "float", "Default correlation", "0.3"),
            FieldDef("recovery", "float", "Recovery rate", "0.4"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
        ],
    ),
    "bermudan_swaption": SpecSchema(
        class_name="BermudanSwaptionPayoff",
        spec_name="BermudanSwaptionSpec",
        requirements=["discount", "forward_rate", "black_vol"],
        fields=[
            FieldDef("notional", "float", "Notional amount"),
            FieldDef("strike", "float", "Strike rate"),
            FieldDef("exercise_dates", "str", "Comma-separated ISO exercise dates"),
            FieldDef("swap_end", "date", "Underlying swap end date"),
            FieldDef("swap_frequency", "Frequency", "Swap frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
            FieldDef("rate_index", "str | None", "Forecast curve key", "None"),
            FieldDef("is_payer", "bool", "True=payer, False=receiver", "True"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Planning functions
# ---------------------------------------------------------------------------

def plan_build(
    payoff_description: str,
    requirements: set[str],
    model: str = "o3-mini",
) -> BuildPlan:
    """Create a build plan. Uses static specs for known instruments."""
    satisfied, missing = analyze_gap(requirements)

    if missing:
        raise NotImplementedError(
            f"Cannot build payoff: missing capabilities {sorted(missing)}. "
            f"Available: {sorted(satisfied)}"
        )

    return _plan_static(payoff_description, requirements, satisfied, missing)


def _plan_static(
    description: str,
    requirements: set[str],
    satisfied: set[str],
    missing: set[str],
) -> BuildPlan:
    """Static planning with deterministic spec schemas for known instruments."""
    desc_lower = description.lower()
    spec_schema = None
    class_name = None
    module = None

    # Match against STATIC_SPECS — longer keys first to avoid partial matches
    for key in sorted(STATIC_SPECS.keys(), key=len, reverse=True):
        if key.replace("_", " ") in desc_lower or key in desc_lower:
            spec_schema = STATIC_SPECS[key]
            class_name = spec_schema.class_name
            module_name = class_name.lower().replace("payoff", "")
            module = f"instruments/_agent/{module_name}.py"
            break

    if class_name is None:
        words = description.split()[:2]
        class_name = "".join(w.capitalize() for w in words) + "Payoff"
        module = f"instruments/_agent/{class_name.lower()}.py"

    step = BuildStep(
        module_path=module,
        description=f"Build {class_name} implementing Payoff protocol",
        acceptance_criteria=(
            "protocol_conformance",
            "non_negativity",
        ),
    )

    return BuildPlan(
        steps=[step],
        description=f"Build plan for: {description}",
        payoff_class_name=class_name,
        requirements=frozenset(requirements),
        satisfied=frozenset(satisfied),
        missing=frozenset(missing),
        spec_schema=spec_schema,
    )
