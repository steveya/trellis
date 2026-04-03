"""Build planning for the agent — gap analysis, spec schemas, step decomposition."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from trellis.core.capabilities import analyze_gap, normalize_capability_name


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldDef:
    """One field definition for a payoff specification (name, type, description, optional default)."""
    name: str
    type: str  # "float", "int", "str", "bool", "date", "str | None", "tuple[date, ...]", ...
    description: str
    default: str | None = None  # None = required; otherwise a Python literal string


@dataclass(frozen=True)
class SpecSchema:
    """Complete field schema for a payoff specification class, used to generate code without LLM involvement."""
    class_name: str
    spec_name: str
    requirements: list[str]
    fields: list[FieldDef]


@dataclass(frozen=True)
class BuildStep:
    """One ordered engineering task in the deterministic build plan."""
    module_path: str
    description: str
    depends_on: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    capability_provided: str | None = None


@dataclass(frozen=True)
class BuildPlan:
    """Deterministic build plan plus gap-analysis context for one product."""
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
        requirements=["discount_curve", "forward_curve", "black_vol_surface"],
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
    "zcb_option": SpecSchema(
        class_name="ZCBOptionPayoff",
        spec_name="ZCBOptionSpec",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Face value / notional of the underlying bond"),
            FieldDef("strike", "float", "Strike quoted per unit face or on the stated notional"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("bond_maturity_date", "date", "Underlying zero-coupon bond maturity date"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
        ],
    ),
    "cap": SpecSchema(
        class_name="AgentCapPayoff",
        spec_name="AgentCapSpec",
        requirements=["discount_curve", "forward_curve", "black_vol_surface"],
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
        requirements=["discount_curve", "forward_curve", "black_vol_surface"],
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
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Face value / notional"),
            FieldDef("coupon", "float", "Annual coupon rate"),
            FieldDef("start_date", "date", "Bond issue / settlement date"),
            FieldDef("end_date", "date", "Maturity date"),
            FieldDef("call_dates", "tuple[date, ...]", "Ordered callable exercise dates"),
            FieldDef("call_price", "float", "Call price (typically par)", "100.0"),
            FieldDef("frequency", "Frequency", "Coupon frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "puttable_bond": SpecSchema(
        class_name="PuttableBondPayoff",
        spec_name="PuttableBondSpec",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Face value / notional"),
            FieldDef("coupon", "float", "Annual coupon rate"),
            FieldDef("start_date", "date", "Bond issue / settlement date"),
            FieldDef("end_date", "date", "Maturity date"),
            FieldDef("put_dates", "tuple[date, ...]", "Ordered put exercise dates"),
            FieldDef("put_price", "float", "Put price (typically par)", "100.0"),
            FieldDef("frequency", "Frequency", "Coupon frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "barrier_option": SpecSchema(
        class_name="BarrierOptionPayoff",
        spec_name="BarrierOptionSpec",
        requirements=["discount_curve", "black_vol_surface"],
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
        requirements=["discount_curve", "black_vol_surface"],
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
    "basket_option": SpecSchema(
        class_name="BasketOptionPayoff",
        spec_name="BasketOptionSpec",
        requirements=["discount_curve", "spot"],
        fields=[
            FieldDef("notional", "float", "Notional / number of basket units"),
            FieldDef("underliers", "str", "Comma-separated basket underlier names"),
            FieldDef("spots", "str", "Comma-separated current spots aligned to underliers"),
            FieldDef("strike", "float", "Basket strike"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("weights", "str | None", "Comma-separated basket weights aligned to underliers", "None"),
            FieldDef("vols", "str | None", "Comma-separated volatilities aligned to underliers", "None"),
            FieldDef("correlation", "str", "Correlation matrix encoded as semicolon-separated rows"),
            FieldDef("dividend_yields", "str | None", "Comma-separated dividend yields aligned to underliers", "None"),
            FieldDef("basket_style", "str", "Payoff style: 'weighted_sum', 'spread', 'best_of', or 'worst_of'", "'weighted_sum'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("averaging_type", "str | None", "Optional averaging style: None, 'arithmetic', or 'geometric'", "None"),
            FieldDef("n_observations", "int | None", "Optional number of observation dates for averaging or autocall logic", "None"),
            FieldDef("barrier_level", "float | None", "Optional basket-level barrier", "None"),
            FieldDef("barrier_direction", "str | None", "Optional barrier direction: 'up' or 'down'", "None"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "cdo": SpecSchema(
        class_name="CDOTranchePayoff",
        spec_name="CDOTrancheSpec",
        requirements=["discount_curve", "credit_curve"],
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
        requirements=["discount_curve", "credit_curve"],
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
        requirements=["discount_curve", "forward_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Notional amount"),
            FieldDef("strike", "float", "Strike rate"),
            FieldDef("exercise_dates", "tuple[date, ...]", "Ordered Bermudan exercise dates"),
            FieldDef("swap_end", "date", "Underlying swap end date"),
            FieldDef("swap_frequency", "Frequency", "Swap frequency", "Frequency.SEMI_ANNUAL"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
            FieldDef("rate_index", "str | None", "Forecast curve key", "None"),
            FieldDef("is_payer", "bool", "True=payer, False=receiver", "True"),
        ],
    ),
    "cds": SpecSchema(
        class_name="CDSPayoff",
        spec_name="CDSSpec",
        requirements=["discount_curve", "credit_curve"],
        fields=[
            FieldDef("notional", "float", "Protection notional"),
            FieldDef("spread", "float", "CDS running spread in decimal form (e.g. 0.015 = 150bps, not 150.0)"),
            FieldDef("recovery", "float", "Recovery rate", "0.4"),
            FieldDef("start_date", "date", "Protection start date"),
            FieldDef("end_date", "date", "Protection end date"),
            FieldDef("frequency", "Frequency", "Premium payment frequency", "Frequency.QUARTERLY"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
        ],
    ),
}

SPECIALIZED_SPECS: dict[str, SpecSchema] = {
    "cds_monte_carlo": SpecSchema(
        class_name="CDSPayoff",
        spec_name="CDSSpec",
        requirements=["discount_curve", "credit_curve"],
        fields=[
            FieldDef("notional", "float", "Protection notional"),
            FieldDef("spread", "float", "CDS running spread in decimal form (e.g. 0.015 = 150bps, not 150.0)"),
            FieldDef("recovery", "float", "Recovery rate", "0.4"),
            FieldDef("start_date", "date", "Protection start date"),
            FieldDef("end_date", "date", "Protection end date"),
            FieldDef("frequency", "Frequency", "Premium payment frequency", "Frequency.QUARTERLY"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_360"),
            FieldDef(
                "n_paths",
                "int",
                "Number of Monte Carlo paths for comparison-quality CDS pricing",
                "250000",
            ),
        ],
    ),
    "fx_vanilla_analytical": SpecSchema(
        class_name="FXVanillaAnalyticalPayoff",
        spec_name="FXVanillaOptionSpec",
        requirements=["discount_curve", "forward_curve", "black_vol_surface", "fx_rates"],
        fields=[
            FieldDef("notional", "float", "Option notional in foreign units"),
            FieldDef("strike", "float", "Strike in domestic currency per unit of foreign currency"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("fx_pair", "str", "FX quote key such as 'EURUSD'"),
            FieldDef("foreign_discount_key", "str", "Foreign discount curve key such as 'EUR-DISC'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "fx_vanilla_monte_carlo": SpecSchema(
        class_name="FXVanillaMonteCarloPayoff",
        spec_name="FXVanillaOptionSpec",
        requirements=["discount_curve", "forward_curve", "black_vol_surface", "fx_rates"],
        fields=[
            FieldDef("notional", "float", "Option notional in foreign units"),
            FieldDef("strike", "float", "Strike in domestic currency per unit of foreign currency"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("fx_pair", "str", "FX quote key such as 'EURUSD'"),
            FieldDef("foreign_discount_key", "str", "Foreign discount curve key such as 'EUR-DISC'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
            FieldDef("n_paths", "int", "Number of Monte Carlo paths", "50000"),
            FieldDef("n_steps", "int", "Number of Monte Carlo time steps", "252"),
        ],
    ),
    "quanto_option_analytical": SpecSchema(
        class_name="QuantoOptionAnalyticalPayoff",
        spec_name="QuantoOptionSpec",
        requirements=["discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"],
        fields=[
            FieldDef("notional", "float", "Option notional in foreign-underlier units"),
            FieldDef("strike", "float", "Strike in payout currency terms"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("fx_pair", "str", "FX quote key such as 'EURUSD'"),
            FieldDef("underlier_currency", "str", "Currency of the underlying asset", "'EUR'"),
            FieldDef("domestic_currency", "str", "Payout currency", "'USD'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("quanto_correlation_key", "str | None", "Key or alias for the underlier/FX correlation input", "None"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "quanto_option_monte_carlo": SpecSchema(
        class_name="QuantoOptionMonteCarloPayoff",
        spec_name="QuantoOptionSpec",
        requirements=["discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"],
        fields=[
            FieldDef("notional", "float", "Option notional in foreign-underlier units"),
            FieldDef("strike", "float", "Strike in payout currency terms"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("fx_pair", "str", "FX quote key such as 'EURUSD'"),
            FieldDef("underlier_currency", "str", "Currency of the underlying asset", "'EUR'"),
            FieldDef("domestic_currency", "str", "Payout currency", "'USD'"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("quanto_correlation_key", "str | None", "Key or alias for the underlier/FX correlation input", "None"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
            FieldDef("n_paths", "int", "Number of Monte Carlo paths", "50000"),
            FieldDef("n_steps", "int", "Number of Monte Carlo time steps", "252"),
        ],
    ),
    "european_option_analytical": SpecSchema(
        class_name="EuropeanOptionAnalyticalPayoff",
        spec_name="EuropeanOptionSpec",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Notional / number of shares"),
            FieldDef("spot", "float", "Current spot price"),
            FieldDef("strike", "float", "Option strike price"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
        ],
    ),
    "european_option_monte_carlo": SpecSchema(
        class_name="EuropeanOptionMonteCarloPayoff",
        spec_name="EuropeanOptionSpec",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Notional / number of shares"),
            FieldDef("spot", "float", "Current spot price"),
            FieldDef("strike", "float", "Option strike price"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
            FieldDef("n_paths", "int", "Number of Monte Carlo paths", "50000"),
            FieldDef("n_steps", "int", "Number of Monte Carlo time steps", "252"),
        ],
    ),
    "european_local_vol_monte_carlo": SpecSchema(
        class_name="EuropeanLocalVolMonteCarloPayoff",
        spec_name="EuropeanLocalVolOptionSpec",
        requirements=["discount_curve", "spot", "local_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Notional / number of shares"),
            FieldDef("strike", "float", "Option strike price"),
            FieldDef("expiry_date", "date", "Option expiry date"),
            FieldDef("option_type", "str", "Option type: 'call' or 'put'", "'call'"),
            FieldDef("day_count", "DayCountConvention", "Day count convention", "DayCountConvention.ACT_365"),
            FieldDef("n_paths", "int", "Number of Monte Carlo paths", "50000"),
            FieldDef("n_steps", "int", "Number of Monte Carlo time steps", "252"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Planning functions
# ---------------------------------------------------------------------------

def plan_build(
    payoff_description: str,
    requirements: set[str],
    model: str = "gpt-5.4-mini",
    instrument_type: str | None = None,
    preferred_method: str | None = None,
    spec_schema_hint: str | None = None,
) -> BuildPlan:
    """Create a build plan. Uses static specs for known instruments."""
    normalized_by_requirement = {
        requirement: normalize_capability_name(requirement)
        for requirement in requirements
    }
    satisfied_canonical, missing_canonical = analyze_gap(requirements)
    satisfied = {
        requirement
        for requirement, canonical in normalized_by_requirement.items()
        if canonical in satisfied_canonical
    }
    missing = {
        requirement
        for requirement, canonical in normalized_by_requirement.items()
        if canonical in missing_canonical
    }

    if missing:
        raise NotImplementedError(
            f"Cannot build payoff: missing capabilities {sorted(missing)}. "
            f"Available: {sorted(satisfied)}"
        )

    return _plan_static(
        payoff_description,
        requirements,
        satisfied,
        missing,
        instrument_type=instrument_type,
        preferred_method=preferred_method,
        spec_schema_hint=spec_schema_hint,
    )


def _plan_static(
    description: str,
    requirements: set[str],
    satisfied: set[str],
    missing: set[str],
    *,
    instrument_type: str | None = None,
    preferred_method: str | None = None,
    spec_schema_hint: str | None = None,
) -> BuildPlan:
    """Static planning with deterministic spec schemas for known instruments."""
    desc_lower = description.lower()
    normalized_requirements = {normalize_capability_name(requirement) for requirement in requirements}
    spec_schema = None
    class_name = None
    module = None

    # Contract-declared spec hint takes priority over regex matching
    if spec_schema_hint:
        _hint_lower = spec_schema_hint.lower().replace(" ", "_")
        if _hint_lower in SPECIALIZED_SPECS:
            spec_schema = SPECIALIZED_SPECS[_hint_lower]
            class_name = spec_schema.class_name
            module_name = class_name.lower().replace("payoff", "")
            module = f"instruments/_agent/{module_name}.py"
        elif _hint_lower in STATIC_SPECS:
            spec_schema = STATIC_SPECS[_hint_lower]
            class_name = spec_schema.class_name
            module_name = class_name.lower().replace("payoff", "")
            module = f"instruments/_agent/{module_name}.py"

    if spec_schema is None:
        specialized = _select_specialized_spec(
            description=description,
            instrument_type=instrument_type,
            normalized_requirements=normalized_requirements,
            preferred_method=preferred_method,
        )
        if specialized is not None:
            spec_schema = specialized
            class_name = spec_schema.class_name
            module_name = class_name.lower().replace("payoff", "")
            module = f"instruments/_agent/{module_name}.py"

    if spec_schema is None and instrument_type:
        # Direct lookup by normalized instrument type (more reliable than text search)
        norm = instrument_type.lower().replace(" ", "_")
        if norm in STATIC_SPECS:
            spec_schema = STATIC_SPECS[norm]
            class_name = spec_schema.class_name
            module_name = class_name.lower().replace("payoff", "")
            module = f"instruments/_agent/{module_name}.py"

    if spec_schema is None:
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


def _select_specialized_spec(
    *,
    description: str,
    instrument_type: str | None,
    normalized_requirements: set[str],
    preferred_method: str | None,
) -> SpecSchema | None:
    """Return a route-aware deterministic schema for common vanilla option families."""
    desc_lower = description.lower()
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    method_hint = _infer_method_hint(desc_lower, preferred_method=preferred_method)

    if normalized_instrument == "quanto_option" or "quanto" in desc_lower:
        if method_hint == "monte_carlo":
            return SPECIALIZED_SPECS["quanto_option_monte_carlo"]
        return SPECIALIZED_SPECS["quanto_option_analytical"]

    if normalized_instrument in {"cds", "credit_default_swap"}:
        if method_hint == "monte_carlo":
            return SPECIALIZED_SPECS["cds_monte_carlo"]

    if _looks_like_fx_vanilla(description, desc_lower, normalized_requirements):
        if method_hint == "monte_carlo":
            return SPECIALIZED_SPECS["fx_vanilla_monte_carlo"]
        return SPECIALIZED_SPECS["fx_vanilla_analytical"]

    if "local vol" in desc_lower or "local_vol_surface" in normalized_requirements:
        if normalized_instrument == "european_option" or "european" in desc_lower:
            return SPECIALIZED_SPECS["european_local_vol_monte_carlo"]

    if normalized_instrument == "european_option" or (
        "european" in desc_lower and "option" in desc_lower
    ):
        if method_hint == "monte_carlo":
            return SPECIALIZED_SPECS["european_option_monte_carlo"]
        return SPECIALIZED_SPECS["european_option_analytical"]

    return None


def _looks_like_fx_vanilla(
    description: str,
    desc_lower: str,
    normalized_requirements: set[str],
) -> bool:
    """Whether the build request clearly targets a vanilla FX option route."""
    if {"fx_rates", "forward_curve"} <= normalized_requirements:
        return True
    fx_tokens = ("fx option", "fx vanilla", "forex option", "garman-kohlhagen", "garman kohlhagen")
    if any(token in desc_lower for token in fx_tokens):
        return True
    return any(re.fullmatch(r"[A-Z]{6}", token) for token in re.findall(r"\b[A-Za-z]{6}\b", description))


def _infer_method_hint(desc_lower: str, *, preferred_method: str | None = None) -> str | None:
    """Infer the preferred route family from the build description."""
    if preferred_method:
        normalized = preferred_method.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"monte_carlo", "mc"}:
            return "monte_carlo"
        if normalized in {"analytical", "garman_kohlhagen", "gk_analytical"}:
            return "analytical"
    preferred_match = re.search(r"preferred method family:\s*([a-zA-Z_]+)", desc_lower)
    if preferred_match:
        return preferred_match.group(1).strip().lower()
    if "implementation target" in desc_lower and "mc" in desc_lower:
        return "monte_carlo"
    if "monte carlo" in desc_lower or " monte_carlo" in desc_lower:
        return "monte_carlo"
    if "analytical" in desc_lower or "garman_kohlhagen" in desc_lower:
        return "analytical"
    return None
