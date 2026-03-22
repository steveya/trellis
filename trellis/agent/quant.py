"""Quant agent: selects pricing method and identifies data requirements.

This agent makes the *financial* decision: given an instrument, which
computational method is appropriate and what market data does it need?
It does NOT write code — that's the builder agent's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from trellis.core.capabilities import (
    MARKET_DATA,
    METHODS,
    _MARKET_DATA_NAMES,
    check_market_data,
)


@dataclass(frozen=True)
class PricingPlan:
    """The quant agent's output: method + data requirements + modeling constraints."""

    method: str                     # "analytical", "rate_tree", "monte_carlo", "pde", "fft"
    method_modules: list[str]       # specific imports the builder should use
    required_market_data: set[str]  # {"discount", "black_vol"}
    model_to_build: str | None      # None if library has it; description if custom needed
    reasoning: str                  # why this method was chosen
    modeling_requirements: tuple[str, ...] = ()  # constraints the implementation must satisfy


# ---------------------------------------------------------------------------
# Modeling requirements — general principles for correct pricing
# ---------------------------------------------------------------------------

# Rate tree requirements (callable bonds, Bermudan swaptions, etc.)
RATE_TREE_REQUIREMENTS = (
    "CALIBRATION: The rate tree must be calibrated to reprice the input yield curve. "
    "At each time step, solve for theta(t) so that the tree-implied zero-coupon bond "
    "price matches curve.discount(T). Verify: tree_zcb(T) ≈ curve.discount(T) at key tenors.",

    "DISCRETE CASHFLOWS: Bond coupons and other scheduled payments must be embedded at "
    "their actual payment dates (mapped to the nearest tree step), NOT spread uniformly "
    "across all steps. Uniform spreading introduces >10% pricing error.",

    "EXERCISE LOGIC: The call/put decision at each node must compare the CONTINUATION "
    "VALUE (discounted expected future value) against the exercise price. The continuation "
    "value is computed from the tree's own discount factors, not the flat input rate.",
)

# Monte Carlo requirements
MC_REQUIREMENTS = (
    "CONVERGENCE: Use at least 10,000 paths. Verify that the standard error is <1% "
    "of the price. If path-dependent, use at least 100 time steps per year.",

    "DISCRETE OBSERVATIONS: For instruments with discrete fixing/observation dates "
    "(Asian options, barriers), simulate paths that pass through those exact dates. "
    "Do not interpolate between steps.",
)

# Copula requirements
COPULA_REQUIREMENTS = (
    "CALIBRATION: Default probabilities must be consistent with the input credit curve. "
    "For each name: P(default by T) = 1 - S(T) where S(T) = credit_curve.survival_probability(T).",
)


# ---------------------------------------------------------------------------
# Static rules for common instruments
# ---------------------------------------------------------------------------

STATIC_PLANS: dict[str, PricingPlan] = {
    "bond": PricingPlan(
        method="analytical",
        method_modules=[],
        required_market_data={"discount"},
        model_to_build=None,
        reasoning="Deterministic cashflows — discount each coupon and principal.",
    ),
    "swap": PricingPlan(
        method="analytical",
        method_modules=[],
        required_market_data={"discount", "forward_rate"},
        model_to_build=None,
        reasoning="Fixed and floating legs are deterministic given the forward curve.",
    ),
    "cap": PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount", "forward_rate", "black_vol"},
        model_to_build=None,
        reasoning="Each caplet is a European option on a forward rate — use Black76.",
    ),
    "floor": PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount", "forward_rate", "black_vol"},
        model_to_build=None,
        reasoning="Each floorlet is a European put on a forward rate — use Black76.",
    ),
    "swaption": PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount", "forward_rate", "black_vol"},
        model_to_build=None,
        reasoning="European swaption — Black76 on the forward swap rate.",
    ),
    "callable_bond": PricingPlan(
        method="rate_tree",
        method_modules=[
            "trellis.models.trees.lattice",
        ],
        required_market_data={"discount", "black_vol"},
        model_to_build=None,
        reasoning="Early exercise requires backward induction on a mean-reverting rate tree.",
        modeling_requirements=RATE_TREE_REQUIREMENTS,
    ),
    "puttable_bond": PricingPlan(
        method="rate_tree",
        method_modules=[
            "trellis.models.trees.lattice",
        ],
        required_market_data={"discount", "black_vol"},
        model_to_build=None,
        reasoning="Early exercise requires backward induction on a mean-reverting rate tree.",
        modeling_requirements=RATE_TREE_REQUIREMENTS,
    ),
    "bermudan_swaption": PricingPlan(
        method="rate_tree",
        method_modules=[
            "trellis.models.trees.lattice",
        ],
        required_market_data={"discount", "forward_rate", "black_vol"},
        model_to_build=None,
        reasoning="Bermudan exercise dates require backward induction on a rate tree.",
        modeling_requirements=RATE_TREE_REQUIREMENTS,
    ),
    "barrier_option": PricingPlan(
        method="monte_carlo",
        method_modules=[
            "trellis.models.monte_carlo.engine",
            "trellis.models.processes.gbm",
        ],
        required_market_data={"discount", "black_vol"},
        model_to_build=None,
        reasoning="Path-dependent barrier monitoring needs simulation.",
        modeling_requirements=MC_REQUIREMENTS,
    ),
    "asian_option": PricingPlan(
        method="monte_carlo",
        method_modules=[
            "trellis.models.monte_carlo.engine",
            "trellis.models.processes.gbm",
        ],
        required_market_data={"discount", "black_vol"},
        model_to_build=None,
        reasoning="Path-dependent averaging needs simulation.",
        modeling_requirements=MC_REQUIREMENTS,
    ),
    "cdo": PricingPlan(
        method="copula",
        method_modules=[
            "trellis.models.copulas.factor",
            "trellis.models.copulas.gaussian",
        ],
        required_market_data={"discount", "credit"},
        model_to_build=None,
        reasoning="Portfolio credit tranching uses copula for default correlation.",
        modeling_requirements=COPULA_REQUIREMENTS,
    ),
    "nth_to_default": PricingPlan(
        method="copula",
        method_modules=[
            "trellis.models.copulas.gaussian",
        ],
        required_market_data={"discount", "credit"},
        model_to_build=None,
        reasoning="Correlated defaults simulated via copula.",
        modeling_requirements=COPULA_REQUIREMENTS,
    ),
    "mbs": PricingPlan(
        method="monte_carlo",
        method_modules=[
            "trellis.models.monte_carlo.engine",
            "trellis.models.cashflow_engine.waterfall",
            "trellis.models.cashflow_engine.prepayment",
        ],
        required_market_data={"discount"},
        model_to_build=None,
        reasoning="Rate-path-dependent prepayment needs MC + waterfall engine.",
    ),
}


# ---------------------------------------------------------------------------
# Quant agent
# ---------------------------------------------------------------------------

def select_pricing_method(
    instrument_description: str,
    instrument_type: str | None = None,
    model: str | None = None,
) -> PricingPlan:
    """Select the appropriate pricing method for an instrument.

    Tries static rules first, falls back to LLM.
    """
    # Normalize type
    if instrument_type:
        itype = instrument_type.lower().replace(" ", "_").replace("-", "_")
    else:
        itype = _extract_type(instrument_description)

    # Static lookup
    if itype in STATIC_PLANS:
        return STATIC_PLANS[itype]

    # LLM fallback
    return _select_via_llm(instrument_description, model)


def _extract_type(description: str) -> str:
    """Extract instrument type keyword from description."""
    desc = description.lower()
    # Check longer keywords first to avoid partial matches
    # (e.g. "callable_bond" before "bond", "bermudan_swaption" before "swaption")
    sorted_keywords = sorted(STATIC_PLANS.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword.replace("_", " ") in desc or keyword in desc:
            return keyword
    return "unknown"


def _select_via_llm(description: str, model: str | None = None) -> PricingPlan:
    """Use LLM to select pricing method for unknown instruments."""
    from trellis.agent.config import llm_generate_json, get_default_model

    model = model or get_default_model()

    method_info = "\n".join(
        f"- {m.name}: {m.description} (requires: {list(m.requires_market_data)})"
        for m in METHODS
    )

    prompt = f"""You are a quantitative analyst selecting a pricing method.

## Instrument
{description}

## Available computational methods
{method_info}

## Available market data types
{', '.join(c.name for c in MARKET_DATA)}

## Rules
- Deterministic cashflows (bonds, swaps) → "analytical" (no special method needed)
- European options on forwards → "analytical" (Black76)
- Early exercise (callable, puttable, Bermudan) → "rate_tree" (backward induction)
- Path-dependent (barriers, Asian, lookback) → "monte_carlo"
- Portfolio credit (CDO, nth-to-default) → "copula"
- Stochastic vol options → "fft_pricing" (Heston characteristic function)
- American options on equity → "monte_carlo" with LSM or "pde_solver" with PSOR
- MBS/ABS → "monte_carlo" with waterfall engine

## Output
Return a JSON object:
{{
    "method": "rate_tree" | "monte_carlo" | "pde_solver" | "fft_pricing" | "copula" | "analytical",
    "method_modules": ["trellis.models.trees.binomial", ...],
    "required_market_data": ["discount", "black_vol"],
    "model_to_build": null or "description of custom model needed",
    "reasoning": "one sentence explanation"
}}

Return ONLY the JSON object."""

    try:
        data = llm_generate_json(prompt, model=model)
    except Exception:
        # Fallback: conservative default
        return PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.models.monte_carlo.engine"],
            required_market_data={"discount", "black_vol"},
            model_to_build=None,
            reasoning="Fallback: Monte Carlo is the most general method.",
        )

    return PricingPlan(
        method=data.get("method", "monte_carlo"),
        method_modules=data.get("method_modules", []),
        required_market_data=set(data.get("required_market_data", ["discount"])),
        model_to_build=data.get("model_to_build"),
        reasoning=data.get("reasoning", ""),
    )


def check_data_availability(
    pricing_plan: PricingPlan,
    market_state,
) -> list[str]:
    """Check if the required market data is available in MarketState.

    Returns list of user-friendly error messages. Empty = all good.
    """
    return check_market_data(pricing_plan.required_market_data, market_state)
