"""Quant agent: selects pricing methods from canonical decompositions.

This layer should not maintain a second hand-written policy table. Canonical
method selection lives in ``trellis.agent.knowledge.canonical.decompositions``;
the quant agent converts those decompositions into ``PricingPlan`` objects for
the build pipeline and falls back to LLM decomposition only for genuinely novel
products.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from trellis.agent.knowledge import get_store
from trellis.agent.knowledge.decompose import decompose, decompose_to_ir
from trellis.agent.knowledge.methods import CANONICAL_METHODS, normalize_method
from trellis.agent.sensitivity_support import (
    SensitivitySupport,
    normalize_requested_measures,
    rank_sensitivity_support,
    support_for_method,
)
from trellis.core.capabilities import (
    check_market_data,
    normalize_market_data_requirements,
)


@dataclass(frozen=True)
class PricingPlan:
    """The quant agent's output: method + data requirements + modeling constraints."""

    method: str
    method_modules: list[str]
    required_market_data: set[str]
    model_to_build: str | None
    reasoning: str
    modeling_requirements: tuple[str, ...] = ()
    sensitivity_support: SensitivitySupport | None = None


def _plan_from_decomposition(decomposition) -> PricingPlan:
    """Convert a canonical product decomposition into a PricingPlan."""
    method = normalize_method(decomposition.method)
    return PricingPlan(
        method=method,
        method_modules=list(decomposition.method_modules),
        required_market_data=normalize_market_data_requirements(
            decomposition.required_market_data
        ),
        model_to_build=None,
        reasoning=decomposition.reasoning,
        modeling_requirements=tuple(decomposition.modeling_requirements),
        sensitivity_support=support_for_method(method),
    )


def _load_static_plans() -> dict[str, PricingPlan]:
    """Load static pricing plans from canonical decompositions.

    These are "static" only in the sense that they are repo-configured. The
    source of truth remains the canonical YAML, not this Python module.
    """
    store = get_store()
    return {
        instrument: _plan_from_decomposition(decomposition)
        for instrument, decomposition in store._decompositions.items()
    }


# Backward-compatible public constant used by tests and callers.
STATIC_PLANS: dict[str, PricingPlan] = _load_static_plans()


_DEFAULT_METHOD_MODULES = {
    "analytical": ["trellis.models.black"],
    "rate_tree": ["trellis.models.trees.lattice"],
    "monte_carlo": ["trellis.models.monte_carlo.engine"],
    "qmc": ["trellis.models.qmc"],
    "fft_pricing": ["trellis.models.transforms.fft_pricer"],
    "pde_solver": ["trellis.models.pde.theta_method"],
    "copula": ["trellis.models.copulas.gaussian"],
    "waterfall": ["trellis.models.cashflow_engine.waterfall"],
}


def select_pricing_method(
    instrument_description: str,
    instrument_type: str | None = None,
    model: str | None = None,
    requested_measures: list[str] | tuple[str, ...] | None = None,
) -> PricingPlan:
    """Select the appropriate pricing method for an instrument.

    Uses canonical decompositions for known instruments, then falls back to the
    decomposition workflow for unknown or composite products.
    """
    requested = normalize_requested_measures(requested_measures)
    if instrument_type:
        key = _normalise_instrument_type(instrument_type)
        if key in STATIC_PLANS and not requested:
            return _apply_contextual_overrides(
                STATIC_PLANS[key],
                instrument_description,
                instrument_type=instrument_type,
            )
    else:
        key = _extract_type(instrument_description)
        if key in STATIC_PLANS and not requested:
            return _apply_contextual_overrides(
                STATIC_PLANS[key],
                instrument_description,
                instrument_type=instrument_type or key,
            )

    if requested and key in STATIC_PLANS:
        product_ir = decompose_to_ir(
            instrument_description,
            instrument_type=instrument_type or key,
        )
        return select_pricing_method_for_product_ir(
            product_ir,
            context_description=instrument_description,
            requested_measures=requested,
        )

    decomposition = decompose(
        instrument_description,
        instrument_type=instrument_type,
        model=model,
    )
    return _apply_contextual_overrides(
        _plan_from_decomposition(decomposition),
        instrument_description,
        instrument_type=instrument_type,
    )


def select_pricing_method_for_product_ir(
    product_ir,
    *,
    preferred_method: str | None = None,
    requested_measures: list[str] | tuple[str, ...] | None = None,
    context_description: str | None = None,
) -> PricingPlan:
    """Build a pricing plan directly from ``ProductIR`` semantics."""
    store = get_store()
    requested = normalize_requested_measures(requested_measures)
    method = normalize_method(
        preferred_method
        or getattr(product_ir, "preferred_method", None)
        or _method_from_candidates(
            getattr(product_ir, "candidate_engine_families", ()),
            requested_measures=requested,
        ),
    )
    requirements_entry = store._load_requirements(method)
    plan = PricingPlan(
        method=method,
        method_modules=list(_DEFAULT_METHOD_MODULES.get(method, ())),
        required_market_data=normalize_market_data_requirements(
            getattr(product_ir, "required_market_data", ()) or ()
        ),
        model_to_build=getattr(product_ir, "instrument", None),
        reasoning="product_ir_compiler",
        modeling_requirements=(
            requirements_entry.requirements if requirements_entry is not None else ()
        ),
        sensitivity_support=support_for_method(method),
    )
    return _apply_contextual_overrides(
        plan,
        context_description,
        instrument_type=getattr(product_ir, "instrument", None),
    )


def _normalise_instrument_type(text: str) -> str:
    """Normalize an explicit instrument type key."""
    return text.lower().strip().replace(" ", "_").replace("-", "_")


def _extract_type(description: str) -> str:
    """Extract an instrument type keyword from free-form text.

    This is intentionally lightweight and only used as a fast path before the
    richer decomposition flow. Longest-key matching avoids ``bond`` winning over
    ``callable_bond``.
    """
    desc = description.lower()
    for keyword in sorted(STATIC_PLANS.keys(), key=len, reverse=True):
        if keyword.replace("_", " ") in desc or keyword in desc:
            return keyword
    return "unknown"


def _method_from_candidates(
    candidate_engine_families: tuple[str, ...] | list[str],
    *,
    requested_measures: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Map candidate engine-family labels onto the canonical quant method name."""
    mapping = {
        "analytical": "analytical",
        "lattice": "rate_tree",
        "tree": "rate_tree",
        "exercise": "monte_carlo",
        "monte_carlo": "monte_carlo",
        "transform": "fft_pricing",
        "transforms": "fft_pricing",
        "fft": "fft_pricing",
        "pde": "pde_solver",
    }
    candidates = [
        mapping[family]
        for family in candidate_engine_families
        if family in mapping
    ]
    if not candidates:
        return "analytical"
    if requested_measures:
        best = max(
            candidates,
            key=lambda method: rank_sensitivity_support(
                support_for_method(method),
                requested_measures,
            ),
        )
        return best
    for method in candidates:
        if method:
            return method
    return "analytical"


def known_methods() -> tuple[str, ...]:
    """Return the canonical method-family labels understood by the quant agent."""
    return tuple(sorted(CANONICAL_METHODS))


def _apply_contextual_overrides(
    plan: PricingPlan,
    description: str | None,
    *,
    instrument_type: str | None = None,
) -> PricingPlan:
    """Apply conservative context-derived overrides without changing the core ontology."""
    plan = _apply_local_vol_overrides(
        plan,
        description,
        instrument_type=instrument_type,
    )
    if not _looks_like_fx_option(description, instrument_type=instrument_type):
        return plan

    required_market_data = normalize_market_data_requirements(
        set(plan.required_market_data)
        | {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"}
    )
    reasoning = plan.reasoning
    fx_reason = "fx_vanilla_context_requires_garman_kohlhagen_inputs"
    if fx_reason not in reasoning:
        reasoning = f"{reasoning}; {fx_reason}" if reasoning else fx_reason
    return replace(
        plan,
        required_market_data=required_market_data,
        reasoning=reasoning,
    )


def _apply_local_vol_overrides(
    plan: PricingPlan,
    description: str | None,
    *,
    instrument_type: str | None = None,
) -> PricingPlan:
    """Narrow vanilla local-vol requests onto the supported MC/PDE substrate."""
    if not _looks_like_local_vol_context(description, instrument_type=instrument_type):
        return plan

    method = plan.method
    if method == "analytical":
        method = "monte_carlo"

    method_modules = list(_DEFAULT_METHOD_MODULES.get(method, plan.method_modules))
    if method == "monte_carlo":
        for module_path in (
            "trellis.models.monte_carlo.local_vol",
            "trellis.models.processes.local_vol",
        ):
            if module_path not in method_modules:
                method_modules.append(module_path)
    elif method == "pde_solver":
        module_path = "trellis.models.processes.local_vol"
        if module_path not in method_modules:
            method_modules.append(module_path)

    required_market_data = normalize_market_data_requirements(
        (set(plan.required_market_data) - {"black_vol_surface"})
        | {"discount_curve", "spot", "local_vol_surface"}
    )
    reasoning = plan.reasoning
    local_vol_reason = "local_vol_context_requires_surface_driven_route"
    if local_vol_reason not in reasoning:
        reasoning = f"{reasoning}; {local_vol_reason}" if reasoning else local_vol_reason
    return replace(
        plan,
        method=method,
        method_modules=method_modules,
        required_market_data=required_market_data,
        reasoning=reasoning,
        sensitivity_support=support_for_method(method),
    )


def _looks_like_fx_option(
    description: str | None,
    *,
    instrument_type: str | None = None,
) -> bool:
    """Detect a vanilla FX-option context from the user-facing request text."""
    if instrument_type == "fx_option":
        return True
    if not description:
        return False
    lower = description.lower()
    if any(token in lower for token in ("fx option", "fx vanilla", "forex option", "garman-kohlhagen", "gk analytical")):
        return True
    return re.search(r"\b[A-Z]{6}\b", description) is not None


def _looks_like_local_vol_context(
    description: str | None,
    *,
    instrument_type: str | None = None,
) -> bool:
    """Detect a bounded vanilla local-vol request from user-facing text."""
    if instrument_type == "local_vol_option":
        return True
    if not description:
        return False
    lower = description.lower()
    return any(
        token in lower
        for token in (
            "local vol",
            "local volatility",
            "dupire",
            "local_vol_mc",
            "local_vol_pde",
        )
    )


def check_data_availability(
    pricing_plan: PricingPlan,
    market_state,
) -> list[str]:
    """Check if the required market data is available in MarketState.

    Returns list of user-friendly error messages. Empty = all good.
    """
    return check_market_data(pricing_plan.required_market_data, market_state)
