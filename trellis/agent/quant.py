"""Quant agent: selects pricing methods from canonical decompositions.

This layer should not maintain a second hand-written policy table. Canonical
method selection lives in ``trellis.agent.knowledge.canonical.decompositions``;
the quant agent converts those decompositions into ``PricingPlan`` objects for
the build pipeline and falls back to LLM decomposition only for genuinely novel
products.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
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
class QuantMethodCandidate:
    """One candidate method considered by the quant agent."""

    method: str
    status: str
    rejection_reason: str = ""
    priority_rank: int = 999
    sensitivity_level: str | None = None
    supported_measures: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuantChallengerPacket:
    """Structured quant handoff for downstream review stages."""

    selected_method: str
    method_identity: str
    route_family: str
    selection_reason: str
    candidate_methods: tuple[QuantMethodCandidate, ...]
    assumption_basis: tuple[str, ...]
    required_market_data: tuple[str, ...]
    requested_measures: tuple[str, ...]
    expected_executable_checks: tuple[str, ...]
    residual_risk_handoff: tuple[str, ...]


@dataclass(frozen=True)
class PricingPlan:
    """Output of the quant agent's method selection step.

    Contains the chosen pricing method (e.g. 'analytical', 'monte_carlo'),
    the modules that implement it, the market data it needs, any modeling
    constraints or assumptions, and the structured challenger packet consumed
    by downstream validation stages.
    """

    method: str
    method_modules: list[str]
    required_market_data: set[str]
    model_to_build: str | None
    reasoning: str
    modeling_requirements: tuple[str, ...] = ()
    sensitivity_support: SensitivitySupport | None = None
    selection_reason: str = ""
    assumption_summary: tuple[str, ...] = ()
    challenger_packet: QuantChallengerPacket | Mapping[str, object] | None = None


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

# Legacy _FAMILY_BLUEPRINT_ROUTE_MODULES removed.
# Route → module mappings are now sourced from the route registry
# via route_registry.get_route_modules().

_METHOD_PRIORITY_ORDER = {
    "analytical": 0,
    "rate_tree": 1,
    "pde_solver": 2,
    "fft_pricing": 3,
    "monte_carlo": 4,
    "qmc": 5,
    "copula": 6,
    "waterfall": 7,
}

_METHOD_ASSUMPTION_SUMMARIES = {
    "analytical": (
        "simplest_valid_assumption_set",
        "closed_form_or_quasi_closed_form_route",
        "no_path_sampling_required",
    ),
    "rate_tree": (
        "simplest_valid_assumption_set",
        "discrete_time_lattice_route",
        "exercise_and_cashflow_discretization_acceptable",
    ),
    "pde_solver": (
        "simplest_valid_assumption_set",
        "finite_difference_discretization_route",
        "boundary_conditions_required",
    ),
    "fft_pricing": (
        "simplest_valid_assumption_set",
        "transform_route_available",
        "characteristic_function_route_available",
    ),
    "monte_carlo": (
        "simplest_valid_assumption_set",
        "simulation_based_valuation_route",
        "path_sampling_required",
    ),
    "qmc": (
        "simplest_valid_assumption_set",
        "low_discrepancy_simulation_route",
        "path_sampling_required",
    ),
    "copula": (
        "simplest_valid_assumption_set",
        "dependence_modeling_route",
        "correlation_or_copula_structure_required",
    ),
    "waterfall": (
        "simplest_valid_assumption_set",
        "cashflow_waterfall_route",
        "cashflow_schedule_required",
    ),
}


def _normalize_string_tuple(values) -> tuple[str, ...]:
    """Normalize arbitrary values into a deduplicated tuple of strings."""
    if not values:
        return ()
    normalized: list[str] = []
    for value in values:
        text = str(getattr(value, "value", value)).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _merge_unique_strings(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Merge string collections while preserving order and dropping duplicates."""
    merged: list[str] = []
    for group in groups:
        for value in group or ():
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return tuple(merged)


def _method_priority(method: str) -> int:
    """Return the default simplicity/latency rank for a method family."""
    return _METHOD_PRIORITY_ORDER.get(normalize_method(method), len(_METHOD_PRIORITY_ORDER))


def _candidate_methods_from_packet(packet) -> tuple[str, ...]:
    """Recover candidate method identities from an existing challenger packet."""
    if packet is None:
        return ()
    if isinstance(packet, QuantChallengerPacket):
        return tuple(candidate.method for candidate in packet.candidate_methods)
    if isinstance(packet, Mapping):
        candidates = packet.get("candidate_methods") or ()
        methods: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                method = normalize_method(candidate.get("method"))
            else:
                method = normalize_method(getattr(candidate, "method", None))
            if method and method not in methods:
                methods.append(method)
        return tuple(methods)
    return ()


def _rejection_reason_for_candidate(
    candidate_method: str,
    selected_method: str,
    *,
    selection_reason: str,
    requested_measures: tuple[str, ...],
    selected_support: SensitivitySupport | None,
) -> str:
    """Return the stable reason a non-selected candidate lost."""
    if candidate_method == selected_method:
        return ""
    if "explicit_preference" in selection_reason:
        return "not_explicit_preference"
    if requested_measures:
        candidate_rank = rank_sensitivity_support(
            support_for_method(candidate_method),
            requested_measures,
        )[:3]
        selected_rank = rank_sensitivity_support(
            selected_support or support_for_method(selected_method),
            requested_measures,
        )[:3]
        if candidate_rank < selected_rank:
            return "lower_requested_measure_support"
        return "lower_measure_selection_rank"
    return "higher_complexity_than_selected_default"


def _candidate_summary_for_method(
    method: str,
    *,
    selected_method: str,
    selection_reason: str,
    requested_measures: tuple[str, ...],
    selected_support: SensitivitySupport | None,
) -> QuantMethodCandidate:
    """Build one structured candidate entry for the challenger packet."""
    normalized = normalize_method(method)
    support = support_for_method(normalized)
    selected = normalized == selected_method
    return QuantMethodCandidate(
        method=normalized,
        status="selected" if selected else "rejected",
        rejection_reason="" if selected else _rejection_reason_for_candidate(
            normalized,
            selected_method,
            selection_reason=selection_reason,
            requested_measures=requested_measures,
            selected_support=selected_support,
        ),
        priority_rank=_method_priority(normalized),
        sensitivity_level=support.level,
        supported_measures=tuple(support.supported_measures),
    )


def _ordered_candidate_methods(
    selected_method: str,
    candidate_methods: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Return candidate methods with the selected method first, then simplicity order."""
    methods: list[str] = []
    for method in (selected_method, *(candidate_methods or ())):
        normalized = normalize_method(method)
        if normalized and normalized not in methods:
            methods.append(normalized)
    return tuple(
        sorted(
            methods,
            key=lambda method: (
                0 if method == selected_method else 1,
                _method_priority(method),
                method,
            ),
        )
    )


def _expected_executable_checks_for_packet(
    *,
    candidate_methods: tuple[str, ...],
    requested_measures: tuple[str, ...],
) -> tuple[str, ...]:
    """Return quant-side executable checks expected downstream."""
    checks = ["market_data_capability_check", "deterministic_validation_bundle"]
    if len(candidate_methods) > 1:
        checks.append("alternative_method_challenge")
    if requested_measures:
        checks.append("requested_measure_support_check")
    return tuple(checks)


def _residual_risk_handoff_for_packet(
    *,
    assumption_basis: tuple[str, ...],
    sensitivity_support: SensitivitySupport | None,
) -> tuple[str, ...]:
    """Return quant-side residual risk ids for model-validation handoff."""
    risk_assumptions = {
        "multiple_valid_methods_available",
        "multi_asset_context",
        "schedule_dependent_product",
        "path_dependent_product",
        "early_exercise_sensitive_product",
        "path_sampling_required",
        "boundary_conditions_required",
        "exercise_and_cashflow_discretization_acceptable",
        "dependence_modeling_route",
        "cashflow_schedule_required",
        "local_vol_surface_driven_context",
    }
    risks = [
        f"quant:{assumption}"
        for assumption in assumption_basis
        if assumption in risk_assumptions
    ]
    if sensitivity_support is not None and sensitivity_support.level == "experimental":
        risks.append("quant:experimental_sensitivity_support")
    return _merge_unique_strings(tuple(risks))


def _build_challenger_packet(
    *,
    selected_method: str,
    selection_reason: str,
    assumption_basis: tuple[str, ...],
    required_market_data,
    candidate_methods: tuple[str, ...] | list[str] | None = None,
    requested_measures: tuple[str, ...] | list[str] | None = None,
    sensitivity_support: SensitivitySupport | None = None,
) -> QuantChallengerPacket:
    """Build the stable quant packet consumed by review and validation."""
    method = normalize_method(selected_method)
    requested = normalize_requested_measures(requested_measures)
    ordered_candidates = _ordered_candidate_methods(method, candidate_methods)
    normalized_assumptions = _normalize_string_tuple(assumption_basis)
    return QuantChallengerPacket(
        selected_method=method,
        method_identity=method,
        route_family=method,
        selection_reason=selection_reason or "pricing_plan_selection",
        candidate_methods=tuple(
            _candidate_summary_for_method(
                candidate,
                selected_method=method,
                selection_reason=selection_reason or "",
                requested_measures=requested,
                selected_support=sensitivity_support,
            )
            for candidate in ordered_candidates
        ),
        assumption_basis=normalized_assumptions,
        required_market_data=_normalize_string_tuple(sorted(required_market_data or ())),
        requested_measures=_normalize_string_tuple(requested),
        expected_executable_checks=_expected_executable_checks_for_packet(
            candidate_methods=ordered_candidates,
            requested_measures=requested,
        ),
        residual_risk_handoff=_residual_risk_handoff_for_packet(
            assumption_basis=normalized_assumptions,
            sensitivity_support=sensitivity_support,
        ),
    )


def _attach_challenger_packet(
    plan: PricingPlan,
    *,
    candidate_methods: tuple[str, ...] | list[str] | None = None,
    requested_measures: tuple[str, ...] | list[str] | None = None,
) -> PricingPlan:
    """Return ``plan`` with a fresh challenger packet matching its fields."""
    candidates = tuple(candidate_methods or ()) or _candidate_methods_from_packet(
        plan.challenger_packet
    ) or (plan.method,)
    packet = _build_challenger_packet(
        selected_method=plan.method,
        selection_reason=plan.selection_reason,
        assumption_basis=plan.assumption_summary,
        required_market_data=plan.required_market_data,
        candidate_methods=candidates,
        requested_measures=requested_measures,
        sensitivity_support=plan.sensitivity_support,
    )
    return replace(plan, challenger_packet=packet)


def quant_challenger_packet_summary(
    pricing_plan: PricingPlan | None = None,
    *,
    packet: QuantChallengerPacket | Mapping[str, object] | None = None,
    deterministic_check_ids: tuple[str, ...] | list[str] | None = None,
    validation_contract_id: str | None = None,
    route_family: str | None = None,
    residual_risks: tuple[str, ...] | list[str] | None = None,
) -> dict[str, object]:
    """Return a YAML-safe quant challenger packet summary."""
    raw_packet = packet or getattr(pricing_plan, "challenger_packet", None)
    if raw_packet is None and pricing_plan is not None:
        selected_method = getattr(pricing_plan, "method", "")
        raw_packet = _build_challenger_packet(
            selected_method=selected_method,
            selection_reason=getattr(pricing_plan, "selection_reason", ""),
            assumption_basis=tuple(getattr(pricing_plan, "assumption_summary", ()) or ()),
            required_market_data=getattr(pricing_plan, "required_market_data", ()) or (),
            candidate_methods=(selected_method,),
            sensitivity_support=getattr(pricing_plan, "sensitivity_support", None),
        )
    if raw_packet is None:
        return {}

    if isinstance(raw_packet, QuantChallengerPacket):
        payload = {
            "selected_method": raw_packet.selected_method,
            "method_identity": raw_packet.method_identity,
            "route_family": raw_packet.route_family,
            "selection_reason": raw_packet.selection_reason,
            "candidate_methods": [
                {
                    "method": candidate.method,
                    "status": candidate.status,
                    "rejection_reason": candidate.rejection_reason,
                    "priority_rank": candidate.priority_rank,
                    "sensitivity_level": candidate.sensitivity_level,
                    "supported_measures": list(candidate.supported_measures),
                }
                for candidate in raw_packet.candidate_methods
            ],
            "assumption_basis": list(raw_packet.assumption_basis),
            "required_market_data": list(raw_packet.required_market_data),
            "requested_measures": list(raw_packet.requested_measures),
            "expected_executable_checks": list(raw_packet.expected_executable_checks),
            "residual_risk_handoff": list(raw_packet.residual_risk_handoff),
        }
    else:
        payload = dict(raw_packet)
        payload["candidate_methods"] = [
            dict(candidate) if isinstance(candidate, Mapping) else {
                "method": getattr(candidate, "method", ""),
                "status": getattr(candidate, "status", ""),
                "rejection_reason": getattr(candidate, "rejection_reason", ""),
            }
            for candidate in payload.get("candidate_methods", ()) or ()
        ]
        for key in (
            "assumption_basis",
            "required_market_data",
            "requested_measures",
            "expected_executable_checks",
            "residual_risk_handoff",
        ):
            payload[key] = list(payload.get(key) or ())

    if route_family:
        payload["route_family"] = route_family
    if deterministic_check_ids:
        payload["expected_executable_checks"] = list(
            _merge_unique_strings(
                tuple(payload.get("expected_executable_checks") or ()),
                _normalize_string_tuple(deterministic_check_ids),
            )
        )
    if residual_risks:
        payload["residual_risk_handoff"] = list(
            _merge_unique_strings(
                tuple(payload.get("residual_risk_handoff") or ()),
                tuple(f"validation:{risk}" for risk in _normalize_string_tuple(residual_risks)),
            )
        )
    if validation_contract_id:
        payload["validation_contract_id"] = validation_contract_id
    return payload


def _assumption_summary_for_method(
    method: str,
    *,
    candidate_methods: tuple[str, ...] | list[str] | None = None,
    product_ir=None,
    context_tags: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Return the stable assumption basis for one selected pricing method."""
    normalized_method = normalize_method(method)
    assumptions = list(
        _METHOD_ASSUMPTION_SUMMARIES.get(
            normalized_method,
            ("simplest_valid_assumption_set",),
        )
    )
    if candidate_methods and len(tuple(candidate_methods)) > 1:
        assumptions.append("multiple_valid_methods_available")
    if product_ir is not None:
        if getattr(product_ir, "multi_asset", False):
            assumptions.append("multi_asset_context")
        if getattr(product_ir, "schedule_dependence", False):
            assumptions.append("schedule_dependent_product")
        if getattr(product_ir, "state_dependence", "") == "path_dependent":
            assumptions.append("path_dependent_product")
        if getattr(product_ir, "exercise_style", "") in {"american", "bermudan"}:
            assumptions.append("early_exercise_sensitive_product")
    if context_tags:
        assumptions.extend(context_tags)
    return _merge_unique_strings(assumptions)


def _selection_reason_for_method(
    *,
    explicit_preference: bool = False,
    requested_measures: tuple[str, ...] | list[str] | None = None,
    candidate_methods: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Return the canonical explanation label for one pricing-method choice."""
    if explicit_preference:
        return "explicit_preference"
    if requested_measures:
        return "measure_priority"
    candidate_count = len(tuple(candidate_methods or ()))
    if candidate_count > 1:
        return "simplest_valid_default"
    if candidate_count == 1:
        return "canonical_method"
    return "fallback_default"


def _candidate_methods_from_engine_families(
    candidate_engine_families: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Map candidate engine-family labels onto canonical method names."""
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
        "copula": "copula",
        "waterfall": "waterfall",
        "qmc": "qmc",
    }
    candidates: list[str] = []
    for family in candidate_engine_families or ():
        method = mapping.get(family)
        if method and method not in candidates:
            candidates.append(method)
    return tuple(candidates)


def _plan_from_decomposition(decomposition) -> PricingPlan:
    """Convert a canonical product decomposition into a PricingPlan."""
    method = normalize_method(decomposition.method)
    selection_reason = "canonical_decomposition"
    assumption_summary = _assumption_summary_for_method(method)
    plan = PricingPlan(
        method=method,
        method_modules=list(decomposition.method_modules),
        required_market_data=normalize_market_data_requirements(
            decomposition.required_market_data
        ),
        model_to_build=None,
        reasoning=decomposition.reasoning,
        modeling_requirements=tuple(decomposition.modeling_requirements),
        sensitivity_support=support_for_method(method),
        selection_reason=selection_reason,
        assumption_summary=assumption_summary,
    )
    return _attach_challenger_packet(plan, candidate_methods=(method,))


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
    candidate_methods = _candidate_methods_from_engine_families(
        getattr(product_ir, "candidate_engine_families", ())
    )
    if preferred_method:
        method = normalize_method(preferred_method)
    elif _prefer_transform_route_for_requested_sensitivities(
        product_ir,
        candidate_methods=candidate_methods,
        requested_measures=requested,
    ):
        method = "fft_pricing"
    else:
        method = _choose_method_from_candidates(
            candidate_methods,
            requested_measures=requested,
        )
    selection_reason = _selection_reason_for_method(
        explicit_preference=preferred_method is not None,
        requested_measures=requested,
        candidate_methods=candidate_methods,
    )
    assumption_summary = _assumption_summary_for_method(
        method,
        candidate_methods=candidate_methods,
        product_ir=product_ir,
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
        selection_reason=selection_reason,
        assumption_summary=assumption_summary,
    )
    return _apply_contextual_overrides(
        plan,
        context_description,
        instrument_type=getattr(product_ir, "instrument", None),
        candidate_methods=candidate_methods,
        requested_measures=requested,
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
    """Backward-compatible wrapper for candidate-method selection."""
    return _choose_method_from_candidates(
        _candidate_methods_from_engine_families(candidate_engine_families),
        requested_measures=requested_measures,
    )


def _choose_method_from_candidates(
    candidates: tuple[str, ...] | list[str],
    *,
    requested_measures: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Choose the simplest valid method, or the method best aligned with requests."""
    candidate_methods = tuple(normalize_method(method) for method in candidates if normalize_method(method))
    if not candidate_methods:
        return "analytical"
    requested = normalize_requested_measures(requested_measures)
    if requested:
        ranked = sorted(
            candidate_methods,
            key=lambda method: (
                rank_sensitivity_support(
                    support_for_method(method),
                    requested,
                )[:3],
                -_method_priority(method),
            ),
            reverse=True,
        )
        return ranked[0]
    ranked = sorted(candidate_methods, key=_method_priority)
    return ranked[0] if ranked else "analytical"


def _prefer_transform_route_for_requested_sensitivities(
    product_ir,
    *,
    candidate_methods: tuple[str, ...] | list[str],
    requested_measures: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Prefer semi-analytical transform routes for European stochastic-vol Greeks.

    This keeps Heston-like European sensitivity requests on the cheaper
    characteristic-function path when both transform and PDE routes are
    available and otherwise equally capable.
    """
    candidates = {normalize_method(method) for method in candidate_methods if method}
    if "fft_pricing" not in candidates or "pde_solver" not in candidates:
        return False
    requested = set(normalize_requested_measures(requested_measures))
    if not requested.intersection({"vega", "vanna", "volga", "vomma"}):
        return False

    instrument = str(getattr(product_ir, "instrument", "") or "").strip().lower()
    exercise_style = str(getattr(product_ir, "exercise_style", "") or "").strip().lower()
    payoff_traits = {
        str(trait).strip().lower()
        for trait in (getattr(product_ir, "payoff_traits", ()) or ())
        if str(trait).strip()
    }
    state_dependence = str(getattr(product_ir, "state_dependence", "") or "").strip().lower()

    stochastic_vol_context = instrument == "heston_option" or (
        {"stochastic_vol", "heston"} & payoff_traits
    )
    european_context = not exercise_style or exercise_style == "european"
    path_simple_context = state_dependence in {"", "terminal", "terminal_markov", "state_only"}
    return stochastic_vol_context and european_context and path_simple_context


def known_methods() -> tuple[str, ...]:
    """Return the canonical method-family labels understood by the quant agent."""
    return tuple(sorted(CANONICAL_METHODS))


def _apply_contextual_overrides(
    plan: PricingPlan,
    description: str | None,
    *,
    instrument_type: str | None = None,
    candidate_methods: tuple[str, ...] | list[str] | None = None,
    requested_measures: tuple[str, ...] | list[str] | None = None,
) -> PricingPlan:
    """Apply conservative context-derived overrides without changing the core ontology."""
    plan = _apply_local_vol_overrides(
        plan,
        description,
        instrument_type=instrument_type,
        requested_measures=requested_measures,
    )
    if not _looks_like_fx_option(description, instrument_type=instrument_type):
        return _attach_challenger_packet(
            plan,
            candidate_methods=candidate_methods,
            requested_measures=requested_measures,
        )

    required_market_data = normalize_market_data_requirements(
        set(plan.required_market_data)
        | {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"}
    )
    reasoning = plan.reasoning
    fx_reason = "fx_vanilla_context_requires_garman_kohlhagen_inputs"
    if fx_reason not in reasoning:
        reasoning = f"{reasoning}; {fx_reason}" if reasoning else fx_reason
    selection_reason = _append_selection_reason(plan.selection_reason, "fx_context_override")
    assumption_summary = _merge_unique_strings(
        plan.assumption_summary,
        ("fx_cross_currency_context", "garman_kohlhagen_or_equivalent_context"),
    )
    return _attach_challenger_packet(
        replace(
            plan,
            required_market_data=required_market_data,
            reasoning=reasoning,
            selection_reason=selection_reason,
            assumption_summary=assumption_summary,
        ),
        candidate_methods=candidate_methods,
        requested_measures=requested_measures,
    )


def _append_selection_reason(existing: str, note: str) -> str:
    """Append a compact explanation note while preserving the prior label."""
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"


def _apply_local_vol_overrides(
    plan: PricingPlan,
    description: str | None,
    *,
    instrument_type: str | None = None,
    requested_measures: tuple[str, ...] | list[str] | None = None,
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
    selection_reason = _append_selection_reason(plan.selection_reason, "local_vol_context_override")
    preserved_context_tags = tuple(
        tag
        for tag in plan.assumption_summary
        if tag in {
            "multiple_valid_methods_available",
            "multi_asset_context",
            "schedule_dependent_product",
            "path_dependent_product",
            "early_exercise_sensitive_product",
        }
    )
    assumption_summary = _merge_unique_strings(
        _assumption_summary_for_method(
            method,
            context_tags=("local_vol_surface_driven_context",),
        ),
        preserved_context_tags,
    )
    return _attach_challenger_packet(
        replace(
            plan,
            method=method,
            method_modules=method_modules,
            required_market_data=required_market_data,
            reasoning=reasoning,
            sensitivity_support=support_for_method(method),
            selection_reason=selection_reason,
            assumption_summary=assumption_summary,
        ),
        candidate_methods=_merge_unique_strings(
            _candidate_methods_from_packet(plan.challenger_packet),
            (plan.method, method),
        ),
        requested_measures=requested_measures,
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
