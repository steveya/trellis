"""Compiler for validated semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from trellis.agent.knowledge.decompose import build_product_ir
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.quant import select_pricing_method_for_product_ir
from trellis.agent.semantic_contract_validation import validate_semantic_contract
from trellis.agent.sensitivity_support import (
    normalize_requested_measures,
    rank_sensitivity_support,
    support_for_method,
)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Convert a mutable dict (or None) into a read-only MappingProxyType for use in frozen dataclasses."""
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class SemanticImplementationBlueprint:
    """Deterministic blueprint emitted from a validated semantic contract."""

    semantic_id: str
    contract: object
    product_ir: object
    pricing_plan: object | None
    preferred_method: str
    candidate_methods: tuple[str, ...]
    required_market_data: tuple[str, ...]
    derivable_market_data: tuple[str, ...]
    route_modules: tuple[str, ...] = ()
    selection_reason: str = ""
    assumption_summary: tuple[str, ...] = ()
    connector_binding_hints: Mapping[str, object] = field(default_factory=dict)
    estimation_hints: Mapping[str, object] = field(default_factory=dict)
    spec_schema_hint: str | None = None
    primitive_routes: tuple[str, ...] = ()
    adapter_steps: tuple[str, ...] = ()
    validation_bundle_hint: str | None = None
    target_modules: tuple[str, ...] = ()
    proving_tasks: tuple[str, ...] = ()
    unsupported_paths: tuple[str, ...] = ()
    requested_measures: tuple[str, ...] = ()
    measure_support_warnings: tuple[str, ...] = ()
    event_machine_skeleton: str | None = None
    calibration_step: object | None = None  # CalibrationContract when present

    def __post_init__(self):
        """Freeze mapping metadata for stable traces and tests."""
        object.__setattr__(self, "connector_binding_hints", _freeze_mapping(self.connector_binding_hints))
        object.__setattr__(self, "estimation_hints", _freeze_mapping(self.estimation_hints))


def compile_semantic_contract(
    spec,
    *,
    requested_measures: tuple[str, ...] | list[str] | None = None,
    preferred_method: str | None = None,
) -> SemanticImplementationBlueprint:
    """Compile a validated semantic contract into a deterministic blueprint."""
    report = validate_semantic_contract(spec)
    if not report.ok or report.normalized_contract is None:
        joined = "; ".join(report.errors) or "unknown validation error"
        raise ValueError(f"Cannot compile invalid semantic contract: {joined}")

    contract = report.normalized_contract
    preferred_method = _select_preferred_method(
        contract,
        requested_measures=requested_measures,
        preferred_method=preferred_method,
    )
    product_ir = build_product_ir(
        description=contract.description or contract.semantic_id,
        instrument=contract.product.instrument_class,
        payoff_family=contract.product.payoff_family,
        payoff_traits=contract.product.payoff_traits,
        exercise_style=contract.product.exercise_style,
        state_dependence=contract.product.state_dependence,
        schedule_dependence=contract.product.schedule_dependence,
        model_family=contract.product.model_family,
        candidate_engine_families=_engine_families_from_methods(contract.methods.candidate_methods),
        required_market_data=frozenset(_required_capabilities(contract)),
        reusable_primitives=contract.blueprint.target_modules,
        supported=not bool(contract.blueprint.blocked_by),
        preferred_method=preferred_method,
        event_machine=getattr(contract.product, "event_machine", None),
    )
    pricing_plan = select_pricing_method_for_product_ir(
        product_ir,
        preferred_method=preferred_method,
        requested_measures=requested_measures,
        context_description=contract.description,
    )
    _calibration_modules: tuple[str, ...] = ()
    _calibration = getattr(contract, "calibration", None)
    if _calibration is not None:
        _prim = getattr(_calibration, "proven_primitive", "")
        if _prim:
            from trellis.agent.calibration_contract import _KNOWN_PRIMITIVES
            _cal_mod = _KNOWN_PRIMITIVES.get(_prim, "")
            if _cal_mod:
                _calibration_modules = (_cal_mod,)
    route_modules = tuple(
        dict.fromkeys((*pricing_plan.method_modules, *contract.blueprint.target_modules, *_calibration_modules))
    )
    # Normalize requested measures and check support coverage
    normalized_measures = tuple(normalize_requested_measures(requested_measures))
    measure_warnings: list[str] = []
    if normalized_measures and pricing_plan.sensitivity_support is not None:
        supported = set(pricing_plan.sensitivity_support.supported_measures)
        for m in normalized_measures:
            m_val = m.value  # DslMeasure is a str-enum; .value is the canonical lowercase key
            if m_val not in supported:
                measure_warnings.append(
                    f"Requested measure '{m_val}' is not in {preferred_method}'s "
                    f"supported set {sorted(supported)}; analytics engine "
                    f"will attempt bump-and-reprice"
                )

    return SemanticImplementationBlueprint(
        semantic_id=contract.semantic_id,
        contract=contract,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        preferred_method=preferred_method,
        candidate_methods=tuple(contract.methods.candidate_methods),
        required_market_data=tuple(item.input_id for item in contract.market_data.required_inputs),
        derivable_market_data=tuple(contract.market_data.derivable_inputs),
        route_modules=route_modules,
        selection_reason=pricing_plan.selection_reason,
        assumption_summary=tuple(pricing_plan.assumption_summary),
        connector_binding_hints=_connector_binding_hints(contract),
        estimation_hints=_estimation_hints(contract),
        spec_schema_hint=_spec_schema_hint(contract, preferred_method),
        primitive_routes=_primitive_routes(contract),
        adapter_steps=tuple(contract.blueprint.adapter_obligations),
        validation_bundle_hint=contract.validation.bundle_hints[0] if contract.validation.bundle_hints else None,
        target_modules=tuple(contract.blueprint.target_modules),
        proving_tasks=tuple(contract.blueprint.proving_tasks),
        unsupported_paths=tuple(dict.fromkeys((*contract.methods.unsupported_variants, *contract.blueprint.blocked_by))),
        requested_measures=normalized_measures,
        measure_support_warnings=tuple(measure_warnings),
        event_machine_skeleton=_emit_event_skeleton(contract),
        calibration_step=getattr(contract, "calibration", None),
    )


def _select_preferred_method(
    contract,
    *,
    requested_measures=None,
    preferred_method: str | None = None,
) -> str:
    """Select the preferred method using the current sensitivity policy."""
    requested = normalize_requested_measures(requested_measures)
    if preferred_method:
        normalized = normalize_method(preferred_method)
        if normalized not in contract.methods.candidate_methods:
            raise ValueError(
                f"Preferred method `{normalized}` is not a candidate for semantic `{contract.semantic_id}`."
            )
        return normalized
    if not requested:
        if contract.methods.preferred_method:
            return contract.methods.preferred_method
        if contract.methods.reference_methods:
            return contract.methods.reference_methods[0]
        if contract.methods.production_methods:
            return contract.methods.production_methods[0]
        return contract.methods.candidate_methods[0]

    ranked = max(
        enumerate(contract.methods.candidate_methods),
        key=lambda item: (
            rank_sensitivity_support(
                support_for_method(item[1]),
                requested,
            ),
            -item[0],
        ),
    )
    return ranked[1]


def _required_capabilities(contract) -> tuple[str, ...]:
    """Return canonical MarketState capabilities required by the contract."""
    capabilities: list[str] = []
    for item in contract.market_data.required_inputs:
        if item.capability and item.capability not in capabilities:
            capabilities.append(item.capability)
    return tuple(capabilities)


def _connector_binding_hints(contract) -> dict[str, object]:
    """Return per-input binding hints for runtime connector resolution."""
    return {
        item.input_id: {
            "capability": item.capability,
            "aliases": list(item.aliases),
            "connector_hint": item.connector_hint,
            "allowed_provenance": list(item.allowed_provenance),
        }
        for item in (*contract.market_data.required_inputs, *contract.market_data.optional_inputs)
    }


def _estimation_hints(contract) -> dict[str, object]:
    """Return estimation and provenance policy hints."""
    return {
        "derivable_inputs": list(contract.market_data.derivable_inputs),
        "estimation_policy": list(contract.market_data.estimation_policy),
        "provenance_requirements": list(contract.market_data.provenance_requirements),
        "missing_data_error_policy": list(contract.market_data.missing_data_error_policy),
    }


def _spec_schema_hint(contract, preferred_method: str) -> str | None:
    """Select the most relevant spec-schema hint for the chosen method."""
    hints = contract.blueprint.spec_schema_hints
    if not hints:
        return None
    for hint in hints:
        lowered = hint.lower()
        if preferred_method == "analytical" and "analytical" in lowered:
            return hint
        if preferred_method == "monte_carlo" and "monte_carlo" in lowered:
            return hint
    return hints[0]


def _primitive_routes(contract) -> tuple[str, ...]:
    """Return the deterministic primitive-route hints for the contract."""
    routes = list(contract.blueprint.primitive_families)
    if contract.product.payoff_family == "basket_path_payoff" and "correlated_basket_monte_carlo" not in routes:
        routes.append("correlated_basket_monte_carlo")
    return tuple(routes)


def _engine_families_from_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    """Map method families onto ProductIR engine-family hints."""
    mapping = {
        "analytical": ("analytical",),
        "rate_tree": ("lattice",),
        "monte_carlo": ("monte_carlo",),
        "qmc": ("qmc",),
        "pde_solver": ("pde",),
        "fft_pricing": ("transforms",),
        "copula": ("copula",),
        "waterfall": ("cashflow",),
    }
    families: list[str] = []
    for method in methods:
        for family in mapping.get(method, ()):
            if family not in families:
                families.append(family)
    return tuple(families)


def _emit_event_skeleton(contract) -> str | None:
    """Emit an event machine skeleton if the contract has one."""
    machine = getattr(getattr(contract, "product", None), "event_machine", None)
    if machine is None:
        return None
    try:
        from trellis.agent.event_machine import emit_event_machine_skeleton
        return emit_event_machine_skeleton(machine)
    except Exception:
        return None
