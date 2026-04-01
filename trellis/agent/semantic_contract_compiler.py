"""Compiler for validated semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from trellis.agent.codegen_guardrails import rank_primitive_routes
from trellis.agent.knowledge.decompose import build_product_ir
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.market_binding import (
    build_market_binding_spec,
    build_required_data_spec,
)
from trellis.agent.quant import select_pricing_method_for_product_ir
from trellis.agent.dsl_lowering import lower_semantic_blueprint
from trellis.agent.semantic_contract_validation import validate_semantic_contract
from trellis.agent.sensitivity_support import (
    normalize_requested_measures,
    normalize_requested_outputs,
    rank_sensitivity_support,
    support_for_method,
)
from trellis.agent.valuation_context import normalize_valuation_context


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Convert a mutable dict (or None) into a read-only MappingProxyType for use in frozen dataclasses."""
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class SemanticImplementationBlueprint:
    """Deterministic blueprint emitted from a validated semantic contract.

    The blueprint now carries two related views:

    - route/module hints for the existing build pipeline
    - a conservative `dsl_lowering` companion object that lowers supported
      semantic routes onto the semiring/Bellman DSL and checked-in helper
      targets
    """

    semantic_id: str
    contract: object
    product_ir: object
    pricing_plan: object | None
    preferred_method: str
    candidate_methods: tuple[str, ...]
    required_market_data: tuple[str, ...]
    derivable_market_data: tuple[str, ...]
    valuation_context: object | None = None
    required_data_spec: object | None = None
    market_binding_spec: object | None = None
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
    requested_outputs: tuple[str, ...] = ()
    requested_measures: tuple[str, ...] = ()
    measure_support_warnings: tuple[str, ...] = ()
    event_machine_skeleton: str | None = None
    calibration_step: object | None = None  # CalibrationContract when present
    dsl_lowering: object | None = None

    def __post_init__(self):
        """Freeze mapping metadata for stable traces and tests."""
        object.__setattr__(self, "connector_binding_hints", _freeze_mapping(self.connector_binding_hints))
        object.__setattr__(self, "estimation_hints", _freeze_mapping(self.estimation_hints))


def compile_semantic_contract(
    spec,
    *,
    valuation_context=None,
    requested_outputs: tuple[str, ...] | list[str] | None = None,
    requested_measures: tuple[str, ...] | list[str] | None = None,
    preferred_method: str | None = None,
) -> SemanticImplementationBlueprint:
    """Compile a validated semantic contract into a deterministic blueprint.

    The result preserves the existing route/module selection fields and also
    attaches a conservative DSL-lowering companion for helper-backed routes.
    Unsupported lowering paths remain explicit through
    ``blueprint.dsl_lowering.admissibility_errors``.
    """
    report = validate_semantic_contract(spec)
    if not report.ok or report.normalized_contract is None:
        joined = "; ".join(report.errors) or "unknown validation error"
        raise ValueError(f"Cannot compile invalid semantic contract: {joined}")

    contract = report.normalized_contract
    resolved_valuation_context = normalize_valuation_context(
        valuation_context,
        requested_outputs=requested_outputs,
        requested_measures=requested_measures,
        reporting_currency=(
            getattr(getattr(contract.product, "conventions", None), "reporting_currency", "")
            or getattr(getattr(contract.product, "conventions", None), "payment_currency", "")
            or ""
        ),
    )
    required_data_spec = build_required_data_spec(contract)
    market_binding_spec = build_market_binding_spec(
        contract,
        valuation_context=resolved_valuation_context,
        required_data_spec=required_data_spec,
    )
    preferred_method = _select_preferred_method(
        contract,
        requested_measures=resolved_valuation_context.requested_outputs,
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
        required_market_data=frozenset(required_data_spec.required_capabilities),
        reusable_primitives=contract.blueprint.target_modules,
        supported=not bool(contract.blueprint.blocked_by),
        preferred_method=preferred_method,
        event_machine=getattr(contract.product, "event_machine", None),
    )
    pricing_plan = select_pricing_method_for_product_ir(
        product_ir,
        preferred_method=preferred_method,
        requested_measures=resolved_valuation_context.requested_outputs,
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
    # Normalize requested measures and check support coverage
    normalized_outputs = tuple(
        normalize_requested_outputs(resolved_valuation_context.requested_outputs)
    )
    normalized_measures = tuple(normalize_requested_measures(normalized_outputs))
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

    primitive_routes = _primitive_routes(
        contract,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
    )
    dsl_lowering = lower_semantic_blueprint(
        contract,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        primitive_routes=primitive_routes,
        valuation_context=resolved_valuation_context,
        market_binding_spec=market_binding_spec,
    )
    route_modules = tuple(
        dict.fromkeys(
            (
                *pricing_plan.method_modules,
                *contract.blueprint.target_modules,
                *getattr(dsl_lowering, "helper_modules", ()),
                *_calibration_modules,
            )
        )
    )

    return SemanticImplementationBlueprint(
        semantic_id=contract.semantic_id,
        contract=contract,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        preferred_method=preferred_method,
        candidate_methods=tuple(contract.methods.candidate_methods),
        required_market_data=required_data_spec.required_input_ids,
        derivable_market_data=required_data_spec.derivable_inputs,
        valuation_context=resolved_valuation_context,
        required_data_spec=required_data_spec,
        market_binding_spec=market_binding_spec,
        route_modules=route_modules,
        selection_reason=pricing_plan.selection_reason,
        assumption_summary=tuple(pricing_plan.assumption_summary),
        connector_binding_hints=_connector_binding_hints(market_binding_spec),
        estimation_hints=_estimation_hints(required_data_spec),
        spec_schema_hint=_spec_schema_hint(contract, preferred_method),
        primitive_routes=primitive_routes,
        adapter_steps=tuple(contract.blueprint.adapter_obligations),
        validation_bundle_hint=contract.validation.bundle_hints[0] if contract.validation.bundle_hints else None,
        target_modules=tuple(contract.blueprint.target_modules),
        proving_tasks=tuple(contract.blueprint.proving_tasks),
        unsupported_paths=tuple(dict.fromkeys((*contract.methods.unsupported_variants, *contract.blueprint.blocked_by))),
        requested_outputs=normalized_outputs,
        requested_measures=normalized_measures,
        measure_support_warnings=tuple(measure_warnings),
        event_machine_skeleton=_emit_event_skeleton(contract),
        calibration_step=getattr(contract, "calibration", None),
        dsl_lowering=dsl_lowering,
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


def _connector_binding_hints(market_binding_spec) -> dict[str, object]:
    """Return backward-compatible runtime binding hints from compiled bindings."""
    if market_binding_spec is None:
        return {}
    return market_binding_spec.to_connector_binding_hints()


def _estimation_hints(required_data_spec) -> dict[str, object]:
    """Return backward-compatible estimation hints from the compiled data spec."""
    if required_data_spec is None:
        return {}
    return required_data_spec.to_estimation_hints()


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


def _primitive_routes(
    contract,
    *,
    product_ir,
    pricing_plan,
) -> tuple[str, ...]:
    """Return deterministic primitive-route hints aligned with route ranking.

    The semantic compiler used to rely only on the static blueprint
    ``primitive_families`` hints. That made lowering insensitive to the
    selected method, for example keeping vanilla-option PDE requests pinned to
    ``analytical_black76``. Reuse the live primitive-plan ranking so semantic
    blueprints and generation plans expose the same route ordering.
    """
    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )
    routes: list[str] = []
    if ranked:
        routes.append(ranked[0].route)
    else:
        routes.extend(contract.blueprint.primitive_families)
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
