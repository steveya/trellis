"""Compiler for validated semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import logging
import re
from types import MappingProxyType
from typing import Mapping

from trellis.agent.codegen_guardrails import rank_primitive_routes
from trellis.agent.contract_ir import ContractIR
from trellis.agent.knowledge.decompose import (
    build_product_ir,
    decompose_to_contract_ir,
    decompose_to_static_leg_contract_ir,
)
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.market_binding import (
    build_market_binding_spec,
    build_required_data_spec,
)
from trellis.agent.quant import select_pricing_method_for_product_ir
from trellis.agent.dsl_lowering import lower_semantic_blueprint
from trellis.agent.lane_obligations import compile_lane_construction_plan
from trellis.agent.semantic_contract_validation import validate_semantic_contract
from trellis.agent.sensitivity_support import (
    normalize_requested_measures,
    normalize_requested_outputs,
    rank_sensitivity_support,
    support_for_method,
)
from trellis.agent.valuation_context import normalize_valuation_context

_LOG = logging.getLogger(__name__)


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
    lane_plan: object | None = None
    contract_ir: ContractIR | None = None
    contract_ir_solver_selection: object | None = None
    contract_ir_solver_shadow: object | None = None
    static_leg_contract_ir: object | None = None
    dynamic_contract_ir: object | None = None
    static_leg_lowering_selection: object | None = None
    static_leg_admission_blockers: tuple[object, ...] = ()

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
    preferred_method = _select_preferred_method(
        contract,
        requested_measures=resolved_valuation_context.requested_outputs,
        preferred_method=preferred_method,
    )
    contract = _specialize_contract_for_preferred_method(contract, preferred_method)
    required_data_spec = build_required_data_spec(contract)
    market_binding_spec = build_market_binding_spec(
        contract,
        valuation_context=resolved_valuation_context,
        required_data_spec=required_data_spec,
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
        derivative_family=getattr(contract.product, "derivative_family", ""),
        underlying_asset_class=getattr(
            getattr(contract.product, "underlying", None),
            "asset_class",
            "",
        ),
        underlying_identifiers=getattr(
            getattr(contract.product, "underlying", None),
            "identifiers",
            (),
        ),
        option_type=getattr(contract.product, "option_type", ""),
    )
    product_ir = _augment_product_ir_with_contract_route_hints(product_ir, contract)
    contract_ir = None
    try:
        contract_ir = decompose_to_contract_ir(
            contract.description or contract.semantic_id,
            instrument_type=(
                getattr(getattr(contract, "product", None), "instrument_class", None)
                or getattr(product_ir, "instrument", None)
            ),
            product_ir=product_ir,
        )
    except Exception:
        _LOG.warning(
            "Contract IR decomposition failed for semantic contract %s",
            getattr(contract, "semantic_id", "<unknown>"),
            exc_info=True,
        )
    static_leg_contract_ir = None
    try:
        static_leg_contract_ir = _lower_static_leg_contract_ir_from_semantic_contract(
            contract
        )
        if static_leg_contract_ir is None:
            static_leg_contract_ir = decompose_to_static_leg_contract_ir(
                contract.description or contract.semantic_id,
                instrument_type=(
                    getattr(getattr(contract, "product", None), "instrument_class", None)
                    or getattr(product_ir, "instrument", None)
                ),
                product_ir=product_ir,
            )
    except Exception:
        _LOG.warning(
            "Static-leg Contract IR decomposition failed for semantic contract %s",
            getattr(contract, "semantic_id", "<unknown>"),
            exc_info=True,
        )
    dynamic_contract_ir = None
    try:
        dynamic_contract_ir = _lower_dynamic_contract_ir_from_semantic_contract(
            contract,
            static_leg_contract_ir=static_leg_contract_ir,
        )
    except Exception:
        _LOG.warning(
            "Dynamic Contract IR decomposition failed for semantic contract %s",
            getattr(contract, "semantic_id", "<unknown>"),
            exc_info=True,
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

    legacy_primitive_routes = _primitive_routes(
        contract,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
    )
    legacy_dsl_lowering = lower_semantic_blueprint(
        contract,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        primitive_routes=legacy_primitive_routes,
        valuation_context=resolved_valuation_context,
        market_binding_spec=market_binding_spec,
    )
    contract_ir_solver_selection = _compile_contract_ir_solver_selection(
        contract=contract,
        contract_ir=contract_ir,
        preferred_method=preferred_method,
        requested_outputs=normalized_outputs,
        valuation_context=resolved_valuation_context,
    )
    static_leg_lowering_selection = _compile_static_leg_lowering_selection(
        static_leg_contract_ir=static_leg_contract_ir,
        preferred_method=preferred_method,
    )
    static_leg_admission_blockers = _compile_static_leg_admission_blockers(
        static_leg_contract_ir
    )
    if contract_ir_solver_selection is not None and bool(
        getattr(contract_ir_solver_selection, "generated_route_authority", True)
    ):
        primitive_routes = ()
        dsl_lowering = _route_free_lowering_from_structural_selection(
            contract_ir_solver_selection,
            preferred_method=preferred_method,
            fallback_lowering=legacy_dsl_lowering,
            selection_note_prefix="contract_ir_selection",
        )
    elif static_leg_lowering_selection is not None:
        primitive_routes = ()
        dsl_lowering = _route_free_lowering_from_structural_selection(
            static_leg_lowering_selection,
            preferred_method=preferred_method,
            fallback_lowering=legacy_dsl_lowering,
            selection_note_prefix="static_leg_selection",
        )
    else:
        primitive_routes = legacy_primitive_routes
        dsl_lowering = legacy_dsl_lowering
    lane_plan = compile_lane_construction_plan(
        preferred_method=preferred_method,
        required_market_data=required_data_spec.required_input_ids,
        dsl_lowering=dsl_lowering,
        unsupported_paths=tuple(
            dict.fromkeys(
                (
                    *contract.methods.unsupported_variants,
                    *contract.blueprint.blocked_by,
                )
            )
        ),
    )
    route_method_modules = tuple(pricing_plan.method_modules)
    if (
        not primitive_routes
        and (
            static_leg_lowering_selection is not None
            or contract_ir_solver_selection is None
        )
    ):
        route_method_modules = ()

    route_modules = tuple(
        dict.fromkeys(
            (
                *route_method_modules,
                *contract.blueprint.target_modules,
                *getattr(dsl_lowering, "helper_modules", ()),
                *_calibration_modules,
            )
        )
    )
    contract_ir_solver_shadow = _compile_contract_ir_solver_shadow(
        contract=contract,
        contract_ir=contract_ir,
        preferred_method=preferred_method,
        requested_outputs=normalized_outputs,
        valuation_context=resolved_valuation_context,
        market_snapshot=getattr(resolved_valuation_context, "market_snapshot", None),
        primitive_routes=legacy_primitive_routes,
        route_modules=route_modules,
        dsl_lowering=legacy_dsl_lowering,
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
        lane_plan=lane_plan,
        contract_ir=contract_ir,
        contract_ir_solver_selection=contract_ir_solver_selection,
        contract_ir_solver_shadow=contract_ir_solver_shadow,
        static_leg_contract_ir=static_leg_contract_ir,
        dynamic_contract_ir=dynamic_contract_ir,
        static_leg_lowering_selection=static_leg_lowering_selection,
        static_leg_admission_blockers=static_leg_admission_blockers,
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


def _specialize_contract_for_preferred_method(contract, preferred_method: str):
    """Rebuild contracts whose blueprint/helper surface depends on method selection."""
    from trellis.agent.semantic_contracts import specialize_semantic_contract_for_method

    return specialize_semantic_contract_for_method(
        contract,
        preferred_method=preferred_method,
    )


def _lower_static_leg_contract_ir_from_semantic_contract(contract):
    """Lower semantic contracts whose normalized fields define static-leg IR."""

    semantic_id = str(getattr(contract, "semantic_id", "") or "").strip().lower()
    product = getattr(contract, "product", None)
    product_semantic_id = str(getattr(product, "semantic_id", "") or "").strip().lower()
    if semantic_id != "range_accrual" and product_semantic_id != "range_accrual":
        return None
    return _lower_range_accrual_to_static_leg_contract_ir(contract)


def _lower_dynamic_contract_ir_from_semantic_contract(
    contract,
    *,
    static_leg_contract_ir,
):
    """Lower semantic contracts whose dynamic metadata has an admitted wrapper."""

    semantic_id = str(getattr(contract, "semantic_id", "") or "").strip().lower()
    product = getattr(contract, "product", None)
    product_semantic_id = str(getattr(product, "semantic_id", "") or "").strip().lower()
    if semantic_id != "range_accrual" and product_semantic_id != "range_accrual":
        return None
    if static_leg_contract_ir is None:
        return None

    from trellis.agent.dynamic_contract_ir import (
        ActionSpec,
        ControlProgram,
        DecisionEvent,
        DynamicContractIR,
        EventProgram,
        EventTimeBucket,
        TerminationRule,
    )

    term_fields = dict(getattr(product, "term_fields", {}) or {})
    callability = dict(term_fields.get("callability") or {})
    dynamic_features = dict(term_fields.get("dynamic_features") or {})
    if dynamic_features:
        return None
    call_schedule = tuple(
        str(item).strip()
        for item in (
            callability.get("call_schedule")
            or callability.get("call_dates")
            or ()
        )
        if str(item).strip()
    )
    if not call_schedule:
        return None
    call_style = str(callability.get("call_style") or "issuer_callable").strip().lower()
    if call_style not in {"issuer_callable", "issuer_call", "callable"}:
        return None
    try:
        call_dates = tuple(date.fromisoformat(item) for item in call_schedule)
    except ValueError:
        return None

    redeem = ActionSpec("redeem", "terminate", "redeem at par")
    continue_ = ActionSpec("continue", "continue", "continue outstanding")
    buckets = tuple(
        EventTimeBucket(
            event_date=call_date,
            phase_sequence=("decision", "termination"),
            events=(
                DecisionEvent(
                    label=f"call_{call_date.isoformat()}",
                    schedule_role="call_date",
                    action_set=(redeem, continue_),
                    controller_role="issuer",
                ),
            ),
        )
        for call_date in call_dates
    )
    termination_rules = tuple(
        TerminationRule(
            label=f"terminate_{call_date.isoformat()}",
            trigger="action == redeem",
            settlement_expression="par_redemption",
            event_label=f"call_{call_date.isoformat()}",
        )
        for call_date in call_dates
    )
    return DynamicContractIR(
        base_contract=static_leg_contract_ir,
        semantic_family="callable_range_accrual",
        base_track="static_leg",
        event_program=EventProgram(
            buckets=buckets,
            termination_rules=termination_rules,
        ),
        control_program=ControlProgram(
            controller_role="issuer",
            decision_style="bermudan",
            decision_event_labels=tuple(
                f"call_{call_date.isoformat()}" for call_date in call_dates
            ),
            admissible_actions=(redeem, continue_),
        ),
        settlement=static_leg_contract_ir.settlement,
    )


def _lower_range_accrual_to_static_leg_contract_ir(contract):
    """Lower the bounded single-index range-accrual semantic contract to static legs."""

    from trellis.agent.semantic_observables import (
        BetweenPredicate,
        ObservationMetadata,
        RateIndexObservable,
    )
    from trellis.agent.static_leg_contract import (
        ConditionalAccrualLeg,
        ConditionalAccrualPeriod,
        FixedCouponFormula,
        KnownCashflow,
        KnownCashflowLeg,
        NotionalSchedule,
        NotionalStep,
        SettlementRule,
        SignedLeg,
        StaticLegContractIR,
    )

    product = getattr(contract, "product", None)
    term_fields = dict(getattr(product, "term_fields", {}) or {})
    reference_index = str(term_fields.get("reference_index") or "").strip().upper()
    schedule = tuple(str(item).strip() for item in getattr(product, "observation_schedule", ()) or ())
    if not reference_index or len(schedule) < 2:
        return None
    try:
        observation_dates = tuple(date.fromisoformat(item) for item in schedule)
    except ValueError:
        return None
    if any(left >= right for left, right in zip(observation_dates, observation_dates[1:])):
        return None

    coupon_definition = dict(term_fields.get("coupon_definition") or {})
    range_condition = dict(term_fields.get("range_condition") or {})
    settlement_profile = dict(term_fields.get("settlement_profile") or {})
    callability = dict(term_fields.get("callability") or {})
    dynamic_features = dict(term_fields.get("dynamic_features") or {})
    if (
        coupon_definition.get("coupon_rate") is None
        or range_condition.get("lower_bound") is None
        or range_condition.get("upper_bound") is None
    ):
        return None

    first_gap = observation_dates[1] - observation_dates[0]
    accrual_start_dates = (observation_dates[0] - first_gap, *observation_dates[:-1])
    payment_dates = observation_dates
    currency = _range_accrual_currency(contract, reference_index)
    index_name, tenor = _split_range_accrual_reference_index(reference_index)
    principal_redemption = _range_accrual_principal_redemption(
        term_fields,
        settlement_profile,
    )
    day_count = (
        str(getattr(getattr(product, "conventions", None), "day_count_convention", "") or "").strip()
        or "ACT/365"
    )

    spec_fields = {
        "reference_index": reference_index,
        "coupon_rate": float(coupon_definition["coupon_rate"]),
        "lower_bound": float(range_condition["lower_bound"]),
        "upper_bound": float(range_condition["upper_bound"]),
        "observation_dates": schedule,
        "accrual_start_dates": tuple(item.isoformat() for item in accrual_start_dates),
        "payment_dates": tuple(item.isoformat() for item in payment_dates),
        "principal_redemption": principal_redemption,
        "inclusive_lower": bool(range_condition.get("inclusive_lower", True)),
        "inclusive_upper": bool(range_condition.get("inclusive_upper", True)),
        "day_count": day_count,
    }
    notional_schedule = NotionalSchedule(
        (
            NotionalStep(
                start_date=accrual_start_dates[0],
                end_date=observation_dates[-1],
                amount=1.0,
            ),
        )
    )
    condition = BetweenPredicate(
        observable=RateIndexObservable(
            observable_id="reference_rate_fixing",
            index_name=index_name,
            tenor=tenor,
            observation=ObservationMetadata(
                schedule_role="observation_dates",
                fixing_date_role="fixing_dates",
                missing_fixing_policy="project_forward_for_future_only",
            ),
        ),
        lower_bound=float(range_condition["lower_bound"]),
        upper_bound=float(range_condition["upper_bound"]),
        inclusive_lower=bool(range_condition.get("inclusive_lower", True)),
        inclusive_upper=bool(range_condition.get("inclusive_upper", True)),
    )
    periods = tuple(
        ConditionalAccrualPeriod(
            accrual_start=start,
            accrual_end=observation,
            observation_date=observation,
            payment_date=payment,
            fixing_date=observation,
        )
        for start, observation, payment in zip(
            accrual_start_dates,
            observation_dates,
            payment_dates,
        )
    )
    conditional_leg = ConditionalAccrualLeg(
        currency=currency,
        notional_schedule=notional_schedule,
        accrual_periods=periods,
        coupon_formula=FixedCouponFormula(float(coupon_definition["coupon_rate"])),
        day_count=day_count,
        payment_frequency=_infer_range_accrual_frequency(observation_dates),
        accrual_condition_ref="reference_rate_fixing_in_range",
        accrual_counter_ref="in_range_coupon_count",
        settlement_rule=str(
            settlement_profile.get("coupon_settlement", "coupon_period_cash_settlement")
        ).strip()
        or "coupon_period_cash_settlement",
        label="range_accrual_coupon",
        metadata={
            "semantic_family": "range_accrual",
            "lowering_source": "semantic_contract.term_fields",
            "reference_index": reference_index,
            "coupon_definition": coupon_definition,
            "range_condition": range_condition,
            "settlement_profile": settlement_profile,
            "callability": callability,
            "dynamic_features": dynamic_features,
            "range_accrual_spec_fields": spec_fields,
        },
        accrual_condition=condition,
    )
    legs = [SignedLeg(direction="receive", leg=conditional_leg)]
    if principal_redemption != 0.0:
        principal_leg = KnownCashflowLeg(
            currency=currency,
            cashflows=(
                KnownCashflow(
                    payment_date=payment_dates[-1],
                    amount=principal_redemption,
                    currency=currency,
                    label="principal_redemption",
                ),
            ),
            label="range_accrual_principal",
        )
        legs.append(SignedLeg(direction="receive", leg=principal_leg))
    return StaticLegContractIR(
        legs=tuple(legs),
        settlement=SettlementRule(
            payout_currency=currency,
            settlement_kind="cash",
            settlement_lag_days=0,
        ),
        metadata={
            "semantic_id": "range_accrual",
            "semantic_family": "range_accrual",
            "lowering_source": "semantic_contract.term_fields",
            "callability": callability,
            "dynamic_features": dynamic_features,
            "range_accrual_spec_fields": spec_fields,
        },
    )


def _range_accrual_currency(contract, reference_index: str) -> str:
    product = getattr(contract, "product", None)
    conventions = getattr(product, "conventions", None)
    for value in (
        getattr(conventions, "payment_currency", ""),
        getattr(conventions, "reporting_currency", ""),
    ):
        currency = str(value or "").strip().upper()
        if currency:
            return currency
    parts = [part for part in re.split(r"[-_/ ]+", reference_index.upper()) if part]
    if parts and len(parts[0]) == 3:
        return parts[0]
    if reference_index.upper() in {"SOFR", "FF", "FEDFUNDS"}:
        return "USD"
    return "USD"


def _split_range_accrual_reference_index(reference_index: str) -> tuple[str, str]:
    normalized = str(reference_index or "").strip().upper()
    parts = [part for part in re.split(r"[-_/ ]+", normalized) if part]
    if len(parts) >= 2 and re.fullmatch(r"\d+[DWMY]", parts[-1]):
        return "-".join(parts[:-1]), parts[-1]
    return normalized, ""


def _range_accrual_principal_redemption(
    term_fields: Mapping[str, object],
    settlement_profile: Mapping[str, object],
) -> float:
    value = term_fields.get("principal_redemption")
    if value in {None, ""}:
        value = settlement_profile.get("principal_redemption", 1.0)
    return float(value)


def _infer_range_accrual_frequency(observation_dates: tuple[date, ...]) -> str:
    if len(observation_dates) < 2:
        return "scheduled"
    average_days = sum(
        (right - left).days
        for left, right in zip(observation_dates, observation_dates[1:])
    ) / float(len(observation_dates) - 1)
    if average_days <= 40:
        return "monthly"
    if average_days <= 100:
        return "quarterly"
    if average_days <= 200:
        return "semiannual"
    return "annual"


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
    explicit_routes = tuple(contract.blueprint.primitive_families)
    semantic_route_families = tuple(getattr(product_ir, "route_families", ()) or ())
    if not explicit_routes and not semantic_route_families:
        return ()

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )
    routes: list[str] = []
    if explicit_routes:
        routes.extend(explicit_routes)
    elif ranked:
        routes.append(ranked[0].route)
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


def _augment_product_ir_with_contract_route_hints(product_ir, contract):
    """Merge explicit semantic primitive-family hints back into ProductIR authority."""
    primitive_routes = tuple(getattr(getattr(contract, "blueprint", None), "primitive_families", ()) or ())
    if not primitive_routes:
        return product_ir

    from trellis.agent.route_registry import find_route_by_id, load_route_registry, resolve_route_family

    registry = load_route_registry()
    route_families = list(getattr(product_ir, "route_families", ()) or ())
    engine_families = list(getattr(product_ir, "candidate_engine_families", ()) or ())
    preferred_method = str(getattr(getattr(contract, "methods", None), "preferred_method", "") or "").strip() or None
    for route_id in primitive_routes:
        spec = find_route_by_id(route_id, registry)
        if spec is None:
            continue
        resolved_family = str(
            resolve_route_family(
                spec,
                product_ir,
                method=preferred_method,
            )
            or spec.route_family
            or ""
        ).strip()
        if resolved_family and resolved_family not in route_families:
            route_families.append(resolved_family)
        engine_family = str(getattr(spec, "engine_family", "") or "").strip()
        if engine_family and engine_family not in engine_families:
            engine_families.append(engine_family)
    return replace(
        product_ir,
        route_families=tuple(route_families),
        candidate_engine_families=tuple(engine_families),
    )


def _split_import_ref(ref: str) -> tuple[str, str]:
    """Split one fully qualified import ref into module and symbol components."""
    module_name, _, symbol = str(ref or "").rpartition(".")
    return module_name.strip(), symbol.strip()


def _compile_contract_ir_solver_selection(
    *,
    contract,
    contract_ir: ContractIR | None,
    preferred_method: str,
    requested_outputs: tuple[str, ...],
    valuation_context,
):
    """Return the authoritative structural selection when the family is admitted.

    Phase 4 promotes structural selection onto the fresh-build authority path
    only for declarations that the bounded ContractIR compiler can already
    admit. Everything else keeps the legacy route-based lowering.
    """

    if contract_ir is None:
        return None

    from trellis.agent.contract_ir_solver_compiler import (
        ContractIRSolverCompileError,
        build_contract_ir_term_environment,
        select_contract_ir_solver,
    )

    try:
        return select_contract_ir_solver(
            contract_ir,
            term_environment=build_contract_ir_term_environment(contract),
            valuation_context=valuation_context,
            preferred_method=preferred_method,
            requested_outputs=requested_outputs,
        )
    except ContractIRSolverCompileError:
        _LOG.debug(
            "Contract IR structural selection did not admit semantic %s",
            getattr(contract, "semantic_id", "<unknown>"),
            exc_info=True,
        )
        return None


def _route_free_lowering_bindings_from_selection(selection) -> tuple[object, ...]:
    """Project structural selection provenance onto lowering target bindings."""
    from trellis.agent.dsl_lowering import DslTargetBinding

    def _append(
        bindings: list[DslTargetBinding],
        seen: set[tuple[str, str, str]],
        ref: str,
        *,
        role: str,
    ) -> None:
        module_name, symbol = _split_import_ref(ref)
        if not module_name or not symbol:
            return
        key = (module_name, symbol, role)
        if key in seen:
            return
        seen.add(key)
        bindings.append(DslTargetBinding(module=module_name, symbol=symbol, role=role))

    bindings: list[DslTargetBinding] = []
    seen: set[tuple[str, str, str]] = set()
    callable_role = "route_helper" if str(getattr(selection, "call_style", "") or "") == "helper_call" else "pricing_kernel"
    _append(
        bindings,
        seen,
        str(getattr(selection, "callable_ref", "") or ""),
        role=callable_role,
    )
    for role, refs in (
        ("route_helper", getattr(selection, "helper_refs", ())),
        ("pricing_kernel", getattr(selection, "pricing_kernel_refs", ())),
        ("schedule_builder", getattr(selection, "schedule_builder_refs", ())),
        ("cashflow_engine", getattr(selection, "cashflow_engine_refs", ())),
        ("market_binding", getattr(selection, "market_binding_refs", ())),
    ):
        for ref in refs or ():
            _append(bindings, seen, str(ref or ""), role=role)
    return tuple(bindings)


def _route_free_lowering_from_structural_selection(
    selection,
    *,
    preferred_method: str,
    fallback_lowering,
    selection_note_prefix: str,
):
    """Build the authoritative route-free lowering from structural selection."""
    from trellis.agent.dsl_lowering import SemanticDslLowering

    route_family = (
        str(getattr(fallback_lowering, "route_family", "") or "").strip()
        or str(preferred_method or "").strip()
        or None
    )
    legacy_notes = tuple(getattr(fallback_lowering, "notes", ()) or ())
    selection_note = f"{selection_note_prefix}:{selection.declaration_id}"
    notes = legacy_notes if selection_note in legacy_notes else (*legacy_notes, selection_note)
    return SemanticDslLowering(
        route_id=None,
        route_family=route_family,
        family_ir=getattr(fallback_lowering, "family_ir", None),
        expr=getattr(fallback_lowering, "expr", None),
        normalized_expr=getattr(fallback_lowering, "normalized_expr", None),
        target_bindings=_route_free_lowering_bindings_from_selection(selection),
        adapters=tuple(getattr(fallback_lowering, "adapters", ()) or ()),
        notes=notes,
        errors=(),
        binding_id=str(getattr(selection, "callable_ref", "") or ""),
    )


def _route_free_lowering_from_contract_ir_selection(
    selection,
    *,
    preferred_method: str,
    fallback_lowering,
):
    """Backward-compatible wrapper for ContractIR structural selection."""
    return _route_free_lowering_from_structural_selection(
        selection,
        preferred_method=preferred_method,
        fallback_lowering=fallback_lowering,
        selection_note_prefix="contract_ir_selection",
    )


def _compile_static_leg_lowering_selection(
    *,
    static_leg_contract_ir,
    preferred_method: str,
):
    """Return the authoritative bounded static-leg lowering selection when admitted."""

    if static_leg_contract_ir is None:
        return None

    from trellis.agent.static_leg_admission import (
        StaticLegLoweringAmbiguityError,
        StaticLegLoweringNoMatchError,
        select_static_leg_lowering,
    )

    try:
        return select_static_leg_lowering(
            static_leg_contract_ir,
            requested_method=preferred_method,
        )
    except (StaticLegLoweringNoMatchError, StaticLegLoweringAmbiguityError):
        _LOG.debug(
            "Static-leg lowering selection did not admit semantic %s",
            getattr(getattr(static_leg_contract_ir, "metadata", None), "semantic_id", "<unknown>"),
            exc_info=True,
        )
        return None


def _compile_static_leg_admission_blockers(static_leg_contract_ir):
    """Return bounded static-leg admission blockers for recognized semantic routes."""

    if static_leg_contract_ir is None:
        return ()
    from trellis.agent.static_leg_contract import CouponLeg

    if any(
        isinstance(getattr(signed_leg, "leg", None), CouponLeg)
        for signed_leg in getattr(static_leg_contract_ir, "legs", ()) or ()
    ):
        from trellis.agent.static_leg_admission import (
            static_coupon_obligation_admission_blockers,
        )

        return static_coupon_obligation_admission_blockers(static_leg_contract_ir)
    if not _is_range_accrual_static_leg_contract_ir(static_leg_contract_ir):
        return ()

    from trellis.agent.static_leg_admission import (
        conditional_range_accrual_admission_blockers,
    )

    return conditional_range_accrual_admission_blockers(static_leg_contract_ir)


def _is_range_accrual_static_leg_contract_ir(static_leg_contract_ir) -> bool:
    metadata = dict(getattr(static_leg_contract_ir, "metadata", {}) or {})
    if (
        str(metadata.get("semantic_family") or metadata.get("semantic_id") or "")
        .strip()
        .lower()
        == "range_accrual"
    ):
        return True
    for signed_leg in getattr(static_leg_contract_ir, "legs", ()) or ():
        leg_metadata = dict(getattr(getattr(signed_leg, "leg", None), "metadata", {}) or {})
        if (
            str(leg_metadata.get("semantic_family") or "")
            .strip()
            .lower()
            == "range_accrual"
        ):
            return True
    return False


def _compile_contract_ir_solver_shadow(
    *,
    contract,
    contract_ir: ContractIR | None,
    preferred_method: str,
    requested_outputs: tuple[str, ...],
    valuation_context,
    market_snapshot,
    primitive_routes: tuple[str, ...],
    route_modules: tuple[str, ...],
    dsl_lowering,
):
    """Compile the additive structural-shadow record when a bound market exists.

    Shadow compilation is deliberately non-authoritative in Phase 3: failure to
    match or bind the structural solver must not perturb the legacy route path.
    """

    if contract_ir is None:
        return None

    from trellis.agent.contract_ir_solver_compiler import (
        ContractIRSolverCompileError,
        build_contract_ir_term_environment,
        compile_contract_ir_solver_shadow,
    )
    from trellis.core.market_state import MarketState

    if not isinstance(market_snapshot, MarketState):
        return None

    legacy_route_id = (
        str(getattr(dsl_lowering, "route_id", None) or "").strip()
        or (primitive_routes[0] if primitive_routes else "")
    )
    legacy_route_family = str(getattr(dsl_lowering, "route_family", None) or "").strip()
    legacy_modules = tuple(
        dict.fromkeys(
            (
                *tuple(getattr(dsl_lowering, "helper_modules", ()) or ()),
                *tuple(route_modules or ()),
            )
        )
    )

    try:
        return compile_contract_ir_solver_shadow(
            contract_ir,
            term_environment=build_contract_ir_term_environment(contract),
            valuation_context=valuation_context,
            market_state=market_snapshot,
            preferred_method=preferred_method,
            requested_outputs=requested_outputs,
            legacy_route_id=legacy_route_id,
            legacy_route_family=legacy_route_family,
            legacy_route_modules=legacy_modules,
        )
    except ContractIRSolverCompileError:
        _LOG.debug(
            "Contract IR structural shadow compilation did not bind for semantic %s",
            getattr(contract, "semantic_id", "<unknown>"),
            exc_info=True,
        )
        return None


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
