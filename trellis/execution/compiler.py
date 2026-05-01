"""Compiler entrypoints for contract execution IR.

Most entrypoints remain conservative and return explicit unsupported execution
artifacts unless a bounded lowering has been admitted. The static-leg compiler
now lowers the first checked cohort into route-free execution artifacts, while
generic and dynamic semantic inputs still fail closed at the execution seam.
"""

from __future__ import annotations

from trellis.execution.ir import (
    ContractExecutionIR,
    ContingentSettlement,
    CouponLegExecution,
    CurveQuoteObservableRef,
    DecisionAction,
    DecisionProgram,
    ExecutionEvent,
    ExecutionEventPlan,
    ExecutionMetadata,
    ExecutionStateField,
    ExecutionStateSchema,
    ForwardRateObservableRef,
    KnownCashflowObligation,
    ObservableBinding,
    PeriodRateOptionStripExecution,
    RequirementHints,
    SettlementProgram,
    SettlementStep,
    SourceTrack,
    SpotObservableRef,
    SurfaceQuoteObservableRef,
)


class UnsupportedExecutionSemantics(ValueError):
    """Raised when fail-closed execution lowering cannot admit the source."""


def compile_contract_execution_ir(
    source: object,
    *,
    source_track: SourceTrack | None = None,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile an upstream semantic object into a conservative execution IR."""
    resolved_source_track = source_track or infer_source_track(source)
    reason = (
        f"execution lowering not implemented for {resolved_source_track.source_kind}"
    )
    if fail_on_unsupported:
        raise UnsupportedExecutionSemantics(reason)
    return ContractExecutionIR.empty(
        source_track=resolved_source_track,
        unsupported_reasons=(reason,),
    )


def compile_semantic_execution_ir(
    semantic_contract: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a semantic contract onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        semantic_contract,
        fail_on_unsupported=fail_on_unsupported,
    )


def compile_contract_ir_execution_ir(
    contract_ir: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a payoff-expression ContractIR onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        contract_ir,
        source_track=infer_source_track(contract_ir, default_source_kind="contract_ir"),
        fail_on_unsupported=fail_on_unsupported,
    )


def compile_static_leg_execution_ir(
    static_leg_contract_ir: object,
    *,
    requested_method: str | None = None,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile an admitted StaticLegContractIR into execution IR."""
    try:
        from trellis.agent.static_leg_contract import StaticLegContractIR
    except ImportError:  # pragma: no cover - defensive import boundary
        StaticLegContractIR = None
    if StaticLegContractIR is not None and isinstance(
        static_leg_contract_ir,
        StaticLegContractIR,
    ):
        return lower_static_leg_contract_ir_to_execution_ir(
            static_leg_contract_ir,
            requested_method=requested_method,
            fail_on_unsupported=fail_on_unsupported,
        )
    return compile_contract_execution_ir(
        static_leg_contract_ir,
        source_track=infer_source_track(
            static_leg_contract_ir,
            default_source_kind="static_leg_contract_ir",
        ),
        fail_on_unsupported=fail_on_unsupported,
    )


def lower_static_leg_contract_ir_to_execution_ir(
    static_leg_contract_ir: object,
    *,
    requested_method: str | None = None,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Lower an admitted StaticLegContractIR into a route-free execution IR."""
    from trellis.agent.static_leg_admission import (
        StaticLegLoweringNoMatchError,
        select_static_leg_lowering,
    )
    from trellis.agent.static_leg_contract import StaticLegContractIR

    if not isinstance(static_leg_contract_ir, StaticLegContractIR):
        raise TypeError("static_leg_contract_ir must be a StaticLegContractIR")

    try:
        selection = select_static_leg_lowering(
            static_leg_contract_ir,
            requested_method=requested_method,
        )
    except StaticLegLoweringNoMatchError as exc:
        reason = f"static-leg execution lowering not admitted: {exc}"
        source_track = infer_source_track(
            static_leg_contract_ir,
            default_source_kind="static_leg_contract_ir",
        )
        if fail_on_unsupported:
            raise UnsupportedExecutionSemantics(reason) from exc
        return ContractExecutionIR.empty(
            source_track=source_track,
            unsupported_reasons=(reason,),
            tags=("static_leg", "unsupported_static_leg_execution"),
        )

    family = _static_leg_product_family(
        static_leg_contract_ir,
        selection.declaration_id,
    )
    source_track = SourceTrack(
        source_kind="static_leg_contract_ir",
        semantic_id=str(static_leg_contract_ir.metadata.get("semantic_id", "") or ""),
        product_family=family,
        instrument_class="static_leg",
        source_ref=f"static_leg_contract_ir:{family}",
        source_metadata={
            "static_leg_lowering_declaration_id": selection.declaration_id,
            "validation_bundle_id": selection.validation_bundle_id,
            "requested_method": selection.method or "",
            "callable_ref": selection.callable_ref,
        },
    )
    try:
        currency = _primary_static_leg_currency(static_leg_contract_ir)
        _validate_static_leg_execution_terms(
            static_leg_contract_ir,
            selection.declaration_id,
        )
        obligations = _static_leg_execution_obligations(static_leg_contract_ir)
        events = _static_leg_execution_events(static_leg_contract_ir)
        observables = _static_leg_execution_observables(static_leg_contract_ir, currency)
        requirement_hints = _static_leg_requirement_hints(
            static_leg_contract_ir,
            currency=currency,
        )
        settlement_steps = _static_leg_settlement_steps(static_leg_contract_ir)
    except (NotImplementedError, UnsupportedExecutionSemantics) as exc:
        reason = f"static-leg execution lowering not admitted: {exc}"
        if fail_on_unsupported:
            raise UnsupportedExecutionSemantics(reason) from exc
        return ContractExecutionIR.empty(
            source_track=source_track,
            unsupported_reasons=(reason,),
            tags=("static_leg", "unsupported_static_leg_execution"),
        )

    return ContractExecutionIR(
        source_track=source_track,
        obligations=obligations,
        observables=observables,
        event_plan=ExecutionEventPlan(
            events=events,
            phase_order=("fixing", "accrual_boundary", "payment", "settlement"),
        ),
        settlement_program=SettlementProgram(steps=settlement_steps),
        requirement_hints=requirement_hints,
        execution_metadata=ExecutionMetadata(
            tags=("route_free_static_leg_execution", family),
            metadata={
                "static_leg_lowering_declaration_id": selection.declaration_id,
                "validation_bundle_id": selection.validation_bundle_id,
                "callable_ref": selection.callable_ref,
            },
        ),
    )


def _static_leg_product_family(contract: object, declaration_id: str) -> str:
    metadata = getattr(contract, "metadata", {})
    semantic_family = str(metadata.get("semantic_family", "") or "").strip()
    if semantic_family:
        return semantic_family
    text = str(declaration_id or "").strip()
    if text.startswith("static_leg_"):
        text = text[len("static_leg_") :]
    for suffix in ("_analytical", "_monte_carlo"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text or "static_leg"


def _primary_static_leg_currency(contract: object) -> str:
    currencies = tuple(getattr(contract, "currencies", ()) or ())
    return str(currencies[0] if currencies else "USD").strip().upper() or "USD"


def _validate_static_leg_execution_terms(contract: object, declaration_id: str) -> None:
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        FixedCouponFormula,
        FloatingCouponFormula,
        KnownCashflowLeg,
    )

    if declaration_id == "static_leg_fixed_float_swap":
        notionals = tuple(
            _constant_notional(signed_leg.leg)
            for signed_leg in contract.legs
            if isinstance(signed_leg.leg, CouponLeg)
        )
        if len(set(round(notional, 12) for notional in notionals)) != 1:
            raise UnsupportedExecutionSemantics(
                "fixed-float execution lowering requires matching constant notionals"
            )
        return

    if declaration_id == "static_leg_fixed_coupon_bond":
        coupon_notionals = tuple(
            _constant_notional(signed_leg.leg)
            for signed_leg in contract.legs
            if isinstance(signed_leg.leg, CouponLeg)
            and isinstance(signed_leg.leg.coupon_formula, FixedCouponFormula)
        )
        redemption_amounts = tuple(
            cashflow.amount
            for signed_leg in contract.legs
            if isinstance(signed_leg.leg, KnownCashflowLeg)
            for cashflow in signed_leg.leg.cashflows
        )
        if len(coupon_notionals) != 1 or len(redemption_amounts) != 1:
            raise UnsupportedExecutionSemantics(
                "fixed-coupon-bond execution lowering requires one coupon leg and one redemption"
            )
        if abs(float(redemption_amounts[0]) - float(coupon_notionals[0])) > 1e-12:
            raise UnsupportedExecutionSemantics(
                "fixed-coupon-bond execution lowering requires redemption to match notional"
            )
        return

    if declaration_id == "static_leg_basis_swap":
        for signed_leg in contract.legs:
            leg = signed_leg.leg
            if isinstance(leg, CouponLeg) and isinstance(
                leg.coupon_formula,
                FloatingCouponFormula,
            ):
                _constant_notional(leg)


def _static_leg_execution_obligations(contract: object) -> tuple[object, ...]:
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        KnownCashflowLeg,
        PeriodRateOptionStripLeg,
    )

    obligations: list[object] = []
    for index, signed_leg in enumerate(contract.legs):
        leg = signed_leg.leg
        leg_id = _leg_id(leg, index)
        if isinstance(leg, CouponLeg):
            obligations.append(
                CouponLegExecution(
                    obligation_id=f"coupon_leg:{leg_id}",
                    leg_id=leg_id,
                    currency=leg.currency,
                    schedule_role="payment_dates",
                    formula_ref=_coupon_formula_ref(leg.coupon_formula),
                    metadata={
                        "direction": signed_leg.direction,
                        "notional": _constant_notional(leg),
                        "day_count": leg.day_count,
                        "payment_frequency": leg.payment_frequency,
                        "periods": _coupon_period_metadata(leg),
                        **_coupon_formula_metadata(leg.coupon_formula),
                    },
                )
            )
            continue
        if isinstance(leg, KnownCashflowLeg):
            for cashflow_index, cashflow in enumerate(leg.cashflows):
                label = str(cashflow.label or f"cashflow_{cashflow_index}").strip()
                obligations.append(
                    KnownCashflowObligation(
                        obligation_id=f"known_cashflow:{label}:{cashflow_index}",
                        payment_date=cashflow.payment_date,
                        currency=cashflow.currency,
                        amount=cashflow.amount,
                        payer=_payer_for_direction(signed_leg.direction),
                        receiver=_receiver_for_direction(signed_leg.direction),
                    )
                )
            continue
        if isinstance(leg, PeriodRateOptionStripLeg):
            obligations.append(
                PeriodRateOptionStripExecution(
                    obligation_id=f"period_rate_option_strip:{leg_id}",
                    strip_id=leg_id,
                    currency=leg.currency,
                    schedule_role="payment_dates",
                    option_style=leg.option_side,
                    metadata={
                        "direction": signed_leg.direction,
                        "notional": _constant_notional(leg),
                        "strike": leg.strike,
                        "day_count": leg.day_count,
                        "payment_frequency": leg.payment_frequency,
                        "rate_index": _rate_index_identifier(leg.rate_index),
                        "periods": _option_period_metadata(leg),
                    },
                )
            )
    return tuple(obligations)


def _static_leg_execution_events(contract: object) -> tuple[ExecutionEvent, ...]:
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        KnownCashflowLeg,
        PeriodRateOptionStripLeg,
    )

    events: list[ExecutionEvent] = []
    seen: set[str] = set()

    def _append(event: ExecutionEvent) -> None:
        if event.event_id in seen:
            return
        seen.add(event.event_id)
        events.append(event)

    for index, signed_leg in enumerate(contract.legs):
        leg = signed_leg.leg
        leg_id = _leg_id(leg, index)
        if isinstance(leg, CouponLeg):
            needs_fixing = _coupon_formula_ref(leg.coupon_formula) == "floating_coupon"
            for period_index, period in enumerate(leg.coupon_periods):
                if needs_fixing and period.fixing_date is not None:
                    _append(
                        ExecutionEvent(
                            event_id=f"fixing:{leg_id}:{period_index}",
                            event_kind="fixing",
                            schedule_role="fixing_dates",
                            phase="fixing",
                            event_date=period.fixing_date,
                            metadata={"leg_id": leg_id, "period_index": period_index},
                        )
                    )
                _append(
                    ExecutionEvent(
                        event_id=f"payment:{leg_id}:{period_index}",
                        event_kind="payment",
                        schedule_role="payment_dates",
                        phase="payment",
                        event_date=period.payment_date,
                        metadata={"leg_id": leg_id, "period_index": period_index},
                    )
                )
            continue
        if isinstance(leg, KnownCashflowLeg):
            for cashflow_index, cashflow in enumerate(leg.cashflows):
                label = str(cashflow.label or f"cashflow_{cashflow_index}").strip()
                _append(
                    ExecutionEvent(
                        event_id=f"payment:known_cashflow:{label}:{cashflow_index}",
                        event_kind="payment",
                        schedule_role="payment_dates",
                        phase="payment",
                        event_date=cashflow.payment_date,
                        metadata={"leg_id": leg_id, "cashflow_index": cashflow_index},
                    )
                )
            continue
        if isinstance(leg, PeriodRateOptionStripLeg):
            for period_index, period in enumerate(leg.option_periods):
                _append(
                    ExecutionEvent(
                        event_id=f"fixing:{leg_id}:{period_index}",
                        event_kind="fixing",
                        schedule_role="fixing_dates",
                        phase="fixing",
                        event_date=period.fixing_date,
                        metadata={"leg_id": leg_id, "period_index": period_index},
                    )
                )
                _append(
                    ExecutionEvent(
                        event_id=f"payment:{leg_id}:{period_index}",
                        event_kind="payment",
                        schedule_role="payment_dates",
                        phase="payment",
                        event_date=period.payment_date,
                        metadata={"leg_id": leg_id, "period_index": period_index},
                    )
                )
    return tuple(sorted(events, key=lambda event: (event.event_date, event.phase, event.event_id)))


def _static_leg_execution_observables(
    contract: object,
    currency: str,
) -> tuple[object, ...]:
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        FloatingCouponFormula,
        PeriodRateOptionStripLeg,
    )

    observables: list[object] = [
        CurveQuoteObservableRef(
            observable_id=f"discount_curve:{currency}",
            source_ref=f"market.discount_curve:{currency}",
            currency=currency,
            tags=("discounting", "static_leg"),
        )
    ]
    seen = {observables[0].observable_id}
    for signed_leg in contract.legs:
        leg = signed_leg.leg
        if isinstance(leg, CouponLeg) and isinstance(leg.coupon_formula, FloatingCouponFormula):
            index_name = _rate_index_identifier(leg.coupon_formula.rate_index)
            observable_id = f"forward_curve:{index_name}"
            if observable_id not in seen:
                seen.add(observable_id)
                observables.append(
                    ForwardRateObservableRef(
                        observable_id=observable_id,
                        source_ref=f"market.forward_curve:{index_name}",
                        currency=leg.currency,
                        tags=("forward_curve", "static_leg"),
                        metadata={"rate_index": index_name},
                    )
                )
        if isinstance(leg, PeriodRateOptionStripLeg):
            index_name = _rate_index_identifier(leg.rate_index)
            for observable_id, kind, cls in (
                (f"forward_curve:{index_name}", "forward_curve", ForwardRateObservableRef),
                (f"black_vol_surface:{index_name}", "black_vol_surface", SurfaceQuoteObservableRef),
            ):
                if observable_id in seen:
                    continue
                seen.add(observable_id)
                observables.append(
                    cls(
                        observable_id=observable_id,
                        source_ref=f"market.{kind}:{index_name}",
                        currency=leg.currency,
                        tags=(kind, "static_leg"),
                        metadata={"rate_index": index_name},
                    )
                )
    return tuple(observables)


def _static_leg_requirement_hints(
    contract: object,
    *,
    currency: str,
) -> RequirementHints:
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        FloatingCouponFormula,
        PeriodRateOptionStripLeg,
    )

    market_inputs = {f"discount_curve:{currency}"}
    state_variables: set[str] = set()
    timeline_roles = {"payment_dates"}
    for signed_leg in contract.legs:
        leg = signed_leg.leg
        if isinstance(leg, CouponLeg) and isinstance(leg.coupon_formula, FloatingCouponFormula):
            market_inputs.add(f"forward_curve:{_rate_index_identifier(leg.coupon_formula.rate_index)}")
            timeline_roles.add("fixing_dates")
        if isinstance(leg, PeriodRateOptionStripLeg):
            index_name = _rate_index_identifier(leg.rate_index)
            market_inputs.add(f"forward_curve:{index_name}")
            market_inputs.add(f"black_vol_surface:{index_name}")
            timeline_roles.add("fixing_dates")
            state_variables.add("period_rate_option_strip")
    return RequirementHints(
        market_inputs=frozenset(market_inputs),
        state_variables=frozenset(state_variables),
        timeline_roles=frozenset(timeline_roles),
    )


def _static_leg_settlement_steps(contract: object) -> tuple[SettlementStep, ...]:
    currency = _primary_static_leg_currency(contract)
    return (
        SettlementStep(
            step_id="static_leg_present_value",
            settlement_kind="present_value",
            expression="sum(execution_obligations)",
            currency=currency,
            payment_date_role="payment_dates",
        ),
    )


def _leg_id(leg: object, index: int) -> str:
    label = str(getattr(leg, "label", "") or "").strip()
    return label or f"leg_{index}"


def _payer_for_direction(direction: str) -> str:
    return "counterparty" if direction == "receive" else "holder"


def _receiver_for_direction(direction: str) -> str:
    return "holder" if direction == "receive" else "counterparty"


def _constant_notional(leg: object) -> float:
    steps = tuple(getattr(getattr(leg, "notional_schedule", None), "steps", ()) or ())
    if len(steps) != 1:
        raise UnsupportedExecutionSemantics(
            "static-leg execution lowering currently requires constant notional"
        )
    return float(steps[0].amount)


def _coupon_formula_ref(formula: object) -> str:
    from trellis.agent.static_leg_contract import FixedCouponFormula, FloatingCouponFormula

    if isinstance(formula, FixedCouponFormula):
        return "fixed_coupon"
    if isinstance(formula, FloatingCouponFormula):
        return "floating_coupon"
    return "quoted_coupon"


def _coupon_formula_metadata(formula: object) -> dict[str, object]:
    from trellis.agent.static_leg_contract import FixedCouponFormula, FloatingCouponFormula

    if isinstance(formula, FixedCouponFormula):
        return {"formula_kind": "fixed", "fixed_rate": formula.rate}
    if isinstance(formula, FloatingCouponFormula):
        return {
            "formula_kind": "floating",
            "rate_index": _rate_index_identifier(formula.rate_index),
            "spread": formula.spread,
            "gearing": formula.gearing,
        }
    return {"formula_kind": "quoted"}


def _coupon_period_metadata(leg: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            period.accrual_start,
            period.accrual_end,
            period.payment_date,
            period.fixing_date,
        )
        for period in leg.coupon_periods
    )


def _option_period_metadata(leg: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            period.accrual_start,
            period.accrual_end,
            period.fixing_date,
            period.payment_date,
        )
        for period in leg.option_periods
    )


def _rate_index_identifier(index: object) -> str:
    from trellis.agent.static_leg_contract import CmsRateIndex, OvernightRateIndex, TermRateIndex

    if isinstance(index, OvernightRateIndex):
        return index.name
    if isinstance(index, TermRateIndex):
        return f"{index.name}-{index.tenor}"
    if isinstance(index, CmsRateIndex):
        return f"{index.curve_id}-{index.tenor}"
    return str(index or "").strip()


def compile_dynamic_execution_ir(
    dynamic_contract_ir: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a DynamicContractIR onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        dynamic_contract_ir,
        source_track=infer_source_track(
            dynamic_contract_ir,
            default_source_kind="dynamic_contract_ir",
        ),
        fail_on_unsupported=fail_on_unsupported,
    )


def compile_bermudan_best_of_basket_execution_ir(
    *,
    semantic_id: str,
    underliers: tuple[str, ...] | list[str],
    strike: float,
    expiry_date: object,
    observation_dates: tuple[object, ...] | list[object],
    exercise_dates: tuple[object, ...] | list[object],
    notional: float = 1.0,
    currency: str = "USD",
    requested_outputs: tuple[str, ...] | list[str] = (),
    validation_policy: str = "",
    source_ref: str = "",
) -> ContractExecutionIR:
    """Compile the bounded Bermudan best-of basket operator shape.

    This is an execution/semantic representation only. It records the named
    observables, payoff expression, holder exercise decisions, and market
    requirements without selecting a pricing route or model family.
    """
    names = _normalize_unique_text(underliers)
    if len(names) < 2:
        raise UnsupportedExecutionSemantics(
            "bermudan best-of basket execution requires at least two underliers"
        )
    observations = _normalize_non_empty_tuple(observation_dates, "observation_dates")
    exercises = _normalize_non_empty_tuple(exercise_dates, "exercise_dates")
    expiry = expiry_date
    if expiry in {None, ""}:
        raise UnsupportedExecutionSemantics("expiry_date is required")

    strike_value = float(strike)
    notional_value = float(notional)
    currency_text = str(currency or "USD").strip().upper()
    underlier_csv = ",".join(names)
    spot_terms = ", ".join(f"spot[{name}]" for name in names)
    payoff_expression = (
        f"notional * max(max({spot_terms}) - {strike_value}, 0.0)"
    )

    observables = tuple(
        SpotObservableRef(
            observable_id=f"spot:{name}",
            source_ref=f"market.spot:{name}",
            currency=currency_text,
            tags=("underlier", "basket_constituent"),
            metadata={
                "underlier": name,
                "vector_index": index,
            },
        )
        for index, name in enumerate(names)
    ) + tuple(
        SurfaceQuoteObservableRef(
            observable_id=f"black_vol_surface:{name}",
            source_ref=f"market.black_vol_surface:{name}",
            currency=currency_text,
            tags=("volatility", "basket_constituent"),
            metadata={
                "underlier": name,
                "vector_index": index,
            },
        )
        for index, name in enumerate(names)
    ) + (
        ObservableBinding(
            observable_id=f"correlation_matrix:{underlier_csv}",
            observable_kind="correlation_matrix",
            source_ref=f"market.correlation_matrix:{underlier_csv}",
            tags=("correlation", "basket"),
            metadata={"underliers": names},
        ),
        CurveQuoteObservableRef(
            observable_id=f"discount_curve:{currency_text}",
            source_ref=f"market.discount_curve:{currency_text}",
            currency=currency_text,
            tags=("discounting",),
        ),
    )

    events = tuple(
        ExecutionEvent(
            event_id=f"observation:{event_date}",
            event_kind="observation",
            schedule_role="observation_dates",
            phase="observation",
            event_date=event_date,
            metadata={"underliers": names},
        )
        for event_date in observations
    ) + tuple(
        ExecutionEvent(
            event_id=f"decision:{event_date}",
            event_kind="decision",
            schedule_role="exercise_dates",
            phase="decision",
            event_date=event_date,
            metadata={"controller_role": "holder"},
        )
        for event_date in exercises
    ) + (
        ExecutionEvent(
            event_id=f"settlement:{expiry}",
            event_kind="settlement",
            schedule_role="expiry_date",
            phase="settlement",
            event_date=expiry,
            metadata={"settlement_ref": "payoff:best_of_call"},
        ),
    )

    decision_actions = tuple(
        DecisionAction(
            action_id=f"holder-exercise:{event_date}",
            action_type="holder_max",
            controller_role="holder",
            schedule_role="exercise_dates",
            state_updates=("exercise_decision", "contract_alive"),
        )
        for event_date in exercises
    )

    return ContractExecutionIR(
        source_track=SourceTrack(
            source_kind="semantic_contract",
            semantic_id=str(semantic_id or "").strip(),
            product_family="bermudan_best_of_basket",
            instrument_class="basket_option",
            source_ref=source_ref or f"semantic_contract:{semantic_id}",
            source_metadata={
                "underliers": names,
                "payoff_operator": "best_of_call",
                "exercise_style": "bermudan",
                "strike": strike_value,
                "notional": notional_value,
                "currency": currency_text,
            },
        ),
        obligations=(
            ContingentSettlement(
                obligation_id="bermudan-best-of-call-settlement",
                condition_ref="holder_exercise_or_expiry",
                settlement_ref="payoff:best_of_call",
                currency=currency_text,
            ),
        ),
        observables=observables,
        event_plan=ExecutionEventPlan(
            events=events,
            phase_order=("observation", "decision", "settlement"),
        ),
        state_schema=ExecutionStateSchema(
            fields=(
                *(
                    ExecutionStateField(
                        name=f"spot:{name}",
                        domain="real",
                        tags=("underlier",),
                    )
                    for name in names
                ),
                ExecutionStateField(name="contract_alive", domain="boolean"),
                ExecutionStateField(name="exercise_value", domain="real"),
                ExecutionStateField(name="continuation_value", domain="real"),
            ),
        ),
        decision_program=DecisionProgram(
            actions=decision_actions,
            controller_role="holder",
        ),
        settlement_program=SettlementProgram(
            steps=(
                SettlementStep(
                    step_id="payoff:best_of_call",
                    settlement_kind="best_of_call_payoff",
                    expression=payoff_expression,
                    currency=currency_text,
                    payment_date_role="exercise_or_expiry",
                ),
            ),
        ),
        requirement_hints=RequirementHints(
            market_inputs=frozenset(
                {
                    *(f"spot:{name}" for name in names),
                    *(f"black_vol_surface:{name}" for name in names),
                    f"correlation_matrix:{underlier_csv}",
                    f"discount_curve:{currency_text}",
                }
            ),
            state_variables=frozenset(
                {
                    "multi_asset_spot_state",
                    "contract_alive",
                    "exercise_value",
                    "continuation_value",
                }
            ),
            timeline_roles=frozenset(
                {
                    "observation_dates",
                    "exercise_dates",
                    "expiry_date",
                    "settlement",
                }
            ),
        ),
        execution_metadata=ExecutionMetadata(
            tags=(
                "route_free_operator_ir",
                "bermudan",
                "best_of",
                "basket",
            ),
            metadata={
                "requested_outputs": tuple(_normalize_unique_text(requested_outputs)),
                "validation_policy": str(validation_policy or "").strip(),
            },
        ),
    )


def infer_source_track(
    source: object,
    *,
    default_source_kind: str | None = None,
) -> SourceTrack:
    """Infer minimal source metadata without importing agent-owned modules."""
    source_kind = (
        _attr_text(source, "source_kind")
        or _attr_text(source, "track")
        or default_source_kind
        or _source_kind_from_type(source)
    )
    semantic_id = (
        _attr_text(source, "semantic_id")
        or _attr_text(source, "declaration_id")
        or _attr_text(source, "contract_id")
    )
    product = getattr(source, "product", None)
    instrument_class = (
        _attr_text(product, "instrument_class")
        or _attr_text(source, "instrument_class")
        or _attr_text(source, "instrument_type")
    )
    product_family = (
        _attr_text(product, "payoff_family")
        or _attr_text(source, "payoff_family")
        or _attr_text(source, "product_family")
    )
    source_ref = (
        _attr_text(source, "source_ref")
        or (f"{source_kind}:{semantic_id}" if semantic_id else source_kind)
    )
    return SourceTrack(
        source_kind=source_kind,
        semantic_id=semantic_id,
        product_family=product_family,
        instrument_class=instrument_class,
        source_ref=source_ref,
    )


def _attr_text(source: object, name: str) -> str:
    if source is None:
        return ""
    value = getattr(source, name, "")
    if callable(value):
        return ""
    return str(value or "").strip()


def _source_kind_from_type(source: object) -> str:
    name = type(source).__name__.lower()
    if "staticleg" in name or "static_leg" in name:
        return "static_leg_contract_ir"
    if "dynamic" in name:
        return "dynamic_contract_ir"
    if "contractir" in name or "contract_ir" in name:
        return "contract_ir"
    if "semantic" in name:
        return "semantic_contract"
    return "unknown"


def _normalize_unique_text(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalize_non_empty_tuple(values: object, label: str) -> tuple[object, ...]:
    if isinstance(values, str):
        values = (values,)
    result = tuple(value for value in values or () if value not in {None, ""})
    if not result:
        raise UnsupportedExecutionSemantics(f"{label} must be non-empty")
    return result
