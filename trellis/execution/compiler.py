"""Compiler entrypoints for the XIR.0 execution seam.

These entrypoints intentionally do not perform product lowering yet.  They
accept upstream semantic objects and return an explicit empty/unsupported
execution artifact so downstream code can wire against the seam without making
pricing-behavior claims.
"""

from __future__ import annotations

from trellis.execution.ir import (
    ContractExecutionIR,
    ContingentSettlement,
    CurveQuoteObservableRef,
    DecisionAction,
    DecisionProgram,
    ExecutionEvent,
    ExecutionEventPlan,
    ExecutionMetadata,
    ExecutionStateField,
    ExecutionStateSchema,
    ObservableBinding,
    RequirementHints,
    SettlementProgram,
    SettlementStep,
    SourceTrack,
    SpotObservableRef,
    SurfaceQuoteObservableRef,
)


class UnsupportedExecutionSemantics(ValueError):
    """Raised when fail-closed execution lowering is requested for XIR.0."""


def compile_contract_execution_ir(
    source: object,
    *,
    source_track: SourceTrack | None = None,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile an upstream semantic object into a conservative execution IR.

    XIR.0 is an authority boundary only.  Until later XIR tickets add concrete
    visitors/lowerers, the compiler returns an empty execution IR with an
    explicit unsupported reason instead of inventing schedules, obligations, or
    pricing semantics.
    """
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
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a StaticLegContractIR onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        static_leg_contract_ir,
        source_track=infer_source_track(
            static_leg_contract_ir,
            default_source_kind="static_leg_contract_ir",
        ),
        fail_on_unsupported=fail_on_unsupported,
    )


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
