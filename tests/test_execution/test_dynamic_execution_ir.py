from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.agent.dynamic_contract_ir import (
    ActionSpec,
    AutomaticTerminationEvent,
    ControlProgram,
    DecisionEvent,
    DynamicContractIR,
    EventProgram,
    EventTimeBucket,
    TerminationRule,
)
from trellis.agent.knowledge.decompose import decompose_to_dynamic_contract_ir
from trellis.agent.semantic_observables import (
    BetweenPredicate,
    ObservationMetadata,
    RateIndexObservable,
)
from trellis.agent.static_leg_contract import (
    ConditionalAccrualLeg,
    ConditionalAccrualPeriod,
    CouponLeg,
    CouponPeriod,
    FixedCouponFormula,
    KnownCashflow,
    KnownCashflowLeg,
    NotionalSchedule,
    NotionalStep,
    SettlementRule,
    SignedLeg,
    StaticLegContractIR,
)
from trellis.core.market_state import MarketState
from trellis.core.payoff import ExecutionBackedPayoff
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.execution import (
    admit_execution_capabilities,
    compile_dynamic_execution_ir,
    contract_execution_summary,
)
from trellis.execution.compiler import UnsupportedExecutionSemantics
from trellis.execution.runtime import price_dynamic_execution_ir
from trellis.execution.visitors.event_compile import (
    compile_callable_bond_spec_from_execution_ir,
)
from trellis.instruments.callable_bond import CallableBondSpec
from trellis.models.callable_bond_pde import price_callable_bond_pde
from trellis.models.callable_bond_tree import price_callable_bond_tree
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _reference_callable_spec() -> CallableBondSpec:
    return CallableBondSpec(
        notional=1_000_000.0,
        coupon=0.05,
        start_date=date(2025, 1, 15),
        end_date=date(2035, 1, 15),
        call_dates=(
            date(2028, 1, 15),
            date(2030, 1, 15),
            date(2032, 1, 15),
        ),
        call_price=100.0,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )


def _callable_bond_contract() -> DynamicContractIR:
    start = date(2025, 1, 15)
    end = date(2035, 1, 15)
    periods = tuple(
        CouponPeriod(
            accrual_start=date(year, month, 15),
            accrual_end=date(next_year, next_month, 15),
            payment_date=date(next_year, next_month, 15),
        )
        for year, month, next_year, next_month in (
            (2025, 1, 2025, 7),
            (2025, 7, 2026, 1),
            (2026, 1, 2026, 7),
            (2026, 7, 2027, 1),
            (2027, 1, 2027, 7),
            (2027, 7, 2028, 1),
            (2028, 1, 2028, 7),
            (2028, 7, 2029, 1),
            (2029, 1, 2029, 7),
            (2029, 7, 2030, 1),
            (2030, 1, 2030, 7),
            (2030, 7, 2031, 1),
            (2031, 1, 2031, 7),
            (2031, 7, 2032, 1),
            (2032, 1, 2032, 7),
            (2032, 7, 2033, 1),
            (2033, 1, 2033, 7),
            (2033, 7, 2034, 1),
            (2034, 1, 2034, 7),
            (2034, 7, 2035, 1),
        )
    )
    base_contract = StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=NotionalSchedule(
                        (
                            NotionalStep(
                                start_date=start,
                                end_date=end,
                                amount=1_000_000.0,
                            ),
                        )
                    ),
                    coupon_periods=periods,
                    coupon_formula=FixedCouponFormula(0.05),
                    day_count="ACT/365",
                    payment_frequency="semiannual",
                    label="coupon_leg",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=KnownCashflowLeg(
                    currency="USD",
                    cashflows=(
                        KnownCashflow(
                            payment_date=end,
                            amount=1_000_000.0,
                            currency="USD",
                            label="principal_redemption",
                        ),
                    ),
                    label="principal_leg",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
        metadata={"semantic_family": "fixed_coupon_bond"},
    )
    redeem = ActionSpec("redeem", "terminate", "redeem at par")
    continue_ = ActionSpec("continue", "continue", "continue outstanding")
    call_dates = (
        date(2028, 1, 15),
        date(2030, 1, 15),
        date(2032, 1, 15),
    )
    return DynamicContractIR(
        base_contract=base_contract,
        semantic_family="callable_bond",
        base_track="static_leg",
        event_program=EventProgram(
            buckets=tuple(
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
            ),
            termination_rules=tuple(
                TerminationRule(
                    label=f"terminate_{call_date.isoformat()}",
                    trigger="action == redeem",
                    settlement_expression="par_redemption",
                    event_label=f"call_{call_date.isoformat()}",
                )
                for call_date in call_dates
            ),
        ),
        control_program=ControlProgram(
            controller_role="issuer",
            decision_style="bermudan",
            decision_event_labels=tuple(f"call_{call_date.isoformat()}" for call_date in call_dates),
            admissible_actions=(redeem, continue_),
        ),
        settlement=SettlementRule(payout_currency="USD"),
    )


def _range_accrual_base_contract() -> StaticLegContractIR:
    observation_dates = (
        date(2025, 1, 15),
        date(2025, 4, 15),
        date(2025, 7, 15),
        date(2025, 10, 15),
    )
    accrual_starts = (
        date(2024, 10, 15),
        date(2025, 1, 15),
        date(2025, 4, 15),
        date(2025, 7, 15),
    )
    condition = BetweenPredicate(
        observable=RateIndexObservable(
            observable_id="reference_rate_fixing",
            index_name="SOFR",
            observation=ObservationMetadata(
                schedule_role="observation_dates",
                fixing_date_role="fixing_dates",
                missing_fixing_policy="project_forward_for_future_only",
            ),
        ),
        lower_bound=0.01,
        upper_bound=0.08,
    )
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=ConditionalAccrualLeg(
                    currency="USD",
                    notional_schedule=NotionalSchedule(
                        (
                            NotionalStep(
                                start_date=accrual_starts[0],
                                end_date=observation_dates[-1],
                                amount=1_000_000.0,
                            ),
                        )
                    ),
                    accrual_periods=tuple(
                        ConditionalAccrualPeriod(
                            accrual_start=start,
                            accrual_end=observation,
                            observation_date=observation,
                            fixing_date=observation,
                            payment_date=observation,
                        )
                        for start, observation in zip(accrual_starts, observation_dates)
                    ),
                    coupon_formula=FixedCouponFormula(0.0525),
                    day_count="ACT/365",
                    payment_frequency="quarterly",
                    accrual_condition_ref="reference_rate_fixing_in_range",
                    accrual_counter_ref="in_range_coupon_count",
                    settlement_rule="coupon_period_cash_settlement",
                    label="range_accrual_coupon",
                    metadata={"semantic_family": "range_accrual"},
                    accrual_condition=condition,
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=KnownCashflowLeg(
                    currency="USD",
                    cashflows=(
                        KnownCashflow(
                            payment_date=observation_dates[-1],
                            amount=1_000_000.0,
                            currency="USD",
                            label="principal_redemption",
                        ),
                    ),
                    label="range_accrual_principal",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
        metadata={"semantic_family": "range_accrual"},
    )


def _callable_range_accrual_contract() -> DynamicContractIR:
    call_dates = (
        date(2025, 4, 15),
        date(2025, 7, 15),
    )
    redeem = ActionSpec("redeem", "terminate", "redeem at par")
    continue_ = ActionSpec("continue", "continue", "continue outstanding")
    return DynamicContractIR(
        base_contract=_range_accrual_base_contract(),
        semantic_family="callable_range_accrual",
        base_track="static_leg",
        event_program=EventProgram(
            buckets=tuple(
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
            ),
            termination_rules=tuple(
                TerminationRule(
                    label=f"terminate_{call_date.isoformat()}",
                    trigger="action == redeem",
                    settlement_expression="par_redemption",
                    event_label=f"call_{call_date.isoformat()}",
                )
                for call_date in call_dates
            ),
        ),
        control_program=ControlProgram(
            controller_role="issuer",
            decision_style="bermudan",
            decision_event_labels=tuple(f"call_{call_date.isoformat()}" for call_date in call_dates),
            admissible_actions=(redeem, continue_),
        ),
        settlement=SettlementRule(payout_currency="USD"),
    )


def test_compile_dynamic_execution_ir_lowers_callable_bond_shape():
    ir = compile_dynamic_execution_ir(_callable_bond_contract())

    assert ir.source_track.source_kind == "dynamic_contract_ir"
    assert ir.source_track.product_family == "callable_bond"
    assert ir.source_track.instrument_class == "callable_bond"
    assert tuple(event.event_kind for event in ir.event_plan.events).count("decision") == 3
    assert tuple(event.event_kind for event in ir.event_plan.events).count("payment") >= 4

    summary = contract_execution_summary(ir)
    assert summary["unsupported_reasons"] == ()
    assert summary["decision_action_types"] == ("continue", "terminate")
    assert summary["requirement_markets"] == (
        "black_vol_surface:USD",
        "discount_curve:USD",
    )
    assert summary["timeline_roles"] == ("call_date", "payment_dates")


def test_compile_dynamic_execution_ir_can_fail_closed_for_unadmitted_dynamic_lane():
    contract = decompose_to_dynamic_contract_ir(
        "Phoenix autocallable note on SPX notional 1000000 coupon 8% "
        "autocall barrier 100% observation dates 2025-07-15, 2026-01-15, "
        "2026-07-15, 2027-01-15 maturity 2027-01-15",
        instrument_type="autocallable",
    )
    assert contract is not None

    with pytest.raises(UnsupportedExecutionSemantics, match="autocallable"):
        compile_dynamic_execution_ir(contract, fail_on_unsupported=True)


def test_compile_dynamic_execution_ir_fails_closed_for_nonsensical_coupon_scale():
    contract = _callable_bond_contract()
    coupon_leg = contract.base_contract.legs[0].leg
    bad_coupon_leg = replace(
        coupon_leg,
        coupon_formula=FixedCouponFormula(5.0),
    )
    bad_base = replace(
        contract.base_contract,
        legs=(
            replace(contract.base_contract.legs[0], leg=bad_coupon_leg),
            contract.base_contract.legs[1],
        ),
    )
    bad_contract = replace(contract, base_contract=bad_base)

    with pytest.raises(UnsupportedExecutionSemantics, match="decimal form"):
        compile_dynamic_execution_ir(bad_contract, fail_on_unsupported=True)


def test_callable_bond_event_compiler_round_trips_helper_spec():
    ir = compile_dynamic_execution_ir(_callable_bond_contract())

    compiled = compile_callable_bond_spec_from_execution_ir(ir)

    assert compiled == _reference_callable_spec()


def test_callable_bond_execution_ir_is_admitted_for_lattice_and_pde():
    ir = compile_dynamic_execution_ir(_callable_bond_contract())

    lattice = admit_execution_capabilities(ir, method="lattice")
    pde = admit_execution_capabilities(ir, method="pde")

    assert lattice.admitted is True
    assert lattice.engine_family == "lattice"
    assert pde.admitted is True
    assert pde.engine_family == "pde"


def test_price_dynamic_execution_ir_matches_callable_bond_tree_and_pde_helpers():
    market_state = _market_state()
    ir = compile_dynamic_execution_ir(_callable_bond_contract())
    spec = _reference_callable_spec()

    lattice_price = price_dynamic_execution_ir(
        ir,
        market_state,
        method="lattice",
        terms={"model": "hull_white", "n_steps": 120},
    )
    pde_price = price_dynamic_execution_ir(
        ir,
        market_state,
        method="pde",
        terms={"n_r": 121, "n_t": 160},
    )

    assert lattice_price == pytest.approx(
        price_callable_bond_tree(
            market_state,
            spec,
            model="hull_white",
            n_steps=120,
        ),
        rel=1e-12,
    )
    assert pde_price == pytest.approx(
        price_callable_bond_pde(
            market_state,
            spec,
            n_r=121,
            n_t=160,
        ),
        rel=1e-12,
    )


def test_execution_backed_payoff_prices_callable_bond_execution_ir():
    market_state = _market_state()
    ir = compile_dynamic_execution_ir(_callable_bond_contract())
    payoff = ExecutionBackedPayoff(
        ir,
        method="lattice",
        execution_terms={"model": "hull_white", "n_steps": 120},
    )

    assert payoff.requirements == {"black_vol_surface", "discount_curve"}
    assert price_payoff(payoff, market_state) == pytest.approx(
        price_dynamic_execution_ir(
            ir,
            market_state,
            method="lattice",
            terms={"model": "hull_white", "n_steps": 120},
        ),
        rel=1e-12,
    )


def test_compile_dynamic_execution_ir_lowers_callable_range_accrual_shape():
    ir = compile_dynamic_execution_ir(_callable_range_accrual_contract())

    assert ir.source_track.source_kind == "dynamic_contract_ir"
    assert ir.source_track.product_family == "callable_range_accrual"
    assert ir.source_track.instrument_class == "callable_range_accrual"
    assert {obligation.obligation_kind for obligation in ir.obligations} == {
        "conditional_accrual_leg"
    }
    summary = contract_execution_summary(ir)
    assert summary["unsupported_reasons"] == ()
    assert summary["decision_action_types"] == ("continue", "terminate")
    assert summary["requirement_markets"] == (
        "discount_curve:USD",
        "fixing_history:SOFR",
        "forward_curve:SOFR",
    )
    assert summary["settlement_kinds"] == (
        "conditional_accrual_present_value",
        "issuer_call_redemption",
    )
    assert dict(ir.source_track.source_metadata)["validation_bundle_id"] == (
        "callable_range_accrual_deterministic_v1"
    )


def test_compile_dynamic_execution_ir_rejects_pay_side_callable_range_accrual():
    contract = _callable_range_accrual_contract()
    base_contract = contract.base_contract
    bad_base = replace(
        base_contract,
        legs=(
            replace(base_contract.legs[0], direction="pay"),
            *base_contract.legs[1:],
        ),
    )

    with pytest.raises(UnsupportedExecutionSemantics, match="receive-side"):
        compile_dynamic_execution_ir(
            replace(contract, base_contract=bad_base),
            fail_on_unsupported=True,
        )


def test_price_dynamic_execution_ir_prices_callable_range_accrual_deterministically():
    from trellis.execution import compile_static_leg_execution_ir
    from trellis.execution.runtime import price_static_leg_execution_ir

    market_state = _market_state()
    contract = _callable_range_accrual_contract()
    ir = compile_dynamic_execution_ir(contract)
    static_ir = compile_static_leg_execution_ir(contract.base_contract)

    dynamic_price = price_dynamic_execution_ir(
        ir,
        market_state,
        method="deterministic",
    )
    static_price = price_static_leg_execution_ir(static_ir, market_state)

    assert dynamic_price > 0.0
    assert dynamic_price < static_price


def test_compile_dynamic_execution_ir_keeps_interrupted_range_accrual_blocked():
    contract = DynamicContractIR(
        base_contract=_range_accrual_base_contract(),
        semantic_family="interrupted_range_accrual",
        base_track="static_leg",
        event_program=EventProgram(
            buckets=(
                EventTimeBucket(
                    event_date=date(2025, 4, 15),
                    phase_sequence=("termination",),
                    events=(
                        AutomaticTerminationEvent(
                            label="interrupt_2025-04-15",
                            trigger="interruption_event",
                            settlement_expression="suspend_accrual",
                        ),
                    ),
                ),
            ),
            termination_rules=(
                TerminationRule(
                    label="terminate_interrupt_2025-04-15",
                    trigger="interruption_event",
                    settlement_expression="suspend_accrual",
                    event_label="interrupt_2025-04-15",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
    )

    with pytest.raises(UnsupportedExecutionSemantics, match="interrupted_range_accrual|automatic"):
        compile_dynamic_execution_ir(contract, fail_on_unsupported=True)
