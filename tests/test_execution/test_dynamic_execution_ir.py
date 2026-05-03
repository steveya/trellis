from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.agent.dynamic_contract_ir import (
    ActionSpec,
    ControlProgram,
    DecisionEvent,
    DynamicContractIR,
    EventProgram,
    EventTimeBucket,
    TerminationRule,
)
from trellis.agent.knowledge.decompose import decompose_to_dynamic_contract_ir
from trellis.agent.static_leg_contract import (
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
