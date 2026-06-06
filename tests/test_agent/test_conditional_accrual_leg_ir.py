from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from trellis.agent.static_leg_contract import (
    ConditionalAccrualLeg,
    ConditionalAccrualPeriod,
    FixedCouponFormula,
    NotionalSchedule,
    NotionalStep,
    SignedLeg,
    StaticLegContractIR,
    StaticLegIRWellFormednessError,
)
from trellis.execution import compile_static_leg_execution_ir
from trellis.execution.compiler import UnsupportedExecutionSemantics


def _notional() -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(
                start_date=date(2026, 1, 15),
                end_date=date(2026, 10, 15),
                amount=1_000_000.0,
            ),
        )
    )


def _conditional_periods() -> tuple[ConditionalAccrualPeriod, ...]:
    return (
        ConditionalAccrualPeriod(
            accrual_start=date(2026, 1, 15),
            accrual_end=date(2026, 4, 15),
            observation_date=date(2026, 4, 15),
            payment_date=date(2026, 4, 17),
            fixing_date=date(2026, 4, 15),
        ),
        ConditionalAccrualPeriod(
            accrual_start=date(2026, 4, 15),
            accrual_end=date(2026, 7, 15),
            observation_date=date(2026, 7, 15),
            payment_date=date(2026, 7, 17),
            fixing_date=date(2026, 7, 15),
        ),
    )


def _conditional_accrual_leg() -> ConditionalAccrualLeg:
    return ConditionalAccrualLeg(
        currency="usd",
        notional_schedule=_notional(),
        accrual_periods=_conditional_periods(),
        coupon_formula=FixedCouponFormula(0.0525),
        day_count="ACT/365",
        payment_frequency="quarterly",
        accrual_condition_ref="sofr_in_range",
        accrual_counter_ref="in_range_coupon_count",
        settlement_rule="coupon_period_cash_settlement",
        label="range_coupon",
        metadata={"semantic_family": "range_accrual"},
    )


def test_conditional_accrual_leg_is_frozen_and_normalized():
    leg = _conditional_accrual_leg()

    assert leg.currency == "USD"
    assert leg.payment_frequency == "quarterly"
    assert leg.accrual_condition_ref == "sofr_in_range"
    assert leg.accrual_counter_ref == "in_range_coupon_count"
    assert leg.settlement_rule == "coupon_period_cash_settlement"
    assert leg.metadata["semantic_family"] == "range_accrual"

    with pytest.raises(FrozenInstanceError):
        leg.currency = "EUR"
    with pytest.raises(TypeError):
        leg.metadata["semantic_family"] = "other"


def test_conditional_accrual_leg_rejects_missing_condition_or_coupon():
    with pytest.raises(
        StaticLegIRWellFormednessError,
        match="ConditionalAccrualLeg.accrual_condition_ref",
    ):
        ConditionalAccrualLeg(
            currency="USD",
            notional_schedule=_notional(),
            accrual_periods=_conditional_periods(),
            coupon_formula=FixedCouponFormula(0.0525),
            day_count="ACT/365",
            payment_frequency="quarterly",
            accrual_condition_ref="",
            accrual_counter_ref="in_range_coupon_count",
        )

    with pytest.raises(
        StaticLegIRWellFormednessError,
        match="ConditionalAccrualLeg.coupon_formula",
    ):
        ConditionalAccrualLeg(
            currency="USD",
            notional_schedule=_notional(),
            accrual_periods=_conditional_periods(),
            coupon_formula=None,
            day_count="ACT/365",
            payment_frequency="quarterly",
            accrual_condition_ref="sofr_in_range",
            accrual_counter_ref="in_range_coupon_count",
        )


def test_conditional_accrual_period_rejects_observation_after_accrual_window():
    with pytest.raises(
        StaticLegIRWellFormednessError,
        match="observation_date must be on or before accrual_end",
    ):
        ConditionalAccrualPeriod(
            accrual_start=date(2026, 1, 15),
            accrual_end=date(2026, 4, 15),
            observation_date=date(2026, 4, 16),
            payment_date=date(2026, 4, 17),
            fixing_date=date(2026, 4, 16),
        )


def test_conditional_accrual_leg_represents_single_index_range_accrual_terms():
    leg = _conditional_accrual_leg()

    assert leg.notional_schedule.initial_notional == pytest.approx(1_000_000.0)
    assert tuple(period.observation_date for period in leg.accrual_periods) == (
        date(2026, 4, 15),
        date(2026, 7, 15),
    )
    assert leg.coupon_formula.rate == pytest.approx(0.0525)
    assert leg.day_count == "ACT/365"

    contract = StaticLegContractIR(
        legs=(SignedLeg(direction="receive", leg=leg),),
        metadata={"semantic_family": "range_accrual"},
    )

    assert contract.currencies == ("USD",)
    execution_ir = compile_static_leg_execution_ir(contract)
    assert execution_ir.obligations == ()
    assert execution_ir.execution_metadata.unsupported_reasons
    assert execution_ir.execution_metadata.unsupported_reasons == (
        "static-leg execution lowering not admitted: No admissible static-leg lowering declaration was found.",
    )

    with pytest.raises(UnsupportedExecutionSemantics, match="No admissible static-leg lowering declaration was found"):
        compile_static_leg_execution_ir(contract, fail_on_unsupported=True)
