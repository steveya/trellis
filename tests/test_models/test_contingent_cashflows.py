"""Tests for reusable event primitives and contingent cashflow kernels."""

from __future__ import annotations

import pytest

from trellis.models.contingent_cashflows import (
    CouponAccrual,
    PrincipalPayment,
    ProtectionPayment,
    TriggerSettlement,
    coupon_cashflow_pv,
    interval_default_probability_from_survival,
    nth_to_default_probability,
    principal_payment_pv,
    project_prepayment_step,
    protection_payment_pv,
    trigger_settlement_pv,
)


def test_interval_default_probability_from_survival_ratios():
    default_prob = interval_default_probability_from_survival(0.98, 0.95)
    assert default_prob == pytest.approx(1.0 - 0.95 / 0.98)


def test_coupon_cashflow_pv_respects_weight_and_sign():
    pv = coupon_cashflow_pv(
        CouponAccrual(
            notional=1_000_000,
            rate=0.04,
            accrual=0.5,
            discount_factor=0.97,
            weight=0.92,
            sign=-1.0,
        )
    )
    assert pv == pytest.approx(-17_848.0)


def test_protection_payment_pv_respects_recovery_and_sign():
    pv = protection_payment_pv(
        ProtectionPayment(
            notional=2_000_000,
            recovery=0.4,
            default_probability=0.03,
            discount_factor=0.95,
        )
    )
    assert pv == pytest.approx(34_200.0)


def test_principal_payment_pv_combines_scheduled_and_prepaid_principal():
    pv = principal_payment_pv(
        PrincipalPayment(
            scheduled_principal=8_000.0,
            prepaid_principal=2_500.0,
            discount_factor=0.99,
        )
    )
    assert pv == pytest.approx(10_395.0)


def test_trigger_settlement_pv_respects_trigger_weight():
    pv = trigger_settlement_pv(
        TriggerSettlement(
            amount=15_000.0,
            discount_factor=0.96,
            trigger_weight=0.35,
        )
    )
    assert pv == pytest.approx(5_040.0)


def test_project_prepayment_step_preserves_notional_evolution():
    step = project_prepayment_step(
        beginning_balance=100_000.0,
        scheduled_interest=500.0,
        scheduled_principal=1_000.0,
        smm=0.10,
    )

    assert step.prepaid_principal == pytest.approx(9_900.0)
    assert step.total_principal == pytest.approx(10_900.0)
    assert step.remaining_balance == pytest.approx(89_100.0)


def test_nth_to_default_probability_decreases_with_later_trigger():
    first = nth_to_default_probability(5, 1, 0.20, 0.25)
    second = nth_to_default_probability(5, 2, 0.20, 0.25)

    assert 0.0 <= second <= first <= 1.0
