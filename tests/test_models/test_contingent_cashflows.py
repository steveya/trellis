"""Tests for reusable event primitives and contingent cashflow kernels."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import build_period_schedule
from trellis.core.types import Frequency
from trellis.models.contingent_cashflows import (
    CouponAccrual,
    DefaultEventGrid,
    FirstEventWeights,
    PrincipalPayment,
    ProtectionPayment,
    TriggerSettlement,
    build_default_event_grid,
    conditional_event_probabilities_from_curve,
    coupon_cashflow_pv,
    expected_first_event_weights,
    interval_default_probability_from_survival,
    nth_to_default_probability,
    principal_payment_pv,
    project_prepayment_step,
    protection_payment_pv,
    sample_first_event_weights,
    trigger_settlement_pv,
)


class _FlatHazardCurve:
    def __init__(self, hazard: float):
        self.hazard = hazard

    def survival_probability(self, t: float) -> float:
        from math import exp

        return exp(-self.hazard * t)


def test_default_event_grid_keeps_curve_time_separate_from_coupon_day_count():
    schedule = build_period_schedule(
        date(2025, 1, 15),
        date(2025, 7, 15),
        Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        time_origin=date(2025, 2, 15),
    )

    grid = build_default_event_grid(schedule, steps_per_year=4)

    assert isinstance(grid, DefaultEventGrid)
    assert grid.periods == schedule.periods
    assert grid.period_interval_stops == (1, 2)
    assert grid.intervals[0].start_time == pytest.approx(0.0)
    assert grid.intervals[0].end_time == pytest.approx(59.0 / 365.0)
    assert grid.period_payment_times[0] == pytest.approx(59.0 / 365.0)
    assert grid.elapsed_period_fractions[0] == pytest.approx(31.0 / 90.0)
    assert grid.elapsed_period_fractions[1] == 0.0
    with pytest.raises(FrozenInstanceError):
        grid.period_interval_stops = ()


def test_default_event_grid_uses_coupon_day_count_for_elapsed_fraction():
    schedule = build_period_schedule(
        date(2025, 1, 31),
        date(2025, 4, 30),
        Frequency.QUARTERLY,
        day_count=DayCountConvention.THIRTY_E_360,
        time_origin=date(2025, 2, 28),
    )

    grid = build_default_event_grid(schedule, steps_per_year=4)

    assert schedule.periods[0].accrual_fraction == pytest.approx(0.25)
    assert grid.elapsed_period_fractions[0] == pytest.approx(28.0 / 90.0)
    assert grid.intervals[0].settlement_date == date(2025, 3, 31)
    assert grid.intervals[0].period_fraction_elapsed == pytest.approx(60.0 / 90.0)


def test_default_event_grid_requires_measured_periods_and_time_origin():
    schedule = build_period_schedule(
        date(2025, 1, 15),
        date(2025, 4, 15),
        Frequency.QUARTERLY,
    )

    with pytest.raises(ValueError, match="time_origin"):
        build_default_event_grid(schedule)


def test_expected_first_event_weights_preserve_unconditional_event_mass():
    weights = expected_first_event_weights((0.10, 0.20, 0.25))

    assert isinstance(weights, FirstEventWeights)
    assert weights.event_weights == pytest.approx((0.10, 0.18, 0.18))
    assert weights.survival_weights == pytest.approx((0.90, 0.72, 0.54))
    with pytest.raises(FrozenInstanceError):
        weights.event_weights = ()


def test_first_event_weights_include_survival_to_a_forward_start():
    exact = expected_first_event_weights(
        (0.10, 0.20),
        initial_survival_weight=0.80,
    )
    sampled = sample_first_event_weights(
        (0.10, 0.20),
        initial_survival_weight=0.80,
        n_paths=400_000,
        seed=17,
    )

    assert exact.event_weights == pytest.approx((0.08, 0.144))
    assert exact.survival_weights == pytest.approx((0.72, 0.576))
    assert sampled.event_weights == pytest.approx(exact.event_weights, abs=1.2e-3)
    assert sampled.survival_weights == pytest.approx(exact.survival_weights, abs=1.2e-3)


def test_curve_probabilities_and_sampled_first_event_weights_are_reproducible():
    schedule = build_period_schedule(
        date(2025, 1, 1),
        date(2026, 1, 1),
        Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_365,
        time_origin=date(2025, 1, 1),
    )
    grid = build_default_event_grid(schedule, steps_per_year=4)
    probabilities = conditional_event_probabilities_from_curve(
        _FlatHazardCurve(0.08),
        grid.intervals,
    )

    first = sample_first_event_weights(probabilities, n_paths=200_000, seed=17)
    second = sample_first_event_weights(probabilities, n_paths=200_000, seed=17)
    expected = expected_first_event_weights(probabilities)

    assert first == second
    assert first.event_weights == pytest.approx(expected.event_weights, abs=8e-4)
    assert first.survival_weights == pytest.approx(expected.survival_weights, abs=8e-4)


def test_sample_first_event_weights_rejects_non_positive_path_count():
    with pytest.raises(ValueError, match="n_paths"):
        sample_first_event_weights((0.1,), n_paths=0, seed=42)


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
