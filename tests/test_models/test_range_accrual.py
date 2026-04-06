"""Tests for the deterministic range-accrual pricing route."""

from __future__ import annotations

from datetime import date

import pytest


def _range_accrual_spec():
    from trellis.models.range_accrual import RangeAccrualSpec

    return RangeAccrualSpec(
        reference_index="SOFR",
        notional=1_000_000.0,
        coupon_rate=0.0525,
        lower_bound=0.015,
        upper_bound=0.0325,
        observation_dates=(
            date(2026, 1, 15),
            date(2026, 4, 15),
            date(2026, 7, 15),
            date(2026, 10, 15),
        ),
        accrual_start_dates=(
            date(2025, 10, 15),
            date(2026, 1, 15),
            date(2026, 4, 15),
            date(2026, 7, 15),
        ),
        payment_dates=(
            date(2026, 1, 15),
            date(2026, 4, 15),
            date(2026, 7, 15),
            date(2026, 10, 15),
        ),
    )


def test_price_range_accrual_returns_coupon_principal_risk_and_validation_bundle():
    from trellis.core.date_utils import year_fraction
    from trellis.curves.forward_curve import ForwardCurve
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.range_accrual import price_range_accrual

    as_of = date(2026, 4, 4)
    discount_curve = YieldCurve.flat(0.04)
    forecast_curve = YieldCurve.flat(0.025)
    spec = _range_accrual_spec()

    result = price_range_accrual(
        spec,
        as_of=as_of,
        discount_curve=discount_curve,
        forecast_curve=forecast_curve,
        fixing_history={date(2026, 1, 15): 0.02},
    )

    projected = ForwardCurve(forecast_curve)
    expected_coupon_leg = 0.0
    for start_date, observation_date, payment_date in zip(
        spec.accrual_start_dates,
        spec.observation_dates,
        spec.payment_dates,
    ):
        if payment_date <= as_of:
            continue
        accrual = year_fraction(start_date, observation_date)
        horizon = year_fraction(as_of, observation_date)
        projected_rate = projected.forward_rate(0.0, horizon, compounding="simple")
        assert spec.lower_bound <= projected_rate <= spec.upper_bound
        payment_time = year_fraction(as_of, payment_date)
        expected_coupon_leg += (
            spec.notional
            * spec.coupon_rate
            * accrual
            * discount_curve.discount(payment_time)
        )
    expected_principal_leg = spec.notional * discount_curve.discount(
        year_fraction(as_of, spec.payment_dates[-1])
    )

    assert result.price == pytest.approx(expected_coupon_leg + expected_principal_leg)
    assert result.coupon_leg_pv == pytest.approx(expected_coupon_leg)
    assert result.principal_leg_pv == pytest.approx(expected_principal_leg)
    assert result.observed_coupon_count == 1
    assert result.projected_coupon_count == 3
    assert result.risk.parallel_curve_pv01 > 0.0
    assert len(result.scenarios) == 4
    assert result.scenarios[-1].name == "rates_up_100bp"
    assert result.scenarios[-1].price < result.price
    assert result.validation_bundle.route_id == "range_accrual_discounted_cashflow_v1"
    checks = {check.check_id: check for check in result.validation_bundle.checks}
    assert checks["historical_fixing_coverage"].status == "passed"
    assert checks["coupon_leg_reference_bound"].status == "passed"
    assert result.validation_bundle.reference_metrics["max_coupon_leg_pv"] == pytest.approx(
        result.coupon_leg_pv
    )


def test_price_range_accrual_requires_fixings_for_observed_coupon_dates():
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.range_accrual import price_range_accrual

    with pytest.raises(ValueError, match="Missing fixing history"):
        price_range_accrual(
            _range_accrual_spec(),
            as_of=date(2026, 4, 4),
            discount_curve=YieldCurve.flat(0.04),
            forecast_curve=YieldCurve.flat(0.025),
            fixing_history={},
        )
