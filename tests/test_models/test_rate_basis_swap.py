from __future__ import annotations

from datetime import date

import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
from trellis.models.rate_basis_swap import (
    BasisSwapFloatingLegPeriod,
    BasisSwapFloatingLegSpec,
    RateBasisSwapSpec,
    price_rate_basis_swap,
)


SETTLE = date(2024, 11, 15)


def test_rate_basis_swap_prices_from_explicit_leg_periods():
    discount_curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0325,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    sofr_curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0410,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    ff_curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0365,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=discount_curve,
        forecast_curves={
            "SOFR": sofr_curve,
            "FF": ff_curve,
        },
    )
    spec = RateBasisSwapSpec(
        pay_leg=BasisSwapFloatingLegSpec(
            notional=1_000_000.0,
            periods=(
                BasisSwapFloatingLegPeriod(
                    accrual_start=date(2025, 1, 15),
                    accrual_end=date(2025, 4, 15),
                    payment_date=date(2025, 4, 15),
                    fixing_date=date(2025, 1, 15),
                ),
                BasisSwapFloatingLegPeriod(
                    accrual_start=date(2025, 4, 15),
                    accrual_end=date(2025, 7, 15),
                    payment_date=date(2025, 7, 15),
                    fixing_date=date(2025, 4, 15),
                ),
            ),
            day_count=DayCountConvention.ACT_360,
            rate_index="SOFR",
            spread=0.0,
        ),
        receive_leg=BasisSwapFloatingLegSpec(
            notional=1_000_000.0,
            periods=(
                BasisSwapFloatingLegPeriod(
                    accrual_start=date(2025, 1, 15),
                    accrual_end=date(2025, 4, 15),
                    payment_date=date(2025, 4, 15),
                    fixing_date=date(2025, 1, 15),
                ),
                BasisSwapFloatingLegPeriod(
                    accrual_start=date(2025, 4, 15),
                    accrual_end=date(2025, 7, 15),
                    payment_date=date(2025, 7, 15),
                    fixing_date=date(2025, 4, 15),
                ),
            ),
            day_count=DayCountConvention.ACT_360,
            rate_index="FF",
            spread=0.0025,
        ),
    )

    pv = price_rate_basis_swap(market_state, spec)

    expected = 0.0
    for period in spec.receive_leg.periods:
        accrual = year_fraction(
            period.accrual_start,
            period.accrual_end,
            DayCountConvention.ACT_360,
        )
        forward = ff_curve.forward_rate_dates(
            period.accrual_start,
            period.accrual_end,
            day_count=DayCountConvention.ACT_360,
        )
        discount_factor = discount_curve.discount_date(period.payment_date)
        expected += 1_000_000.0 * (forward + 0.0025) * accrual * discount_factor
    for period in spec.pay_leg.periods:
        accrual = year_fraction(
            period.accrual_start,
            period.accrual_end,
            DayCountConvention.ACT_360,
        )
        forward = sofr_curve.forward_rate_dates(
            period.accrual_start,
            period.accrual_end,
            day_count=DayCountConvention.ACT_360,
        )
        discount_factor = discount_curve.discount_date(period.payment_date)
        expected -= 1_000_000.0 * forward * accrual * discount_factor

    assert pv == pytest.approx(expected)
