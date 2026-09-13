"""Parallel shifts retain dated flat-curve economics and conventions."""

from dataclasses import FrozenInstanceError
from datetime import date
from math import exp

import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
from trellis.curves.forward_curve import ForwardCurve


@pytest.mark.parametrize("bps", [-75.0, 0.0, 25.0])
@pytest.mark.parametrize(
    "curve_day_count",
    [
        DayCountConvention.ACT_ACT_ISDA,
        DayCountConvention.ACT_365,
        DayCountConvention.ACT_360,
    ],
)
def test_parallel_shift_preserves_date_conventions_and_original(bps, curve_day_count):
    value_date = date(2024, 2, 15)
    end_date = date(2025, 2, 15)
    curve = DateAwareFlatYieldCurve(
        value_date, 0.05, curve_day_count=curve_day_count, max_tenor=12.5,
    )

    shifted = curve.shift(bps)

    assert type(shifted) is type(curve)
    assert shifted is not curve
    assert shifted.value_date == value_date
    assert shifted.curve_day_count is curve_day_count
    assert shifted.max_tenor == 12.5
    shifted_rate = 0.05 + bps / 10_000.0
    assert shifted.flat_rate == pytest.approx(shifted_rate)
    assert shifted.zero_rate(3.0) == pytest.approx(shifted_rate)
    assert shifted.discount(3.0) == pytest.approx(exp(-shifted_rate * 3.0))
    maturity = year_fraction(value_date, end_date, curve_day_count)
    assert shifted.discount_date(end_date) == pytest.approx(exp(-shifted_rate * maturity))
    assert shifted.discount_date(value_date) == 1.0
    assert curve.flat_rate == 0.05
    assert curve.discount_date(end_date) == pytest.approx(exp(-0.05 * maturity))
    with pytest.raises(FrozenInstanceError):
        shifted.flat_rate = 0.07


@pytest.mark.parametrize("bps", [-75.0, 0.0, 25.0])
@pytest.mark.parametrize("compounding", ["simple", "continuous"])
def test_shifted_dated_forward_preserves_separate_accrual_convention(bps, compounding):
    curve = DateAwareFlatYieldCurve(
        date(2024, 2, 15), 0.05, curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    ).shift(bps)
    start_date, end_date = date(2024, 11, 15), date(2025, 2, 15)
    t_start = year_fraction(curve.value_date, start_date, curve.curve_day_count)
    t_end = year_fraction(curve.value_date, end_date, curve.curve_day_count)
    alpha = year_fraction(start_date, end_date, DayCountConvention.ACT_360)
    log_growth = (0.05 + bps / 10_000.0) * (t_end - t_start)
    expected = (
        (exp(log_growth) - 1.0) / alpha
        if compounding == "simple" else log_growth / alpha
    )

    for forward_source in (curve, ForwardCurve(curve)):
        assert forward_source.forward_rate_dates(
            start_date,
            end_date,
            day_count=DayCountConvention.ACT_360,
            compounding=compounding,
        ) == pytest.approx(expected, rel=1e-12)
