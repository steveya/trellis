"""WP5: Conventions and edge cases verification.

Day counts against ISDA references, calendar holidays, schedule edge cases,
and pathological pricing inputs.
"""

from datetime import date

import numpy as np
import pytest

from trellis.conventions.day_count import DayCountConvention, year_fraction
from trellis.conventions.calendar import (
    WEEKEND_ONLY, US_SETTLEMENT, TARGET, UK_SETTLEMENT, BusinessDayAdjustment,
)
from trellis.conventions.schedule import generate_schedule, StubType, RollConvention
from trellis.core.types import Frequency
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond


SETTLE = date(2024, 11, 15)


# ---------------------------------------------------------------------------
# Day count conventions — ISDA reference calculations
# ---------------------------------------------------------------------------

class TestDayCountISDA:

    def test_act_360_90_days(self):
        """Jan 1 to Apr 1 2024 = 91 days / 360."""
        frac = year_fraction(date(2024, 1, 1), date(2024, 4, 1), DayCountConvention.ACT_360)
        assert frac == pytest.approx(91 / 360)

    def test_act_365_90_days(self):
        frac = year_fraction(date(2024, 1, 1), date(2024, 4, 1), DayCountConvention.ACT_365)
        assert frac == pytest.approx(91 / 365)

    def test_act_act_isda_full_leap_year(self):
        """Full leap year 2024: Jan 1 to Jan 1 = exactly 1.0."""
        frac = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_ACT)
        assert frac == pytest.approx(1.0, rel=1e-10)

    def test_act_act_isda_full_non_leap_year(self):
        """Full non-leap year 2023: exactly 1.0."""
        frac = year_fraction(date(2023, 1, 1), date(2024, 1, 1), DayCountConvention.ACT_ACT)
        assert frac == pytest.approx(1.0, rel=1e-10)

    def test_act_act_isda_same_year(self):
        """Within a single leap year: 182 days / 366."""
        frac = year_fraction(date(2024, 1, 1), date(2024, 7, 1), DayCountConvention.ACT_ACT)
        assert frac == pytest.approx(182 / 366, rel=1e-6)

    def test_act_act_isda_span_boundary(self):
        """Nov 1 2023 to Mar 1 2024: 61/365 + 60/366."""
        frac = year_fraction(date(2023, 11, 1), date(2024, 3, 1), DayCountConvention.ACT_ACT)
        expected = 61 / 365 + 60 / 366
        assert frac == pytest.approx(expected, rel=1e-6)

    def test_act_act_isda_two_years(self):
        """2Y span: exactly 2.0."""
        frac = year_fraction(date(2023, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_ACT)
        assert frac == pytest.approx(2.0, rel=1e-10)

    def test_30_360_standard(self):
        """Feb 1 to Aug 1 = (360*(0) + 30*6 + 0) / 360 = 0.5."""
        frac = year_fraction(date(2024, 2, 1), date(2024, 8, 1), DayCountConvention.THIRTY_360)
        assert frac == pytest.approx(0.5, rel=1e-6)

    def test_30e_360_month_end(self):
        """Jan 31 to Mar 31: both clamped to 30. = 60/360."""
        frac = year_fraction(date(2024, 1, 31), date(2024, 3, 31), DayCountConvention.THIRTY_E_360)
        assert frac == pytest.approx(60 / 360)

    def test_act_365_25(self):
        """Full year = 366/365.25 for leap year."""
        frac = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_365_25)
        assert frac == pytest.approx(366 / 365.25)

    def test_one_one_always_one(self):
        assert year_fraction(date(2024, 1, 1), date(2024, 1, 2), DayCountConvention.ONE_ONE) == 1.0
        assert year_fraction(date(2024, 1, 1), date(2034, 1, 1), DayCountConvention.ONE_ONE) == 1.0

    def test_bus_252_with_brazil(self):
        from trellis.conventions.calendar import BRAZIL
        frac = year_fraction(
            date(2024, 1, 1), date(2024, 1, 8),
            DayCountConvention.BUS_252,
            calendar=BRAZIL,
        )
        # Business days Jan 2-8 (Jan 1 = holiday): Jan 2,3,4,5(Fri)=4 days, Jan 8(Mon)=5
        assert frac > 0
        assert frac < 1.0


# ---------------------------------------------------------------------------
# Calendar holidays
# ---------------------------------------------------------------------------

class TestCalendarHolidays:

    def test_us_christmas_2024(self):
        assert not US_SETTLEMENT.is_business_day(date(2024, 12, 25))

    def test_us_thanksgiving_2024(self):
        assert not US_SETTLEMENT.is_business_day(date(2024, 11, 28))

    def test_us_july4_2025_friday(self):
        assert not US_SETTLEMENT.is_business_day(date(2025, 7, 4))

    def test_us_regular_business_day(self):
        assert US_SETTLEMENT.is_business_day(date(2024, 11, 18))

    def test_target_good_friday_2024(self):
        assert not TARGET.is_business_day(date(2024, 3, 29))

    def test_target_labour_day(self):
        assert not TARGET.is_business_day(date(2024, 5, 1))

    def test_uk_boxing_day(self):
        assert not UK_SETTLEMENT.is_business_day(date(2024, 12, 26))

    def test_weekend_not_business(self):
        assert not WEEKEND_ONLY.is_business_day(date(2024, 11, 16))  # Saturday
        assert not WEEKEND_ONLY.is_business_day(date(2024, 11, 17))  # Sunday

    def test_modified_following_month_end(self):
        """Sat Nov 30 → modified following → Fri Nov 29 (not Dec 2)."""
        result = WEEKEND_ONLY.adjust(date(2024, 11, 30), BusinessDayAdjustment.MODIFIED_FOLLOWING)
        assert result == date(2024, 11, 29)

    def test_following_basic(self):
        result = WEEKEND_ONLY.adjust(date(2024, 11, 16), BusinessDayAdjustment.FOLLOWING)
        assert result == date(2024, 11, 18)


# ---------------------------------------------------------------------------
# Schedule generation edge cases
# ---------------------------------------------------------------------------

class TestScheduleEdgeCases:

    def test_backward_compat(self):
        """generate_schedule(start, end, freq) matches original behavior."""
        from trellis.core.date_utils import add_months
        start = date(2024, 1, 15)
        end = date(2026, 1, 15)
        result = generate_schedule(start, end, Frequency.SEMI_ANNUAL)
        expected = [date(2024, 7, 15), date(2025, 1, 15), date(2025, 7, 15), date(2026, 1, 15)]
        assert result == expected

    def test_end_date_always_included(self):
        result = generate_schedule(date(2024, 1, 15), date(2026, 3, 20), Frequency.SEMI_ANNUAL)
        assert result[-1] == date(2026, 3, 20)

    def test_eom_roll(self):
        """EOM roll: start Jan 31, monthly → all dates end-of-month."""
        import calendar
        result = generate_schedule(
            date(2024, 1, 31), date(2024, 6, 30),
            Frequency.MONTHLY, roll_convention=RollConvention.EOM,
        )
        for d in result:
            assert d.day == calendar.monthrange(d.year, d.month)[1]


# ---------------------------------------------------------------------------
# Pricing edge cases
# ---------------------------------------------------------------------------

class TestPricingEdgeCases:

    def test_zero_coupon_bond(self):
        zcb = Bond(face=100, coupon=0.0, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2, issue_date=SETTLE)
        result = price_instrument(zcb, YieldCurve.flat(0.05), SETTLE, greeks="all")
        assert result.dirty_price > 0
        assert result.greeks["dv01"] > 0

    def test_very_short_maturity(self):
        """Bond maturing in 6 months."""
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2025, 5, 15),
                     maturity=1, frequency=2)
        result = price_instrument(bond, YieldCurve.flat(0.05), SETTLE, greeks=None)
        assert result.dirty_price > 0

    def test_high_coupon_bond(self):
        """20% coupon bond (extreme but valid)."""
        bond = Bond(face=100, coupon=0.20, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        result = price_instrument(bond, YieldCurve.flat(0.05), SETTLE, greeks=None)
        assert result.dirty_price > 200  # premium bond

    def test_very_low_rates(self):
        """Near-zero rates."""
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        result = price_instrument(bond, YieldCurve.flat(0.001), SETTLE, greeks=None)
        assert result.dirty_price > 100  # all bonds premium at near-zero rates

    def test_high_rates(self):
        """20% rates."""
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        result = price_instrument(bond, YieldCurve.flat(0.20), SETTLE, greeks=None)
        assert result.dirty_price < 50  # deep discount

    def test_flat_curve_shift_symmetry(self):
        """Shift up and down should produce symmetric-ish price changes for small bumps."""
        curve = YieldCurve.flat(0.05)
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        p_base = price_instrument(bond, curve, SETTLE, greeks=None).dirty_price
        p_up = price_instrument(bond, curve.shift(+1), SETTLE, greeks=None).dirty_price
        p_dn = price_instrument(bond, curve.shift(-1), SETTLE, greeks=None).dirty_price
        # For small bumps, changes should be approximately symmetric
        change_up = p_up - p_base
        change_dn = p_dn - p_base
        assert change_up < 0  # rates up → price down
        assert change_dn > 0  # rates down → price up
        # Approximate symmetry (convexity causes slight asymmetry)
        assert abs(change_up + change_dn) < abs(change_up) * 0.1
