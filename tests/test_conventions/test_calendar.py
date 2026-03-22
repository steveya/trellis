"""Tests for Calendar class and built-in calendars."""

from datetime import date, timedelta

import pytest

from trellis.conventions.calendar import (
    BRAZIL, BusinessDayAdjustment, Calendar, JointCalendar,
    TARGET, UK_SETTLEMENT, US_SETTLEMENT, WEEKEND_ONLY,
)


class TestWeekendOnly:

    def test_weekday_is_business(self):
        assert WEEKEND_ONLY.is_business_day(date(2024, 11, 18))  # Monday

    def test_saturday_is_not_business(self):
        assert not WEEKEND_ONLY.is_business_day(date(2024, 11, 16))

    def test_sunday_is_not_business(self):
        assert not WEEKEND_ONLY.is_business_day(date(2024, 11, 17))


class TestUSSettlement:

    def test_christmas_2024(self):
        assert not US_SETTLEMENT.is_business_day(date(2024, 12, 25))

    def test_july_4_observed_2025(self):
        # July 4, 2025 is a Friday
        assert not US_SETTLEMENT.is_business_day(date(2025, 7, 4))

    def test_regular_business_day(self):
        assert US_SETTLEMENT.is_business_day(date(2024, 11, 18))  # Monday

    def test_thanksgiving_2024(self):
        # 4th Thursday of November 2024 = Nov 28
        assert not US_SETTLEMENT.is_business_day(date(2024, 11, 28))


class TestTARGET:

    def test_good_friday_2024(self):
        assert not TARGET.is_business_day(date(2024, 3, 29))

    def test_labour_day(self):
        assert not TARGET.is_business_day(date(2024, 5, 1))


class TestBrazil:

    def test_carnival_2024(self):
        # Carnival Tuesday 2024 = Feb 13
        assert not BRAZIL.is_business_day(date(2024, 2, 13))

    def test_independence_day(self):
        assert not BRAZIL.is_business_day(date(2024, 9, 7))


class TestAdjustment:

    def test_following(self):
        cal = WEEKEND_ONLY
        # Saturday Nov 16 → Monday Nov 18
        assert cal.adjust(date(2024, 11, 16), BusinessDayAdjustment.FOLLOWING) == date(2024, 11, 18)

    def test_preceding(self):
        cal = WEEKEND_ONLY
        # Saturday Nov 16 → Friday Nov 15
        assert cal.adjust(date(2024, 11, 16), BusinessDayAdjustment.PRECEDING) == date(2024, 11, 15)

    def test_modified_following_same_month(self):
        cal = WEEKEND_ONLY
        # Saturday Nov 16 → Monday Nov 18 (same month, ok)
        assert cal.adjust(date(2024, 11, 16), BusinessDayAdjustment.MODIFIED_FOLLOWING) == date(2024, 11, 18)

    def test_modified_following_crosses_month(self):
        cal = WEEKEND_ONLY
        # Saturday Nov 30 → following would be Dec 2, but modified goes back to Fri Nov 29
        assert cal.adjust(date(2024, 11, 30), BusinessDayAdjustment.MODIFIED_FOLLOWING) == date(2024, 11, 29)

    def test_unadjusted(self):
        cal = WEEKEND_ONLY
        assert cal.adjust(date(2024, 11, 16), BusinessDayAdjustment.UNADJUSTED) == date(2024, 11, 16)

    def test_business_day_unchanged(self):
        cal = WEEKEND_ONLY
        assert cal.adjust(date(2024, 11, 18), BusinessDayAdjustment.FOLLOWING) == date(2024, 11, 18)


class TestBusinessDaysBetween:

    def test_one_week(self):
        # Mon Jan 1 to Mon Jan 8: Tue-Fri + Mon = 5 business days
        assert WEEKEND_ONLY.business_days_between(date(2024, 1, 1), date(2024, 1, 8)) == 5

    def test_same_day(self):
        assert WEEKEND_ONLY.business_days_between(date(2024, 1, 1), date(2024, 1, 1)) == 0

    def test_with_holiday(self):
        # Dec 23-27, 2024: Mon(23), Tue(24), Wed(25=Christmas), Thu(26), Fri(27)
        # Business days (d1, d2] = Tue, Thu, Fri = 3
        assert US_SETTLEMENT.business_days_between(date(2024, 12, 23), date(2024, 12, 27)) == 3


class TestAddBusinessDays:

    def test_add_forward(self):
        # From Monday, add 5 business days → next Monday
        assert WEEKEND_ONLY.add_business_days(date(2024, 1, 1), 5) == date(2024, 1, 8)

    def test_add_backward(self):
        assert WEEKEND_ONLY.add_business_days(date(2024, 1, 8), -5) == date(2024, 1, 1)


class TestJointCalendar:

    def test_union_holidays(self):
        joint = JointCalendar(US_SETTLEMENT, UK_SETTLEMENT)
        # Boxing Day (UK) = Dec 26
        assert not joint.is_business_day(date(2024, 12, 26))
        # Thanksgiving (US) = Nov 28
        assert not joint.is_business_day(date(2024, 11, 28))
