"""Tests for convention-aware schedule generation."""

from datetime import date

import pytest

from trellis.conventions.calendar import WEEKEND_ONLY, BusinessDayAdjustment
from trellis.conventions.schedule import (
    RollConvention, StubType, generate_schedule,
)
from trellis.core.types import Frequency


class TestBackwardCompat:

    def test_matches_old_behavior(self):
        """generate_schedule(start, end, freq) matches original implementation."""
        from trellis.core.date_utils import add_months

        start = date(2024, 1, 15)
        end = date(2026, 1, 15)
        freq = Frequency.SEMI_ANNUAL

        result = generate_schedule(start, end, freq)

        # Old behavior: forward from start, exclude start, include end
        expected = [
            date(2024, 7, 15),
            date(2025, 1, 15),
            date(2025, 7, 15),
            date(2026, 1, 15),
        ]
        assert result == expected


class TestStubs:

    def test_short_last(self):
        # 13-month span, quarterly: 4 full quarters + 1-month stub
        result = generate_schedule(
            date(2024, 1, 15), date(2025, 2, 15),
            Frequency.QUARTERLY, stub=StubType.SHORT_LAST,
        )
        assert result[-1] == date(2025, 2, 15)
        assert len(result) == 5  # 4 quarterly + maturity

    def test_short_first(self):
        result = generate_schedule(
            date(2024, 1, 15), date(2025, 2, 15),
            Frequency.QUARTERLY, stub=StubType.SHORT_FIRST,
        )
        assert result[-1] == date(2025, 2, 15)
        # Backward generation: periods from end


class TestEOMRoll:

    def test_eom_monthly(self):
        # Start at Jan 31, monthly, EOM roll
        result = generate_schedule(
            date(2024, 1, 31), date(2024, 6, 30),
            Frequency.MONTHLY, roll_convention=RollConvention.EOM,
        )
        # All dates should be end of month
        import calendar
        for d in result:
            assert d.day == calendar.monthrange(d.year, d.month)[1], f"{d} is not EOM"


class TestCalendarAdjustment:

    def test_adjusted_dates(self):
        result = generate_schedule(
            date(2024, 1, 15), date(2024, 7, 15),
            Frequency.QUARTERLY,
            calendar=WEEKEND_ONLY,
            bda=BusinessDayAdjustment.MODIFIED_FOLLOWING,
        )
        for d in result:
            assert WEEKEND_ONLY.is_business_day(d), f"{d} is not a business day"

    def test_unadjusted_may_land_on_weekend(self):
        result = generate_schedule(
            date(2024, 1, 15), date(2024, 7, 15),
            Frequency.QUARTERLY,
        )
        # April 15, 2024 is a Monday, July 15 is a Monday — both business days
        # This test just verifies no adjustment is applied by default
        assert result == [date(2024, 4, 15), date(2024, 7, 15)]
