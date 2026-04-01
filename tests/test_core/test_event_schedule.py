"""Tests for the explicit periodized schedule substrate."""

from datetime import date

import pytest

from trellis.conventions.calendar import WEEKEND_ONLY, BusinessDayAdjustment
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.date_utils import (
    build_contract_timeline,
    build_exercise_timeline_from_dates,
    build_period_schedule,
)
from trellis.core.types import DayCountConvention, Frequency, TimelineRole


class TestEventSchedule:

    def test_build_period_schedule_returns_explicit_periods(self):
        schedule = build_period_schedule(
            date(2024, 1, 15),
            date(2025, 1, 15),
            Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_360,
            time_origin=date(2024, 1, 15),
        )

        assert len(schedule) == 4
        first = schedule.periods[0]
        assert first.start_date == date(2024, 1, 15)
        assert first.end_date == date(2024, 4, 15)
        assert first.payment_date == date(2024, 4, 15)
        assert first.accrual_fraction == pytest.approx(91 / 360)
        assert first.t_start == pytest.approx(0.0)
        assert first.t_end == pytest.approx(91 / 360)
        assert first.t_payment == pytest.approx(91 / 360)
        assert schedule[0] == first
        assert schedule[-1].end_date == date(2025, 1, 15)

    def test_build_period_schedule_preserves_stub_structure(self):
        schedule = build_period_schedule(
            date(2024, 1, 15),
            date(2025, 2, 15),
            Frequency.QUARTERLY,
            stub=StubType.SHORT_FIRST,
        )

        assert schedule.periods[0].start_date == date(2024, 1, 15)
        assert schedule.periods[0].end_date == date(2024, 2, 15)
        assert schedule.periods[-1].end_date == date(2025, 2, 15)

    def test_build_period_schedule_supports_eom_and_payment_lag(self):
        schedule = build_period_schedule(
            date(2024, 1, 31),
            date(2024, 3, 31),
            Frequency.MONTHLY,
            roll_convention=RollConvention.EOM,
            calendar=WEEKEND_ONLY,
            bda=BusinessDayAdjustment.FOLLOWING,
            payment_lag_days=2,
        )

        first = schedule.periods[0]
        assert first.end_date == date(2024, 2, 29)
        assert first.payment_date == date(2024, 3, 4)
        second = schedule.periods[1]
        assert second.end_date == date(2024, 4, 1)
        assert second.payment_date == date(2024, 4, 3)

    def test_time_origin_requires_day_count(self):
        with pytest.raises(ValueError, match="day_count is required"):
            build_period_schedule(
                date(2024, 1, 15),
                date(2024, 7, 15),
                Frequency.SEMI_ANNUAL,
                time_origin=date(2024, 1, 15),
            )

    def test_build_contract_timeline_carries_role(self):
        timeline = build_contract_timeline(
            date(2024, 1, 15),
            date(2024, 7, 15),
            Frequency.QUARTERLY,
            role=TimelineRole.PAYMENT,
            day_count=DayCountConvention.ACT_360,
            time_origin=date(2024, 1, 15),
        )

        assert timeline.role is TimelineRole.PAYMENT
        assert timeline.label is None
        assert timeline.event_dates[0] == date(2024, 4, 15)
        assert timeline[0].accrual_fraction == pytest.approx(91 / 360)

    def test_build_exercise_timeline_from_dates_supports_explicit_events(self):
        timeline = build_exercise_timeline_from_dates(
            [date(2026, 3, 15), date(2025, 9, 15), date(2025, 3, 15)],
            day_count=DayCountConvention.ACT_365,
            time_origin=date(2025, 3, 15),
        )

        assert timeline.role is TimelineRole.EXERCISE
        assert timeline.frequency is None
        assert timeline.event_dates == (
            date(2025, 3, 15),
            date(2025, 9, 15),
            date(2026, 3, 15),
        )
        assert timeline[0].t_payment == pytest.approx(0.0)
        assert timeline[-1].t_payment is not None
