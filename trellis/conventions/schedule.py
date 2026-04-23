"""Convention-aware schedule generation."""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Iterable, Union

from datetime import datetime

from trellis.conventions.day_count import DayCountConvention, year_fraction
from trellis.conventions.calendar import BusinessDayAdjustment, Calendar
from trellis.core.types import ContractTimeline, EventSchedule, Frequency, SchedulePeriod, TimelineRole

DateLike = Union[date, datetime]


def _to_date(d: DateLike) -> date:
    """Normalize ``date``/``datetime`` inputs to plain ``date`` objects."""
    if isinstance(d, datetime):
        return d.date()
    return d


def _add_months(dt: date, months: int) -> date:
    """Add calendar months, clamping the day to the new month's max."""
    import calendar as _cal
    total_months = dt.month - 1 + months
    year = dt.year + total_months // 12
    month = total_months % 12 + 1
    day = min(dt.day, _cal.monthrange(year, month)[1])
    return date(year, month, day)


class StubType(Enum):
    """Placement rule for irregular first or last coupon periods."""
    SHORT_FIRST = "short_first"
    SHORT_LAST = "short_last"
    LONG_FIRST = "long_first"
    LONG_LAST = "long_last"


class RollConvention(Enum):
    """Month-roll rule applied when stepping a schedule by calendar months."""
    NONE = "none"
    EOM = "eom"
    IMM = "imm"


_IMM_MONTHS = (3, 6, 9, 12)


def _is_eom(d: date) -> bool:
    """Return whether ``d`` falls on the last calendar day of its month."""
    import calendar as _cal
    return d.day == _cal.monthrange(d.year, d.month)[1]


def _adjust_eom(d: date) -> date:
    """Move date to end of its month."""
    import calendar as _cal
    return d.replace(day=_cal.monthrange(d.year, d.month)[1])


def generate_schedule(
    start: DateLike,
    end: DateLike,
    frequency: Frequency,
    *,
    calendar: Calendar | None = None,
    bda: BusinessDayAdjustment = BusinessDayAdjustment.UNADJUSTED,
    stub: StubType = StubType.SHORT_LAST,
    roll_convention: RollConvention = RollConvention.NONE,
) -> list[date]:
    """Generate payment dates from *start* (exclusive) to *end* (inclusive).

    Backward compatible: ``generate_schedule(start, end, freq)`` behaves
    identically to the original implementation.
    """
    start_d, end_d = _to_date(start), _to_date(end)
    months_per_period = 12 // frequency.value
    use_eom = (roll_convention == RollConvention.EOM and _is_eom(start_d))

    if roll_convention == RollConvention.IMM:
        if stub != StubType.SHORT_FIRST:
            raise ValueError("IMM roll schedules currently support only SHORT_FIRST stubs")
        return _generate_imm_roll_schedule(
            start_d,
            end_d,
            months_per_period,
            calendar=calendar,
            bda=bda,
        )

    if stub in (StubType.SHORT_LAST, StubType.LONG_LAST):
        dates = _generate_forward(start_d, end_d, months_per_period, use_eom)
        if stub == StubType.LONG_LAST and len(dates) >= 2:
            # Merge the last short stub into the previous period
            if dates[-1] != end_d:
                dates.pop(-1)
    else:
        dates = _generate_backward(start_d, end_d, months_per_period, use_eom)
        if stub == StubType.LONG_FIRST and len(dates) >= 2:
            # Merge the first short stub into the next period
            dates.pop(0)

    # Ensure end date is present
    if not dates or dates[-1] != end_d:
        dates.append(end_d)

    # Apply business day adjustment
    if calendar is not None and bda != BusinessDayAdjustment.UNADJUSTED:
        dates = [calendar.adjust(d, bda) for d in dates]

    return dates


def build_period_schedule(
    start: DateLike,
    end: DateLike,
    frequency: Frequency,
    *,
    calendar: Calendar | None = None,
    bda: BusinessDayAdjustment = BusinessDayAdjustment.UNADJUSTED,
    stub: StubType = StubType.SHORT_LAST,
    roll_convention: RollConvention = RollConvention.NONE,
    day_count: DayCountConvention | None = None,
    time_origin: DateLike | None = None,
    payment_lag_days: int = 0,
) -> EventSchedule:
    """Build an explicit periodized schedule for pricing routes.

    Unlike :func:`generate_schedule`, this returns accrual periods with
    explicit period boundaries, payment dates, optional accrual fractions, and
    optional model times measured from ``time_origin``.
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    origin_d = _to_date(time_origin) if time_origin is not None else None
    if origin_d is not None and day_count is None:
        raise ValueError("day_count is required when time_origin is provided")

    payment_dates = generate_schedule(
        start_d,
        end_d,
        frequency,
        calendar=calendar,
        bda=bda,
        stub=stub,
        roll_convention=roll_convention,
    )
    period_starts = [start_d] + payment_dates[:-1]
    periods: list[SchedulePeriod] = []

    for period_start, period_end in zip(period_starts, payment_dates):
        payment_date = period_end + timedelta(days=payment_lag_days)
        if calendar is not None and bda != BusinessDayAdjustment.UNADJUSTED:
            payment_date = calendar.adjust(payment_date, bda)

        accrual_fraction = None
        if day_count is not None:
            accrual_fraction = year_fraction(
                period_start,
                period_end,
                day_count,
                ref_start=period_start,
                ref_end=period_end,
                frequency=frequency,
                calendar=calendar,
            )

        t_start = t_end = t_payment = None
        if origin_d is not None and day_count is not None:
            t_start = year_fraction(origin_d, period_start, day_count, calendar=calendar)
            t_end = year_fraction(origin_d, period_end, day_count, calendar=calendar)
            t_payment = year_fraction(origin_d, payment_date, day_count, calendar=calendar)

        periods.append(
            SchedulePeriod(
                start_date=period_start,
                end_date=period_end,
                payment_date=payment_date,
                accrual_fraction=accrual_fraction,
                t_start=t_start,
                t_end=t_end,
                t_payment=t_payment,
            )
        )

    return EventSchedule(
        start_date=start_d,
        end_date=end_d,
        frequency=frequency,
        day_count=day_count,
        time_origin=origin_d,
        periods=tuple(periods),
    )


def build_contract_timeline(
    start: DateLike,
    end: DateLike,
    frequency: Frequency,
    *,
    role: TimelineRole,
    label: str | None = None,
    **kwargs,
) -> ContractTimeline:
    """Build a role-typed timeline over a periodic schedule."""
    schedule = build_period_schedule(start, end, frequency, **kwargs)
    return ContractTimeline(
        role=role,
        start_date=schedule.start_date,
        end_date=schedule.end_date,
        frequency=schedule.frequency,
        day_count=schedule.day_count,
        time_origin=schedule.time_origin,
        periods=schedule.periods,
        label=label,
    )


def build_contract_timeline_from_dates(
    dates: Iterable[DateLike],
    *,
    role: TimelineRole,
    day_count: DayCountConvention | None = None,
    time_origin: DateLike | None = None,
    label: str | None = None,
) -> ContractTimeline:
    """Build a role-typed point-event timeline from explicit dates.

    Each event date becomes a zero-width schedule period whose start, end, and
    payment dates are identical. This is appropriate for exercise,
    observation, reset, or settlement events supplied as explicit dates.
    """
    ordered = tuple(sorted(_to_date(item) for item in dates))
    if not ordered:
        raise ValueError("explicit event timelines require at least one date")
    origin_d = _to_date(time_origin) if time_origin is not None else None
    if origin_d is not None and day_count is None:
        raise ValueError("day_count is required when time_origin is provided")

    periods: list[SchedulePeriod] = []
    for event_date in ordered:
        t_event = None
        if origin_d is not None and day_count is not None:
            t_event = year_fraction(origin_d, event_date, day_count)
        periods.append(
            SchedulePeriod(
                start_date=event_date,
                end_date=event_date,
                payment_date=event_date,
                accrual_fraction=0.0,
                t_start=t_event,
                t_end=t_event,
                t_payment=t_event,
            )
        )

    return ContractTimeline(
        role=role,
        start_date=ordered[0],
        end_date=ordered[-1],
        frequency=None,
        day_count=day_count,
        time_origin=origin_d,
        periods=tuple(periods),
        label=label,
    )


def _generate_forward(start: date, end: date, months: int, eom: bool) -> list[date]:
    """Step forward in fixed month increments until the next date would exceed ``end``."""
    dates: list[date] = []
    i = 1
    while True:
        d = _add_months(start, months * i)
        if eom:
            d = _adjust_eom(d)
        if d > end:
            break
        dates.append(d)
        i += 1
    return dates


def _generate_backward(start: date, end: date, months: int, eom: bool) -> list[date]:
    """Step backward from ``end`` in fixed month increments until crossing ``start``."""
    dates: list[date] = []
    i = 1
    while True:
        d = _add_months(end, -months * i)
        if eom:
            d = _adjust_eom(d)
        if d <= start:
            break
        dates.insert(0, d)
        i += 1
    return dates


def _adjust_roll_date(
    d: date,
    *,
    calendar: Calendar | None,
    bda: BusinessDayAdjustment,
) -> date:
    """Return ``d`` after the optional business-day adjustment."""
    if calendar is not None and bda != BusinessDayAdjustment.UNADJUSTED:
        return calendar.adjust(d, bda)
    return d


def _next_imm_roll_after(start: date) -> date:
    """Return the next CDS-style IMM roll date after ``start``."""
    for month in _IMM_MONTHS:
        candidate = date(start.year, month, 20)
        if candidate > start:
            return candidate
    return date(start.year + 1, _IMM_MONTHS[0], 20)


def _generate_imm_roll_schedule(
    start: date,
    end: date,
    months: int,
    *,
    calendar: Calendar | None,
    bda: BusinessDayAdjustment,
) -> list[date]:
    """Generate CDS-style IMM roll dates, using the 20th of IMM months."""
    dates: list[date] = []
    candidate = _next_imm_roll_after(start)
    while True:
        adjusted = _adjust_roll_date(candidate, calendar=calendar, bda=bda)
        if adjusted >= end:
            break
        dates.append(adjusted)
        candidate = _add_months(candidate, months)
        candidate = candidate.replace(day=20)

    final_end = _adjust_roll_date(end, calendar=calendar, bda=bda)
    if not dates or dates[-1] != final_end:
        dates.append(final_end)
    return dates
