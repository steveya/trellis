"""Convention-aware schedule generation."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Union

from datetime import datetime

from trellis.conventions.calendar import BusinessDayAdjustment, Calendar
from trellis.core.types import Frequency

DateLike = Union[date, datetime]


def _to_date(d: DateLike) -> date:
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
    SHORT_FIRST = "short_first"
    SHORT_LAST = "short_last"
    LONG_FIRST = "long_first"
    LONG_LAST = "long_last"


class RollConvention(Enum):
    NONE = "none"
    EOM = "eom"
    IMM = "imm"


def _is_eom(d: date) -> bool:
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


def _generate_forward(start: date, end: date, months: int, eom: bool) -> list[date]:
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
