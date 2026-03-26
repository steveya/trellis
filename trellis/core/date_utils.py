"""Date utilities: day count conventions, schedule generation, year fractions."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Union

from trellis.core.types import DayCountConvention, Frequency

DateLike = Union[date, datetime]


def _to_date(d: DateLike) -> date:
    """Normalize ``date``/``datetime`` inputs to plain ``date`` objects."""
    if isinstance(d, datetime):
        return d.date()
    return d


def add_months(dt: DateLike, months: int) -> date:
    """Add *months* calendar months, clamping the day to the new month's max."""
    d = _to_date(dt)
    total_months = d.month - 1 + months
    year = d.year + total_months // 12
    month = total_months % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def year_fraction(d1: DateLike, d2: DateLike,
                  convention: DayCountConvention = DayCountConvention.ACT_365,
                  **kwargs) -> float:
    """Compute the year fraction between two dates under *convention*.

    Delegates to :func:`trellis.conventions.day_count.year_fraction` which
    supports all conventions including ACT/ACT ICMA and BUS/252.
    """
    from trellis.conventions.day_count import year_fraction as _yf
    return _yf(d1, d2, convention, **kwargs)


def generate_schedule(start: DateLike, end: DateLike,
                      frequency: Frequency, **kwargs) -> list[date]:
    """Generate coupon dates from *start* (exclusive) to *end* (inclusive).

    Delegates to :func:`trellis.conventions.schedule.generate_schedule` which
    supports calendars, stub types, and roll conventions.
    """
    from trellis.conventions.schedule import generate_schedule as _gs
    return _gs(start, end, frequency, **kwargs)


def get_bracketing_dates(start: DateLike, end: DateLike,
                         frequency: Frequency, query: DateLike) -> tuple[date, date]:
    """Return (period_start, period_end) containing *query*."""
    start_d, end_d, query_d = _to_date(start), _to_date(end), _to_date(query)
    if query_d < start_d or query_d > end_d:
        raise ValueError("Date is not in the schedule")
    months_per_period = 12 // frequency.value
    for i in range(1, frequency.value + 1):
        upper = add_months(start_d, months_per_period * i)
        if query_d <= upper:
            lower = add_months(start_d, months_per_period * (i - 1))
            return (lower, upper)
    raise ValueError("Date is not in the schedule")


def get_accrual_fraction(start: DateLike, end: DateLike,
                         frequency: Frequency, query: DateLike) -> float:
    """Return the accrual fraction for the period containing *query*."""
    lower, upper = get_bracketing_dates(start, end, frequency, query)
    return (query if isinstance(query, datetime) else datetime.combine(query, datetime.min.time())).__sub__(
        datetime.combine(lower, datetime.min.time())
    ).days / (datetime.combine(upper, datetime.min.time()) - datetime.combine(lower, datetime.min.time())).days
