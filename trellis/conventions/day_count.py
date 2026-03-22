"""Day count conventions and year fraction computation."""

from __future__ import annotations

import calendar as _cal
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Union

from trellis.core.types import Frequency

if TYPE_CHECKING:
    from trellis.conventions.calendar import Calendar

DateLike = Union[date, datetime]


class DayCountConvention(Enum):
    """Day count conventions for year fraction computation."""

    # Original values (backward compat — these are canonical)
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"

    # New canonical names (aliases for existing values)
    ACT_365_FIXED = "ACT/365"       # alias for ACT_365
    ACT_ACT_ISDA = "ACT/ACT"        # alias for ACT_ACT
    THIRTY_360_US = "30/360"         # alias for THIRTY_360

    # New conventions with distinct values
    ACT_ACT_ICMA = "ACT/ACT ICMA"
    THIRTY_E_360 = "30E/360"
    THIRTY_E_360_ISDA = "30E/360 ISDA"
    ACT_365_25 = "ACT/365.25"
    BUS_252 = "BUS/252"
    ONE_ONE = "1/1"


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    return d


def year_fraction(
    d1: DateLike,
    d2: DateLike,
    convention: DayCountConvention = DayCountConvention.ACT_365,
    *,
    ref_start: DateLike | None = None,
    ref_end: DateLike | None = None,
    frequency: Frequency | None = None,
    calendar: Calendar | None = None,
) -> float:
    """Compute the year fraction between two dates under *convention*.

    Parameters
    ----------
    d1, d2 : date or datetime
        Start and end dates.
    convention : DayCountConvention
        Day count convention.
    ref_start, ref_end : date or None
        Accrual period boundaries (required for ACT/ACT ICMA).
    frequency : Frequency or None
        Coupons per year (required for ACT/ACT ICMA).
    calendar : Calendar or None
        Business day calendar (required for BUS/252).
    """
    d1, d2 = _to_date(d1), _to_date(d2)
    delta_days = (d2 - d1).days

    if convention in (DayCountConvention.ACT_360,):
        return delta_days / 360.0

    elif convention in (DayCountConvention.ACT_365, DayCountConvention.ACT_365_FIXED):
        return delta_days / 365.0

    elif convention == DayCountConvention.ACT_365_25:
        return delta_days / 365.25

    elif convention in (DayCountConvention.ACT_ACT, DayCountConvention.ACT_ACT_ISDA):
        return _act_act_isda(d1, d2)

    elif convention == DayCountConvention.ACT_ACT_ICMA:
        if ref_start is None or ref_end is None or frequency is None:
            raise ValueError(
                "ACT/ACT ICMA requires ref_start, ref_end, and frequency"
            )
        rs, re = _to_date(ref_start), _to_date(ref_end)
        period_days = (re - rs).days
        if period_days == 0:
            return 0.0
        return delta_days / (frequency.value * period_days)

    elif convention in (DayCountConvention.THIRTY_360, DayCountConvention.THIRTY_360_US):
        return _thirty_360_us(d1, d2)

    elif convention == DayCountConvention.THIRTY_E_360:
        return _thirty_e_360(d1, d2)

    elif convention == DayCountConvention.THIRTY_E_360_ISDA:
        return _thirty_e_360_isda(d1, d2)

    elif convention == DayCountConvention.BUS_252:
        if calendar is None:
            raise ValueError("BUS/252 requires a calendar")
        return calendar.business_days_between(d1, d2) / 252.0

    elif convention == DayCountConvention.ONE_ONE:
        return 1.0

    else:
        raise ValueError(f"Unsupported convention: {convention}")


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

def _act_act_isda(d1: date, d2: date) -> float:
    if d1.year == d2.year:
        days_in_year = 366 if _cal.isleap(d1.year) else 365
        return (d2 - d1).days / days_in_year
    frac = 0.0
    # First partial year: d1 to Jan 1 of next year
    start_of_next = date(d1.year + 1, 1, 1)
    days_first = (start_of_next - d1).days
    days_in_first_year = 366 if _cal.isleap(d1.year) else 365
    frac += days_first / days_in_first_year
    # Full middle years
    for y in range(d1.year + 1, d2.year):
        frac += 1.0
    # Last partial year: Jan 1 of d2's year to d2
    start_of_last_year = date(d2.year, 1, 1)
    days_last = (d2 - start_of_last_year).days
    days_in_last_year = 366 if _cal.isleap(d2.year) else 365
    frac += days_last / days_in_last_year
    return frac


def _thirty_360_us(d1: date, d2: date) -> float:
    y1, m1, day1 = d1.year, d1.month, min(d1.day, 30)
    y2, m2, day2 = d2.year, d2.month, d2.day
    # US rule: d2 clamped to 30 only if d1 was clamped (d1.day >= 30)
    if day2 == 31 and day1 >= 30:
        day2 = 30
    else:
        day2 = min(day2, 31)  # no clamp unless d1 triggered it
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0


def _thirty_e_360(d1: date, d2: date) -> float:
    y1, m1, day1 = d1.year, d1.month, min(d1.day, 30)
    y2, m2, day2 = d2.year, d2.month, min(d2.day, 30)
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0


def _thirty_e_360_isda(d1: date, d2: date) -> float:
    y1, m1, day1 = d1.year, d1.month, d1.day
    y2, m2, day2 = d2.year, d2.month, d2.day
    if day1 == _cal.monthrange(y1, m1)[1]:
        day1 = 30
    if day2 == _cal.monthrange(y2, m2)[1] and m2 != 2:
        day2 = 30
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0
