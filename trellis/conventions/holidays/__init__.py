"""Holiday computation helpers."""

from __future__ import annotations

from datetime import date, timedelta


def easter(year: int) -> date:
    """Compute Easter Sunday (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def observed(d: date) -> date:
    """Shift Saturday holidays to Friday, Sunday to Monday."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of weekday in month/year.

    weekday: 0=Monday ... 6=Sunday. n: 1-based (1=first, -1=last).
    """
    import calendar as cal
    if n > 0:
        first_day = date(year, month, 1)
        # Days until first occurrence of weekday
        offset = (weekday - first_day.weekday()) % 7
        return first_day + timedelta(days=offset + 7 * (n - 1))
    else:
        last_day_num = cal.monthrange(year, month)[1]
        last_day = date(year, month, last_day_num)
        offset = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=offset + 7 * (-n - 1))
