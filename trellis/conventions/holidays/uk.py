"""UK Settlement holidays."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.conventions.holidays import easter, nth_weekday, observed


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """UK bank holidays."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(observed(date(y, 1, 1)))                        # New Year
        e = easter(y)
        dates.add(e - timedelta(days=2))                          # Good Friday
        dates.add(e + timedelta(days=1))                          # Easter Monday
        dates.add(nth_weekday(y, 5, 0, 1))                       # Early May Bank Holiday
        dates.add(nth_weekday(y, 5, 0, -1))                      # Spring Bank Holiday
        dates.add(nth_weekday(y, 8, 0, -1))                      # Summer Bank Holiday
        dates.add(observed(date(y, 12, 25)))                      # Christmas
        dates.add(observed(date(y, 12, 26)))                      # Boxing Day
    return frozenset(dates)
