"""Sydney (Australia) holidays."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.conventions.holidays import easter, nth_weekday, observed


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """Australian national holidays (Sydney)."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(observed(date(y, 1, 1)))                        # New Year
        dates.add(date(y, 1, 26))                                 # Australia Day
        e = easter(y)
        dates.add(e - timedelta(days=2))                          # Good Friday
        dates.add(e + timedelta(days=1))                          # Easter Monday
        dates.add(date(y, 4, 25))                                 # ANZAC Day
        dates.add(nth_weekday(y, 6, 0, 2))                       # Queen's Birthday (2nd Mon Jun)
        dates.add(observed(date(y, 12, 25)))                      # Christmas
        dates.add(observed(date(y, 12, 26)))                      # Boxing Day
    return frozenset(dates)
