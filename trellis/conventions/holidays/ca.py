"""Toronto (Canada) holidays."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.conventions.holidays import easter, nth_weekday, observed


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """Canadian national holidays (Toronto)."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(observed(date(y, 1, 1)))                        # New Year
        dates.add(nth_weekday(y, 2, 0, 3))                       # Family Day (3rd Mon Feb)
        e = easter(y)
        dates.add(e - timedelta(days=2))                          # Good Friday
        mon_before_may25 = date(y, 5, 25)
        while mon_before_may25.weekday() != 0:
            mon_before_may25 -= timedelta(days=1)
        dates.add(mon_before_may25)                               # Victoria Day
        dates.add(observed(date(y, 7, 1)))                        # Canada Day
        dates.add(nth_weekday(y, 8, 0, 1))                       # Civic Holiday (1st Mon Aug)
        dates.add(nth_weekday(y, 9, 0, 1))                       # Labour Day (1st Mon Sep)
        dates.add(nth_weekday(y, 10, 0, 2))                      # Thanksgiving (2nd Mon Oct)
        dates.add(observed(date(y, 12, 25)))                      # Christmas
        dates.add(observed(date(y, 12, 26)))                      # Boxing Day
    return frozenset(dates)
