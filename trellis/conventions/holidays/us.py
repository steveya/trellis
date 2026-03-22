"""US Settlement holidays."""

from __future__ import annotations

from datetime import date

from trellis.conventions.holidays import easter, nth_weekday, observed


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """US federal holidays (observed rules)."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(observed(date(y, 1, 1)))                    # New Year
        dates.add(nth_weekday(y, 1, 0, 3))                    # MLK Day (3rd Mon Jan)
        dates.add(nth_weekday(y, 2, 0, 3))                    # Presidents Day (3rd Mon Feb)
        dates.add(nth_weekday(y, 5, 0, -1))                   # Memorial Day (last Mon May)
        dates.add(observed(date(y, 6, 19)))                    # Juneteenth
        dates.add(observed(date(y, 7, 4)))                     # Independence Day
        dates.add(nth_weekday(y, 9, 0, 1))                    # Labor Day (1st Mon Sep)
        dates.add(nth_weekday(y, 10, 0, 2))                   # Columbus Day (2nd Mon Oct)
        dates.add(observed(date(y, 11, 11)))                   # Veterans Day
        dates.add(nth_weekday(y, 11, 3, 4))                   # Thanksgiving (4th Thu Nov)
        dates.add(observed(date(y, 12, 25)))                   # Christmas
    return frozenset(dates)
