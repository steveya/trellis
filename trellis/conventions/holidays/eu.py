"""TARGET (Trans-European Automated Real-time Gross settlement) holidays."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.conventions.holidays import easter


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """TARGET calendar holidays."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(date(y, 1, 1))                                 # New Year
        e = easter(y)
        dates.add(e - timedelta(days=2))                          # Good Friday
        dates.add(e + timedelta(days=1))                          # Easter Monday
        dates.add(date(y, 5, 1))                                 # Labour Day
        dates.add(date(y, 12, 25))                                # Christmas
        dates.add(date(y, 12, 26))                                # Boxing Day
    return frozenset(dates)
