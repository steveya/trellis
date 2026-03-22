"""Zurich (Switzerland) holidays."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.conventions.holidays import easter, observed


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """Swiss holidays (Zurich)."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(observed(date(y, 1, 1)))                        # New Year
        dates.add(observed(date(y, 1, 2)))                        # Berchtoldstag
        e = easter(y)
        dates.add(e - timedelta(days=2))                          # Good Friday
        dates.add(e + timedelta(days=1))                          # Easter Monday
        dates.add(date(y, 5, 1))                                  # Labour Day
        dates.add(e + timedelta(days=39))                         # Ascension
        dates.add(e + timedelta(days=50))                         # Whit Monday
        dates.add(date(y, 8, 1))                                  # Swiss National Day
        dates.add(observed(date(y, 12, 25)))                      # Christmas
        dates.add(observed(date(y, 12, 26)))                      # St. Stephen's
    return frozenset(dates)
