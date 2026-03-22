"""Brazil holidays."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.conventions.holidays import easter, observed


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """Brazilian national holidays."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(date(y, 1, 1))                                  # New Year
        e = easter(y)
        dates.add(e - timedelta(days=48))                         # Carnival Monday
        dates.add(e - timedelta(days=47))                         # Carnival Tuesday
        dates.add(e - timedelta(days=2))                          # Good Friday
        dates.add(date(y, 4, 21))                                 # Tiradentes
        dates.add(date(y, 5, 1))                                  # Labour Day
        dates.add(e + timedelta(days=60))                         # Corpus Christi
        dates.add(date(y, 9, 7))                                  # Independence
        dates.add(date(y, 10, 12))                                # Our Lady of Aparecida
        dates.add(date(y, 11, 2))                                 # All Souls
        dates.add(date(y, 11, 15))                                # Republic Proclamation
        dates.add(date(y, 12, 25))                                # Christmas
    return frozenset(dates)
