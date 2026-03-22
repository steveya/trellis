"""Tokyo (Japan) holidays."""

from __future__ import annotations

import math
from datetime import date

from trellis.conventions.holidays import nth_weekday, observed


def _vernal_equinox(year: int) -> date:
    """Approximate vernal equinox for Japan (valid ~1980-2099)."""
    day = int(20.8431 + 0.242194 * (year - 1980) - int((year - 1980) / 4))
    return date(year, 3, day)


def _autumnal_equinox(year: int) -> date:
    """Approximate autumnal equinox for Japan (valid ~1980-2099)."""
    day = int(23.2488 + 0.242194 * (year - 1980) - int((year - 1980) / 4))
    return date(year, 9, day)


def holidays(start_year: int = 2000, end_year: int = 2075) -> frozenset[date]:
    """Japanese national holidays."""
    dates: set[date] = set()
    for y in range(start_year, end_year + 1):
        dates.add(date(y, 1, 1))                                  # New Year
        dates.add(date(y, 1, 2))                                  # Bank Holiday
        dates.add(date(y, 1, 3))                                  # Bank Holiday
        dates.add(nth_weekday(y, 1, 0, 2))                       # Coming of Age (2nd Mon)
        dates.add(date(y, 2, 11))                                 # National Foundation
        if y >= 2020:
            dates.add(date(y, 2, 23))                             # Emperor's Birthday
        dates.add(_vernal_equinox(y))                             # Vernal Equinox
        dates.add(date(y, 4, 29))                                 # Showa Day
        dates.add(date(y, 5, 3))                                  # Constitution Day
        dates.add(date(y, 5, 4))                                  # Greenery Day
        dates.add(date(y, 5, 5))                                  # Children's Day
        dates.add(nth_weekday(y, 7, 0, 3))                       # Marine Day (3rd Mon)
        if y >= 2016:
            dates.add(date(y, 8, 11))                             # Mountain Day
        dates.add(nth_weekday(y, 9, 0, 3))                       # Respect for Aged (3rd Mon)
        dates.add(_autumnal_equinox(y))                           # Autumnal Equinox
        dates.add(nth_weekday(y, 10, 0, 2))                      # Sports Day (2nd Mon)
        dates.add(date(y, 11, 3))                                 # Culture Day
        dates.add(date(y, 11, 23))                                # Labour Thanksgiving
        dates.add(date(y, 12, 31))                                # Bank Holiday
    return frozenset(dates)
