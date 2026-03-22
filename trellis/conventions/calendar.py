"""Business day calendars and adjustment conventions."""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum


class BusinessDayAdjustment(Enum):
    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"


class Calendar:
    """Business day calendar based on weekend mask + holiday set.

    Parameters
    ----------
    name : str
        Human-readable name.
    weekend_days : frozenset[int]
        Non-business days (0=Monday ... 6=Sunday). Default: {5, 6}.
    holidays : frozenset[date]
        Specific holiday dates.
    """

    def __init__(
        self,
        name: str = "WeekendOnly",
        weekend_days: frozenset[int] = frozenset({5, 6}),
        holidays: frozenset[date] = frozenset(),
    ):
        self.name = name
        self._weekend_days = weekend_days
        self._holidays = holidays

    def is_business_day(self, d: date) -> bool:
        return d.weekday() not in self._weekend_days and d not in self._holidays

    def adjust(
        self,
        d: date,
        convention: BusinessDayAdjustment = BusinessDayAdjustment.MODIFIED_FOLLOWING,
    ) -> date:
        """Adjust date to a business day per the given convention."""
        if convention == BusinessDayAdjustment.UNADJUSTED:
            return d

        if convention == BusinessDayAdjustment.FOLLOWING:
            return self._next_business_day(d)

        if convention == BusinessDayAdjustment.PRECEDING:
            return self._prev_business_day(d)

        if convention == BusinessDayAdjustment.MODIFIED_FOLLOWING:
            adjusted = self._next_business_day(d)
            if adjusted.month != d.month:
                adjusted = self._prev_business_day(d)
            return adjusted

        if convention == BusinessDayAdjustment.MODIFIED_PRECEDING:
            adjusted = self._prev_business_day(d)
            if adjusted.month != d.month:
                adjusted = self._next_business_day(d)
            return adjusted

        raise ValueError(f"Unknown convention: {convention}")

    def business_days_between(self, d1: date, d2: date) -> int:
        """Count business days in the half-open interval (d1, d2]."""
        if d2 <= d1:
            return 0
        count = 0
        current = d1 + timedelta(days=1)
        while current <= d2:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def add_business_days(self, d: date, n: int) -> date:
        """Add n business days to d. Negative n goes backward."""
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        current = d
        while remaining > 0:
            current += timedelta(days=step)
            if self.is_business_day(current):
                remaining -= 1
        return current

    @property
    def holidays(self) -> frozenset[date]:
        return self._holidays

    def _next_business_day(self, d: date) -> date:
        while not self.is_business_day(d):
            d += timedelta(days=1)
        return d

    def _prev_business_day(self, d: date) -> date:
        while not self.is_business_day(d):
            d -= timedelta(days=1)
        return d

    def __repr__(self) -> str:
        return f"Calendar({self.name!r})"


class JointCalendar(Calendar):
    """Combines multiple calendars. A day is a business day only if it
    is a business day in ALL constituent calendars."""

    def __init__(self, *calendars: Calendar, name: str | None = None):
        combined_name = name or " + ".join(c.name for c in calendars)
        combined_weekends = frozenset().union(*(c._weekend_days for c in calendars))
        combined_holidays = frozenset().union(*(c._holidays for c in calendars))
        super().__init__(combined_name, combined_weekends, combined_holidays)


# ---------------------------------------------------------------------------
# Built-in calendar singletons
# ---------------------------------------------------------------------------

def _build_calendar(name: str, holiday_module: str) -> Calendar:
    import importlib
    mod = importlib.import_module(f"trellis.conventions.holidays.{holiday_module}")
    return Calendar(name, holidays=mod.holidays())


# Lazy singletons to avoid import-time overhead
_CALENDAR_CACHE: dict[str, Calendar] = {}


def _get_calendar(name: str, module: str) -> Calendar:
    if name not in _CALENDAR_CACHE:
        _CALENDAR_CACHE[name] = _build_calendar(name, module)
    return _CALENDAR_CACHE[name]


# Weekend-only (no holidays, always available)
WEEKEND_ONLY = Calendar("WeekendOnly")


class _CalendarProxy:
    """Lazy proxy that builds the calendar on first access."""

    def __init__(self, name: str, module: str):
        self._name = name
        self._module = module
        self._calendar: Calendar | None = None

    def _resolve(self) -> Calendar:
        if self._calendar is None:
            self._calendar = _build_calendar(self._name, self._module)
        return self._calendar

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)

    def __repr__(self):
        return f"Calendar({self._name!r})"


US_SETTLEMENT: Calendar = _CalendarProxy("USSettlement", "us")  # type: ignore[assignment]
UK_SETTLEMENT: Calendar = _CalendarProxy("UKSettlement", "uk")  # type: ignore[assignment]
TARGET: Calendar = _CalendarProxy("TARGET", "eu")  # type: ignore[assignment]
TOKYO: Calendar = _CalendarProxy("Tokyo", "jp")  # type: ignore[assignment]
SYDNEY: Calendar = _CalendarProxy("Sydney", "au")  # type: ignore[assignment]
TORONTO: Calendar = _CalendarProxy("Toronto", "ca")  # type: ignore[assignment]
ZURICH: Calendar = _CalendarProxy("Zurich", "ch")  # type: ignore[assignment]
BRAZIL: Calendar = _CalendarProxy("Brazil", "br")  # type: ignore[assignment]
