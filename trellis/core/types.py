"""Core protocols, enums, and data structures for Trellis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Greek specification types
# ---------------------------------------------------------------------------

GreeksSpec = None | list[str] | str  # None=price only, "all", or list of names
KNOWN_GREEKS = frozenset({
    "dv01", "duration", "modified_duration", "convexity", "key_rate_durations",
})


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Frequency(Enum):
    """Coupon or reset frequency expressed as payments per year."""
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12



# DayCountConvention is defined in trellis.conventions.day_count but
# re-exported here for backward compatibility.  Lazy import via __getattr__
# avoids a circular import (conventions.day_count imports Frequency from here).
def __getattr__(name: str):
    """Lazily expose backward-compatible symbols without creating import cycles."""
    if name == "DayCountConvention":
        from trellis.conventions.day_count import DayCountConvention
        globals()["DayCountConvention"] = DayCountConvention  # cache
        return DayCountConvention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CashflowSchedule:
    """A sequence of cashflow dates and amounts."""
    dates: list[date]
    amounts: list[float]


@dataclass
class PricingResult:
    """Container for pricing output."""
    clean_price: float
    dirty_price: float
    accrued_interest: float
    ytm: float | None = None
    greeks: dict[str, float] = field(default_factory=dict)
    curve_sensitivities: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class Instrument(Protocol):
    """Any priceable instrument."""

    def cashflows(self, settlement: date | None = None) -> CashflowSchedule:
        """Return scheduled future cashflows, optionally filtered from ``settlement`` onward."""
        ...

    def price(self, curve: DiscountCurve, settlement: date | None = None) -> float:
        """Return the model price under ``curve`` and optional ``settlement`` date."""
        ...


@runtime_checkable
class DiscountCurve(Protocol):
    """Anything that can produce discount factors."""

    def discount(self, t: float) -> float:
        """Return the discount factor for time *t* (in years)."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return the continuously compounded zero rate for time *t*."""
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Fetches market data."""

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Return {tenor_years: yield} for the given date."""
        ...
