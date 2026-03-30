"""Core protocols, enums, and data structures shared across Trellis.

Defines the fundamental types that most other modules depend on:
frequency enums, day count conventions, the DiscountCurve and Instrument
protocols, and simple data containers like CashflowSchedule and PricingResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Greek specification types
# ---------------------------------------------------------------------------

# What risk sensitivities to compute alongside the price.
#   None       — price only, no sensitivities
#   "all"      — every known sensitivity (dv01, duration, convexity, etc.)
#   list[str]  — specific sensitivities by name (e.g. ["dv01", "duration"])
GreeksSpec = None | list[str] | str
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
    """Paired lists of payment dates and dollar amounts.

    dates[i] and amounts[i] describe the same cashflow. Both lists must
    have equal length and dates should be in ascending order.
    """
    dates: list[date]
    amounts: list[float]


@dataclass
class PricingResult:
    """Full pricing output for a bond or similar instrument.

    Attributes:
        clean_price: Price excluding interest accrued since the last coupon.
        dirty_price: Price including accrued interest (what you actually pay).
        accrued_interest: Interest accumulated since the last coupon date.
        ytm: Yield to maturity (annualized return if held to maturity), or
            None if not computed.
        greeks: Risk sensitivities keyed by name (e.g. {"dv01": 0.045}).
        curve_sensitivities: Sensitivity of price to each curve tenor
            (e.g. {"1Y": -0.02, "5Y": -0.08}).
    """
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
    """Protocol for curves that convert future values to present values.

    A discount factor D(t) answers: "what is $1 received at time t worth today?"
    For example, D(1.0) = 0.95 means $1 in one year is worth $0.95 now.
    """

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
