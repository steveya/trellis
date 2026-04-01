"""Core protocols, enums, and data structures shared across Trellis.

Defines the fundamental types that most other modules depend on:
frequency enums, day count conventions, the DiscountCurve and Instrument
protocols, and simple data containers like schedules and pricing results.
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


class DslMeasure(str, Enum):
    """Declarative measure type for the DSL pipeline.

    Each member's value is the canonical lowercase string used throughout
    the pipeline.  Inherits from ``str`` so existing code comparing
    ``measure == "dv01"`` or ``measure in some_set`` works unchanged.
    """

    PRICE = "price"
    DV01 = "dv01"
    DURATION = "duration"
    CONVEXITY = "convexity"
    KEY_RATE_DURATIONS = "key_rate_durations"
    VEGA = "vega"
    DELTA = "delta"
    GAMMA = "gamma"
    THETA = "theta"
    RHO = "rho"
    OAS = "oas"
    Z_SPREAD = "z_spread"
    SCENARIO_PNL = "scenario_pnl"


_DSL_MEASURE_ALIASES: dict[str, DslMeasure] = {
    "krd": DslMeasure.KEY_RATE_DURATIONS,
    "modified_duration": DslMeasure.DURATION,
    "pv": DslMeasure.PRICE,
    "npv": DslMeasure.PRICE,
    "pv01": DslMeasure.DV01,
    "zspread": DslMeasure.Z_SPREAD,
    "z-spread": DslMeasure.Z_SPREAD,
    "scenario": DslMeasure.SCENARIO_PNL,
}


def normalize_dsl_measure(name: str) -> DslMeasure:
    """Resolve a string measure name to the canonical ``DslMeasure`` enum.

    Raises ``ValueError`` for unknown names.
    """
    key = name.strip().lower().replace(" ", "_")
    alias = _DSL_MEASURE_ALIASES.get(key)
    if alias is not None:
        return alias
    try:
        return DslMeasure(key)
    except ValueError:
        raise ValueError(
            f"Unknown measure: {key!r}. "
            f"Known: {sorted(m.value for m in DslMeasure)}"
        ) from None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Frequency(Enum):
    """Coupon or reset frequency expressed as payments per year."""
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12


class TimelineRole(Enum):
    """Semantic role carried by a contract timeline."""

    PAYMENT = "payment"
    OBSERVATION = "observation"
    EXERCISE = "exercise"
    RESET = "reset"
    SETTLEMENT = "settlement"



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


@dataclass(frozen=True)
class SchedulePeriod:
    """One explicit accrual/payment period in a pricing schedule.

    The date fields are always populated. Time and accrual fields are optional
    so routes can use the same object with or without a chosen time origin and
    day-count convention.
    """

    start_date: date
    end_date: date
    payment_date: date
    accrual_fraction: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    t_payment: float | None = None


@dataclass(frozen=True)
class EventSchedule:
    """Immutable sequence of explicit schedule periods for event-style routes."""

    start_date: date
    end_date: date
    frequency: Frequency
    day_count: DayCountConvention | None
    time_origin: date | None
    periods: tuple[SchedulePeriod, ...]

    def __iter__(self):
        return iter(self.periods)

    def __len__(self) -> int:
        return len(self.periods)

    def __getitem__(self, item):
        return self.periods[item]

    @property
    def payment_dates(self) -> tuple[date, ...]:
        """Return the payment dates in order."""
        return tuple(period.payment_date for period in self.periods)

    @property
    def period_end_dates(self) -> tuple[date, ...]:
        """Return the accrual end dates in order."""
        return tuple(period.end_date for period in self.periods)


@dataclass(frozen=True)
class ContractTimeline:
    """Role-typed timeline used by contract and route helpers.

    This wraps explicit schedule periods with a semantic role so downstream
    code can distinguish payment, observation, exercise, reset, and settlement
    timelines without rebuilding those meanings from local context.
    """

    role: TimelineRole
    start_date: date
    end_date: date
    frequency: Frequency | None
    day_count: DayCountConvention | None
    time_origin: date | None
    periods: tuple[SchedulePeriod, ...]
    label: str | None = None

    def __iter__(self):
        return iter(self.periods)

    def __len__(self) -> int:
        return len(self.periods)

    def __getitem__(self, item):
        return self.periods[item]

    @property
    def event_dates(self) -> tuple[date, ...]:
        """Return the primary event dates in order."""
        return tuple(period.payment_date for period in self.periods)

    @property
    def payment_dates(self) -> tuple[date, ...]:
        """Return payment dates for timelines that emit cashflows."""
        return tuple(period.payment_date for period in self.periods)

    @property
    def period_end_dates(self) -> tuple[date, ...]:
        """Return the period end dates in order."""
        return tuple(period.end_date for period in self.periods)


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
