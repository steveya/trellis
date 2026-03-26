"""Payoff protocol and deterministic cashflow adapter."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Instrument


@runtime_checkable
class Payoff(Protocol):
    """Protocol for anything that can be priced from a MarketState.

    ``evaluate()`` returns the present value as a float.
    Each payoff handles its own discounting internally.
    """

    @property
    def requirements(self) -> set[str]:
        """Capability names this payoff needs from MarketState."""
        ...

    def evaluate(self, market_state: MarketState) -> float:
        """Compute the present value given market data.

        The payoff is responsible for all discounting. The returned
        float is the final PV — ``price_payoff()`` returns it directly.
        """
        ...


class DeterministicCashflowPayoff:
    """Adapter: wraps any Instrument (e.g. Bond) into the Payoff protocol.

    Discounts each cashflow using ``market_state.discount``. This is the
    bridge between instrument objects that expose dated cashflows and the
    payoff protocol that expects a single present-value evaluation.

    Mathematically, this computes

    .. math::

       PV = \sum_i CF_i \cdot D(t_i)

    where ``CF_i`` are the future cashflows and ``D(t_i)`` are discount
    factors from the session or market state.
    """

    def __init__(self, instrument: Instrument,
                 day_count: DayCountConvention = DayCountConvention.ACT_365):
        """Wrap an instrument plus the day-count convention for discount timing."""
        self._instrument = instrument
        self._day_count = day_count

    @property
    def instrument(self) -> Instrument:
        """Return the wrapped instrument object."""
        return self._instrument

    @property
    def requirements(self) -> set[str]:
        """Declare that deterministic cashflow pricing needs a discount curve."""
        return {"discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Discount each future cashflow date and sum the resulting PV."""
        schedule = self._instrument.cashflows(market_state.settlement)
        pv = 0.0
        for d, amt in zip(schedule.dates, schedule.amounts):
            t = year_fraction(market_state.settlement, d, self._day_count)
            pv += amt * market_state.discount.discount(t)
        return pv


# Backward-compat aliases (deprecated)
class Cashflows:
    """Deprecated. evaluate() now returns float directly."""
    def __init__(self, flows):
        """Store legacy raw cashflow tuples for backward compatibility only."""
        self.flows = flows

class PresentValue:
    """Deprecated. evaluate() now returns float directly."""
    def __init__(self, pv):
        """Store a legacy pre-discounted scalar PV for backward compatibility."""
        self.pv = pv
