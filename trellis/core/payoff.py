"""Payoff protocol, return types, and deterministic cashflow adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Union, runtime_checkable

from trellis.core.market_state import MarketState
from trellis.core.types import Instrument


# ---------------------------------------------------------------------------
# Return types for evaluate()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cashflows:
    """Undiscounted cashflows — price_payoff() will discount each one.

    Use this for analytical/cashflow-based pricing (bonds, caps, swaps, swaptions).

    Example::

        return Cashflows([(date(2025,5,15), 2.5), (date(2025,11,15), 102.5)])
    """

    flows: list[tuple[date, float]]


@dataclass(frozen=True)
class PresentValue:
    """Already-discounted present value — price_payoff() returns it directly.

    Use this when the pricing method handles its own discounting
    (tree backward induction, Monte Carlo, PDE solvers).

    Example::

        tree_price = backward_induction(tree, payoff_fn, r)
        return PresentValue(tree_price)
    """

    pv: float


# Union type for evaluate() return
EvaluateResult = Union[Cashflows, PresentValue, list[tuple[date, float]]]


# ---------------------------------------------------------------------------
# Payoff protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Payoff(Protocol):
    """Protocol for anything that can be priced from a MarketState."""

    @property
    def requirements(self) -> set[str]:
        """Capability names this payoff needs from MarketState."""
        ...

    def evaluate(self, market_state: MarketState) -> EvaluateResult:
        """Evaluate the payoff given market data.

        Returns one of:
        - ``Cashflows([(date, amount), ...])`` — undiscounted cashflows.
          ``price_payoff()`` will discount each one.
        - ``PresentValue(pv)`` — already-discounted PV.
          ``price_payoff()`` returns it directly.
        - ``list[tuple[date, float]]`` — backward-compatible, treated as Cashflows.
        """
        ...


# ---------------------------------------------------------------------------
# Deterministic cashflow adapter
# ---------------------------------------------------------------------------

class DeterministicCashflowPayoff:
    """Adapter: wraps any Instrument (e.g. Bond) into the Payoff protocol.

    The wrapped instrument's cashflows are deterministic (no model dependency),
    so the only requirement is "discount" for pricing.
    """

    def __init__(self, instrument: Instrument):
        self._instrument = instrument

    @property
    def instrument(self) -> Instrument:
        return self._instrument

    @property
    def requirements(self) -> set[str]:
        return {"discount"}

    def evaluate(self, market_state: MarketState) -> Cashflows:
        """Generate cashflows from the underlying instrument."""
        schedule = self._instrument.cashflows(market_state.settlement)
        return Cashflows(list(zip(schedule.dates, schedule.amounts)))
