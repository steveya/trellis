"""Core domain layer for Trellis.

This package collects the financial-domain primitives that pricing engines and
agent-generated payoffs depend on: market state, payoff protocols, state-space
objects, core types, and capability checks.
"""

from trellis.core.capabilities import (
    analyze_gap,
    capability_summary,
    check_market_data,
    discover_capabilities,
)
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import (
    Cashflows,
    DeterministicCashflowPayoff,
    Payoff,
    PresentValue,
)
from trellis.core.state_space import StateSpace
from trellis.core.types import DayCountConvention, Frequency, PricingResult

__all__ = [
    "MarketState",
    "MissingCapabilityError",
    "Payoff",
    "DeterministicCashflowPayoff",
    "Cashflows",
    "PresentValue",
    "StateSpace",
    "Frequency",
    "DayCountConvention",
    "PricingResult",
    "analyze_gap",
    "check_market_data",
    "discover_capabilities",
    "capability_summary",
]
