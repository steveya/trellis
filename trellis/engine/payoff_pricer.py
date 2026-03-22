"""Pricing function for Payoff objects."""

from __future__ import annotations

from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import Payoff


def price_payoff(
    payoff: Payoff,
    market_state: MarketState,
    **kwargs,
) -> float:
    """Price a payoff against a MarketState.

    Each payoff handles its own discounting — this function simply
    checks market data requirements, calls ``evaluate()``, and returns
    the result.

    Parameters
    ----------
    payoff : Payoff
    market_state : MarketState
    **kwargs
        Ignored (kept for backward compatibility with ``day_count=`` callers).

    Raises
    ------
    MissingCapabilityError
        If market_state lacks required market data.
    """
    from trellis.core.capabilities import check_market_data, _MARKET_DATA_NAMES
    market_data_reqs = payoff.requirements & _MARKET_DATA_NAMES
    errors = check_market_data(market_data_reqs, market_state)
    if errors:
        raise MissingCapabilityError(
            payoff.requirements - market_state.available_capabilities,
            market_state.available_capabilities,
            details=errors,
        )

    return payoff.evaluate(market_state)
