"""Pricing function for Payoff objects."""

from __future__ import annotations

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import Cashflows, Payoff, PresentValue
from trellis.core.types import DayCountConvention


def price_payoff(
    payoff: Payoff,
    market_state: MarketState,
    *,
    day_count: DayCountConvention = DayCountConvention.ACT_365,
) -> float:
    """Price a payoff against a MarketState.

    Dispatches on the return type of ``payoff.evaluate()``:

    - ``Cashflows`` → discounts each cashflow using ``market_state.discount``
    - ``PresentValue`` → returns the PV directly (method already discounted)
    - ``list[tuple[date, float]]`` → backward-compatible, treated as Cashflows

    Raises
    ------
    MissingCapabilityError
        If market_state lacks required market data.
    """
    # Check market data requirements with helpful error messages
    from trellis.core.capabilities import check_market_data, _MARKET_DATA_NAMES
    market_data_reqs = payoff.requirements & _MARKET_DATA_NAMES
    errors = check_market_data(market_data_reqs, market_state)
    if errors:
        raise MissingCapabilityError(
            payoff.requirements - market_state.available_capabilities,
            market_state.available_capabilities,
            details=errors,
        )

    result = payoff.evaluate(market_state)

    # Dispatch on return type
    if isinstance(result, PresentValue):
        return result.pv

    if isinstance(result, Cashflows):
        flows = result.flows
    elif isinstance(result, list):
        flows = result  # backward-compatible: list[tuple[date, float]]
    else:
        raise TypeError(
            f"evaluate() returned {type(result).__name__}, "
            f"expected Cashflows, PresentValue, or list[tuple[date, float]]"
        )

    # Discount each cashflow
    pv = 0.0
    for cf_date, amount in flows:
        t = year_fraction(market_state.settlement, cf_date, day_count)
        pv += amount * market_state.discount.discount(t)

    return pv
