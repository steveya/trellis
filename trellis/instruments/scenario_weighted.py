"""ScenarioWeightedPayoff: probability-weighted pricing across discrete states."""

from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import Payoff
from trellis.core.types import DayCountConvention
from trellis.engine.payoff_pricer import price_payoff as _price_payoff


class ScenarioWeightedPayoff:
    """Wraps any Payoff, prices it under each state, returns probability-weighted PV.

    The inner payoff's requirements must be satisfied by each conditional
    MarketState in the StateSpace, not by the outer MarketState.
    """

    def __init__(
        self,
        inner: Payoff,
        *,
        day_count: DayCountConvention = DayCountConvention.ACT_365,
    ):
        self._inner = inner
        self._day_count = day_count

    @property
    def inner(self) -> Payoff:
        return self._inner

    @property
    def requirements(self) -> set[str]:
        return {"state_space"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price inner payoff under each scenario, return weighted PV."""
        state_space = market_state.state_space
        weighted_pv = 0.0

        for state_name in state_space.state_names:
            prob = state_space.probability(state_name)
            cond_ms = state_space.market_state(state_name)

            missing = self._inner.requirements - cond_ms.available_capabilities
            if missing:
                raise MissingCapabilityError(missing, cond_ms.available_capabilities)

            pv = self._inner.evaluate(cond_ms)
            weighted_pv += prob * pv

        return weighted_pv
