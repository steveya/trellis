"""Agent-generated payoff: Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import build_payment_timeline
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.instruments._agent._period_rate_option_static_leg import (
    build_period_rate_option_execution_payoff,
)



@dataclass(frozen=True)
class AgentFloorSpec:
    """Specification for Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""
    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None


class AgentFloorPayoff:
    """Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""

    def __init__(self, spec: AgentFloorSpec):
        self._spec = spec

    @property
    def spec(self) -> AgentFloorSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        timeline = build_payment_timeline(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            day_count=spec.day_count,
            time_origin=market_state.settlement,
            label="agent_floor_timeline",
        )
        if not timeline:
            return 0.0
        payoff = build_period_rate_option_execution_payoff(
            spec,
            timeline,
            option_side="put",
            label="agent_floor_timeline",
        )
        return float(payoff.evaluate(market_state))
