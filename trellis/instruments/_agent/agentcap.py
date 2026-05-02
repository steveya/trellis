"""Agent-generated payoff: Build a pricer for: FinancePy parity: USD cap strip Black

Price a cap strip under the declared benchmark rates surface.

Instrument class: cap.
Strike: 0.04.
Notional: 1000000.0.
Start date: 2024-11-15.
End date: 2029-11-15.
Payment frequency: quarterly.
Day count: ACT/360.
Rate index: USD-SOFR-3M.

Preferred method family: analytical
FinancePy binding: financepy.rates.cap_floor.black
Benchmark product: period_rate_option_strip

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

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
class AgentCapSpec:
    """Specification for Build a pricer for: FinancePy parity: USD cap strip Black

Price a cap strip under the declared benchmark rates surface.

Instrument class: cap.
Strike: 0.04.
Notional: 1000000.0.
Start date: 2024-11-15.
End date: 2029-11-15.
Payment frequency: quarterly.
Day count: ACT/360.
Rate index: USD-SOFR-3M.

Preferred method family: analytical
FinancePy binding: financepy.rates.cap_floor.black
Benchmark product: period_rate_option_strip

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""
    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    model: str | None = None
    shift: float | None = None
    sabr: dict[str, float] | None = None


class AgentCapPayoff:
    """Build a pricer for: FinancePy parity: USD cap strip Black

Price a cap strip under the declared benchmark rates surface.

Instrument class: cap.
Strike: 0.04.
Notional: 1000000.0.
Start date: 2024-11-15.
End date: 2029-11-15.
Payment frequency: quarterly.
Day count: ACT/360.
Rate index: USD-SOFR-3M.

Preferred method family: analytical
FinancePy binding: financepy.rates.cap_floor.black
Benchmark product: period_rate_option_strip

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: AgentCapSpec):
        self._spec = spec

    @property
    def spec(self) -> AgentCapSpec:
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
            label="agent_cap_timeline",
        )
        if not timeline:
            return 0.0
        payoff = build_period_rate_option_execution_payoff(
            spec,
            timeline,
            option_side="call",
            label="agent_cap_timeline",
        )
        return float(payoff.evaluate(market_state))
