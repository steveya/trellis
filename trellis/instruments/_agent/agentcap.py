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
Benchmark product: rate_cap_floor_strip

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
from trellis.models.rate_cap_floor import price_rate_cap_floor_strip_analytical



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
Benchmark product: rate_cap_floor_strip

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
Benchmark product: rate_cap_floor_strip

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
        return float(
            price_rate_cap_floor_strip_analytical(
                market_state,
                spec=spec,
                instrument_class="cap",
                notional=spec.notional,
                strike=spec.strike,
                start_date=spec.start_date,
                end_date=spec.end_date,
                frequency=spec.frequency,
                day_count=spec.day_count,
                rate_index=spec.rate_index,
                model=spec.model,
                shift=spec.shift,
                sabr=spec.sabr,
            )
        )
