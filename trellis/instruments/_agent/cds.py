"""Agent-generated payoff: Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: analytical_cds
Preferred method family: analytical

Implementation target: analytical_cds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.credit_default_swap import build_cds_schedule, price_cds_analytical



@dataclass(frozen=True)
class CDSSpec:
    """Specification for Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: analytical_cds
Preferred method family: analytical

Implementation target: analytical_cds."""
    notional: float
    spread: float
    start_date: date
    end_date: date
    recovery: float = 0.4
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360


class CDSPayoff:
    """Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: analytical_cds
Preferred method family: analytical

Implementation target: analytical_cds."""

    def __init__(self, spec: CDSSpec):
        self._spec = spec

    @property
    def spec(self) -> CDSSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"credit_curve", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        spec = self._spec
        if market_state.credit_curve is None:
            raise ValueError("CDSPayoff requires market_state.credit_curve")
        if market_state.discount is None:
            raise ValueError("CDSPayoff requires market_state.discount")

        spread = spec.spread
        if spread > 1.0:
            spread = spread / 10000.0

        schedule = build_cds_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            spec.day_count,
            time_origin=spec.start_date,
        )

        return float(
            price_cds_analytical(
                notional=spec.notional,
                spread_quote=spread,
                recovery=spec.recovery,
                schedule=schedule,
                credit_curve=market_state.credit_curve,
                discount_curve=market_state.discount,
            )
        )
