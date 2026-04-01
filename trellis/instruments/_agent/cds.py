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

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



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
        return {"credit", "discount"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        spec = self._spec
        from trellis.models.credit_default_swap import build_cds_schedule, price_cds_analytical

        if market_state.discount is None:
            raise ValueError("CDSPayoff.evaluate requires a discount curve in market_state")
        if market_state.credit_curve is None:
            raise ValueError("CDSPayoff.evaluate requires a credit curve in market_state")

        spread = float(spec.spread)
        if spread > 1.0:
            spread *= 1e-4

        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        if not schedule or schedule[0] != spec.start_date:
            schedule = [spec.start_date] + [d for d in schedule if d > spec.start_date]
        if schedule[-1] != spec.end_date:
            schedule = [d for d in schedule if d < spec.end_date] + [spec.end_date]

        periods = build_cds_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            day_count=spec.day_count,
            time_origin=spec.start_date,
        )

        try:
            return float(
                price_cds_analytical(
                    notional=spec.notional,
                    spread=spread,
                    recovery=spec.recovery,
                    periods=periods,
                    market_state=market_state,
                )
            )
        except TypeError:
            premium_leg = 0.0
            protection_leg = 0.0
            prev_t = 0.0
            prev_surv = 1.0

            for pay_date in schedule[1:]:
                t = year_fraction(spec.start_date, pay_date, spec.day_count)
                surv = market_state.credit_curve.survival_probability(t)
                df = market_state.discount.discount(t)
                accrual = t - prev_t
                premium_leg += spread * spec.notional * accrual * df * surv
                protection_leg += spec.notional * (1.0 - spec.recovery) * df * (prev_surv - surv)
                prev_t = t
                prev_surv = surv

            return float(protection_leg - premium_leg)
