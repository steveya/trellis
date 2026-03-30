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
class CDSPricingSpec:
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
    strike: float
    start_date: date
    end_date: date
    day_count: DayCountConvention
    frequency: Frequency
    rate_index: str | None
    valuation_date: date
    recovery_rate: float = 0.4
    is_payer: bool = True
    include_accrued: bool = True
    settlement_days: int = 3
    pricing_model: str = 'analytical_cds'
    component: str = 'cds_pricing'


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

    def __init__(self, spec: CDSPricingSpec):
        self._spec = spec

    @property
    def spec(self) -> CDSPricingSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"credit_curve", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        credit_curve = market_state.credit_curve
        discount_curve = market_state.discount
        if credit_curve is None:
            raise ValueError("CDSPayoff requires a credit_curve in MarketState")
        if discount_curve is None:
            raise ValueError("CDSPayoff requires a discount curve in MarketState")

        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        period_starts = [spec.start_date] + schedule[:-1]

        premium_leg = 0.0
        protection_leg = 0.0

        prev_survival = credit_curve.survival_probability(
            year_fraction(market_state.settlement, spec.start_date, spec.day_count)
        ) if spec.start_date > market_state.settlement else 1.0

        for p_start, p_end in zip(period_starts, schedule):
            if p_end <= market_state.settlement:
                continue

            accrual = year_fraction(p_start, p_end, spec.day_count)
            t_start = year_fraction(market_state.settlement, p_start, spec.day_count)
            t_end = year_fraction(market_state.settlement, p_end, spec.day_count)

            if t_end < 0.0:
                continue

            surv_start = credit_curve.survival_probability(max(t_start, 0.0))
            surv_end = credit_curve.survival_probability(max(t_end, 0.0))
            df_end = discount_curve.discount(max(t_end, 0.0))

            premium_leg += spec.notional * spec.strike * accrual * df_end * surv_end

            default_prob = max(surv_start - surv_end, 0.0)
            protection_leg += spec.notional * (1.0 - spec.recovery_rate) * df_end * default_prob

            if spec.include_accrued:
                protection_leg += (
                    spec.notional
                    * spec.strike
                    * 0.5
                    * accrual
                    * df_end
                    * default_prob
                )

            prev_survival = surv_end

        pv = protection_leg - premium_leg if spec.is_payer else premium_leg - protection_leg
        return float(pv)