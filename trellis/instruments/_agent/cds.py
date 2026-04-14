"""Agent-generated payoff: Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: mc_cds
Preferred method family: monte_carlo

Implementation target: mc_cds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.core.differentiable import get_numpy
from trellis.models.credit_default_swap import build_cds_schedule, price_cds_monte_carlo, interval_default_probability



@dataclass(frozen=True)
class CDSSpec:
    """Specification for Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: mc_cds
Preferred method family: monte_carlo

Implementation target: mc_cds."""
    notional: float
    spread: float
    start_date: date
    end_date: date
    recovery: float = 0.4
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    n_paths: int = 250000


class CDSPayoff:
    """Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: mc_cds
Preferred method family: monte_carlo

Implementation target: mc_cds."""

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
        from trellis.core.date_utils import year_fraction
        from trellis.models.credit_default_swap import build_cds_schedule, price_cds_monte_carlo

        np = get_numpy()
        spec = self._spec

        spread = float(spec.spread)
        if spread > 1.0:
            spread *= 1e-4

        schedule = build_cds_schedule(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=spec.start_date,
        )

        credit_curve = market_state.credit_curve
        discount_curve = market_state.discount

        n_paths = int(spec.n_paths) if getattr(spec, "n_paths", None) is not None else 250000
        if n_paths < 10000:
            n_paths = 10000

        try:
            return float(
            price_cds_monte_carlo(
                notional=spec.notional,
                spread_quote=spread,
                recovery=spec.recovery,
                schedule=schedule,
                credit_curve=credit_curve,
                discount_curve=discount_curve,
                n_paths=n_paths,
                seed=42,
            )
        )
        except Exception:
            premium_leg = 0.0
        protection_leg = 0.0
        prev_t = 0.0
        prev_date = spec.start_date

        for period in schedule:
            t_pay = getattr(period, "t_payment", year_fraction(spec.start_date, period.payment_date, spec.day_count))
            accrual_frac = float(getattr(period, "accrual_fraction", year_fraction(prev_date, period.payment_date, spec.day_count)))

            s_prev = float(credit_curve.survival_probability(prev_t))
            s_pay = float(credit_curve.survival_probability(t_pay))
            default_prob = 0.0 if s_prev <= 0.0 else max(0.0, min(1.0, 1.0 - s_pay / s_prev))

            df = float(discount_curve.discount(t_pay))
            premium_leg += spec.notional * spread * accrual_frac * df * s_pay
            protection_leg += spec.notional * (1.0 - spec.recovery) * df * default_prob

            prev_t = t_pay
            prev_date = period.payment_date

        return float(protection_leg - premium_leg)
