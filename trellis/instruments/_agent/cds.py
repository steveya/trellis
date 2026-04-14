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
from trellis.models.credit_default_swap import (
    build_cds_schedule,
    price_cds_analytical,
    price_cds_monte_carlo,
)



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
    valuation_date: date | None = None
    pricing_method: str = "analytical"
    n_paths: int | None = None


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

        spread = float(spec.spread)
        if spread > 1.0:
            spread *= 1e-4

        schedule = build_cds_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            day_count=spec.day_count,
            time_origin=spec.valuation_date or spec.start_date,
        )

        credit_curve = market_state.credit_curve
        discount_curve = market_state.discount
        pricing_method = str(getattr(spec, "pricing_method", "analytical") or "analytical").strip().lower()
        n_paths = getattr(spec, "n_paths", None)

        if pricing_method == "monte_carlo" or (
            pricing_method not in {"", "analytical"} and n_paths is not None
        ):
            path_count = int(n_paths) if n_paths is not None else 250000
            if path_count < 10000:
                path_count = 10000
            return float(
                price_cds_monte_carlo(
                    notional=spec.notional,
                    spread_quote=spread,
                    recovery=spec.recovery,
                    schedule=schedule,
                    credit_curve=credit_curve,
                    discount_curve=discount_curve,
                    n_paths=path_count,
                    seed=42,
                )
            )
        return float(
            price_cds_analytical(
                notional=spec.notional,
                spread_quote=spread,
                recovery=spec.recovery,
                schedule=schedule,
                credit_curve=credit_curve,
                discount_curve=discount_curve,
            )
        )
