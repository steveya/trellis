"""Agent-generated payoff: Build a pricer for: CDS par spread: hazard rate bootstrap vs closed-form

Bootstrap a hazard rate curve from CDS par spreads, then reprice
each input CDS to verify the curve reproduces the market spreads.
Input CDS maturities and par spreads:
  1Y: 50 bp, 2Y: 80 bp, 3Y: 100 bp, 5Y: 150 bp, 7Y: 200 bp.
Recovery rate: 40%.  Flat risk-free rate: 3%.
Method 1: piecewise-constant hazard rate bootstrap (solve for each
segment so that the model CDS spread matches the market spread).
Method 2: closed-form analytical CDS pricing using the bootstrapped
curve — verify round-trip consistency (repriced spreads should match
input spreads to within 0.1 bp).

Comparison targets: bootstrapped_cds (analytical), analytical_cds (analytical)
Cross-validation harness:
  internal targets: bootstrapped_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_bootstrap

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
    """Specification for Build a pricer for: CDS par spread: hazard rate bootstrap vs closed-form

Bootstrap a hazard rate curve from CDS par spreads, then reprice
each input CDS to verify the curve reproduces the market spreads.
Input CDS maturities and par spreads:
  1Y: 50 bp, 2Y: 80 bp, 3Y: 100 bp, 5Y: 150 bp, 7Y: 200 bp.
Recovery rate: 40%.  Flat risk-free rate: 3%.
Method 1: piecewise-constant hazard rate bootstrap (solve for each
segment so that the model CDS spread matches the market spread).
Method 2: closed-form analytical CDS pricing using the bootstrapped
curve — verify round-trip consistency (repriced spreads should match
input spreads to within 0.1 bp).

Comparison targets: bootstrapped_cds (analytical), analytical_cds (analytical)
Cross-validation harness:
  internal targets: bootstrapped_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_bootstrap

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
    """Build a pricer for: CDS par spread: hazard rate bootstrap vs closed-form

Bootstrap a hazard rate curve from CDS par spreads, then reprice
each input CDS to verify the curve reproduces the market spreads.
Input CDS maturities and par spreads:
  1Y: 50 bp, 2Y: 80 bp, 3Y: 100 bp, 5Y: 150 bp, 7Y: 200 bp.
Recovery rate: 40%.  Flat risk-free rate: 3%.
Method 1: piecewise-constant hazard rate bootstrap (solve for each
segment so that the model CDS spread matches the market spread).
Method 2: closed-form analytical CDS pricing using the bootstrapped
curve — verify round-trip consistency (repriced spreads should match
input spreads to within 0.1 bp).

Comparison targets: bootstrapped_cds (analytical), analytical_cds (analytical)
Cross-validation harness:
  internal targets: bootstrapped_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_bootstrap

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
        spread = float(spec.spread)
        if spread > 1.0:
            spread /= 10000.0

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
