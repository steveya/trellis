"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity barrier Black-Scholes

Price a continuously monitored barrier option under the declared analytical benchmark surface.

Spot: 100.0.
Strike: 100.0.
Barrier: 120.0.
Barrier type: up_and_out.
Option type: call.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.barrier.black_scholes
Benchmark product: barrier_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.core.date_utils import year_fraction



@dataclass(frozen=True)
class BarrierOptionSpec:
    """Specification for Build a pricer for: FinancePy parity: equity barrier Black-Scholes

Price a continuously monitored barrier option under the declared analytical benchmark surface.

Spot: 100.0.
Strike: 100.0.
Barrier: 120.0.
Barrier type: up_and_out.
Option type: call.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.barrier.black_scholes
Benchmark product: barrier_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""
    notional: float
    spot: float
    strike: float
    barrier: float
    expiry_date: date
    barrier_type: str
    option_type: str = 'call'
    rebate: float = 0.0
    observations_per_year: int | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class BarrierOptionPayoff:
    """Build a pricer for: FinancePy parity: equity barrier Black-Scholes

Price a continuously monitored barrier option under the declared analytical benchmark surface.

Spot: 100.0.
Strike: 100.0.
Barrier: 120.0.
Barrier type: up_and_out.
Option type: call.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.barrier.black_scholes
Benchmark product: barrier_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: BarrierOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BarrierOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        from trellis.models.analytical.barrier import barrier_option_price

        if market_state.discount is None:
            raise ValueError("BarrierOptionPayoff requires market_state.discount")
        if market_state.vol_surface is None:
            raise ValueError("BarrierOptionPayoff requires market_state.vol_surface")
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            return 0.0

        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        rate = float(market_state.discount.zero_rate(T))
        carry_rates = dict(getattr(market_state, "model_parameters", None) or {}).get("underlier_carry_rates") or {}
        carry_rate = 0.0
        if isinstance(carry_rates, dict) and len(carry_rates) == 1:
            carry_rate = float(next(iter(carry_rates.values())))
        pv = barrier_option_price(
            float(spec.spot),
            float(spec.strike),
            float(spec.barrier),
            rate,
            sigma,
            T,
            barrier_type=spec.barrier_type,
            option_type=spec.option_type,
            rebate=float(spec.rebate),
            q=carry_rate,
            observations_per_year=spec.observations_per_year,
        )
        return float(spec.notional) * float(pv)
