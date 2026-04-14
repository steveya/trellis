"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity compound option

compound_option

Spot: 100.0.
Outer option type: call.
Inner option type: call.
Outer strike: 12.0.
Inner strike: 100.0.
Outer expiry date: 2025-05-16.
Inner expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.compound.black_scholes
Benchmark product: compound_option

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
from trellis.models.analytical import price_equity_compound_option_analytical



@dataclass(frozen=True)
class CompoundOptionSpec:
    """Specification for Build a pricer for: FinancePy parity: equity compound option

compound_option

Spot: 100.0.
Outer option type: call.
Inner option type: call.
Outer strike: 12.0.
Inner strike: 100.0.
Outer expiry date: 2025-05-16.
Inner expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.compound.black_scholes
Benchmark product: compound_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""
    notional: float
    spot: float
    outer_expiry_date: date
    inner_expiry_date: date
    outer_strike: float
    inner_strike: float
    outer_option_type: str = 'call'
    inner_option_type: str = 'call'
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CompoundOptionPayoff:
    """Build a pricer for: FinancePy parity: equity compound option

compound_option

Spot: 100.0.
Outer option type: call.
Inner option type: call.
Outer strike: 12.0.
Inner strike: 100.0.
Outer expiry date: 2025-05-16.
Inner expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.compound.black_scholes
Benchmark product: compound_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: CompoundOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> CompoundOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        return float(price_equity_compound_option_analytical(market_state, spec))
