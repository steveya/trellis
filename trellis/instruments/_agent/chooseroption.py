"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity chooser option

chooser_option

Spot: 100.0.
Choose date: 2025-05-16.
Call expiry date: 2025-11-15.
Put expiry date: 2025-11-15.
Call strike: 100.0.
Put strike: 100.0.

Preferred method family: analytical
FinancePy binding: financepy.equity.chooser.black_scholes
Benchmark product: chooser_option

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
from trellis.models.analytical import price_equity_chooser_option_analytical



@dataclass(frozen=True)
class ChooserOptionSpec:
    """Specification for Build a pricer for: FinancePy parity: equity chooser option

chooser_option

Spot: 100.0.
Choose date: 2025-05-16.
Call expiry date: 2025-11-15.
Put expiry date: 2025-11-15.
Call strike: 100.0.
Put strike: 100.0.

Preferred method family: analytical
FinancePy binding: financepy.equity.chooser.black_scholes
Benchmark product: chooser_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""
    notional: float
    spot: float
    choose_date: date
    call_expiry_date: date
    put_expiry_date: date
    call_strike: float
    put_strike: float
    day_count: DayCountConvention = DayCountConvention.ACT_365


class ChooserOptionPayoff:
    """Build a pricer for: FinancePy parity: equity chooser option

chooser_option

Spot: 100.0.
Choose date: 2025-05-16.
Call expiry date: 2025-11-15.
Put expiry date: 2025-11-15.
Call strike: 100.0.
Put strike: 100.0.

Preferred method family: analytical
FinancePy binding: financepy.equity.chooser.black_scholes
Benchmark product: chooser_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: ChooserOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> ChooserOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        if market_state.vol_surface is None:
            raise ValueError("ChooserOptionPayoff requires a black vol surface in market_state")
        if market_state.discount is None:
            raise ValueError("ChooserOptionPayoff requires a discount curve in market_state")

        return float(price_equity_chooser_option_analytical(market_state, spec))
