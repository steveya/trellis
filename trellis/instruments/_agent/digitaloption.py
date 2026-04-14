"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity digital cash-or-nothing

digital_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Payout type: cash_or_nothing.
Cash payoff: 10.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.digital.black_scholes
Benchmark product: digital_option

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
from trellis.models.analytical import price_equity_digital_option_analytical



@dataclass(frozen=True)
class DigitalOptionSpec:
    """Specification for Build a pricer for: FinancePy parity: equity digital cash-or-nothing

digital_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Payout type: cash_or_nothing.
Cash payoff: 10.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.digital.black_scholes
Benchmark product: digital_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = 'call'
    payout_type: str = 'cash_or_nothing'
    cash_payoff: float = 1.0
    day_count: DayCountConvention = DayCountConvention.ACT_365


class DigitalOptionPayoff:
    """Build a pricer for: FinancePy parity: equity digital cash-or-nothing

digital_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Payout type: cash_or_nothing.
Cash payoff: 10.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.digital.black_scholes
Benchmark product: digital_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: DigitalOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> DigitalOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        spec = self._spec
        from trellis.models.analytical import price_equity_digital_option_analytical

        return float(price_equity_digital_option_analytical(market_state, spec))
