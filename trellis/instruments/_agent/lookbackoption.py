"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity fixed lookback option

lookback_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Lookback type: fixed_strike.
Running extreme: 100.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.lookback.fixed
Benchmark product: lookback_option

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
from trellis.models.analytical import price_equity_fixed_lookback_option_analytical
from trellis.core.date_utils import year_fraction



@dataclass(frozen=True)
class LookbackOptionSpec:
    """Specification for Build a pricer for: FinancePy parity: equity fixed lookback option

lookback_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Lookback type: fixed_strike.
Running extreme: 100.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.lookback.fixed
Benchmark product: lookback_option

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
    lookback_type: str = 'fixed_strike'
    running_extreme: float | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class LookbackOptionPayoff:
    """Build a pricer for: FinancePy parity: equity fixed lookback option

lookback_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Lookback type: fixed_strike.
Running extreme: 100.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.lookback.fixed
Benchmark product: lookback_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: LookbackOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> LookbackOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        from trellis.models.analytical import price_equity_fixed_lookback_option_analytical

        if self._spec.lookback_type != "fixed_strike":
            raise ValueError(f"Unsupported lookback_type: {self._spec.lookback_type!r}")
        if self._spec.option_type not in {"call", "put"}:
            raise ValueError(f"Unsupported option_type: {self._spec.option_type!r}")
        if not hasattr(market_state, "discount") or market_state.discount is None:
            raise ValueError("market_state.discount is required")
        if not hasattr(market_state, "vol_surface") or market_state.vol_surface is None:
            raise ValueError("market_state.vol_surface is required")

        t = year_fraction(market_state.as_of, self._spec.expiry_date, self._spec.day_count)
        if t < 0:
            raise ValueError("Expiry date must not be before valuation date")

        vol = market_state.vol_surface.black_vol(t, self._spec.strike)
        _ = vol  # keep explicit market binding visible; analytical helper resolves pricing inputs

        return float(price_equity_fixed_lookback_option_analytical(market_state, self._spec))
