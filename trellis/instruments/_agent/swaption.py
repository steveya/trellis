"""Agent-generated payoff: Build a pricer for: European swaption: Black76 vs HW tree vs HW MC

European payer swaption.  Expiry: 1Y.  Underlying: 5Y fixed-for-float
interest rate swap.  Strike (fixed rate): 3%.  Notional: $1,000,000.
Fixed leg: semi-annual, 30/360.  Float leg: quarterly 3M SOFR, Act/360.
Use the USD OIS curve for discounting and the SOFR-3M forecast curve
for forward rate projection from the market snapshot (as_of 2024-11-15).
The yield curve is NOT flat — the agent must project forward swap rates
from the actual term structure.
Hull-White model: mean reversion a=0.05, vol sigma=0.01.
Method 1: Black76 analytical (with forward swap rate and swaption vol).
Method 2: Hull-White tree (calibrated to the OIS curve).
Method 3: Hull-White Monte Carlo (calibrated to the OIS curve).
All three should agree within 5%.

Construct methods: analytical, rate_tree, monte_carlo
Comparison targets: black76 (analytical), hw_tree (rate_tree), hw_mc (monte_carlo)
Cross-validation harness:
  internal targets: black76, hw_tree, hw_mc
  external targets: quantlib, financepy

Implementation target: hw_mc
Preferred method family: monte_carlo

Implementation target: hw_mc."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.rate_style_swaption import price_swaption_monte_carlo



@dataclass(frozen=True)
class SwaptionSpec:
    """Specification for Build a pricer for: European swaption: Black76 vs HW tree vs HW MC

European payer swaption.  Expiry: 1Y.  Underlying: 5Y fixed-for-float
interest rate swap.  Strike (fixed rate): 3%.  Notional: $1,000,000.
Fixed leg: semi-annual, 30/360.  Float leg: quarterly 3M SOFR, Act/360.
Use the USD OIS curve for discounting and the SOFR-3M forecast curve
for forward rate projection from the market snapshot (as_of 2024-11-15).
The yield curve is NOT flat — the agent must project forward swap rates
from the actual term structure.
Hull-White model: mean reversion a=0.05, vol sigma=0.01.
Method 1: Black76 analytical (with forward swap rate and swaption vol).
Method 2: Hull-White tree (calibrated to the OIS curve).
Method 3: Hull-White Monte Carlo (calibrated to the OIS curve).
All three should agree within 5%.

Construct methods: analytical, rate_tree, monte_carlo
Comparison targets: black76 (analytical), hw_tree (rate_tree), hw_mc (monte_carlo)
Cross-validation harness:
  internal targets: black76, hw_tree, hw_mc
  external targets: quantlib, financepy

Implementation target: hw_mc
Preferred method family: monte_carlo

Implementation target: hw_mc."""
    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.THIRTY_360
    rate_index: str | None = 'USD-SOFR-3M'
    is_payer: bool = True


class SwaptionPayoff:
    """Build a pricer for: European swaption: Black76 vs HW tree vs HW MC

European payer swaption.  Expiry: 1Y.  Underlying: 5Y fixed-for-float
interest rate swap.  Strike (fixed rate): 3%.  Notional: $1,000,000.
Fixed leg: semi-annual, 30/360.  Float leg: quarterly 3M SOFR, Act/360.
Use the USD OIS curve for discounting and the SOFR-3M forecast curve
for forward rate projection from the market snapshot (as_of 2024-11-15).
The yield curve is NOT flat — the agent must project forward swap rates
from the actual term structure.
Hull-White model: mean reversion a=0.05, vol sigma=0.01.
Method 1: Black76 analytical (with forward swap rate and swaption vol).
Method 2: Hull-White tree (calibrated to the OIS curve).
Method 3: Hull-White Monte Carlo (calibrated to the OIS curve).
All three should agree within 5%.

Construct methods: analytical, rate_tree, monte_carlo
Comparison targets: black76 (analytical), hw_tree (rate_tree), hw_mc (monte_carlo)
Cross-validation harness:
  internal targets: black76, hw_tree, hw_mc
  external targets: quantlib, financepy

Implementation target: hw_mc
Preferred method family: monte_carlo

Implementation target: hw_mc."""

    def __init__(self, spec: SwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> SwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        return float(price_swaption_monte_carlo(market_state, spec, n_paths=20000, seed=42, mean_reversion=0.05, sigma=0.01))