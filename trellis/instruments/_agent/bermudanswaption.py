"""Agent-generated payoff: Build a pricer for: Bermudan swaption on HW tree.

Construct methods: rate_tree
Comparison targets: hw_tree_bermudan (rate_tree), black76_european_lower_bound (analytical)
Cross-validation harness:
  internal targets: hw_tree_bermudan, black76_european_lower_bound
  external targets: quantlib, financepy
New component: swap_valuation_on_tree

Implementation target: hw_tree_bermudan
Preferred method family: rate_tree
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency

@dataclass(frozen=True)
class BermudanSwaptionSpec:
    """Specification for the checked-in Bermudan swaption tree helper."""
    notional: float
    strike: float
    exercise_dates: str
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class BermudanSwaptionPayoff:
    """Thin adapter around the checked-in Bermudan swaption tree route helper."""

    def __init__(self, spec: BermudanSwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BermudanSwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        from trellis.models.bermudan_swaption_tree import price_bermudan_swaption_tree

        # Thin rate-tree adapter: delegate to the checked-in route helper.
        return float(price_bermudan_swaption_tree(market_state, spec, model="hull_white"))
