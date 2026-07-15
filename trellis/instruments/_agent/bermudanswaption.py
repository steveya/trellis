"""Agent-generated payoff: Build a pricer for: Bermudan swaption on HW tree

Construct methods: rate_tree
Comparison targets: hw_tree_bermudan (rate_tree), black76_european_lower_bound (analytical)
Cross-validation harness:
  internal targets: hw_tree_bermudan, black76_european_lower_bound
  external targets: quantlib, financepy
New component: swap_valuation_on_tree

Implementation target: black76_european_lower_bound
Preferred method family: analytical

Implementation target: black76_european_lower_bound."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import normalize_explicit_dates
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.rate_style_swaption import (
    price_swaption_black76_raw,
    resolve_swaption_black76_inputs,
)


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    """Specification for Build a pricer for: Bermudan swaption on HW tree

Construct methods: rate_tree
Comparison targets: hw_tree_bermudan (rate_tree), black76_european_lower_bound (analytical)
Cross-validation harness:
  internal targets: hw_tree_bermudan, black76_european_lower_bound
  external targets: quantlib, financepy
New component: swap_valuation_on_tree

Implementation target: black76_european_lower_bound
Preferred method family: analytical

Implementation target: black76_european_lower_bound."""
    notional: float
    strike: float
    exercise_dates: tuple[date, ...]
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class BermudanSwaptionPayoff:
    """Build a pricer for: Bermudan swaption on HW tree

Construct methods: rate_tree
Comparison targets: hw_tree_bermudan (rate_tree), black76_european_lower_bound (analytical)
Cross-validation harness:
  internal targets: hw_tree_bermudan, black76_european_lower_bound
  external targets: quantlib, financepy
New component: swap_valuation_on_tree

Implementation target: black76_european_lower_bound
Preferred method family: analytical

Implementation target: black76_european_lower_bound."""

    def __init__(self, spec: BermudanSwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BermudanSwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        exercise_dates = normalize_explicit_dates(spec.exercise_dates)
        valid_exercise_dates = tuple(
            exercise_date
            for exercise_date in exercise_dates
            if market_state.settlement < exercise_date < spec.swap_end
        )
        if not valid_exercise_dates:
            return 0.0
        final_exercise_date = valid_exercise_dates[-1]
        resolved = resolve_swaption_black76_inputs(
            market_state,
            spec,
            expiry_date=final_exercise_date,
        )
        return float(price_swaption_black76_raw(resolved))
