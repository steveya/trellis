"""Agent-generated payoff: Build a pricer for: Puttable bond: exercise_fn=max and puttable-callable symmetry

Construct methods: rate_tree
Comparison targets: puttable_tree (rate_tree), callable_tree_symmetry (rate_tree)
Cross-validation harness:
  internal targets: puttable_tree, callable_tree_symmetry
  external targets: quantlib, financepy

Implementation target: callable_tree_symmetry
Preferred method family: rate_tree

Implementation target: callable_tree_symmetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.callable_bond_tree import price_callable_bond_tree



@dataclass(frozen=True)
class PuttableBondSpec:
    """Specification for Build a pricer for: Puttable bond: exercise_fn=max and puttable-callable symmetry

Construct methods: rate_tree
Comparison targets: puttable_tree (rate_tree), callable_tree_symmetry (rate_tree)
Cross-validation harness:
  internal targets: puttable_tree, callable_tree_symmetry
  external targets: quantlib, financepy

Implementation target: callable_tree_symmetry
Preferred method family: rate_tree

Implementation target: callable_tree_symmetry."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    put_dates: tuple[date, ...]
    put_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class PuttableBondPayoff:
    """Build a pricer for: Puttable bond: exercise_fn=max and puttable-callable symmetry

Construct methods: rate_tree
Comparison targets: puttable_tree (rate_tree), callable_tree_symmetry (rate_tree)
Cross-validation harness:
  internal targets: puttable_tree, callable_tree_symmetry
  external targets: quantlib, financepy

Implementation target: callable_tree_symmetry
Preferred method family: rate_tree

Implementation target: callable_tree_symmetry."""

    def __init__(self, spec: PuttableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> PuttableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        return float(
            price_callable_bond_tree(
                market_state,
                spec,
                model="hull_white",
            )
        )
