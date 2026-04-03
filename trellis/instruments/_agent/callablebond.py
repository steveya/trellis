"""Agent-generated payoff: Build a pricer for: Callable bond: BDT lognormal vs HW normal tree

Construct methods: rate_tree
Comparison targets: bdt_tree (rate_tree), hull_white_tree (rate_tree)
Cross-validation harness:
  internal targets: bdt_tree, hull_white_tree
  external targets: financepy, quantlib
New component: bdt_calibration_via_generic_lattice

Implementation target: hull_white_tree
Preferred method family: rate_tree

Implementation target: hull_white_tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class CallableBondSpec:
    """Specification for Build a pricer for: Callable bond: BDT lognormal vs HW normal tree

Construct methods: rate_tree
Comparison targets: bdt_tree (rate_tree), hull_white_tree (rate_tree)
Cross-validation harness:
  internal targets: bdt_tree, hull_white_tree
  external targets: financepy, quantlib
New component: bdt_calibration_via_generic_lattice

Implementation target: hull_white_tree
Preferred method family: rate_tree

Implementation target: hull_white_tree."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: tuple[date, ...]
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Build a pricer for: Callable bond: BDT lognormal vs HW normal tree

Construct methods: rate_tree
Comparison targets: bdt_tree (rate_tree), hull_white_tree (rate_tree)
Cross-validation harness:
  internal targets: bdt_tree, hull_white_tree
  external targets: financepy, quantlib
New component: bdt_calibration_via_generic_lattice

Implementation target: hull_white_tree
Preferred method family: rate_tree

Implementation target: hull_white_tree."""

    def __init__(self, spec: CallableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        from trellis.models.callable_bond_tree import price_callable_bond_tree

        spec = self._spec

        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        return float(
            price_callable_bond_tree(
                market_state,
                spec,
                model="hull_white",
            )
        )
