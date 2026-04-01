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
from trellis.models.callable_bond_tree import price_callable_bond_tree



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
    call_dates: str
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
        return {"black_vol", "discount"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        spec = self._spec

        if market_state.discount is None:
            raise ValueError("CallableBondPayoff requires a discount curve in market_state")
        if market_state.vol_surface is None:
            raise ValueError("CallableBondPayoff requires a vol_surface in market_state")

        call_dates = [d.strip() for d in spec.call_dates.split(",") if d.strip()]
        call_dates_parsed = [date.fromisoformat(d) for d in call_dates]

        _ = market_state.vol_surface.black_vol(
        max(year_fraction(spec.start_date, spec.end_date, spec.day_count) / 2.0, 1e-6),
        max(market_state.discount.zero_rate(year_fraction(spec.start_date, spec.end_date, spec.day_count)), 1e-6),
        )

        return float(price_callable_bond_tree(market_state, spec, model="hull_white"))
