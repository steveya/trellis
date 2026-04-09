"""Agent-generated payoff: Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree

Price a 10-year callable bond paying a 5% semi-annual coupon, par $100,
callable at par on any coupon date after year 3.
Use the USD OIS discount curve from the market snapshot (as_of 2024-11-15).
Hull-White model: mean reversion a=0.05, short-rate vol sigma=0.01.
Method 1: solve the HW rate PDE backward in time using PSOR to
enforce the call constraint (issuer calls when continuation value > par).
Method 2: Hull-White trinomial/binomial rate tree for comparison.
Both methods should agree within 1%.
Compute callable bond price and option-adjusted spread (OAS).

Construct methods: pde_solver, rate_tree
Comparison targets: hw_pde_psor (pde_solver), hw_rate_tree (rate_tree)
Cross-validation harness:
  internal targets: hw_pde_psor, hw_rate_tree
  external targets: quantlib
New component: hw_rate_pde_operator

Implementation target: hw_rate_tree
Preferred method family: rate_tree

Implementation target: hw_rate_tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.callable_bond_tree import price_callable_bond_tree



@dataclass(frozen=True)
class CallableBondSpec:
    """Specification for Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree

Price a 10-year callable bond paying a 5% semi-annual coupon, par $100,
callable at par on any coupon date after year 3.
Use the USD OIS discount curve from the market snapshot (as_of 2024-11-15).
Hull-White model: mean reversion a=0.05, short-rate vol sigma=0.01.
Method 1: solve the HW rate PDE backward in time using PSOR to
enforce the call constraint (issuer calls when continuation value > par).
Method 2: Hull-White trinomial/binomial rate tree for comparison.
Both methods should agree within 1%.
Compute callable bond price and option-adjusted spread (OAS).

Construct methods: pde_solver, rate_tree
Comparison targets: hw_pde_psor (pde_solver), hw_rate_tree (rate_tree)
Cross-validation harness:
  internal targets: hw_pde_psor, hw_rate_tree
  external targets: quantlib
New component: hw_rate_pde_operator

Implementation target: hw_rate_tree
Preferred method family: rate_tree

Implementation target: hw_rate_tree."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: tuple[date, ...]
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree

Price a 10-year callable bond paying a 5% semi-annual coupon, par $100,
callable at par on any coupon date after year 3.
Use the USD OIS discount curve from the market snapshot (as_of 2024-11-15).
Hull-White model: mean reversion a=0.05, short-rate vol sigma=0.01.
Method 1: solve the HW rate PDE backward in time using PSOR to
enforce the call constraint (issuer calls when continuation value > par).
Method 2: Hull-White trinomial/binomial rate tree for comparison.
Both methods should agree within 1%.
Compute callable bond price and option-adjusted spread (OAS).

Construct methods: pde_solver, rate_tree
Comparison targets: hw_pde_psor (pde_solver), hw_rate_tree (rate_tree)
Cross-validation harness:
  internal targets: hw_pde_psor, hw_rate_tree
  external targets: quantlib
New component: hw_rate_pde_operator

Implementation target: hw_rate_tree
Preferred method family: rate_tree

Implementation target: hw_rate_tree."""

    def __init__(self, spec: CallableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        return float(price_callable_bond_tree(market_state, spec, model="hull_white"))