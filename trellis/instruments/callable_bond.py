"""Callable bond: a bond the issuer can redeem early on specified dates.

Priced using backward induction on a calibrated interest rate tree.
At each call date, the issuer exercises (redeems at par) if the bond's
continuation value exceeds the call price — this is a Bermudan-style
exercise problem (exercise allowed on specific dates, not every day).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import (
    year_fraction,
)
from trellis.core.market_state import MarketState

from trellis.core.types import DayCountConvention, Frequency
from trellis.models.callable_bond_tree import price_callable_bond_tree


@dataclass(frozen=True)
class CallableBondSpec:
    """Contract terms for a callable bond (coupon, maturity, call schedule)."""

    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: list[date]
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Callable bond priced on an interest rate tree.

    Builds a Hull-White rate tree calibrated to the discount curve, then
    works backward from maturity. At each call date, the issuer redeems
    the bond if doing so is cheaper than letting it continue — this caps
    the bond value below the straight (non-callable) bond price.
    """

    def __init__(self, spec: CallableBondSpec):
        """Store the callable-bond terms used for all future tree valuations."""
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        """Return the immutable callable-bond specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Needs a discount curve and an interest rate volatility surface."""
        return {"discount_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        """Build a Hull-White tree through the checked-in callable-bond helper."""
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0
        return price_callable_bond_tree(
            market_state,
            spec,
            model="hull_white",
            mean_reversion=0.1,
        )
