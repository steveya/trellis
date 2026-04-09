"""Agent-generated payoff: Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

Price a 1Y European option on a zero-coupon bond maturing at T=5Y.
Face value $100, strike price $88. Hull-White model: mean reversion
a=0.1, short-rate vol sigma=0.01, initial short rate r0=0.05.
Flat yield curve at 5%.
Cross-validate Ho-Lee tree, Hull-White tree, and Jamshidian analytical
decomposition.  All three should agree within 2%.

Construct methods: rate_tree
Comparison targets: ho_lee_tree (rate_tree), hull_white_tree (rate_tree), jamshidian (analytical)
Cross-validation harness:
  internal targets: ho_lee_tree, hull_white_tree
  analytical benchmark: jamshidian
  external targets: quantlib, financepy
New component: jamshidian_decomposition

Implementation target: jamshidian
Preferred method family: analytical

Implementation target: jamshidian."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.zcb_option import price_zcb_option_jamshidian



@dataclass(frozen=True)
class ZCBOptionSpec:
    """Specification for Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

Price a 1Y European option on a zero-coupon bond maturing at T=5Y.
Face value $100, strike price $88. Hull-White model: mean reversion
a=0.1, short-rate vol sigma=0.01, initial short rate r0=0.05.
Flat yield curve at 5%.
Cross-validate Ho-Lee tree, Hull-White tree, and Jamshidian analytical
decomposition.  All three should agree within 2%.

Construct methods: rate_tree
Comparison targets: ho_lee_tree (rate_tree), hull_white_tree (rate_tree), jamshidian (analytical)
Cross-validation harness:
  internal targets: ho_lee_tree, hull_white_tree
  analytical benchmark: jamshidian
  external targets: quantlib, financepy
New component: jamshidian_decomposition

Implementation target: jamshidian
Preferred method family: analytical

Implementation target: jamshidian."""
    notional: float
    strike: float
    expiry_date: date
    bond_maturity_date: date
    day_count: DayCountConvention = DayCountConvention.ACT_365
    option_type: str = "'call'"


class ZCBOptionPayoff:
    """Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

Price a 1Y European option on a zero-coupon bond maturing at T=5Y.
Face value $100, strike price $88. Hull-White model: mean reversion
a=0.1, short-rate vol sigma=0.01, initial short rate r0=0.05.
Flat yield curve at 5%.
Cross-validate Ho-Lee tree, Hull-White tree, and Jamshidian analytical
decomposition.  All three should agree within 2%.

Construct methods: rate_tree
Comparison targets: ho_lee_tree (rate_tree), hull_white_tree (rate_tree), jamshidian (analytical)
Cross-validation harness:
  internal targets: ho_lee_tree, hull_white_tree
  analytical benchmark: jamshidian
  external targets: quantlib, financepy
New component: jamshidian_decomposition

Implementation target: jamshidian
Preferred method family: analytical

Implementation target: jamshidian."""

    def __init__(self, spec: ZCBOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> ZCBOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        # Resolve and validate the required market data.
        if market_state.discount is None:
            raise ValueError("ZCBOptionPayoff requires a discount curve in market_state")
        if market_state.vol_surface is None:
            raise ValueError("ZCBOptionPayoff requires a black vol surface in market_state")
        if market_state.as_of is None:
            raise ValueError("ZCBOptionPayoff requires market_state.as_of to be set")

        if spec.bond_maturity_date <= spec.expiry_date:
            raise ValueError("bond_maturity_date must be strictly after expiry_date")

        option_type = str(spec.option_type).strip().strip("'").strip('"').lower()
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        # Prefer the checked-in analytical helper for Jamshidian pricing.
        try:
            return float(price_zcb_option_jamshidian(market_state, spec, mean_reversion=0.1))
        except TypeError:
            # Fallback for helper variants that do not accept mean_reversion as a keyword.
            return float(price_zcb_option_jamshidian(market_state, spec))
