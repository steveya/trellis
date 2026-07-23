"""Agent-generated payoff: Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

European call option on a zero-coupon bond.
Settlement / valuation date: 2024-11-15.
Option expiry: 2027-11-15.
Underlying bond maturity: 2033-11-15.
Strike: 63 per 100 face (= 0.63 per unit face).
Face / notional: 100.
Shared short-rate comparison regime:
- flat discount curve at 5%
- flat short-rate volatility sigma = 0.01
- Hull-White mean reversion a = 0.1
- Ho-Lee uses the same sigma with zero mean reversion

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
from trellis.models.analytical.jamshidian import (
    ResolvedJamshidianInputs,
    zcb_option_hw_raw,
)
from trellis.models.resolution.short_rate_claims import (
    resolve_discount_bond_claim_inputs,
)



@dataclass(frozen=True)
class ZCBOptionSpec:
    """Specification for Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

European call option on a zero-coupon bond.
Settlement / valuation date: 2024-11-15.
Option expiry: 2027-11-15.
Underlying bond maturity: 2033-11-15.
Strike: 63 per 100 face (= 0.63 per unit face).
Face / notional: 100.
Shared short-rate comparison regime:
- flat discount curve at 5%
- flat short-rate volatility sigma = 0.01
- Hull-White mean reversion a = 0.1
- Ho-Lee uses the same sigma with zero mean reversion

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
    option_type: str = 'call'


class ZCBOptionPayoff:
    """Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

European call option on a zero-coupon bond.
Settlement / valuation date: 2024-11-15.
Option expiry: 2027-11-15.
Underlying bond maturity: 2033-11-15.
Strike: 63 per 100 face (= 0.63 per unit face).
Face / notional: 100.
Shared short-rate comparison regime:
- flat discount curve at 5%
- flat short-rate volatility sigma = 0.01
- Hull-White mean reversion a = 0.1
- Ho-Lee uses the same sigma with zero mean reversion

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
        claim = resolve_discount_bond_claim_inputs(
            market_state,
            spec,
            model="hull_white",
            default_mean_reversion=0.1,
        )
        if claim.expiry_time <= 0.0:
            payoff_sign = -1.0 if claim.option_type == "put" else 1.0
            intrinsic = max(
                payoff_sign
                * (claim.discount_factor_bond - claim.strike_unit),
                0.0,
            )
            return float(claim.notional * intrinsic)

        resolved = ResolvedJamshidianInputs(
            discount_factor_expiry=claim.discount_factor_expiry,
            discount_factor_bond=claim.discount_factor_bond,
            strike=claim.strike_unit,
            T_exp=claim.expiry_time,
            T_bond=claim.bond_maturity_time,
            sigma=claim.regime.sigma,
            a=claim.regime.mean_reversion,
        )
        raw = zcb_option_hw_raw(resolved)
        return float(claim.notional * float(raw[claim.option_type]))
