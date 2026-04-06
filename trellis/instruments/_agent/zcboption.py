"""Agent-generated payoff: Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

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

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class ZCBOptionSpec:
    """Specification for Build a pricer for: ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical

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
        spec = self._spec
        if spec.bond_maturity_date <= spec.expiry_date:
            raise ValueError("bond_maturity_date must be strictly after expiry_date")

        # Normalize strike quotes quoted on face value to unit-notional strike for the kernel.
        strike = spec.strike / spec.notional if spec.notional != 0 else spec.strike

        # Ensure the required market capabilities exist and are observable.
        _ = market_state.discount
        if market_state.vol_surface is None:
            raise ValueError("market_state.vol_surface is required for ZCB option pricing")

        # Touch the requested market volatility surface per routing contract.
        valuation_date = market_state.settlement or market_state.as_of
        expiry_t = year_fraction(valuation_date, spec.expiry_date, spec.day_count)
        _ = market_state.vol_surface.black_vol(expiry_t, strike)

        # Jamshidian analytical route helper.
        from trellis.models.zcb_option import price_zcb_option_jamshidian

        try:
            pv = price_zcb_option_jamshidian(market_state, spec, mean_reversion=0.1)
        except TypeError:
            # Fallback for helper variants that expect normalized strike or different spec handling.
            normalized_spec = ZCBOptionSpec(
                notional=spec.notional,
                strike=strike,
                expiry_date=spec.expiry_date,
                bond_maturity_date=spec.bond_maturity_date,
                day_count=spec.day_count,
                option_type=spec.option_type,
            )
            pv = price_zcb_option_jamshidian(market_state, normalized_spec, mean_reversion=0.1)

        return float(pv)
