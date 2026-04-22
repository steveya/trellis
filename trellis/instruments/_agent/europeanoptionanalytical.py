"""Agent-generated payoff: Build a pricer for: European call: theta-method convergence order measurement

Construct methods: pde_solver
Comparison targets: theta_0.5 (pde_solver), theta_1.0 (pde_solver), black_scholes (analytical)
Cross-validation harness:
  internal targets: theta_0.5, theta_1.0
  analytical benchmark: black_scholes
  external targets: quantlib
New component: convergence_order_diagnostic

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import PricingValue
from trellis.core.types import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.models.black import black76_call, black76_put, black76_asset_or_nothing_call, black76_asset_or_nothing_put, black76_cash_or_nothing_call, black76_cash_or_nothing_put
from trellis.models.analytical.equity_vanilla_bs import equity_vanilla_bs_outputs



@dataclass(frozen=True)
class EuropeanOptionSpec:
    """Specification for Build a pricer for: European call: theta-method convergence order measurement

Construct methods: pde_solver
Comparison targets: theta_0.5 (pde_solver), theta_1.0 (pde_solver), black_scholes (analytical)
Cross-validation harness:
  internal targets: theta_0.5, theta_1.0
  analytical benchmark: black_scholes
  external targets: quantlib
New component: convergence_order_diagnostic

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = 'call'
    day_count: DayCountConvention = DayCountConvention.ACT_365


class EuropeanOptionAnalyticalPayoff:
    """Build a pricer for: European call: theta-method convergence order measurement

Construct methods: pde_solver
Comparison targets: theta_0.5 (pde_solver), theta_1.0 (pde_solver), black_scholes (analytical)
Cross-validation harness:
  internal targets: theta_0.5, theta_1.0
  analytical benchmark: black_scholes
  external targets: quantlib
New component: convergence_order_diagnostic

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""

    def __init__(self, spec: EuropeanOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> EuropeanOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> PricingValue:
        spec = self._spec
        spec = self._spec
        if market_state.discount is None:
            raise ValueError("market_state.discount is required for Black-Scholes comparison")
        if market_state.vol_surface is None:
            raise ValueError("market_state.vol_surface is required for Black-Scholes comparison")
        T = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)
        spot = spec.spot
        strike = spec.strike
        option_type = str(spec.option_type or "call").strip().lower()
        if T <= 0.0:
            intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
            return spec.notional * intrinsic
        df = market_state.discount.discount(T)
        sigma = market_state.vol_surface.black_vol(max(T, 1e-6), strike)
        forward = spot / max(df, 1e-12)
        if option_type == "call":
            undiscounted = black76_call(forward, strike, sigma, T)
        elif option_type == "put":
            undiscounted = black76_put(forward, strike, sigma, T)
        else:
            raise ValueError(f"Unsupported option_type {spec.option_type!r}")
        return spec.notional * df * undiscounted

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        return dict(equity_vanilla_bs_outputs(market_state, self._spec))