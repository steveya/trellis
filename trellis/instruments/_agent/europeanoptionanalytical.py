"""Agent-generated payoff: Build a pricer for: Finite element method (FEM) vs finite difference for European

Construct methods: pde_solver
Comparison targets: fem_solver (pde_solver), fd_theta_method (pde_solver), black_scholes (analytical)
Cross-validation harness:
  internal targets: fem_solver, fd_theta_method
  analytical benchmark: black_scholes
New component: fem_1d_solver

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class EuropeanOptionSpec:
    """Specification for Build a pricer for: Finite element method (FEM) vs finite difference for European

Construct methods: pde_solver
Comparison targets: fem_solver (pde_solver), fd_theta_method (pde_solver), black_scholes (analytical)
Cross-validation harness:
  internal targets: fem_solver, fd_theta_method
  analytical benchmark: black_scholes
New component: fem_1d_solver

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
    """Build a pricer for: Finite element method (FEM) vs finite difference for European

Construct methods: pde_solver
Comparison targets: fem_solver (pde_solver), fd_theta_method (pde_solver), black_scholes (analytical)
Cross-validation harness:
  internal targets: fem_solver, fd_theta_method
  analytical benchmark: black_scholes
New component: fem_1d_solver

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
        return {"black_vol", "discount"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        if market_state.discount is None:
            raise ValueError("European option pricing requires a discount curve")
        if market_state.vol_surface is None:
            raise ValueError("European option pricing requires a Black volatility surface")

        t = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if t <= 0.0:
            if spec.option_type.lower() == "call":
                return float(spec.notional * max(spec.spot - spec.strike, 0.0))
            if spec.option_type.lower() == "put":
                return float(spec.notional * max(spec.strike - spec.spot, 0.0))
            raise ValueError(f"Unsupported option_type: {spec.option_type!r}")

        df = market_state.discount.discount(t)
        if df <= 0.0:
            raise ValueError("Discount factor must be positive")

        forward = spec.spot / df
        vol = market_state.vol_surface.black_vol(t, spec.strike)

        opt_type = spec.option_type.lower()
        if opt_type == "call":
            undiscounted = black76_call(forward, spec.strike, vol, t)
        elif opt_type == "put":
            undiscounted = black76_put(forward, spec.strike, vol, t)
        else:
            raise ValueError(f"Unsupported option_type: {spec.option_type!r}")

        return float(spec.notional * df * undiscounted)