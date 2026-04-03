"""Agent-generated payoff: Build a pricer for: European equity call under local vol: PDE vs MC

Construct methods: pde_solver, monte_carlo
Comparison targets: local_vol_pde (pde_solver), local_vol_mc (monte_carlo)
Cross-validation harness:
  internal targets: local_vol_pde, local_vol_mc
New component: local_vol_equity_route

Implementation target: local_vol_pde
Preferred method family: pde_solver

Implementation target: local_vol_pde."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class EuropeanLocalVolOptionSpec:
    """Specification for Build a pricer for: European equity call under local vol: PDE vs MC

Construct methods: pde_solver, monte_carlo
Comparison targets: local_vol_pde (pde_solver), local_vol_mc (monte_carlo)
Cross-validation harness:
  internal targets: local_vol_pde, local_vol_mc
New component: local_vol_equity_route

Implementation target: local_vol_pde
Preferred method family: pde_solver

Implementation target: local_vol_pde."""
    notional: float
    strike: float
    expiry_date: date
    option_type: str = "'call'"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class EuropeanLocalVolMonteCarloPayoff:
    """Build a pricer for: European equity call under local vol: PDE vs MC

Construct methods: pde_solver, monte_carlo
Comparison targets: local_vol_pde (pde_solver), local_vol_mc (monte_carlo)
Cross-validation harness:
  internal targets: local_vol_pde, local_vol_mc
New component: local_vol_equity_route

Implementation target: local_vol_pde
Preferred method family: pde_solver

Implementation target: local_vol_pde."""

    def __init__(self, spec: EuropeanLocalVolOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> EuropeanLocalVolOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "local_vol_surface", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        spec = self._spec
        from trellis.models.equity_option_pde import price_vanilla_equity_option_pde

        price = price_vanilla_equity_option_pde(market_state, spec, theta=0.5)
        return float(price) * float(spec.notional)
