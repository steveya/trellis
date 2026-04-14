"""Agent-generated payoff: Build a pricer for: Nth-to-default basket: Gaussian copula vs default-time MC

Construct methods: copula, monte_carlo
Comparison targets: gaussian_copula (analytical), mc_default_times (monte_carlo)
Cross-validation harness:
  internal targets: gaussian_copula, mc_default_times
New component: default_time_mc

Implementation target: mc_default_times
Preferred method family: monte_carlo

Implementation target: mc_default_times."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.basket_option import (
    price_basket_option_analytical,
    price_basket_option_monte_carlo,
)



@dataclass(frozen=True)
class BasketOptionSpec:
    """Specification for Build a pricer for: Nth-to-default basket: Gaussian copula vs default-time MC

Construct methods: copula, monte_carlo
Comparison targets: gaussian_copula (analytical), mc_default_times (monte_carlo)
Cross-validation harness:
  internal targets: gaussian_copula, mc_default_times
New component: default_time_mc

Implementation target: mc_default_times
Preferred method family: monte_carlo

Implementation target: mc_default_times."""
    notional: float
    underliers: str
    spots: str
    strike: float
    expiry_date: date
    correlation: str
    weights: str | None = None
    vols: str | None = None
    dividend_yields: str | None = None
    basket_style: str = "'weighted_sum'"
    option_type: str = "'call'"
    averaging_type: str | None = None
    n_observations: int | None = None
    barrier_level: float | None = None
    barrier_direction: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class BasketOptionPayoff:
    """Build a pricer for: Nth-to-default basket: Gaussian copula vs default-time MC

Construct methods: copula, monte_carlo
Comparison targets: gaussian_copula (analytical), mc_default_times (monte_carlo)
Cross-validation harness:
  internal targets: gaussian_copula, mc_default_times
New component: default_time_mc

Implementation target: mc_default_times
Preferred method family: monte_carlo

Implementation target: mc_default_times."""

    def __init__(self, spec: BasketOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BasketOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "spot", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        requires_path_runtime = any(
            value is not None
            for value in (
                spec.averaging_type,
                spec.n_observations,
                spec.barrier_level,
                spec.barrier_direction,
            )
        )
        if requires_path_runtime:
            return float(price_basket_option_monte_carlo(market_state, spec))
        return float(price_basket_option_analytical(market_state, spec))
