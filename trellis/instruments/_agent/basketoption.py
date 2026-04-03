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

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.resolution.basket_semantics import resolve_basket_semantics



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
        return {"discount_curve", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        from trellis.models.monte_carlo.semantic_basket import (
            price_ranked_observation_basket_monte_carlo,
        )

        spec = self._spec

        underlyings = [item.strip() for item in spec.underliers.split(",") if item.strip()]
        spots = [item.strip() for item in spec.spots.split(",") if item.strip()]

        weights = None
        if spec.weights is not None:
            weights = [item.strip() for item in spec.weights.split(",") if item.strip()]

        vols = None
        if spec.vols is not None:
            vols = [item.strip() for item in spec.vols.split(",") if item.strip()]

        dividend_yields = None
        if spec.dividend_yields is not None:
            dividend_yields = [item.strip() for item in spec.dividend_yields.split(",") if item.strip()]

        correlation = [
            [float(cell.strip()) for cell in row.split(",") if cell.strip()]
            for row in spec.correlation.split(";")
            if row.strip()
        ]

        ranked_spec = BasketOptionSpec(
            notional=spec.notional,
            underliers=",".join(underlyings),
            spots=",".join(spots),
            strike=spec.strike,
            expiry_date=spec.expiry_date,
            correlation=";".join(",".join(str(cell) for cell in row) for row in correlation),
            weights=",".join(weights) if weights is not None else None,
            vols=",".join(vols) if vols is not None else None,
            dividend_yields=",".join(dividend_yields) if dividend_yields is not None else None,
            basket_style=spec.basket_style,
            option_type=spec.option_type,
            averaging_type=spec.averaging_type,
            n_observations=spec.n_observations,
            barrier_level=spec.barrier_level,
            barrier_direction=spec.barrier_direction,
            day_count=spec.day_count,
        )

        resolved = resolve_basket_semantics(market_state, ranked_spec)
        return float(price_ranked_observation_basket_monte_carlo(ranked_spec, resolved))
