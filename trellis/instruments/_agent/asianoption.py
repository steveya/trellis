"""Agent-generated payoff: Build a pricer for: Geometric Asian option: closed-form vs MC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put

# Required imports for Monte Carlo pricing of path‐dependent instruments
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM
import numpy as np


@dataclass(frozen=True)
class AsianOptionSpec:
    """Specification for Build a pricer for: Geometric Asian option: closed-form vs MC."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    averaging_type: str = 'arithmetic'
    option_type: str = 'call'
    n_observations: int = 12
    day_count: DayCountConvention = DayCountConvention.ACT_365


class AsianOptionPayoff:
    """Build a pricer for: Geometric Asian option: closed-form vs MC."""

    def __init__(self, spec: AsianOptionSpec):
        """Store the generated Asian-option specification."""
        self._spec = spec

    @property
    def spec(self) -> AsianOptionSpec:
        """Return the immutable generated Asian-option specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and Black volatility."""
        return {"black_vol", "discount"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated Asian option from Monte Carlo path averages."""
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0:
            return 0.0
        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

        # Determine number of simulation steps.
        # Ensure at least 100 steps per year for convergence.
        min_steps = max(int(100 * T), 1)
        base_steps = spec.n_observations - 1
        n_steps = max(base_steps, min_steps)

        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=50000, n_steps=n_steps, seed=42, method="exact")

        # Identify observation indices corresponding to the desired observation dates.
        # Even if simulation has extra steps, pick equally spaced indices.
        obs_indices = np.linspace(0, n_steps, spec.n_observations, dtype=int)

        def payoff_fn(paths):
            """Map simulated paths to arithmetic- or geometric-average option payoffs."""
            # Extract prices at the discrete observation dates.
            observed_prices = paths[:, obs_indices]
            if spec.averaging_type.lower() == 'geometric':
                # Avoid issues with zeros by taking log; assume prices > 0.
                avg_price = np.exp(np.mean(np.log(observed_prices), axis=1))
            else:  # default to arithmetic averaging
                avg_price = np.mean(observed_prices, axis=1)
            if spec.option_type.lower() == 'call':
                payoffs = np.maximum(avg_price - spec.strike, 0)
            else:
                payoffs = np.maximum(spec.strike - avg_price, 0)
            return payoffs * spec.notional / spec.spot

        result = engine.price(spec.spot, T, payoff_fn, discount_rate=r)
        return result["price"]
