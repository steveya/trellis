"""Agent-generated payoff: Down-and-out barrier call option on equity, spot=100, strike=100, barrier=80, expiry=1Y, notional=1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class DownAndOutBarrierCallSpec:
    """Specification for Down-and-out barrier call option on equity, spot=100, strike=100, barrier=80, expiry=1Y, notional=1."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    barrier: float


class DownAndOutBarrierCall:
    """Down-and-out barrier call option on equity, spot=100, strike=100, barrier=80, expiry=1Y, notional=1."""

    def __init__(self, spec: DownAndOutBarrierCallSpec):
        """Store the generated down-and-out call specification."""
        self._spec = spec

    @property
    def spec(self) -> DownAndOutBarrierCallSpec:
        """Return the immutable generated barrier-option specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and Black volatility."""
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated barrier call by Monte Carlo path simulation."""
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.processes.gbm import GBM
        import numpy as np

        spec = self._spec
        # Use ACT/365 as the day count convention
        T = year_fraction(market_state.settlement, spec.expiry_date, DayCountConvention.ACT_365)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        process = GBM(mu=r, sigma=sigma)
        # Ensure at least 100 steps per year; if T>=1, use approx 252 steps per year for trading days.
        n_steps = int(max(100, 252 * T))
        engine = MonteCarloEngine(process, n_paths=50000, n_steps=n_steps, seed=42, method="exact")

        def payoff_fn(paths):
            """Map simulated paths to down-and-out call payoffs."""
            # For a down-and-out option, if any simulated price is <= barrier, the option knocks out.
            breached = np.any(paths <= spec.barrier, axis=1)
            S_T = paths[:, -1]
            vanilla_payoff = np.maximum(S_T - spec.strike, 0)
            payoffs = np.where(breached, 0.0, vanilla_payoff)
            return payoffs * spec.notional / spec.spot

        result = engine.price(spec.spot, T, payoff_fn, discount_rate=r)
        # Verify that the Monte Carlo standard error is less than 1% of the price.
        if result.get("stderr", 0) > 0.01 * result["price"]:
            raise ValueError("Monte Carlo standard error exceeds 1% tolerance of the price.")
        return result["price"]
