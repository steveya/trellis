"""Agent-generated payoff: Build a pricer for: FX barrier option: analytical vs MC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM


@dataclass(frozen=True)
class BarrierOptionSpec:
    """Specification for Build a pricer for: FX barrier option: analytical vs MC."""
    notional: float
    spot: float
    strike: float
    barrier: float
    expiry_date: date
    barrier_type: str
    option_type: str = 'call'
    day_count: DayCountConvention = DayCountConvention.ACT_365


class BarrierOptionPayoff:
    """Build a pricer for: FX barrier option: analytical vs MC."""

    def __init__(self, spec: BarrierOptionSpec):
        """Store the generated barrier-option specification."""
        self._spec = spec

    @property
    def spec(self) -> BarrierOptionSpec:
        """Return the immutable generated barrier-option specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and Black volatility."""
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated barrier option by Monte Carlo with adaptive path count."""
        import numpy as np

        spec = self._spec
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        process = GBM(mu=r, sigma=sigma)
        # Choose number of steps: at least 100 per year; use daily steps (252) if that is higher.
        n_steps = int(np.ceil(max(100 * T, 252)))
        n_paths = 50000

        def payoff_fn(paths):
            """Map simulated FX paths to knock-in/knock-out option payoffs."""
            S_T = paths[:, -1]
            # Determine if barrier is breached depending on barrier type.
            if "down" in spec.barrier_type:
                breached = np.any(paths <= spec.barrier, axis=1)
            else:  # "up" barrier
                breached = np.any(paths >= spec.barrier, axis=1)
            # Calculate vanilla option payoff.
            if spec.option_type == "call":
                vanilla = np.maximum(S_T - spec.strike, 0)
            else:
                vanilla = np.maximum(spec.strike - S_T, 0)
            # For "out" options, payoff is nullified if barrier is breached;
            # for "in" options, payoff is paid only if barrier was breached.
            if "out" in spec.barrier_type:
                payoffs = np.where(breached, 0.0, vanilla)
            else:  # barrier "in"
                payoffs = np.where(breached, vanilla, 0.0)
            return payoffs * spec.notional / spec.spot

        iteration = 0
        max_iter = 5
        price = None
        st_err = None

        while iteration < max_iter:
            engine = MonteCarloEngine(
                process, n_paths=n_paths, n_steps=n_steps, seed=42, method="exact",
            )
            result = engine.price(spec.spot, T, payoff_fn, discount_rate=r)
            price = result["price"]
            st_err = result["st_err"]
            # If price is nearly zero, standard error check is not applicable.
            if abs(price) < 1e-8 or st_err < 0.01 * abs(price):
                break
            # Increase number of paths to achieve lower standard error.
            n_paths *= 2
            iteration += 1

        return price
