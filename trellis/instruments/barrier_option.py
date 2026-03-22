"""Barrier option — reference implementation using Monte Carlo.

This is the hand-coded reference for MC-based pricing patterns.
The agent uses this as a template for other path-dependent instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState

from trellis.core.types import DayCountConvention
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM


@dataclass(frozen=True)
class BarrierOptionSpec:
    """Specification for a barrier option."""

    notional: float
    spot: float
    strike: float
    barrier: float
    expiry_date: date
    barrier_type: str = "down_and_out"  # up_and_out, down_and_out, up_and_in, down_and_in
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365


class BarrierOptionPayoff:
    """Barrier option priced via Monte Carlo simulation.

    Monitors the barrier continuously (approximated by discrete steps).
    """

    def __init__(self, spec: BarrierOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BarrierOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "black_vol"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(
            process, n_paths=50000, n_steps=252, seed=42, method="exact",
        )

        barrier = spec.barrier
        barrier_type = spec.barrier_type

        def payoff_fn(paths):
            S_T = paths[:, -1]

            # Check barrier breach
            if "down" in barrier_type:
                breached = raw_np.any(paths <= barrier, axis=1)
            else:  # up
                breached = raw_np.any(paths >= barrier, axis=1)

            # Vanilla payoff
            if spec.option_type == "call":
                vanilla = raw_np.maximum(S_T - spec.strike, 0)
            else:
                vanilla = raw_np.maximum(spec.strike - S_T, 0)

            # Apply barrier logic
            if "out" in barrier_type:
                payoffs = raw_np.where(breached, 0.0, vanilla)
            else:  # "in"
                payoffs = raw_np.where(breached, vanilla, 0.0)

            return payoffs * spec.notional / spec.spot

        result = engine.price(spec.spot, T, payoff_fn, discount_rate=r)
        return result["price"]
