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
from trellis.models.monte_carlo.path_state import barrier_payoff
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
        """Store the barrier-option contract specification."""
        self._spec = spec

    @property
    def spec(self) -> BarrierOptionSpec:
        """Return the immutable barrier-option specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that barrier-option valuation needs discounting and Black vol."""
        return {"discount", "black_vol"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the option by Monte Carlo path simulation with barrier state tracking."""
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

        def terminal_payoff(terminal):
            if spec.option_type == "call":
                return raw_np.maximum(terminal - spec.strike, 0.0)
            return raw_np.maximum(spec.strike - terminal, 0.0)

        result = engine.price(
            spec.spot,
            T,
            barrier_payoff(
                barrier=spec.barrier,
                direction="down" if "down" in spec.barrier_type else "up",
                knock="out" if "out" in spec.barrier_type else "in",
                terminal_payoff_fn=terminal_payoff,
                scale=spec.notional / spec.spot,
                name=f"{spec.barrier_type}_{spec.option_type}_barrier_payoff",
            ),
            discount_rate=r,
            return_paths=False,
        )
        return result["price"]
