"""Agent-generated payoff: American put option on equity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState


@dataclass(frozen=True)
class AmericanPutEquitySpec:
    """Specification for the generated American equity put payoff."""
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "put"
    exercise_style: str = "american"


class AmericanOptionPayoff:
    """Generated American put pricer using Longstaff-Schwartz on GBM paths."""
    def __init__(self, spec: AmericanPutEquitySpec):
        """Store the generated American-option specification."""
        self._spec = spec

    @property
    def spec(self) -> AmericanPutEquitySpec:
        """Return the immutable generated American-option specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and Black volatility."""
        return {"discount_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated American put with Longstaff-Schwartz regression."""
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.monte_carlo.lsm import longstaff_schwartz
        from trellis.models.monte_carlo.schemes import LaguerreBasis
        from trellis.models.processes.gbm import GBM

        spec = self._spec
        np = get_numpy()
        T = year_fraction(market_state.settlement, spec.expiry_date)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=4096, n_steps=64, seed=42, method="exact")
        paths = engine.simulate(spec.spot, T)
        dt = T / engine.n_steps
        exercise_dates = list(range(1, engine.n_steps + 1))
        basis = LaguerreBasis()

        def payoff_fn(spots):
            """Return intrinsic put values at the exercise spots used by LSM."""
            return np.maximum(spec.strike - spots, 0.0)

        return float(
            longstaff_schwartz(
                paths,
                exercise_dates,
                payoff_fn,
                discount_rate=r,
                dt=dt,
                basis_fn=basis,
            )
        )
