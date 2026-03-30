"""Agent-generated payoff: Build a pricer for: Digital (cash-or-nothing) option: BS formula vs MC vs COS

Implementation target: mc_digital
Preferred method family: monte_carlo

Implementation target: mc_digital."""

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
class EuropeanOptionSpec:
    """Specification for Build a pricer for: Digital (cash-or-nothing) option: BS formula vs MC vs COS

Implementation target: mc_digital
Preferred method family: monte_carlo

Implementation target: mc_digital."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = 'call'
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class EuropeanOptionMonteCarloPayoff:
    """Build a pricer for: Digital (cash-or-nothing) option: BS formula vs MC vs COS

Implementation target: mc_digital
Preferred method family: monte_carlo

Implementation target: mc_digital."""

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
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            if spec.option_type.lower() == "call":
                return float(spec.notional if spec.spot > spec.strike else 0.0)
            return float(spec.notional if spec.spot < spec.strike else 0.0)

        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        df = float(market_state.discount.discount(T))
        r = -__import__("math").log(df) / T if T > 0.0 and df > 0.0 else 0.0

        n_paths = max(int(spec.n_paths), 10000)
        n_steps = max(int(spec.n_steps), max(1, int(round(100 * T))))

        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=n_paths, n_steps=n_steps, seed=42, method="exact")

        result = engine.simulate(spec.spot, T)
        paths = result
        import numpy as np
        arr = np.asarray(paths, dtype=float)
        if arr.ndim == 2:
            terminal = arr[:, -1]
        elif arr.ndim == 3:
            terminal = arr[:, -1, 0]
        else:
            raise ValueError(f"Unexpected Monte Carlo path shape: {arr.shape}")

        if spec.option_type.lower() == "call":
            payoff = np.where(terminal > spec.strike, 1.0, 0.0)
        else:
            payoff = np.where(terminal < spec.strike, 1.0, 0.0)

        pv = spec.notional * df * float(np.mean(payoff))
        return float(pv)