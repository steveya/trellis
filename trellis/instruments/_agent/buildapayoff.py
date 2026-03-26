from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class EuropeanLocalVolCallSpec:
    """Specification for Build a pricer for: European equity call under local vol: PDE vs MC

Implementation target: local_vol_mc
Preferred method family: monte_carlo

Implementation target: local_vol_mc."""
    strike: float
    expiry_date: date
    notional: float
    is_call: bool = True
    n_paths: int = 100000
    n_steps: int = 252
    antithetic: bool = True
    control_variate: bool = True
    use_qmc: bool = False
    seed: int = 42
    dividend_yield: float = 0.0
    local_vol_surface_source: str | None = auto
    pricing_method: str = monte_carlo
    output_paths: bool = False


class EuropeanLocalVolCallPayoff:
    """Build a pricer for: European equity call under local vol: PDE vs MC

Implementation target: local_vol_mc
Preferred method family: monte_carlo

Implementation target: local_vol_mc."""

    def __init__(self, spec: EuropeanLocalVolCallSpec):
        self._spec = spec

    @property
    def spec(self) -> EuropeanLocalVolCallSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        from trellis.core.differentiable import get_numpy
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.processes.gbm import GBM

        np = get_numpy()

        # Time to expiry in years
        T = year_fraction(market_state.settlement, spec.expiry_date)
        if T <= 0:
            return 0.0

        # Risk-free rate (continuous) and dividend yield
        r = float(market_state.discount.zero_rate(T))
        q = float(spec.dividend_yield)

        # Spot and Black implied vol
        S0 = float(market_state.spot)
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

        # Enforce modelling requirements: sufficient paths and time steps
        n_paths = int(max(spec.n_paths, 10000))
        n_steps = int(max(spec.n_steps, max(1, int(T * 100))))  # >=100 steps per year

        # Use GBM under risk-neutral drift (r - q)
        process = GBM(mu=(r - q), sigma=sigma)

        engine = MonteCarloEngine(
            process,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=spec.seed,
            method="exact",
        )

        strike = spec.strike

        def payoff_fn(paths):
            """Return 1D array of payoffs for each simulated path (undiscounted)."""
            S_T = paths[:, -1]
            if spec.is_call:
                payoffs = np.maximum(S_T - strike, 0.0)
            else:
                payoffs = np.maximum(strike - S_T, 0.0)
            return payoffs * spec.notional

        result = engine.price(S0, T, payoff_fn, discount_rate=r)
        return float(result["price"])