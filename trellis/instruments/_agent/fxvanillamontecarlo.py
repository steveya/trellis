"""Deterministic vanilla FX Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.instruments._agent.fxvanillaanalytical import _resolve_fx_inputs


@dataclass(frozen=True)
class FXVanillaOptionSpec:
    """Specification for a vanilla FX option under risk-neutral FX Monte Carlo."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class FXVanillaMonteCarloPayoff:
    """Deterministic thin adapter over the GBM Monte Carlo engine for vanilla FX."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "forward_rate", "black_vol", "fx"}

    def evaluate(self, market_state: MarketState) -> float:
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.processes.gbm import GBM

        spec = self._spec
        np = get_numpy()
        spot, T, domestic_df, foreign_df = _resolve_fx_inputs(market_state, spec)
        if T <= 0:
            intrinsic = max(spot - spec.strike, 0.0) if spec.option_type.lower() == "call" else max(spec.strike - spot, 0.0)
            return float(spec.notional) * float(intrinsic)

        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        rd = float(-np.log(domestic_df) / T)
        rf = float(-np.log(foreign_df) / T)
        process = GBM(mu=rd - rf, sigma=sigma)
        engine = MonteCarloEngine(
            process,
            n_paths=int(spec.n_paths),
            n_steps=int(spec.n_steps),
            seed=42,
            method="exact",
        )

        option_type = spec.option_type.lower()
        strike = float(spec.strike)
        notional = float(spec.notional)

        def payoff_fn(paths):
            terminal = paths[:, -1]
            if option_type == "call":
                payoffs = np.maximum(terminal - strike, 0.0)
            elif option_type == "put":
                payoffs = np.maximum(strike - terminal, 0.0)
            else:
                raise ValueError(
                    f"Unsupported option_type {spec.option_type!r}; expected 'call' or 'put'"
                )
            return payoffs * notional

        result = engine.price(
            spot,
            T,
            payoff_fn,
            discount_rate=rd,
            return_paths=False,
        )
        return float(result["price"])
