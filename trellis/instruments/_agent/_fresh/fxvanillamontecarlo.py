"""Deterministic vanilla FX Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.fx_vanilla import price_fx_vanilla_monte_carlo


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
    seed: int = 42


class FXVanillaMonteCarloPayoff:
    """Deterministic thin adapter over the semantic-facing FX Monte Carlo helper."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates"}

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_fx_vanilla_monte_carlo(market_state, self._spec))
