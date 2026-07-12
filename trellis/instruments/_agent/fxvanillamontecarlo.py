"""Deterministic vanilla FX Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical import terminal_intrinsic
from trellis.models.fx_vanilla import resolve_fx_vanilla_inputs
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import terminal_value_payoff
from trellis.models.processes.gbm import GBM


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
    """Deterministic adapter over generic terminal-state Monte Carlo primitives."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates"}

    def evaluate(self, market_state: MarketState) -> float:
        resolved = resolve_fx_vanilla_inputs(market_state, self._spec)
        gk = resolved.garman_kohlhagen
        if gk.T <= 0.0:
            return float(resolved.notional) * float(
                terminal_intrinsic(
                    resolved.option_type,
                    spot=gk.spot,
                    strike=gk.strike,
                )
            )

        payoff = terminal_value_payoff(
            lambda terminal: resolved.notional * terminal_intrinsic(
                resolved.option_type,
                spot=terminal,
                strike=gk.strike,
            ),
            name="fx_vanilla_terminal",
        )
        engine = MonteCarloEngine(
            GBM(
                mu=resolved.domestic_rate - resolved.foreign_rate,
                sigma=float(gk.sigma),
            ),
            n_paths=max(int(self._spec.n_paths), 1),
            n_steps=max(int(self._spec.n_steps), 1),
            seed=int(self._spec.seed),
            method="exact",
        )
        result = engine.price(
            float(gk.spot),
            float(gk.T),
            payoff,
            discount_rate=float(resolved.domestic_rate),
            return_paths=False,
        )
        return float(result["price"])
