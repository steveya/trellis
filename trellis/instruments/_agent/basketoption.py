"""Generated terminal-basket adapter composed from public pricing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import implied_zero_rate
from trellis.models.analytical.terminal_basket import (
    two_asset_extremum_option_stulz,
    two_asset_spread_option_kirk,
    two_asset_terminal_basket_gauss_hermite,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.payoffs import terminal_basket_option_payoff
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.terminal_basket import (
    resolve_terminal_basket_inputs,
)

np = get_numpy()

@dataclass(frozen=True)
class BasketOptionSpec:
    """Terminal basket market, payoff, and optional simulation controls."""
    notional: float
    underliers: str
    spots: str
    strike: float
    expiry_date: date
    correlation: str
    weights: str | None = None
    vols: str | None = None
    dividend_yields: str | None = None
    basket_style: str = "weighted_sum"
    option_type: str = "call"
    averaging_type: str | None = None
    n_observations: int | None = None
    barrier_level: float | None = None
    barrier_direction: str | None = None
    n_paths: int = 40_000
    seed: int = 42
    day_count: DayCountConvention = DayCountConvention.ACT_365


class BasketOptionPayoff:
    """Evaluate a terminal basket through explicit public composition."""

    def __init__(self, spec: BasketOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> BasketOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "spot", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        resolved = resolve_terminal_basket_inputs(market_state, spec)
        semantics = resolved.semantics
        requires_path_runtime = any(
            value is not None
            for value in (
                spec.averaging_type,
                spec.n_observations,
                spec.barrier_level,
                spec.barrier_direction,
            )
        )
        if requires_path_runtime:
            if semantics.T <= 0.0:
                intrinsic = terminal_basket_option_payoff(
                    np.asarray([semantics.constituent_spots], dtype=float),
                    weights=resolved.weights,
                    basket_style=resolved.basket_style,
                    strike=resolved.strike,
                    option_type=resolved.option_type,
                )[0]
                return float(spec.notional) * float(intrinsic)
            rate = float(implied_zero_rate(semantics.domestic_df, semantics.T))
            process = CorrelatedGBM(
                mu=[rate for _ in semantics.constituent_names],
                sigma=list(resolved.vols),
                corr=[list(row) for row in resolved.correlation_matrix],
                dividend_yield=list(resolved.carry),
            )

            def payoff_fn(paths):
                return terminal_basket_option_payoff(
                    paths[:, -1, :],
                    weights=resolved.weights,
                    basket_style=resolved.basket_style,
                    strike=resolved.strike,
                    option_type=resolved.option_type,
                )

            result = MonteCarloEngine(
                process,
                n_paths=max(int(spec.n_paths), 8192),
                n_steps=1,
                seed=int(spec.seed),
                method="exact",
            ).price(
                np.asarray(semantics.constituent_spots, dtype=float),
                float(semantics.T),
                payoff_fn,
                discount_rate=rate,
                return_paths=False,
            )
            return float(spec.notional) * float(result["price"])

        if semantics.T <= 0.0:
            intrinsic = terminal_basket_option_payoff(
                np.asarray([semantics.constituent_spots], dtype=float),
                weights=resolved.weights,
                basket_style=resolved.basket_style,
                strike=resolved.strike,
                option_type=resolved.option_type,
            )[0]
            return float(spec.notional) * float(intrinsic)
        if resolved.basket_style in {"best_of", "worst_of"}:
            unit_price = two_asset_extremum_option_stulz(
                spots=resolved.notional_spots,
                strike=resolved.strike,
                T=semantics.T,
                discount_factor=semantics.domestic_df,
                dividend_yields=resolved.carry,
                volatilities=resolved.vols,
                correlation=resolved.correlation_matrix[0][1],
                basket_style=resolved.basket_style,
                option_type=resolved.option_type,
            )
        elif resolved.basket_style == "spread":
            rate = float(implied_zero_rate(semantics.domestic_df, semantics.T))
            forwards = tuple(
                float(spot)
                * exp(
                    (rate - float(dividend_yield))
                    * float(semantics.T)
                )
                for spot, dividend_yield in zip(
                    semantics.constituent_spots,
                    resolved.carry,
                    strict=True,
                )
            )
            unit_price = two_asset_spread_option_kirk(
                forwards=forwards,
                strike=resolved.strike,
                T=semantics.T,
                discount_factor=semantics.domestic_df,
                volatilities=resolved.vols,
                correlation=resolved.correlation_matrix[0][1],
                weights=resolved.weights,
                option_type=resolved.option_type,
            )
        else:
            unit_price = two_asset_terminal_basket_gauss_hermite(
                spots=resolved.notional_spots,
                weights=resolved.weights,
                strike=resolved.strike,
                T=semantics.T,
                discount_factor=semantics.domestic_df,
                dividend_yields=resolved.carry,
                volatilities=resolved.vols,
                correlation=resolved.correlation_matrix[0][1],
                basket_style=resolved.basket_style,
                option_type=resolved.option_type,
            )
        return float(spec.notional) * float(unit_price)
