"""Agent-generated payoff: Price a Himalaya-style basket on AAPL, MSFT, NVDA, and AMZN. Observe monthly on 2026-04-01, 2026-05-01, and 2026-06-01. At each observation, choose the best performer among the remaining names, remove it, lock that simple return, and settle the average locked returns once at maturity. Use discount curve, spot, vol surface, and correlation.."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.analytical.support import implied_zero_rate
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.ranked_observation_payoffs import (
    build_ranked_observation_basket_state_payoff,
    terminal_ranked_observation_basket_payoff,
)
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.basket_semantics import resolve_basket_semantics


@dataclass(frozen=True)
class HimalayaBasketSpec:
    """Specification for Price a Himalaya-style basket on AAPL, MSFT, NVDA, and AMZN. Observe monthly on 2026-04-01, 2026-05-01, and 2026-06-01. At each observation, choose the best performer among the remaining names, remove it, lock that simple return, and settle the average locked returns once at maturity. Use discount curve, spot, vol surface, and correlation.."""
    underlyings: str
    observation_dates: tuple[date, ...]
    expiry_date: date
    notional: float
    correlation_key: str | None
    rate_index: str | None
    strike: float = 0.0
    day_count: DayCountConvention = DayCountConvention.ACT_365


class HimalayaBasketPayoff:
    """Price a Himalaya-style basket on AAPL, MSFT, NVDA, and AMZN. Observe monthly on 2026-04-01, 2026-05-01, and 2026-06-01. At each observation, choose the best performer among the remaining names, remove it, lock that simple return, and settle the average locked returns once at maturity. Use discount curve, spot, vol surface, and correlation.."""

    def __init__(self, spec: HimalayaBasketSpec):
        self._spec = spec

    @property
    def spec(self) -> HimalayaBasketSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {
            "black_vol_surface",
            "discount_curve",
            "forward_curve",
            "model_parameters",
            "spot",
        }

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        resolved = resolve_basket_semantics(
            market_state,
            spec,
            constituents=spec.underlyings,
            observation_dates=spec.observation_dates,
            correlation_matrix_key=spec.correlation_key,
            selection_rule="best_of_remaining",
            lock_rule="remove_selected",
            aggregation_rule="average_locked_returns",
            option_type="call",
            selection_count=1,
            day_count=spec.day_count,
        )
        if resolved.T <= 0.0:
            intrinsic = terminal_ranked_observation_basket_payoff(
                spec,
                [[[float(value) for value in resolved.constituent_spots]]],
                resolved,
            )[0]
            return float(spec.notional) * float(intrinsic)

        domestic_rate = float(implied_zero_rate(resolved.domestic_df, resolved.T))
        process = CorrelatedGBM(
            mu=[domestic_rate - float(carry) for carry in resolved.constituent_carry],
            sigma=[float(value) for value in resolved.constituent_vols],
            corr=[
                [float(cell) for cell in row]
                for row in resolved.correlation_matrix
            ],
        )
        engine_steps = max(
            int(getattr(spec, "n_steps", 252) or 252),
            64,
            int(float(resolved.T) * 252.0 + 0.999999),
            len(resolved.observation_times) * 16 if resolved.observation_times else 64,
        )
        payoff = build_ranked_observation_basket_state_payoff(
            spec,
            resolved,
            n_steps=engine_steps,
        )
        engine = MonteCarloEngine(
            process,
            n_paths=max(int(getattr(spec, "n_paths", 50000) or 50000), 4096),
            n_steps=engine_steps,
            seed=int(getattr(spec, "seed", 42) or 42),
            method=getattr(spec, "mc_method", None) or "exact",
        )
        price_result = engine.price(
            tuple(float(value) for value in resolved.constituent_spots),
            float(resolved.T),
            payoff,
            discount_rate=0.0,
            storage_policy=payoff.path_requirement,
            return_paths=False,
        )
        return (
            float(spec.notional) * float(resolved.domestic_df)
            * float(price_result["price"])
        )
