"""Agent-generated payoff: Price a Himalaya-style basket on AAPL, MSFT, NVDA, and AMZN. Observe monthly on 2026-04-01, 2026-05-01, and 2026-06-01. At each observation, choose the best performer among the remaining names, remove it, lock that simple return, and settle the average locked returns once at maturity. Use discount curve, spot, vol surface, and correlation.."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.resolution.basket_semantics import resolve_basket_semantics
from trellis.models.monte_carlo.semantic_basket import (
    RankedObservationBasketSpec,
    RankedObservationBasketMonteCarloPayoff,
    price_ranked_observation_basket_monte_carlo,
)


@dataclass(frozen=True)
class HimalayaBasketSpec:
    """Specification for Price a Himalaya-style basket on AAPL, MSFT, NVDA, and AMZN. Observe monthly on 2026-04-01, 2026-05-01, and 2026-06-01. At each observation, choose the best performer among the remaining names, remove it, lock that simple return, and settle the average locked returns once at maturity. Use discount curve, spot, vol surface, and correlation.."""
    underlyings: str
    observation_dates: str
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

        # Parse underlyings and observation dates
        underlyings = [u.strip() for u in spec.underlyings.split(",")]
        obs_dates = [
            date.fromisoformat(d.strip())
            for d in spec.observation_dates.split(",")
        ]
        expiry_date = spec.expiry_date
        day_count = spec.day_count

        # Resolve basket semantics: spots, vols, forwards, correlation
        basket_state = resolve_basket_semantics(
            market_state,
            constituents=",".join(underlyings),
            observation_dates=",".join(d.isoformat() for d in obs_dates),
            expiry_date=expiry_date,
            correlation_matrix_key=spec.correlation_key,
            day_count=day_count,
        )

        # Build the ranked-observation basket spec for the Monte Carlo helper
        ranked_spec = RankedObservationBasketSpec(
            expiry_date=expiry_date,
            notional=spec.notional,
            strike=spec.strike,
            constituents=",".join(underlyings),
            observation_dates=",".join(d.isoformat() for d in obs_dates),
            option_type="call",
            day_count=day_count,
        )

        # Build the Monte Carlo payoff wrapper
        ranked_payoff = RankedObservationBasketMonteCarloPayoff(spec=ranked_spec)

        # Price via the shared ranked-observation basket Monte Carlo route
        pv = price_ranked_observation_basket_monte_carlo(
            spec=ranked_spec,
            resolved=basket_state,
        )

        return float(pv)
