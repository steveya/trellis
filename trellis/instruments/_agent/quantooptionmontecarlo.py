"""Deterministic quanto-option Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.quanto_option import price_quanto_option_monte_carlo_from_market_state


REQUIREMENTS = frozenset(
    {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "fx_rates",
        "spot",
        "model_parameters",
    }
)


@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for a single-underlier quanto option."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 42
    mc_method: str = "exact"


class QuantoOptionMonteCarloPayoff:
    """Deterministic thin adapter over the semantic-facing quanto MC helper."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_quanto_option_monte_carlo_from_market_state(market_state, self._spec))
