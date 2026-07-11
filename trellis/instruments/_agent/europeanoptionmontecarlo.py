"""Admitted adapter for a European single-state terminal claim."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.monte_carlo.single_state_diffusion import (
    price_single_state_terminal_claim_monte_carlo_result,
)
from trellis.models.resolution.single_state_diffusion import (
    terminal_intrinsic_from_resolved,
)


@dataclass(frozen=True)
class EuropeanOptionSpec:
    """Minimal European call/put terms consumed by the generic MC estimator."""

    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    dividend_yield: float = 0.0
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class EuropeanOptionMonteCarloPayoff:
    """Compose a vanilla terminal payoff with the single-state MC estimator."""

    def __init__(self, spec: EuropeanOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> EuropeanOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        result = price_single_state_terminal_claim_monte_carlo_result(
            market_state,
            self._spec,
            terminal_payoff=lambda terminal, resolved: terminal_intrinsic_from_resolved(
                terminal,
                resolved,
            ),
            scheme="exact",
            variance_reduction="none",
        )
        return float(result.price)
