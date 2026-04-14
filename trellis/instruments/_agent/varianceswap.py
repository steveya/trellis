"""Deterministic benchmark wrapper for variance-swap parity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical import (
    equity_variance_swap_outputs_analytical,
    price_equity_variance_swap_analytical,
)


@dataclass(frozen=True)
class VarianceSwapSpec:
    """Specification for one equity variance swap."""

    notional: float
    spot: float
    strike_variance: float
    expiry_date: date
    realized_variance: float = 0.0
    replication_strikes: str | None = None
    replication_volatilities: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class VarianceSwapPayoff:
    """Variance-swap payoff backed by the shared analytical helper."""

    def __init__(self, spec: VarianceSwapSpec):
        self._spec = spec

    @property
    def spec(self) -> VarianceSwapSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "black_vol_surface"}

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        return dict(equity_variance_swap_outputs_analytical(market_state, self._spec))

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_equity_variance_swap_analytical(market_state, self._spec))
