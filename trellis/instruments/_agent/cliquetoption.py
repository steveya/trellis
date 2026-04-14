"""Deterministic benchmark wrapper for FinancePy-style cliquet parity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical import price_equity_cliquet_option_analytical


@dataclass(frozen=True)
class CliquetOptionSpec:
    """Specification for one reset-style cliquet option."""

    notional: float
    spot: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.THIRTY_E_360
    time_day_count: DayCountConvention = DayCountConvention.ACT_365


class CliquetOptionPayoff:
    """Reset-style cliquet payoff backed by the shared analytical helper."""

    def __init__(self, spec: CliquetOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> CliquetOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        return float(price_equity_cliquet_option_analytical(market_state, self._spec))
