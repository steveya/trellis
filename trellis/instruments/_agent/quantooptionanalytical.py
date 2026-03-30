"""Compatibility adapter for the quanto analytical payoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.quanto import price_quanto_option_raw
from trellis.models.resolution.quanto import resolve_quanto_inputs


REQUIREMENTS = frozenset(
    {
        "black_vol_surface",
        "discount_curve",
        "forward_curve",
        "fx_rates",
        "model_parameters",
        "spot",
    }
)


@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for the single-name quanto analytical adapter."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class QuantoOptionAnalyticalPayoff:
    """Compatibility payoff that delegates through the shared analytical helper."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def evaluate(self, market_state: MarketState) -> float:
        return float(
            price_quanto_option_raw(
                self._spec,
                resolve_quanto_inputs(market_state, self._spec),
            )
        )
