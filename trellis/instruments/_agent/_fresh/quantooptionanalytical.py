"""Agent-generated payoff: Quanto option."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.quanto import price_quanto_option_analytical
from trellis.models.resolution.quanto import resolve_quanto_inputs


@dataclass(frozen=True)
class QuantoOptionSpec:
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
    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates", "model_parameters", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        resolved = resolve_quanto_inputs(market_state, self._spec)
        return float(price_quanto_option_analytical(self._spec, resolved))