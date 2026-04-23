"""Agent-generated payoff: European payer swaption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import PricingValue
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.rate_style_swaption import price_swaption_black76



@dataclass(frozen=True)
class SwaptionSpec:
    """Specification for European payer swaption."""
    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class SwaptionPayoff:
    """European payer swaption."""

    def __init__(self, spec: SwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> SwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> PricingValue:
        spec = self._spec
        return price_swaption_black76(market_state, spec)