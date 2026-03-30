"""Agent-generated payoff: European payer swaption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.calibration.rates import swaption_terms
from trellis.models.black import black76_call, black76_put


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
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T, annuity, forward_swap_rate, _payment_count = swaption_terms(spec, market_state)
        sigma = market_state.vol_surface.black_vol(T, spec.strike)

        if spec.is_payer:
            black_value = black76_call(forward_swap_rate, spec.strike, sigma, T)
        else:
            black_value = black76_put(forward_swap_rate, spec.strike, sigma, T)

        return spec.notional * annuity * float(black_value)
