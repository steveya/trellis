"""Agent-generated payoff: European payer swaption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
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
        fwd_curve = market_state.forecast_forward_curve(spec.rate_index)

        schedule = generate_schedule(spec.swap_start, spec.swap_end, spec.swap_frequency)
        starts = [spec.swap_start] + schedule[:-1]

        annuity = 0.0
        float_pv = 0.0
        for p_start, p_end in zip(starts, schedule):
            tau = year_fraction(p_start, p_end, spec.day_count)
            t_start = year_fraction(market_state.settlement, p_start, spec.day_count)
            t_end = year_fraction(market_state.settlement, p_end, spec.day_count)
            t_start = max(t_start, 1e-6)

            df = market_state.discount.discount(t_end)
            F = fwd_curve.forward_rate(t_start, t_end)
            annuity += tau * float(df)
            float_pv += float(F) * tau * float(df)

        forward_swap_rate = float_pv / annuity if annuity > 0 else 0.0

        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        sigma = market_state.vol_surface.black_vol(T, spec.strike)

        if spec.is_payer:
            black_value = black76_call(forward_swap_rate, spec.strike, sigma, T)
        else:
            black_value = black76_put(forward_swap_rate, spec.strike, sigma, T)

        return spec.notional * annuity * float(black_value)
