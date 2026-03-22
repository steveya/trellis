"""Cap and Floor payoffs — decomposed into caplets/floorlets via Black76."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class CapFloorSpec:
    """Specification for a cap or floor."""

    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None


def _capfloor_pv(
    spec: CapFloorSpec,
    market_state: MarketState,
    pricing_fn,
) -> float:
    """Shared logic: sum discounted caplet/floorlet values."""
    schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
    period_starts = [spec.start_date] + schedule[:-1]

    pv = 0.0
    for p_start, p_end in zip(period_starts, schedule):
        if p_end <= market_state.settlement:
            continue

        tau = year_fraction(p_start, p_end, spec.day_count)
        t_fix = year_fraction(market_state.settlement, p_start, spec.day_count)
        t_pay = year_fraction(market_state.settlement, p_end, spec.day_count)

        if t_fix <= 0:
            continue

        fwd = market_state.forecast_forward_curve(spec.rate_index)
        F = fwd.forward_rate(t_fix, t_pay)
        sigma = market_state.vol_surface.black_vol(t_fix, spec.strike)

        undiscounted = spec.notional * tau * pricing_fn(F, spec.strike, sigma, t_fix)
        df = market_state.discount.discount(t_pay)
        pv += float(undiscounted) * float(df)

    return pv


class CapPayoff:
    """Interest rate cap priced via Black76."""

    def __init__(self, spec: CapFloorSpec):
        self._spec = spec

    @property
    def spec(self) -> CapFloorSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "forward_rate", "black_vol"}

    def evaluate(self, market_state: MarketState) -> float:
        return _capfloor_pv(self._spec, market_state, black76_call)


class FloorPayoff:
    """Interest rate floor priced via Black76."""

    def __init__(self, spec: CapFloorSpec):
        self._spec = spec

    @property
    def spec(self) -> CapFloorSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "forward_rate", "black_vol"}

    def evaluate(self, market_state: MarketState) -> float:
        return _capfloor_pv(self._spec, market_state, black76_put)
