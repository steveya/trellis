"""Agent-generated payoff: Build a pricer for: Cap/floor: Black76 vs HW tree vs MC rate simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class AgentFloorSpec:
    """Specification for Build a pricer for: Cap/floor: Black76 vs HW tree vs MC rate simulation."""
    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None


class AgentFloorPayoff:
    """Build a pricer for: Cap/floor: Black76 vs HW tree vs MC rate simulation."""

    def __init__(self, spec: AgentFloorSpec):
        """Store the generated floor specification."""
        self._spec = spec

    @property
    def spec(self) -> AgentFloorSpec:
        """Return the immutable generated floor specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare the discount, forward-rate, and vol inputs used by the payoff."""
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated floor as a discounted strip of Black-76 floorlets."""
        spec = self._spec
        fwd_curve = market_state.forecast_forward_curve(spec.rate_index)
        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        period_starts = [spec.start_date] + schedule[:-1]

        pv = 0.0
        for p_start, p_end in zip(period_starts, schedule):
            if p_end <= market_state.settlement:
                continue

            tau = year_fraction(p_start, p_end, spec.day_count)
            t_fix = year_fraction(market_state.settlement, p_start, spec.day_count)
            t_pay = year_fraction(market_state.settlement, p_end, spec.day_count)
            t_fix = max(t_fix, 1e-6)

            F = fwd_curve.forward_rate(t_fix, t_pay)
            sigma = market_state.vol_surface.black_vol(t_fix, spec.strike)

            undiscounted = spec.notional * tau * black76_put(F, spec.strike, sigma, t_fix)
            df = market_state.discount.discount(t_pay)
            pv += float(undiscounted) * float(df)

        return pv
