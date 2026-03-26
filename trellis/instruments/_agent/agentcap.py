"""Agent-generated payoff: Build a pricer for: CMS cap: convexity adjustment (Hagan) vs replication vs MC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class AgentCapSpec:
    """Specification for Build a pricer for: CMS cap: convexity adjustment (Hagan) vs replication vs MC."""
    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None


class AgentCapPayoff:
    """Build a pricer for: CMS cap: convexity adjustment (Hagan) vs replication vs MC."""

    def __init__(self, spec: AgentCapSpec):
        """Store the generated cap specification."""
        self._spec = spec

    @property
    def spec(self) -> AgentCapSpec:
        """Return the immutable generated cap specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare the discount, forward-rate, and vol inputs used by the payoff."""
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated cap as a discounted strip of Black-76 caplets."""
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
            if t_fix <= 0:
                continue
            F = fwd_curve.forward_rate(t_fix, t_pay)
            sigma = market_state.vol_surface.black_vol(t_fix, spec.strike)
            undiscounted = spec.notional * tau * black76_call(F, spec.strike, sigma, t_fix)
            df = market_state.discount.discount(t_pay)
            pv += float(undiscounted) * float(df)
        return pv
