"""Agent-generated payoff: Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class AgentFloorSpec:
    """Specification for Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""
    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None


class AgentFloorPayoff:
    """Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""

    def __init__(self, spec: AgentFloorSpec):
        self._spec = spec

    @property
    def spec(self) -> AgentFloorSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        if not schedule:
            return 0.0

        try:
            forward_curve = (
                market_state.forecast_forward_curve(spec.rate_index)
                if spec.rate_index is not None
                else market_state.forecast_forward_curve()
            )
        except Exception:
            forward_curve = getattr(market_state, "forward_curve", None)

        total_pv = 0.0
        prev_date = spec.start_date

        for pay_date in schedule:
            accrual = year_fraction(prev_date, pay_date, spec.day_count)
            if accrual <= 0.0:
                prev_date = pay_date
                continue

            t = max(year_fraction(spec.start_date, pay_date, spec.day_count), 0.0)

            if forward_curve is not None and hasattr(forward_curve, "forward_rate"):
                try:
                    fwd = forward_curve.forward_rate(prev_date, pay_date)
                except TypeError:
                    fwd = forward_curve.forward_rate(t)
            elif forward_curve is not None and hasattr(forward_curve, "rate"):
                fwd = forward_curve.rate(t)
            else:
                raise ValueError("MarketState does not provide a usable forward rate curve")

            df = market_state.discount.discount(t)
            vol = market_state.vol_surface.black_vol(t, spec.strike)

            if fwd >= spec.strike:
                undiscounted = black76_call(fwd, spec.strike, vol, t)
            else:
                undiscounted = black76_put(fwd, spec.strike, vol, t)

            total_pv += spec.notional * accrual * df * undiscounted
            prev_date = pay_date

        return float(total_pv)