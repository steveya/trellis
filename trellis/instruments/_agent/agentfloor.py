"""Agent-generated payoff: Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation

Implementation target: black76_cap
Preferred method family: analytical

Implementation target: black76_cap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import build_payment_timeline
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
        return {"black_vol_surface", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        timeline = build_payment_timeline(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            day_count=spec.day_count,
            time_origin=market_state.settlement,
            label="agent_floor_timeline",
        )
        if not timeline:
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
        for period in timeline:
            if period.payment_date <= market_state.settlement:
                continue

            accrual = float(period.accrual_fraction or 0.0)
            if accrual <= 0.0:
                continue

            t_start = max(float(period.t_start or 0.0), 0.0)
            t_pay = max(float(period.t_payment or 0.0), 0.0)
            option_time = t_pay

            if forward_curve is not None and hasattr(forward_curve, "forward_rate"):
                try:
                    fwd = forward_curve.forward_rate(max(t_start, 1e-6), t_pay)
                except TypeError:
                    fwd = forward_curve.forward_rate(option_time)
            elif forward_curve is not None and hasattr(forward_curve, "rate"):
                fwd = forward_curve.rate(option_time)
            else:
                raise ValueError("MarketState does not provide a usable forward rate curve")

            df = market_state.discount.discount(t_pay)
            vol = market_state.vol_surface.black_vol(option_time, spec.strike)

            if fwd >= spec.strike:
                undiscounted = black76_call(fwd, spec.strike, vol, option_time)
            else:
                undiscounted = black76_put(fwd, spec.strike, vol, option_time)

            total_pv += spec.notional * accrual * df * undiscounted

        return float(total_pv)
