"""Agent-generated payoff: Callable bond with a call schedule (Bermudan callable)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import datetime

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class CallableBondSpec:
    """Specification for Callable bond with a call schedule (Bermudan callable)."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: str
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Callable bond with a call schedule (Bermudan callable)."""

    def __init__(self, spec: CallableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol", "discount"}

    def evaluate(self, market_state: MarketState) -> float:
        from trellis.core.date_utils import generate_schedule, year_fraction
        from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction

        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        # Use the zero rate at mid‐term to calibrate the tree initial short rate.
        r0 = float(market_state.discount.zero_rate(T / 2))
        # Get Black implied vol at mid‐term then convert to Hull–White absolute volatility.
        black_vol = float(market_state.vol_surface.black_vol(T / 2, r0))
        sigma = black_vol * r0  # conversion: sigma_HW = sigma_Black * forward_rate(~r0)
        mean_reversion = 0.1  # typical Hull–White mean reversion

        # Determine number of steps.
        n_steps = min(200, max(50, int(T * 50)))
        dt = T / n_steps

        # Build the short-rate lattice.
        lattice = build_rate_lattice(r0, sigma, mean_reversion, T, n_steps)

        # Map call dates (from comma‐separated string) to tree step indices.
        exercise_steps = []
        for dstr in spec.call_dates.split(","):
            try:
                cd = datetime.date.fromisoformat(dstr.strip())
            except Exception:
                continue
            t_call = year_fraction(market_state.settlement, cd, spec.day_count)
            step = int(round(t_call / dt))
            if 0 < step < n_steps:
                exercise_steps.append(step)
        exercise_steps = sorted(set(exercise_steps))

        # Embed coupon payments at actual coupon dates.
        # Build a mapping of tree step index to coupon amount.
        coupon_steps: dict[int, float] = {}
        coupon_schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        prev_date = spec.start_date
        for pay_date in coupon_schedule:
            if pay_date <= market_state.settlement:
                prev_date = pay_date
                continue
            tau = year_fraction(prev_date, pay_date, spec.day_count)
            coupon_amt = spec.notional * spec.coupon * tau
            t_pay = year_fraction(market_state.settlement, pay_date, spec.day_count)
            step_index = int(round(t_pay / dt))
            if step_index > n_steps:
                step_index = n_steps
            coupon_steps[step_index] = coupon_steps.get(step_index, 0.0) + coupon_amt
            prev_date = pay_date

        # Attach coupon cashflows to the lattice so that the backward induction
        # algorithm can (internally) use them when rolling cashflows back.
        lattice.coupon_steps = coupon_steps

        # At maturity, the investor receives the notional plus any terminal coupon.
        def payoff_at_node(step, node, lattice):
            coupon = coupon_steps.get(step, 0.0)
            return spec.notional + coupon

        # If exercised (called), the holder receives the call price plus any coupon due.
        def exercise_value(step, node, lattice):
            coupon = coupon_steps.get(step, 0.0)
            return spec.call_price + coupon

        price = lattice_backward_induction(
            lattice, payoff_at_node, exercise_value,
            exercise_type="bermudan", exercise_steps=exercise_steps,
        )

        return price