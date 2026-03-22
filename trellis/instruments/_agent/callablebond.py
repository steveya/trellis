"""Agent-generated payoff: Callable bond with a call schedule (Bermudan callable)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
        from trellis.models.trees.binomial import BinomialTree
        from trellis.models.trees.backward_induction import backward_induction

        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T / 2))
        sigma = float(market_state.vol_surface.black_vol(T / 2, r))
        n_steps = min(200, max(50, int(T * 50)))
        tree = BinomialTree.crr(r, T, n_steps, r, sigma)
        dt = T / n_steps

        # Parse call dates from string; skip any invalid dates.
        call_dates_list = []
        for s in spec.call_dates.split(","):
            s = s.strip()
            try:
                d_call = date.fromisoformat(s)
                call_dates_list.append(d_call)
            except ValueError:
                # Skip invalid date strings
                continue

        # Map valid call dates to tree step indices
        exercise_steps = []
        for cd in call_dates_list:
            t_call = year_fraction(market_state.settlement, cd, spec.day_count)
            step = int(round(t_call / dt))
            if 0 < step < n_steps:
                exercise_steps.append(step)

        # Coupon per step (spread evenly across steps)
        coupon_per_step = spec.notional * spec.coupon * dt

        def payoff_at_node(step, node):
            """Terminal payoff at maturity: final coupon plus notional."""
            return spec.notional + coupon_per_step

        def exercise_value(step, node, tree):
            """Exercise value at a callable date: call price plus coupon."""
            return spec.call_price + coupon_per_step

        price = backward_induction(
            tree, payoff_at_node, r, "bermudan", exercise_steps, exercise_value
        )

        # Calculate PV of coupon stream outside the tree using schedule generation.
        coupon_pv = 0.0
        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        starts = [spec.start_date] + schedule[:-1]
        for p_start, p_end in zip(starts, schedule):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.day_count)
            t_pay = year_fraction(market_state.settlement, p_end, spec.day_count)
            coupon_pv += spec.notional * spec.coupon * tau * float(market_state.discount.discount(t_pay))

        # The straight bond PV (no call option) discounted at maturity.
        straight_bond = float(market_state.discount.discount(T)) * spec.notional

        # Combined value: the callable bond is worth at most the straight bond.
        return min(price, coupon_pv + straight_bond)