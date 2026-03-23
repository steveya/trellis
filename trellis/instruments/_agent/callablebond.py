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
        from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction

        spec = self._spec
        # Total time to maturity in years
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        # Get initial short rate using half-maturity (as proxy for forward rate)
        r0 = float(market_state.discount.zero_rate(T / 2))
        # Get Black vol and convert to Hull-White absolute rate vol
        black_vol = float(market_state.vol_surface.black_vol(T / 2, r0))
        sigma_hw = black_vol * r0
        mean_reversion = 0.1  # typical Hull-White mean reversion

        n_steps = min(200, max(50, int(T * 50)))
        lattice = build_rate_lattice(r0, sigma_hw, mean_reversion, T, n_steps,
                                     discount_curve=market_state.discount)
        dt = T / n_steps

        # Map call dates (from comma-separated ISO strings) to tree step indices
        exercise_steps = []
        for d_str in spec.call_dates.split(","):
            d_str = d_str.strip()
            if not d_str:
                continue
            call_date = date.fromisoformat(d_str)
            t_call = year_fraction(market_state.settlement, call_date, spec.day_count)
            step = int(round(t_call / dt))
            if 0 < step < n_steps:
                exercise_steps.append(step)

        # Use uniform coupon accrual per step for the tree
        coupon_per_step = spec.notional * spec.coupon * dt

        def payoff_at_node(step, node, lattice):
            # Terminal payoff: principal plus final coupon accrual.
            return spec.notional + coupon_per_step

        def exercise_value(step, node, lattice):
            # Exercise value: call price (typically par) plus coupon accrual at this step.
            return spec.call_price + coupon_per_step

        tree_price = lattice_backward_induction(
            lattice, payoff_at_node, exercise_value,
            exercise_type="bermudan", exercise_steps=exercise_steps,
        )

        # Compute PV of discrete coupon cashflows using the actual payment schedule.
        coupon_pv = 0.0
        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        # For each period, the coupon is based on the accrual from the previous coupon date.
        period_starts = [spec.start_date] + schedule[:-1]
        for p_start, p_end in zip(period_starts, schedule):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.day_count)
            t_pay = year_fraction(market_state.settlement, p_end, spec.day_count)
            coupon_pv += spec.notional * spec.coupon * tau * float(market_state.discount.discount(t_pay))

        # Straight bond PV: discounted principal plus PV of coupons.
        straight_bond_pv = coupon_pv + float(market_state.discount.discount(T)) * spec.notional

        # The callable bond cannot be worth more than a straight bond.
        return min(tree_price, straight_bond_pv)