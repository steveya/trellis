"""Callable bond — reference implementation using backward induction on a tree.

This is the hand-coded reference for tree-based pricing patterns.
The agent uses this as a template for other early-exercise instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState

from trellis.core.types import DayCountConvention, Frequency
from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction


@dataclass(frozen=True)
class CallableBondSpec:
    """Specification for a callable bond."""

    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: list[date]
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Callable bond priced via backward induction on a binomial tree.

    The issuer can call (redeem) the bond at par on specified dates.
    This is a Bermudan exercise problem — the issuer exercises when
    the continuation value exceeds the call price.
    """

    def __init__(self, spec: CallableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "black_vol"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        r = float(market_state.discount.zero_rate(T / 2))
        sigma = float(market_state.vol_surface.black_vol(T / 2, r))

        n_steps = min(200, max(50, int(T * 50)))
        tree = BinomialTree.crr(r, T, n_steps, r, sigma)
        dt = T / n_steps

        # Map call dates to tree step indices
        exercise_steps = []
        for cd in spec.call_dates:
            t_call = year_fraction(market_state.settlement, cd, spec.day_count)
            step = int(round(t_call / dt))
            if 0 < step < n_steps:
                exercise_steps.append(step)

        # Coupon per step (simplified: spread evenly)
        coupon_per_step = spec.notional * spec.coupon * dt

        def payoff_at_node(step, node):
            """Terminal payoff: final coupon + notional."""
            return spec.notional + coupon_per_step

        def exercise_value(step, node, tree):
            """Issuer calls at call_price (plus accrued coupon)."""
            return spec.call_price + coupon_per_step

        price = backward_induction(
            tree, payoff_at_node, r, "bermudan",
            exercise_steps, exercise_value,
        )

        # Add PV of intermediate coupons (simplified: coupon stream as annuity)
        # The tree price captures the optionality; we add the coupon stream PV
        # This is an approximation — a full implementation would embed coupons in the tree
        coupon_pv = 0.0
        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        starts = [spec.start_date] + schedule[:-1]
        for p_start, p_end in zip(starts, schedule):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.day_count)
            t_pay = year_fraction(market_state.settlement, p_end, spec.day_count)
            coupon_pv += spec.notional * spec.coupon * tau * float(
                market_state.discount.discount(t_pay)
            )

        # Combine: tree gives the embedded option-adjusted value
        # Total = min(straight_bond_pv, tree_price + coupon_pv)
        # The callable is worth less than or equal to the straight bond
        return min(price, coupon_pv + float(market_state.discount.discount(T)) * spec.notional)
