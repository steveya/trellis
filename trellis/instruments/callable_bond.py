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
from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction


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
    """Callable bond priced via backward induction on a calibrated rate lattice.

    The issuer can call (redeem) the bond at par on specified dates.
    This is a Bermudan exercise problem — the issuer exercises when
    the continuation value exceeds the call price.
    """

    def __init__(self, spec: CallableBondSpec):
        """Store the callable-bond terms used for all future tree valuations."""
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        """Return the immutable callable-bond specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that callable-bond valuation needs discount and rate-vol inputs."""
        return {"discount", "black_vol"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the bond on a calibrated short-rate tree and cap it by straight-bond PV."""
        spec = self._spec
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

        # Map call dates to tree step indices
        exercise_steps = []
        for cd in spec.call_dates:
            t_call = year_fraction(market_state.settlement, cd, spec.day_count)
            step = int(round(t_call / dt))
            if 0 < step < n_steps:
                exercise_steps.append(step)

        # Use uniform coupon accrual per step for the tree
        coupon_per_step = spec.notional * spec.coupon * dt

        def payoff_at_node(step, node, lattice):
            """Terminal payoff: final coupon + notional."""
            return spec.notional + coupon_per_step

        def exercise_value(step, node, lattice):
            """Issuer calls at call_price (plus accrued coupon)."""
            return spec.call_price + coupon_per_step

        def cashflow(step, node, lattice):
            """Intermediate coupon cashflows at each tree step."""
            return coupon_per_step

        # exercise_fn=min: issuer calls to MINIMIZE liability (callable bond)
        tree_price = lattice_backward_induction(
            lattice, payoff_at_node, exercise_value,
            exercise_type="bermudan", exercise_steps=exercise_steps,
            cashflow_at_node=cashflow,
            exercise_fn=min,
        )

        # Compute PV of discrete coupon cashflows using the actual payment schedule
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

        # Straight bond PV: discounted principal plus PV of coupons
        straight_bond_pv = coupon_pv + float(market_state.discount.discount(T)) * spec.notional

        # The callable bond cannot be worth more than a straight bond
        return min(tree_price, straight_bond_pv)
