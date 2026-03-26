"""Agent-generated payoff: Build a pricer for: OAS duration (spread duration) for callable bonds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction


@dataclass(frozen=True)
class CallableBondSpec:
    """Specification for Build a pricer for: OAS duration (spread duration) for callable bonds."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: str
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Build a pricer for: OAS duration (spread duration) for callable bonds."""

    def __init__(self, spec: CallableBondSpec):
        """Store the generated callable-bond specification."""
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        """Return the immutable generated callable-bond specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and rate-vol inputs."""
        return {"black_vol", "discount"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated callable bond on a rate tree and cap it by straight-bond PV."""
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        # Obtain the initial short rate using half-maturity as proxy for the forward rate.
        r0 = float(market_state.discount.zero_rate(T / 2))
        # Convert Black vol to Hull-White absolute rate vol.
        black_vol = float(market_state.vol_surface.black_vol(T / 2, r0))
        sigma_hw = black_vol * r0
        mean_reversion = 0.1  # typical mean-reversion for Hull-White

        n_steps = min(200, max(50, int(T * 50)))
        # Build and calibrate the short-rate lattice using the provided discount curve.
        lattice = build_rate_lattice(r0, sigma_hw, mean_reversion, T, n_steps,
                                     discount_curve=market_state.discount)
        dt = T / n_steps

        # Parse the call dates string and map each call date to a tree step index.
        exercise_steps = []
        for d_str in spec.call_dates.split(','):
            call_date = date.fromisoformat(d_str.strip())
            t_call = year_fraction(market_state.settlement, call_date, spec.day_count)
            step = int(round(t_call / dt))
            if 0 < step < n_steps:
                exercise_steps.append(step)

        # In the tree, we use a uniform coupon accrual per step.
        coupon_per_step = spec.notional * spec.coupon * dt

        def payoff_at_node(step, node, lattice):
            """Return maturity redemption plus the final coupon."""
            # At maturity, the bond pays notional plus the final coupon.
            return spec.notional + coupon_per_step

        def exercise_value(step, node, lattice):
            """Return the issuer call amount at a call step."""
            # When the issuer exercises the call, they pay the call price plus the accrued coupon.
            return spec.call_price + coupon_per_step

        def cashflow(step, node, lattice):
            """Return the uniform per-step coupon cashflow used in the lattice."""
            # At coupon dates (as mapped in the tree) pay the coupon.
            return coupon_per_step

        # Backward induction on the rate tree with Bermudan exercise.
        tree_price = lattice_backward_induction(
            lattice, payoff_at_node, exercise_value,
            exercise_type="bermudan", exercise_steps=exercise_steps,
            cashflow_at_node=cashflow,
            exercise_fn=min,  # issuer calls to minimize liability
        )

        # Compute the PV of coupon cashflows using the actual discrete schedule.
        coupon_pv = 0.0
        schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
        starts = [spec.start_date] + schedule[:-1]
        for p_start, p_end in zip(starts, schedule):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.day_count)
            t_pay = year_fraction(market_state.settlement, p_end, spec.day_count)
            coupon_pv += spec.notional * spec.coupon * tau * float(market_state.discount.discount(t_pay))

        # Compute the PV of a straight bond from the discount curve.
        straight_bond_pv = coupon_pv + float(market_state.discount.discount(T)) * spec.notional

        # The callable bond cannot be worth more than the corresponding straight bond.
        return min(tree_price, straight_bond_pv)
