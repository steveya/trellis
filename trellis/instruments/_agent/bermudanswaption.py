"""Agent-generated payoff: Build a pricer for: Bermudan swaption: tree vs LSM MC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    """Specification for Build a pricer for: Bermudan swaption: tree vs LSM MC."""

    notional: float
    strike: float
    exercise_dates: str
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class BermudanSwaptionPayoff:
    """Build a pricer for: Bermudan swaption: tree vs LSM MC."""

    def __init__(self, spec: BermudanSwaptionSpec):
        """Store the generated Bermudan swaption specification."""
        self._spec = spec

    @property
    def spec(self) -> BermudanSwaptionSpec:
        """Return the immutable generated Bermudan swaption specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting, forwarding, and Black vol."""
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated Bermudan swaption on a calibrated short-rate lattice."""
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.swap_end, spec.day_count)
        if T <= 0:
            return 0.0

        r0 = float(market_state.discount.zero_rate(max(T / 2, 1e-6)))
        black_vol = float(market_state.vol_surface.black_vol(max(T / 2, 1e-6), r0))
        sigma_hw = black_vol * max(r0, 1e-6)
        mean_reversion = 0.1

        n_steps = min(200, max(50, int(T * 20)))
        dt = T / n_steps
        lattice = build_rate_lattice(
            r0,
            sigma_hw,
            mean_reversion,
            T,
            n_steps,
            discount_curve=market_state.discount,
        )

        parsed_exercise_dates = []
        for raw_date in spec.exercise_dates.split(","):
            raw_date = raw_date.strip()
            if not raw_date:
                continue
            try:
                ex_date = date.fromisoformat(raw_date)
            except ValueError:
                continue
            if market_state.settlement < ex_date < spec.swap_end:
                parsed_exercise_dates.append(ex_date)

        if not parsed_exercise_dates:
            return 0.0

        first_exercise = min(parsed_exercise_dates)
        exercise_steps = sorted({
            int(round(year_fraction(market_state.settlement, ex_date, spec.day_count) / dt))
            for ex_date in parsed_exercise_dates
            if 0 < int(round(year_fraction(market_state.settlement, ex_date, spec.day_count) / dt)) < n_steps
        })
        if not exercise_steps:
            return 0.0

        swap_schedule = [first_exercise]
        coupon_months = 12 // spec.swap_frequency.value
        current = first_exercise
        while current < spec.swap_end:
            from trellis.core.date_utils import add_months

            current = add_months(current, coupon_months)
            if current <= spec.swap_end:
                swap_schedule.append(current)

        payment_times = [
            year_fraction(market_state.settlement, pay_date, spec.day_count)
            for pay_date in swap_schedule[1:]
        ]
        accruals = [
            year_fraction(start, end, spec.day_count)
            for start, end in zip(swap_schedule[:-1], swap_schedule[1:])
        ]
        coupon_amounts = [
            spec.notional * spec.strike * accrual
            for accrual in accruals
        ]
        swap_end_step = min(
            int(round(year_fraction(market_state.settlement, spec.swap_end, spec.day_count) / dt)),
            n_steps,
        )
        steps_per_coupon = max(1, int(round((payment_times[0] - year_fraction(market_state.settlement, first_exercise, spec.day_count)) / dt))) if payment_times else max(1, int(round(0.5 / dt)))

        def _compute_swap_values_at_step(exercise_step: int) -> list[float]:
            """Rollback fixed-for-floating swap values from maturity to one exercise step."""
            coupon_steps = []
            step = exercise_step + steps_per_coupon
            while step <= swap_end_step:
                coupon_steps.append(step)
                step += steps_per_coupon

            n_terminal = lattice.n_nodes(swap_end_step)
            final_coupon = coupon_amounts[-1] if coupon_amounts and swap_end_step in coupon_steps else 0.0
            values = [spec.notional + final_coupon] * n_terminal

            coupon_by_step = {
                coupon_step: coupon_amounts[min(idx, len(coupon_amounts) - 1)]
                for idx, coupon_step in enumerate(coupon_steps)
            }

            for step in range(swap_end_step - 1, exercise_step - 1, -1):
                next_values = [0.0] * lattice.n_nodes(step)
                for node in range(lattice.n_nodes(step)):
                    df = lattice.get_discount(step, node)
                    probs = lattice.get_probabilities(step, node)
                    children = lattice.child_indices(step, node)
                    cont = df * sum(prob * values[child] for prob, child in zip(probs, children))
                    if step in coupon_by_step and step > exercise_step:
                        cont += coupon_by_step[step]
                    next_values[node] = cont
                values = next_values

            return [spec.notional - value for value in values]

        swap_values_by_step = {
            ex_step: _compute_swap_values_at_step(ex_step)
            for ex_step in exercise_steps
            if ex_step < swap_end_step
        }
        valid_exercise_steps = sorted(swap_values_by_step)
        if not valid_exercise_steps:
            return 0.0

        def terminal_payoff(step, node, tree):
            """Return zero continuation value beyond the last exercise/maturity node."""
            return 0.0

        def exercise_value(step, node, tree):
            """Return payer or receiver intrinsic swaption value at an exercise node."""
            swap_values = swap_values_by_step.get(step)
            if swap_values is None:
                return 0.0
            swap_value = swap_values[node]
            return max(swap_value, 0.0) if spec.is_payer else max(-swap_value, 0.0)

        return lattice_backward_induction(
            lattice,
            terminal_payoff,
            exercise_value=exercise_value,
            exercise_type="bermudan",
            exercise_steps=valid_exercise_steps,
            exercise_fn=max,
        )
