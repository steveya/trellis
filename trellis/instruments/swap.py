"""Interest rate swap: exchange fixed-rate payments for floating-rate payments.

A swap lets two parties exchange interest payments. One pays a fixed rate,
the other pays a rate that resets periodically to the market rate. The
present value is the difference between the two payment streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency


@dataclass(frozen=True)
class SwapSpec:
    """Contract terms for an interest rate swap.

    A payer swap (is_payer=True) means you pay the fixed rate and receive
    the floating market rate. A receiver swap is the opposite.
    """

    notional: float
    fixed_rate: float
    start_date: date
    end_date: date
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL
    float_frequency: Frequency = Frequency.QUARTERLY
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360
    float_day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True  # True = pay fixed, receive floating


class SwapPayoff:
    """Interest rate swap as a Payoff.

    Payer swap (is_payer=True): receive floating, pay fixed.
    Receiver swap (is_payer=False): pay floating, receive fixed.
    """

    def __init__(self, spec: SwapSpec):
        """Store the fixed/floating swap contract specification."""
        self._spec = spec

    @property
    def spec(self) -> SwapSpec:
        """Return the immutable swap specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Swap pricing needs a discount curve and a forward rate curve."""
        return {"discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        """Compute the net present value: floating payments minus fixed payments (for a payer)."""
        spec = self._spec
        sign = 1.0 if spec.is_payer else -1.0
        pv = 0.0

        # Fixed leg
        fixed_sched = generate_schedule(spec.start_date, spec.end_date, spec.fixed_frequency)
        fixed_starts = [spec.start_date] + fixed_sched[:-1]
        for p_start, p_end in zip(fixed_starts, fixed_sched):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.fixed_day_count)
            t_pay = year_fraction(market_state.settlement, p_end, spec.fixed_day_count)
            df = float(market_state.discount.discount(t_pay))
            pv -= sign * spec.notional * spec.fixed_rate * tau * df

        # Floating leg
        float_sched = generate_schedule(spec.start_date, spec.end_date, spec.float_frequency)
        float_starts = [spec.start_date] + float_sched[:-1]
        fwd_curve = market_state.forecast_forward_curve(spec.rate_index)

        for p_start, p_end in zip(float_starts, float_sched):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.float_day_count)
            t_start = year_fraction(market_state.settlement, p_start, spec.float_day_count)
            t_end = year_fraction(market_state.settlement, p_end, spec.float_day_count)
            t_start = max(t_start, 1e-6)
            F = fwd_curve.forward_rate(t_start, t_end)
            df = float(market_state.discount.discount(t_end))
            pv += sign * spec.notional * float(F) * tau * df

        return pv


def par_swap_rate(spec: SwapSpec, market_state: MarketState) -> float:
    """Find the fixed rate that makes the swap worth zero today.

    This is the market-implied fair rate: the fixed rate at which the present
    value of fixed payments exactly equals the present value of floating payments.
    """
    fwd_curve = market_state.forecast_forward_curve(spec.rate_index)

    # Floating leg PV (without notional, as fraction)
    float_sched = generate_schedule(spec.start_date, spec.end_date, spec.float_frequency)
    float_starts = [spec.start_date] + float_sched[:-1]
    float_pv = 0.0
    for p_start, p_end in zip(float_starts, float_sched):
        if p_end <= market_state.settlement:
            continue
        tau = year_fraction(p_start, p_end, spec.float_day_count)
        t_start = year_fraction(market_state.settlement, p_start, spec.float_day_count)
        t_end = year_fraction(market_state.settlement, p_end, spec.float_day_count)
        t_start = max(t_start, 1e-6)
        F = fwd_curve.forward_rate(t_start, t_end)
        df = market_state.discount.discount(t_end)
        float_pv += float(F) * tau * float(df)

    # Fixed leg annuity
    fixed_sched = generate_schedule(spec.start_date, spec.end_date, spec.fixed_frequency)
    fixed_starts = [spec.start_date] + fixed_sched[:-1]
    annuity = 0.0
    for p_start, p_end in zip(fixed_starts, fixed_sched):
        if p_end <= market_state.settlement:
            continue
        tau = year_fraction(p_start, p_end, spec.fixed_day_count)
        t_end = year_fraction(market_state.settlement, p_end, spec.fixed_day_count)
        df = market_state.discount.discount(t_end)
        annuity += tau * float(df)

    if annuity == 0.0:
        raise ValueError("Fixed leg annuity is zero")

    return float_pv / annuity
