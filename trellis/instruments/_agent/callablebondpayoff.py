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
    call_schedule: str
    coupon_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360


class CallableBondPayoff:
    """Callable bond with a call schedule (Bermudan callable)."""

    def __init__(self, spec: CallableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> list[tuple[date, float]]:
        spec = self._spec
        # Generate the regular coupon payment schedule.
        payment_dates = generate_schedule(spec.start_date, spec.end_date, spec.coupon_frequency)
        # Build base cashflows for a non-callable bond.
        base_cashflows = []
        period_starts = [spec.start_date] + payment_dates[:-1]
        for p_start, p_end in zip(period_starts, payment_dates):
            if p_end <= market_state.settlement:
                continue
            tau = year_fraction(p_start, p_end, spec.day_count)
            amt = spec.notional * spec.coupon * tau
            # At maturity, add principal repayment.
            if p_end == payment_dates[-1]:
                amt += spec.notional
            base_cashflows.append((p_end, amt))

        # Parse the call schedule.
        # Expected format: "YYYY-MM-DD:call_price,YYYY-MM-DD:call_price,..." 
        call_list = []
        if spec.call_schedule.strip():
            for token in spec.call_schedule.split(","):
                token = token.strip()
                if token:
                    try:
                        call_date_str, price_str = token.split(":")
                        cdate = date.fromisoformat(call_date_str.strip())
                        cprice = float(price_str.strip())
                        call_list.append((cdate, cprice))
                    except Exception:
                        raise ValueError("Invalid call_schedule format. Expected 'YYYY-MM-DD:price,...'")
            # Sort by call date ascending.
            call_list.sort(key=lambda x: x[0])

        # Determine if and when the issuer will exercise the call option.
        # The decision is made by comparing the forward (uncalled) bond value at the call date
        # versus the call price. The issuer will call at the first available call date for which
        # the forward value exceeds the call price.
        call_decision = None  # Tuple of (call_date, call_price)
        # We use the market discount curve to compute forward prices.
        for call_date, call_price in call_list:
            if call_date <= market_state.settlement:
                continue
            # Compute the forward price of the non-callable bond at the call date.
            forward_value = 0.0
            t_call = year_fraction(market_state.settlement, call_date, spec.day_count)
            # Use remaining cashflows with payment date >= call_date.
            for cf_date, amt in base_cashflows:
                if cf_date >= call_date:
                    t_cf = year_fraction(market_state.settlement, cf_date, spec.day_count)
                    # Remove discounting to settlement by ratio: DF(cf)/DF(call)
                    df_call = market_state.discount.discount(t_call)
                    df_cf = market_state.discount.discount(t_cf)
                    # Avoid division by zero.
                    if df_call == 0:
                        ratio = 0.0
                    else:
                        ratio = df_cf / df_call
                    forward_value += amt * ratio
            # If the forward (uncallable) value is greater than the call price,
            # the issuer will optimally call at this date.
            if forward_value > call_price:
                call_decision = (call_date, call_price)
                break

        # If a call decision is made, adjust the cashflow schedule.
        if call_decision is not None:
            call_date, call_price = call_decision
            adjusted_cashflows = []
            for cf_date, amt in base_cashflows:
                if cf_date < call_date:
                    adjusted_cashflows.append((cf_date, amt))
                elif cf_date == call_date:
                    # At the call date, the investor receives the call price (instead of the scheduled coupon+principal)
                    adjusted_cashflows.append((cf_date, call_price))
                    # Once called, no further cashflows are paid.
                    break
                else:
                    break
            return adjusted_cashflows
        else:
            # No call is optimal; return the full non-callable cashflow schedule.
            return base_cashflows