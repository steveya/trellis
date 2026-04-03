"""Interest rate caps and floors.

A cap protects against rising interest rates: it pays the holder whenever
the floating rate exceeds a strike rate. A floor protects against falling
rates. Each is priced as a series of individual period options (caplets or
floorlets) using the Black-76 formula for interest rate options.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import build_period_schedule
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class CapFloorSpec:
    """Contract terms for an interest rate cap or floor."""

    notional: float
    strike: float
    start_date: date
    end_date: date
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None


def _capfloor_pv(
    spec: CapFloorSpec,
    market_state: MarketState,
    pricing_fn,
) -> float:
    """Return the sum of discounted Black-76 caplets or floorlets.

    For each future accrual period ``[T_i, T_{i+1}]`` this computes

    .. math::

       PV_i = N \tau_i D(0, T_{i+1}) \operatorname{Black}(F_i, K, \sigma_i, T_i)

    and sums the surviving periods after settlement.
    """
    schedule = build_period_schedule(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=market_state.settlement,
    )

    pv = 0.0
    for period in schedule:
        if period.payment_date <= market_state.settlement:
            continue

        tau = float(period.accrual_fraction or 0.0)
        t_fix = float(period.t_start or 0.0)
        t_pay = float(period.t_payment or 0.0)

        if t_fix <= 0:
            continue

        fwd = market_state.forecast_forward_curve(spec.rate_index)
        F = fwd.forward_rate(t_fix, t_pay)
        sigma = market_state.vol_surface.black_vol(t_fix, spec.strike)

        undiscounted = spec.notional * tau * pricing_fn(F, spec.strike, sigma, t_fix)
        df = market_state.discount.discount(t_pay)
        pv += undiscounted * df

    return pv


class CapPayoff:
    """Interest rate cap: pays out when the floating rate exceeds the strike.

    Priced as a sum of per-period caplets, each valued with the Black-76
    option pricing formula.
    """

    def __init__(self, spec: CapFloorSpec):
        """Store the cap contract specification used for all future valuations."""
        self._spec = spec

    @property
    def spec(self) -> CapFloorSpec:
        """Return the immutable contract specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Cap pricing needs discount, forward rate, and volatility curves."""
        return {"discount_curve", "forward_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        """Sum the present values of all caplets to get the cap price."""
        return _capfloor_pv(self._spec, market_state, black76_call)


class FloorPayoff:
    """Interest rate floor: pays out when the floating rate falls below the strike.

    Priced as a sum of per-period floorlets, each valued with the Black-76
    option pricing formula.
    """

    def __init__(self, spec: CapFloorSpec):
        """Store the floor contract specification used for all future valuations."""
        self._spec = spec

    @property
    def spec(self) -> CapFloorSpec:
        """Return the immutable contract specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Floor pricing needs discount, forward rate, and volatility curves."""
        return {"discount_curve", "forward_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        """Sum the present values of all floorlets to get the floor price."""
        return _capfloor_pv(self._spec, market_state, black76_put)
