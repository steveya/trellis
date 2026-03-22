"""FX spot, forwards, and cross-currency payoff conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, DiscountCurve


@dataclass(frozen=True)
class FXRate:
    """Spot FX rate.

    Convention: 1 unit of foreign buys ``spot`` units of domestic.
    E.g., EURUSD = 1.10 means 1 EUR = 1.10 USD.
    """

    spot: float
    domestic: str
    foreign: str


class FXForward:
    """FX forward rate via covered interest rate parity.

    F(t) = S * df_foreign(t) / df_domestic(t)
    """

    def __init__(self, fx_rate: FXRate,
                 domestic_curve: DiscountCurve,
                 foreign_curve: DiscountCurve):
        self._fx_rate = fx_rate
        self._domestic_curve = domestic_curve
        self._foreign_curve = foreign_curve

    @property
    def spot(self) -> float:
        return self._fx_rate.spot

    def forward(self, t: float) -> float:
        """FX forward rate at time t years."""
        return (self._fx_rate.spot
                * self._foreign_curve.discount(t)
                / self._domestic_curve.discount(t))

    def forward_points(self, t: float) -> float:
        """Forward points = F(t) - S."""
        return self.forward(t) - self._fx_rate.spot


class FXForwardPayoff:
    """Wraps a foreign-currency payoff, converting to domestic via FX forward.

    Each cashflow ``(date, amount_foreign)`` becomes
    ``(date, amount_foreign * F(t))`` where F(t) is the CIP forward rate.
    """

    def __init__(self, inner, fx_pair: str, foreign_discount_key: str):
        """
        Parameters
        ----------
        inner : Payoff
            Foreign-currency payoff.
        fx_pair : str
            Key into ``MarketState.fx_rates``, e.g. ``"EURUSD"``.
        foreign_discount_key : str
            Key into ``MarketState.forecast_curves`` for the foreign
            discount curve.
        """
        self._inner = inner
        self._fx_pair = fx_pair
        self._foreign_discount_key = foreign_discount_key

    @property
    def requirements(self) -> set[str]:
        return self._inner.requirements | {"fx", "discount", "forecast_rate"}

    def evaluate(self, market_state: MarketState) -> list[tuple[date, float]]:
        fx_rate = market_state.fx_rates[self._fx_pair]
        foreign_curve = market_state.forecast_curves[self._foreign_discount_key]
        domestic_curve = market_state.discount

        fx_fwd = FXForward(fx_rate, domestic_curve, foreign_curve)

        result = self._inner.evaluate(market_state)

        # Handle both Cashflows and raw list returns
        from trellis.core.payoff import Cashflows, PresentValue
        if isinstance(result, Cashflows):
            flows = result.flows
        elif isinstance(result, PresentValue):
            # Can't convert FX on a PV — return the PV as-is
            return result
        else:
            flows = result

        domestic_cfs: list[tuple[date, float]] = []
        for cf_date, amount_foreign in flows:
            t = year_fraction(market_state.settlement, cf_date,
                              DayCountConvention.ACT_365)
            domestic_cfs.append((cf_date, amount_foreign * fx_fwd.forward(t)))

        return Cashflows(domestic_cfs)
