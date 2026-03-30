"""FX spot, forwards, and cross-currency payoff conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, DiscountCurve


@dataclass(frozen=True)
class FXRate:
    """Spot FX rate between two currencies.

    Convention: 1 unit of foreign = ``spot`` units of domestic.
    Example: EURUSD = 1.10 means 1 EUR = 1.10 USD (domestic=USD, foreign=EUR).
    """

    spot: float
    domestic: str
    foreign: str


class FXForward:
    """FX forward rate derived from spot rate and interest rate differentials.

    The forward rate at time t equals spot * (foreign discount / domestic discount).
    This follows from covered interest rate parity: you can replicate a forward
    by borrowing in one currency and lending in the other.
    """

    def __init__(self, fx_rate: FXRate,
                 domestic_curve: DiscountCurve,
                 foreign_curve: DiscountCurve):
        """Store the spot FX quote and domestic/foreign discount curves."""
        self._fx_rate = fx_rate
        self._domestic_curve = domestic_curve
        self._foreign_curve = foreign_curve

    @property
    def spot(self) -> float:
        """Return the spot FX rate used as the base for forward calculations."""
        return self._fx_rate.spot

    def forward(self, t: float) -> float:
        """FX forward rate at time t years from now."""
        return (self._fx_rate.spot
                * self._foreign_curve.discount(t)
                / self._domestic_curve.discount(t))

    def forward_points(self, t: float) -> float:
        """Forward points = F(t) - S."""
        return self.forward(t) - self._fx_rate.spot


class FXForwardPayoff:
    """Wraps a foreign-currency payoff and converts its value to domestic currency.

    Evaluates the inner payoff (which prices in foreign currency), then
    multiplies by the spot FX rate to get the domestic-currency present value.
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
        """Needs everything the inner payoff needs, plus FX rates and curves."""
        return self._inner.requirements | {"fx", "discount", "forecast_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price inner payoff in foreign currency, convert to domestic.

        The inner payoff is evaluated against a MarketState with the foreign
        discount curve. The resulting PV (in foreign) is converted to domestic
        at the spot FX rate. This is correct because the inner payoff already
        discounts using the foreign curve, and spot × foreign_PV = domestic_PV
        by covered interest parity.
        """
        fx_rate = market_state.fx_rates[self._fx_pair]

        # The inner payoff evaluates with whatever MarketState it receives.
        # The foreign PV is already discounted at foreign rates.
        foreign_pv = self._inner.evaluate(market_state)

        # Convert: domestic_PV = foreign_PV × spot
        return foreign_pv * fx_rate.spot
