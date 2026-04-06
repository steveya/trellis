"""FX spot, forwards, and cross-currency payoff conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
        return self._inner.requirements | {"fx_rates", "discount_curve", "forward_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price inner payoff in foreign currency, convert to domestic.

        The inner payoff is evaluated against a MarketState with the foreign
        discount curve. The resulting PV (in foreign) is converted to domestic
        at the spot FX rate. This is correct because the inner payoff already
        discounts using the foreign curve, and spot × foreign_PV = domestic_PV
        by covered interest parity.
        """
        if market_state.fx_rates is None or self._fx_pair not in market_state.fx_rates:
            raise ValueError(
                f"market_state.fx_rates must contain fx pair {self._fx_pair!r}"
            )
        if (
            market_state.forecast_curves is None
            or self._foreign_discount_key not in market_state.forecast_curves
        ):
            raise ValueError(
                "market_state.forecast_curves must contain foreign discount key "
                f"{self._foreign_discount_key!r}"
            )

        from trellis.core.runtime_contract import wrap_market_state_with_contract

        fx_rate = market_state.fx_rates[self._fx_pair]
        foreign_curve = market_state.forecast_curves[self._foreign_discount_key]
        selected_curve_names = dict(market_state.selected_curve_names or {})
        selected_curve_names["discount_curve"] = self._foreign_discount_key
        base_market_state = getattr(market_state, "raw_market_state", market_state)
        foreign_market_state = replace(
            base_market_state,
            discount=foreign_curve,
            forward_curve=None,
            selected_curve_names=selected_curve_names or None,
        )
        foreign_market_state = wrap_market_state_with_contract(
            foreign_market_state,
            requirements=self._inner.requirements,
            context=type(self._inner).__name__,
        )

        # Reprice the inner payoff against the foreign discount curve so the
        # resulting PV is expressed in foreign currency before FX conversion.
        foreign_pv = self._inner.evaluate(foreign_market_state)

        # Convert: domestic_PV = foreign_PV × spot
        return foreign_pv * fx_rate.spot
