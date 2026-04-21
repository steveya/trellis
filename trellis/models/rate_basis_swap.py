"""Checked pricing helper for bounded floating-vs-floating rate basis swaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.models.contingent_cashflows import CouponAccrual, coupon_cashflow_pv


@dataclass(frozen=True)
class BasisSwapFloatingLegPeriod:
    accrual_start: date
    accrual_end: date
    payment_date: date
    fixing_date: date | None = None


@dataclass(frozen=True)
class BasisSwapFloatingLegSpec:
    notional: float
    periods: tuple[BasisSwapFloatingLegPeriod, ...]
    day_count: DayCountConvention
    rate_index: str | None = None
    spread: float = 0.0


@dataclass(frozen=True)
class RateBasisSwapSpec:
    pay_leg: BasisSwapFloatingLegSpec
    receive_leg: BasisSwapFloatingLegSpec


def _discount_factor(market_state: MarketState, payment_date: date) -> float:
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("Basis swap pricing requires market_state.discount")
    discount_date = getattr(discount_curve, "discount_date", None)
    if callable(discount_date):
        return float(discount_date(payment_date))
    payment_years = max(
        year_fraction(
            market_state.settlement,
            payment_date,
            DayCountConvention.ACT_365,
        ),
        0.0,
    )
    return float(discount_curve.discount(payment_years))


def _forward_rate(
    market_state: MarketState,
    leg: BasisSwapFloatingLegSpec,
    period: BasisSwapFloatingLegPeriod,
) -> float:
    fixing_history = None
    if period.fixing_date is not None and period.fixing_date <= market_state.settlement:
        try:
            fixing_history = market_state.fixing_history(leg.rate_index)
        except Exception:
            fixing_history = None
    if fixing_history is not None and period.fixing_date in fixing_history:
        return float(fixing_history[period.fixing_date])

    forward_curve = market_state.forecast_forward_curve(leg.rate_index)
    forward_rate_dates = getattr(forward_curve, "forward_rate_dates", None)
    if callable(forward_rate_dates):
        try:
            return float(
                forward_rate_dates(
                    period.accrual_start,
                    period.accrual_end,
                    day_count=leg.day_count,
                )
            )
        except AttributeError:
            pass

    start_years = max(
        year_fraction(
            market_state.settlement,
            period.accrual_start,
            DayCountConvention.ACT_365,
        ),
        0.0,
    )
    end_years = max(
        year_fraction(
            market_state.settlement,
            period.accrual_end,
            DayCountConvention.ACT_365,
        ),
        start_years + 1e-6,
    )
    return float(forward_curve.forward_rate(start_years, end_years))


def _leg_pv(
    market_state: MarketState,
    leg: BasisSwapFloatingLegSpec,
    *,
    sign: float,
) -> float:
    pv = 0.0
    for period in leg.periods:
        if period.payment_date <= market_state.settlement:
            continue
        accrual = year_fraction(
            period.accrual_start,
            period.accrual_end,
            leg.day_count,
        )
        rate = _forward_rate(market_state, leg, period) + float(leg.spread)
        discount_factor = _discount_factor(market_state, period.payment_date)
        pv += coupon_cashflow_pv(
            CouponAccrual(
                notional=float(leg.notional),
                rate=rate,
                accrual=accrual,
                discount_factor=discount_factor,
                sign=sign,
            )
        )
    return float(pv)


def price_rate_basis_swap(
    market_state: MarketState,
    spec: RateBasisSwapSpec,
) -> float:
    """Return receive-leg PV minus pay-leg PV for one bounded basis swap."""

    return float(
        _leg_pv(market_state, spec.receive_leg, sign=1.0)
        + _leg_pv(market_state, spec.pay_leg, sign=-1.0)
    )


__all__ = [
    "BasisSwapFloatingLegPeriod",
    "BasisSwapFloatingLegSpec",
    "RateBasisSwapSpec",
    "price_rate_basis_swap",
]
