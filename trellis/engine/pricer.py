"""High-level pricing: given a bond and a yield curve, produce a full PricingResult.

Combines the price calculation with risk sensitivities (Greeks) computed
via automatic differentiation through the pricing function.
"""

from __future__ import annotations

from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.core.types import GreeksSpec, PricingResult
from trellis.core.date_utils import year_fraction
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.analytics import compute_greeks
from trellis.instruments.bond import Bond

np = get_numpy()


def price_instrument(
    instrument: Bond,
    curve: YieldCurve,
    settlement: date | None = None,
    *,
    greeks: GreeksSpec = "all",
) -> PricingResult:
    """Price an instrument off a curve and compute analytics.

    Parameters
    ----------
    instrument : Bond
        The bond to price.
    curve : YieldCurve
        Discount curve.
    settlement : date or None
        Settlement date (defaults to today).

    Returns
    -------
    PricingResult
    """
    settlement = settlement or date.today()

    # Price off the curve
    dirty_price = instrument.price(curve, settlement)

    # Accrued interest (simple approximation for coupon bonds)
    accrued = _accrued_interest(instrument, settlement)
    clean_price = dirty_price - accrued

    # Build a pricing function over curve rates for autodiff Greeks
    schedule = instrument.cashflows(settlement)
    times = [year_fraction(settlement, d, instrument.day_count) for d in schedule.dates]
    cf_amounts = schedule.amounts

    # Determine which Greeks to compute
    if greeks is None:
        computed_greeks: dict = {}
    else:
        def _price_from_rates(rates_vec):
            """Price as a function of the curve's rate vector (enables automatic differentiation)."""
            from trellis.curves.interpolation import linear_interp
            pv = np.array(0.0)
            for t, amt in zip(times, cf_amounts):
                r = linear_interp(t, curve.tenors, rates_vec)
                pv = pv + amt * np.exp(-r * t)
            return pv

        measures = None if greeks == "all" else greeks
        computed_greeks = compute_greeks(
            _price_from_rates, curve.rates, tenors=curve.tenors, measures=measures,
        )

    return PricingResult(
        clean_price=clean_price,
        dirty_price=dirty_price,
        accrued_interest=accrued,
        ytm=None,  # TODO: solve for YTM
        greeks=computed_greeks,
        curve_sensitivities=computed_greeks.get("key_rate_durations", {}) if computed_greeks else {},
    )


def _accrued_interest(bond: Bond, settlement: date) -> float:
    """Compute interest earned since the last coupon payment up to the settlement date."""
    if bond.coupon_rate == 0.0:
        return 0.0
    if bond.maturity_date is None:
        return 0.0

    from trellis.core.date_utils import add_months
    issue = bond.issue_date or add_months(bond.maturity_date, -12 * (bond.maturity or 10))
    months_per_period = 12 // bond.frequency

    last_coupon = issue
    i = 1
    while True:
        next_d = add_months(issue, months_per_period * i)
        if next_d > settlement:
            break
        last_coupon = next_d
        i += 1

    frac = year_fraction(last_coupon, settlement, bond.day_count)
    period_years = months_per_period / 12.0
    accrued = bond.notional * bond.coupon_rate * (frac / period_years) * period_years
    return accrued
