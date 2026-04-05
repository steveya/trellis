"""High-level pricing: given a bond and a yield curve, produce a full PricingResult.

Combines the price calculation with risk sensitivities (Greeks) computed
via automatic differentiation through the pricing function.
"""

from __future__ import annotations

from datetime import date

from trellis.analytics.measures import KeyRateDurations
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.core.types import Frequency, GreeksSpec, PricingResult
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
        if _needs_key_rate_durations(greeks):
            computed_greeks["key_rate_durations"] = _canonical_key_rate_durations(
                instrument,
                curve,
                settlement,
                dirty_price,
            )

    return PricingResult(
        clean_price=clean_price,
        dirty_price=dirty_price,
        accrued_interest=accrued,
        ytm=_solve_ytm(instrument, dirty_price, settlement),
        greeks=computed_greeks,
        curve_sensitivities=computed_greeks.get("key_rate_durations", {}) if computed_greeks else {},
    )


def _needs_key_rate_durations(greeks: GreeksSpec) -> bool:
    """Return whether the caller requested KRD output."""
    if greeks is None:
        return False
    if greeks == "all":
        return True
    if isinstance(greeks, str):
        return greeks == "key_rate_durations"
    return "key_rate_durations" in greeks


def _canonical_key_rate_durations(
    instrument: Bond,
    curve: YieldCurve,
    settlement: date,
    dirty_price: float,
) -> dict[float, float]:
    """Compute KRDs through the shared interpolation-aware measure path."""
    measure = KeyRateDurations(
        tenors=tuple(float(tenor) for tenor in np.asarray(curve.tenors, dtype=float)),
    )
    payoff = DeterministicCashflowPayoff(instrument, day_count=instrument.day_count)
    market_state = MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=curve,
    )
    context = {"_cache": {"base_price": float(dirty_price)}}
    return measure.compute(payoff, market_state, **context)


def _accrued_interest(bond: Bond, settlement: date) -> float:
    """Compute interest earned since the last coupon payment up to the settlement date."""
    if bond.coupon_rate == 0.0:
        return 0.0
    if bond.maturity_date is None:
        return 0.0
    frequency = _bond_frequency(bond)
    issue = bond.issue_date or _infer_issue_date(bond)
    if settlement <= issue or settlement >= bond.maturity_date:
        return 0.0

    from trellis.conventions.schedule import build_period_schedule

    schedule = build_period_schedule(
        issue,
        bond.maturity_date,
        frequency,
        day_count=bond.day_count,
    )
    coupon_payment = bond.notional * bond.coupon_rate / bond.frequency

    for period in schedule.periods:
        if not (period.start_date < settlement < period.end_date):
            continue
        period_fraction = float(period.accrual_fraction or 0.0)
        if period_fraction <= 0.0:
            return 0.0
        accrued_fraction = year_fraction(
            period.start_date,
            settlement,
            bond.day_count,
            ref_start=period.start_date,
            ref_end=period.end_date,
            frequency=frequency,
        )
        return coupon_payment * accrued_fraction / period_fraction
    return 0.0


def _solve_ytm(bond: Bond, dirty_price: float, settlement: date) -> float | None:
    """Solve the nominal annualized yield that reproduces ``dirty_price``."""
    if dirty_price <= 0.0:
        return None
    if bond.maturity_date is None or settlement >= bond.maturity_date:
        return None

    schedule = bond.cashflows(settlement)
    if not schedule.dates:
        return None

    frequency = float(bond.frequency)
    times = tuple(year_fraction(settlement, cashflow_date, bond.day_count) for cashflow_date in schedule.dates)
    amounts = tuple(float(amount) for amount in schedule.amounts)

    def objective(yield_rate: float) -> float:
        base = 1.0 + yield_rate / frequency
        if base <= 0.0:
            return float("inf")
        present_value = 0.0
        for time_to_cashflow, amount in zip(times, amounts):
            present_value += amount / (base ** (frequency * time_to_cashflow))
        return present_value - dirty_price

    from scipy.optimize import brentq

    lower = -0.99 * frequency
    upper = 0.25
    lower_value = objective(lower)
    upper_value = objective(upper)

    while lower_value * upper_value > 0.0 and upper < 64.0:
        upper *= 2.0
        upper_value = objective(upper)

    if lower_value * upper_value > 0.0:
        return None
    return float(brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12))


def _bond_frequency(bond: Bond) -> Frequency:
    """Return the coupon frequency enum for a bond."""
    try:
        return Frequency(bond.frequency)
    except ValueError as exc:
        raise ValueError(f"Unsupported bond frequency for convention-aware pricing: {bond.frequency!r}") from exc


def _infer_issue_date(bond: Bond) -> date:
    """Infer the issue date using the bond's legacy maturity fallback."""
    from trellis.core.date_utils import add_months

    return add_months(bond.maturity_date, -12 * (bond.maturity or 10))
