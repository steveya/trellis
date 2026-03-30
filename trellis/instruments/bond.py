"""Bond instrument — autograd-compatible, curve-based pricing."""

from __future__ import annotations

from datetime import date
from typing import Optional

from trellis.core.differentiable import get_numpy
from trellis.core.types import (
    CashflowSchedule,
    DayCountConvention,
    DiscountCurve,
    Frequency,
    PricingResult,
)
from trellis.core.date_utils import add_months, year_fraction

np = get_numpy()


class Bond:
    """Fixed-rate bond that pays periodic coupons and returns principal at maturity.

    Has two construction modes:
    - Period-based (older): pass maturity as integer years, frequency as int.
      Used by get_cashflows(), get_price(), get_duration(), get_convexity().
    - Date-based (preferred): pass maturity_date, issue_date. Used by
      cashflows() and price(curve).
    """

    def __init__(
        self,
        notional: float = 100,
        coupon: float = 0.0,
        maturity: int | None = None,
        frequency: int = 2,
        *,
        face: float | None = None,
        maturity_date: date | None = None,
        issue_date: date | None = None,
        day_count: DayCountConvention = DayCountConvention.ACT_ACT,
    ):
        """Initialize bond terms. See class docstring for the two construction modes."""
        self.notional = face if face is not None else notional
        self.coupon_rate = coupon
        self.maturity = maturity
        self.frequency = frequency
        self.maturity_date = maturity_date
        self.issue_date = issue_date
        self.day_count = day_count

    # ------------------------------------------------------------------
    # Legacy interface (period-based, flat-rate discounting)
    # ------------------------------------------------------------------

    def get_cashflows(self):
        """Return array of all cashflow amounts (coupons + final principal). Period-based mode."""
        n_periods = self.maturity * self.frequency
        coupon_payment = self.notional * self.coupon_rate / self.frequency
        # Build without in-place mutation for autograd compatibility
        coupons = np.ones(n_periods) * coupon_payment
        # Add notional to last period
        principal = np.zeros(n_periods)
        principal = np.concatenate([np.zeros(n_periods - 1), np.array([self.notional])])
        return coupons + principal

    def get_price(self, rates):
        """Discount cashflows using a vector of per-period rates. Period-based mode."""
        cashflows = self.get_cashflows()
        n = self.maturity * self.frequency
        discount_factors = np.exp(-rates[:n] / self.frequency)
        return np.dot(cashflows, discount_factors)

    def get_duration(self, rates):
        """Macaulay duration in periods (weighted-average time to cashflows). Period-based mode."""
        cashflows = self.get_cashflows()
        n = self.maturity * self.frequency
        periods = np.arange(1, n + 1, dtype=float)
        discount_factors = np.exp(-rates[:n] / self.frequency)
        return np.dot(periods, cashflows * discount_factors) / self.get_price(rates)

    def get_convexity(self, rates):
        """Convexity in periods (second-order price sensitivity to rates). Period-based mode."""
        cashflows = self.get_cashflows()
        n = self.maturity * self.frequency
        periods = np.arange(1, n + 1, dtype=float)
        discount_factors = np.exp(-rates[:n] / self.frequency)
        return np.dot(periods ** 2, cashflows * discount_factors) / self.get_price(rates)

    # ------------------------------------------------------------------
    # New curve-based interface
    # ------------------------------------------------------------------

    def cashflows(self, settlement: date | None = None) -> CashflowSchedule:
        """Return the list of (date, amount) pairs for all future coupons and principal."""
        if self.maturity_date is None:
            raise ValueError("maturity_date required for dated cashflow schedule")
        issue = self.issue_date or _infer_issue(self.maturity_date, self.maturity, self.frequency)
        months_per_period = 12 // self.frequency
        dates: list[date] = []
        amounts: list[float] = []
        coupon_payment = self.notional * self.coupon_rate / self.frequency
        i = 1
        while True:
            d = add_months(issue, months_per_period * i)
            if settlement and d <= settlement:
                i += 1
                continue
            if d > self.maturity_date:
                break
            dates.append(d)
            amounts.append(coupon_payment)
            i += 1
        # Add maturity cashflow
        if not dates or dates[-1] != self.maturity_date:
            dates.append(self.maturity_date)
            amounts.append(coupon_payment + self.notional)
        else:
            amounts[-1] += self.notional
        return CashflowSchedule(dates=dates, amounts=amounts)

    def price(self, curve: DiscountCurve, settlement: date | None = None) -> float:
        """Compute the present value by discounting each future cashflow using the curve."""
        settlement = settlement or date.today()
        schedule = self.cashflows(settlement)
        pv = 0.0
        for d, amt in zip(schedule.dates, schedule.amounts):
            t = year_fraction(settlement, d, self.day_count)
            pv += amt * curve.discount(t)
        return pv


class ParBond(Bond):
    """A bond initialised at par (convenience wrapper)."""

    def __init__(self, notional: float, maturity: int, frequency: int, coupon_rate: float):
        """Create a bond priced at par (face value) with the given coupon rate."""
        super().__init__(notional, coupon_rate, maturity, frequency)


def _infer_issue(maturity_date: date, maturity_years: int | None, frequency: int) -> date:
    """Best-effort issue date when not supplied."""
    if maturity_years is not None:
        return add_months(maturity_date, -12 * maturity_years)
    # Default: assume 10-year bond
    return add_months(maturity_date, -120)
