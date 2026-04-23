"""Date-aware flat discount curve for benchmark and convention-sensitive pricing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy

np = get_numpy()


@dataclass(frozen=True)
class DateAwareFlatYieldCurve:
    """Flat continuously compounded curve with optional date-aware helpers.

    The standard DiscountCurve protocol in Trellis is time-based. FinancePy
    benchmark tasks, especially rates products, rely on date-based forward
    extraction using different day-count conventions for discounting and
    accrual. This curve keeps the time-based protocol intact while exposing
    date-aware helpers for families that need exact convention alignment.
    """

    value_date: date
    flat_rate: float
    curve_day_count: DayCountConvention = DayCountConvention.ACT_ACT_ISDA
    max_tenor: float = 31.0

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate at time *t*."""
        del t
        return self.flat_rate

    def discount(self, t: float) -> float:
        """Discount factor at time *t* using the supplied time directly."""
        return np.exp(-self.flat_rate * t)

    def discount_date(self, target_date: date) -> float:
        """Discount factor to one concrete date using the curve day count."""
        t = year_fraction(self.value_date, target_date, self.curve_day_count)
        return np.exp(-self.flat_rate * t)

    def forward_rate_dates(
        self,
        start_date: date,
        end_date: date,
        *,
        day_count: DayCountConvention,
        compounding: str = "simple",
    ) -> float:
        """Forward rate between two dates under one accrual day count."""
        alpha = year_fraction(start_date, end_date, day_count)
        if alpha <= 0.0:
            raise ValueError("Forward accrual year fraction must be positive")
        df1 = self.discount_date(start_date)
        df2 = self.discount_date(end_date)
        if compounding == "simple":
            return (df1 / df2 - 1.0) / alpha
        if compounding == "continuous":
            return np.log(df1 / df2) / alpha
        raise ValueError(f"Unknown compounding: {compounding!r}")
