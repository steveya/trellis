"""Forward rate extraction from a discount curve."""

from __future__ import annotations

from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention, DiscountCurve

np = get_numpy()


class ForwardCurve:
    """Extracts forward rates from a DiscountCurve.

    All methods are autograd-compatible.
    """

    def __init__(self, discount_curve: DiscountCurve):
        """Store the underlying discount curve used to extract forwards."""
        self._curve = discount_curve

    def forward_rate(
        self,
        t1: float,
        t2: float,
        compounding: str = "simple",
    ) -> float:
        """Simple or continuous forward rate F(t1, t2).

        Parameters
        ----------
        t1 : float
            Start time in years.
        t2 : float
            End time in years (must be > t1).
        compounding : str
            ``"simple"`` or ``"continuous"``.
        """
        if t2 <= t1:
            raise ValueError(f"t2 must be > t1 (got t1={t1}, t2={t2})")

        df1 = self._curve.discount(t1)
        df2 = self._curve.discount(t2)
        tau = t2 - t1

        if compounding == "simple":
            return (df1 / df2 - 1.0) / tau
        elif compounding == "continuous":
            return np.log(df1 / df2) / tau
        else:
            raise ValueError(f"Unknown compounding: {compounding!r}")

    def discount_date(self, target_date: date) -> float:
        """Date-aware discount factor when the underlying curve supports it."""
        discount_date = getattr(self._curve, "discount_date", None)
        if discount_date is None:
            raise AttributeError("Underlying discount curve has no date-aware discount method")
        return float(discount_date(target_date))

    def forward_rate_dates(
        self,
        start_date: date,
        end_date: date,
        *,
        day_count: DayCountConvention,
        compounding: str = "simple",
    ) -> float:
        """Date-aware forward rate when the underlying curve supports it."""
        curve_method = getattr(self._curve, "forward_rate_dates", None)
        if curve_method is not None:
            return float(
                curve_method(
                    start_date,
                    end_date,
                    day_count=day_count,
                    compounding=compounding,
                )
            )
        discount_date = getattr(self._curve, "discount_date", None)
        if discount_date is None:
            raise AttributeError("Underlying discount curve has no date-aware forward helper")
        alpha = year_fraction(start_date, end_date, day_count)
        if alpha <= 0.0:
            raise ValueError("Forward accrual year fraction must be positive")
        df1 = float(discount_date(start_date))
        df2 = float(discount_date(end_date))
        if compounding == "simple":
            return float((df1 / df2 - 1.0) / alpha)
        if compounding == "continuous":
            return float(np.log(df1 / df2) / alpha)
        raise ValueError(f"Unknown compounding: {compounding!r}")
