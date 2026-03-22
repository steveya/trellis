"""Forward rate extraction from a discount curve."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.core.types import DiscountCurve

np = get_numpy()


class ForwardCurve:
    """Extracts forward rates from a DiscountCurve.

    All methods are autograd-compatible.
    """

    def __init__(self, discount_curve: DiscountCurve):
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
