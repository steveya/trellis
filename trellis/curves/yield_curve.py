"""Yield curve implementing the DiscountCurve protocol."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.curves.interpolation import linear_interp

np = get_numpy()


class YieldCurve:
    """Continuously compounded zero-rate curve.

    Parameters
    ----------
    tenors : array-like
        Maturities in years (e.g. [0.25, 0.5, 1, 2, 5, 10, 30]).
    rates : array-like
        Continuously compounded zero rates for each tenor.
    """

    def __init__(self, tenors, rates):
        self.tenors = np.asarray(tenors, dtype=float)
        self.rates = np.asarray(rates, dtype=float)

    # ------------------------------------------------------------------
    # DiscountCurve protocol
    # ------------------------------------------------------------------

    def zero_rate(self, t: float) -> float:
        """Interpolated continuously compounded zero rate at time *t*."""
        return linear_interp(t, self.tenors, self.rates)

    def discount(self, t: float) -> float:
        """Discount factor at time *t*: exp(-r(t) * t)."""
        r = self.zero_rate(t)
        return np.exp(-r * t)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_treasury_yields(cls, data: dict[float, float]) -> YieldCurve:
        """Build a curve from {tenor: yield} dict (e.g. from FRED).

        Treasury yields are quoted as semi-annual BEY; we convert to
        continuously compounded rates.
        """
        tenors = sorted(data.keys())
        cc_rates = []
        for t in tenors:
            y = data[t]
            # BEY → continuous: r = 2 * ln(1 + y/2)
            r = 2.0 * np.log(1.0 + y / 2.0)
            cc_rates.append(float(r))
        return cls(tenors, cc_rates)

    # ------------------------------------------------------------------
    # Scenario methods (return new instances — immutable)
    # ------------------------------------------------------------------

    def shift(self, bps: float) -> YieldCurve:
        """New curve with parallel shift of *bps* basis points."""
        return YieldCurve(self.tenors.copy(), self.rates + bps / 10_000.0)

    def bump(self, tenor_bumps: dict[float, float]) -> YieldCurve:
        """New curve with per-tenor bumps (in bps). Unmatched tenors unchanged."""
        new_rates = self.rates.copy()
        for tenor, bps_val in tenor_bumps.items():
            mask = np.isclose(self.tenors, tenor)
            new_rates = np.where(mask, new_rates + bps_val / 10_000.0, new_rates)
        return YieldCurve(self.tenors.copy(), new_rates)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def flat(cls, rate: float, max_tenor: float = 30.0) -> YieldCurve:
        """Convenience: flat curve at a single rate."""
        tenors = [0.0, max_tenor]
        return cls(tenors, [rate, rate])
