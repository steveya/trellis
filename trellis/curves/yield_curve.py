"""Yield curve implementing the DiscountCurve protocol."""

from __future__ import annotations

from trellis.curves.shocks import build_curve_shock_surface
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
        """Store tenor and zero-rate grids for interpolation-based discounting."""
        self.tenors = np.asarray(tenors, dtype=float)
        self.rates = np.asarray(rates, dtype=float)
        if self.tenors.ndim != 1 or self.rates.ndim != 1:
            raise ValueError("YieldCurve tenors and rates must be one-dimensional")
        if len(self.tenors) == 0:
            raise ValueError("YieldCurve requires at least one tenor/rate knot")
        if len(self.tenors) != len(self.rates):
            raise ValueError("YieldCurve tenors and rates must have the same length")
        for index in range(len(self.tenors) - 1):
            if float(self.tenors[index + 1]) <= float(self.tenors[index]):
                raise ValueError("YieldCurve tenors must be strictly increasing")

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
        """New curve with per-tenor bucket bumps (in bps).

        Exact-tenor requests still update the matching knot directly. Off-grid
        requests now insert a bumped knot into the curve so later repricing
        paths see the interpolation-aware local shock instead of a silent no-op.
        """
        surface = build_curve_shock_surface(self, tuple(tenor_bumps))
        return surface.apply_bumps(tenor_bumps)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def flat(cls, rate: float, max_tenor: float = 30.0) -> YieldCurve:
        """Convenience: flat curve at a single rate."""
        tenors = [0.0, max_tenor]
        return cls(tenors, [rate, rate])
