"""Credit curve: survival probabilities from hazard rates."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.core.types import DiscountCurve
from trellis.curves.interpolation import linear_interp, to_backend_array, validation_view

np = get_numpy()


class CreditCurve:
    """Survival probability curve from piecewise-constant hazard rates.

    S(t) = exp(-lambda(t) * t) where lambda(t) is interpolated.
    """

    def __init__(self, tenors, hazard_rates):
        """Store tenor and hazard-rate grids for interpolation-based survival queries."""
        self.tenors = to_backend_array(tenors)
        self.hazard_rates = to_backend_array(hazard_rates)
        tenors_view = validation_view(self.tenors)
        hazard_view = validation_view(self.hazard_rates)
        if tenors_view.ndim != 1 or hazard_view.ndim != 1:
            raise ValueError("CreditCurve tenors and hazard_rates must be one-dimensional")
        if len(tenors_view) == 0:
            raise ValueError("CreditCurve requires at least one tenor/hazard-rate knot")
        if len(tenors_view) != len(hazard_view):
            raise ValueError("CreditCurve tenors and hazard_rates must have the same length")
        for index in range(len(tenors_view) - 1):
            if tenors_view[index + 1] <= tenors_view[index]:
                raise ValueError("CreditCurve tenors must be strictly increasing")

    def hazard_rate(self, t: float) -> float:
        """Interpolated hazard rate at time t.

        The interpolation is differentiable in the stored hazard-rate nodes
        and only piecewise differentiable in the query location ``t`` away
        from knot boundaries.
        """
        return linear_interp(t, self.tenors, self.hazard_rates)

    def survival_probability(self, t: float) -> float:
        """S(t) = exp(-lambda(t) * t)."""
        return np.exp(-self.hazard_rate(t) * t)

    def risky_discount(self, t: float, discount_curve: DiscountCurve) -> float:
        """Risky discount factor = S(t) * df(t)."""
        return self.survival_probability(t) * discount_curve.discount(t)

    @classmethod
    def flat(cls, hazard_rate: float, max_tenor: float = 30.0) -> CreditCurve:
        """Flat hazard rate curve."""
        return cls([0.0, max_tenor], [hazard_rate, hazard_rate])

    @classmethod
    def from_spreads(cls, cds_spreads: dict[float, float],
                     recovery: float = 0.4) -> CreditCurve:
        """Bootstrap from CDS par spreads (first-order approximation).

        lambda ≈ spread / (1 - R) for each tenor.
        """
        tenors = sorted(cds_spreads.keys())
        hazard_rates = [cds_spreads[t] / (1.0 - recovery) for t in tenors]
        return cls(tenors, hazard_rates)

    def shift(self, bps: float) -> CreditCurve:
        """New curve with parallel hazard rate shift (bps)."""
        return CreditCurve(self.tenors.copy(),
                           self.hazard_rates + bps / 10_000.0)
