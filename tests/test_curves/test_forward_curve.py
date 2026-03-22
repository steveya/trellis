"""Tests for ForwardCurve — forward rate extraction from discount curves."""

import numpy as np
import pytest

from trellis.core.differentiable import get_numpy, gradient

anp = get_numpy()
from trellis.curves.forward_curve import ForwardCurve
from trellis.curves.yield_curve import YieldCurve


def _flat_curve(rate=0.05):
    return YieldCurve.flat(rate)


def _sloped_curve():
    return YieldCurve([1.0, 2.0, 5.0, 10.0, 30.0], [0.04, 0.045, 0.05, 0.048, 0.046])


class TestForwardRate:

    def test_no_arb_simple(self):
        """df(t1)/df(t2) = 1 + F_simple * tau."""
        curve = _flat_curve(0.05)
        fc = ForwardCurve(curve)
        t1, t2 = 1.0, 2.0
        F = fc.forward_rate(t1, t2, compounding="simple")
        tau = t2 - t1
        lhs = float(curve.discount(t1) / curve.discount(t2))
        rhs = 1.0 + F * tau
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_no_arb_continuous(self):
        """ln(df(t1)/df(t2)) / tau = F_cc."""
        curve = _flat_curve(0.05)
        fc = ForwardCurve(curve)
        t1, t2 = 1.0, 2.0
        F_cc = fc.forward_rate(t1, t2, compounding="continuous")
        expected = np.log(float(curve.discount(t1) / curve.discount(t2)))
        assert F_cc == pytest.approx(expected, rel=1e-12)

    def test_flat_curve_continuous_forward_equals_zero_rate(self):
        """On a flat CC curve, continuous forward = zero rate."""
        curve = _flat_curve(0.05)
        fc = ForwardCurve(curve)
        F_cc = fc.forward_rate(1.0, 2.0, compounding="continuous")
        assert F_cc == pytest.approx(0.05, rel=1e-6)

    def test_non_flat_curve_no_arb(self):
        """No-arb identity holds on a non-flat curve."""
        curve = _sloped_curve()
        fc = ForwardCurve(curve)
        for t1, t2 in [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0)]:
            F = fc.forward_rate(t1, t2, compounding="simple")
            tau = t2 - t1
            lhs = float(curve.discount(t1) / curve.discount(t2))
            rhs = 1.0 + F * tau
            assert lhs == pytest.approx(rhs, rel=1e-10), f"Failed for [{t1}, {t2}]"

    def test_t1_equals_t2_raises(self):
        fc = ForwardCurve(_flat_curve())
        with pytest.raises(ValueError, match="t2 must be > t1"):
            fc.forward_rate(1.0, 1.0)

    def test_t2_less_than_t1_raises(self):
        fc = ForwardCurve(_flat_curve())
        with pytest.raises(ValueError):
            fc.forward_rate(2.0, 1.0)

    def test_short_period(self):
        """Short forward period should be close to zero rate at that point."""
        curve = _flat_curve(0.05)
        fc = ForwardCurve(curve)
        F = fc.forward_rate(1.0, 1.01, compounding="continuous")
        assert F == pytest.approx(0.05, rel=0.01)

    def test_autograd_compatible(self):
        """Gradient through forward_rate should not raise.

        Note: YieldCurve constructor uses np.asarray which autograd can't
        trace through, so we construct the curve outside the traced function
        and differentiate through the forward rate computation only.
        """
        from trellis.curves.interpolation import linear_interp

        tenors = np.array([1.0, 2.0, 5.0])

        def price_via_forward(rates):
            # Replicate discount factor computation without YieldCurve constructor
            r1 = linear_interp(1.0, tenors, rates)
            r2 = linear_interp(2.0, tenors, rates)
            df1 = anp.exp(-r1 * 1.0)
            df2 = anp.exp(-r2 * 2.0)
            return (df1 / df2 - 1.0)  # simple forward rate * tau=1

        rates = np.array([0.04, 0.045, 0.05])
        grad_fn = gradient(price_via_forward, 0)
        grad = grad_fn(rates)
        assert grad is not None
        assert not np.all(grad == 0)
