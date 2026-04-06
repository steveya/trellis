"""Tests for yield curve and interpolation."""
import numpy as np
import pytest
from trellis.curves.yield_curve import YieldCurve
from trellis.curves.interpolation import linear_interp, log_linear_interp


class TestLinearInterp:
    def test_exact_knot(self):
        xs = [1.0, 2.0, 3.0]
        ys = [0.01, 0.02, 0.03]
        assert linear_interp(2.0, xs, ys) == pytest.approx(0.02)

    def test_midpoint(self):
        xs = [1.0, 3.0]
        ys = [0.01, 0.03]
        assert linear_interp(2.0, xs, ys) == pytest.approx(0.02)

    def test_flat_extrapolation_left(self):
        xs = [1.0, 2.0]
        ys = [0.01, 0.02]
        assert linear_interp(0.0, xs, ys) == pytest.approx(0.01)

    def test_flat_extrapolation_right(self):
        xs = [1.0, 2.0]
        ys = [0.01, 0.02]
        assert linear_interp(5.0, xs, ys) == pytest.approx(0.02)


class TestYieldCurve:
    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            YieldCurve([1.0, 2.0], [0.03])

    def test_rejects_unsorted_tenors(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            YieldCurve([5.0, 1.0, 10.0], [0.04, 0.03, 0.05])

    def test_flat_curve_discount(self):
        curve = YieldCurve.flat(0.05)
        df = curve.discount(1.0)
        assert df == pytest.approx(np.exp(-0.05))

    def test_flat_curve_discount_at_zero(self):
        curve = YieldCurve.flat(0.05)
        assert curve.discount(0.0) == pytest.approx(1.0)

    def test_from_treasury_yields(self):
        data = {1.0: 0.04, 5.0: 0.045, 10.0: 0.05}
        curve = YieldCurve.from_treasury_yields(data)
        assert len(curve.tenors) == 3
        # Rates should be continuously compounded (slightly different from BEY)
        assert curve.zero_rate(1.0) == pytest.approx(2 * np.log(1 + 0.04/2))

    def test_discount_decreases_with_time(self):
        curve = YieldCurve.flat(0.05)
        assert curve.discount(5.0) < curve.discount(1.0)

    def test_interpolated_rate(self):
        curve = YieldCurve([1.0, 10.0], [0.03, 0.05])
        r5 = curve.zero_rate(5.5)
        assert 0.03 < r5 < 0.05
