"""Tests for interpolation-aware yield-curve shock surfaces."""

from __future__ import annotations

import pytest

from trellis.curves.shocks import build_curve_shock_surface
from trellis.curves.yield_curve import YieldCurve


def test_curve_shock_surface_reports_off_grid_support_and_sparse_warning():
    curve = YieldCurve([1.0, 10.0, 30.0], [0.02, 0.03, 0.04])

    surface = build_curve_shock_surface(
        curve,
        bucket_tenors=(7.0,),
        support_width_warning_years=5.0,
    )

    bucket = surface.buckets[0]
    assert bucket.tenor == pytest.approx(7.0)
    assert bucket.is_exact_curve_tenor is False
    assert bucket.left_support_tenor == pytest.approx(1.0)
    assert bucket.right_support_tenor == pytest.approx(10.0)
    assert bucket.support_width == pytest.approx(9.0)
    assert [warning.code for warning in bucket.warnings] == ["wide_support_interval"]


def test_curve_shock_surface_apply_inserts_off_grid_bucket():
    curve = YieldCurve([1.0, 5.0, 10.0], [0.02, 0.03, 0.04])

    surface = build_curve_shock_surface(curve, bucket_tenors=(7.0,))
    bumped = surface.apply_bumps({7.0: 25.0})

    assert tuple(float(tenor) for tenor in bumped.tenors) == pytest.approx((1.0, 5.0, 7.0, 10.0))
    assert bumped.zero_rate(7.0) == pytest.approx(curve.zero_rate(7.0) + 0.0025)
    assert bumped.zero_rate(5.0) == pytest.approx(curve.zero_rate(5.0))
    assert bumped.zero_rate(10.0) == pytest.approx(curve.zero_rate(10.0))
