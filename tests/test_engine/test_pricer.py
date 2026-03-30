"""Tests for pricing engine and analytics."""
import numpy as np
import pytest
from datetime import date

from trellis.instruments.bond import Bond
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.pricer import price_instrument
from trellis.engine.analytics import compute_greeks


class TestPricer:
    def test_treasury_bond_pricing(self):
        """End-to-end: price a Treasury bond with mocked yield data."""
        # Mocked FRED-like data (as decimal, already converted from BEY)
        yield_data = {
            1/12: 0.0525, 0.25: 0.053, 0.5: 0.052,
            1.0: 0.048, 2.0: 0.045, 3.0: 0.043,
            5.0: 0.042, 7.0: 0.042, 10.0: 0.043,
            20.0: 0.046, 30.0: 0.045,
        }
        curve = YieldCurve.from_treasury_yields(yield_data)

        bond = Bond(
            face=100, coupon=0.045,
            maturity_date=date(2034, 11, 15),
            maturity=10, frequency=2,
        )

        result = price_instrument(bond, curve, settlement=date(2024, 11, 15))
        assert 80 < result.clean_price < 120
        assert result.dirty_price > 0
        assert "dv01" in result.greeks
        assert "duration" in result.greeks
        assert "convexity" in result.greeks
        assert result.greeks["duration"] > 0
        assert result.greeks["dv01"] > 0
        assert result.curve_sensitivities

    def test_flat_curve_zero_coupon(self):
        """Zero-coupon bond on flat curve should equal face * discount(T)."""
        curve = YieldCurve.flat(0.05)
        bond = Bond(
            face=100, coupon=0.0,
            maturity_date=date(2034, 3, 19),
            maturity=10, frequency=2,
            issue_date=date(2024, 3, 19),
        )
        result = price_instrument(bond, curve, settlement=date(2024, 3, 19))
        expected = 100 * np.exp(-0.05 * 10)
        assert result.dirty_price == pytest.approx(expected, rel=0.01)


class TestAnalytics:
    def test_greeks_finite_difference_consistency(self):
        """Autodiff Greeks should be consistent with finite-difference approximations."""
        curve = YieldCurve([1.0, 5.0, 10.0], [0.04, 0.045, 0.05])
        rates = curve.rates.copy()

        def price_fn(r):
            # Simple 5-year bond pricing function
            from trellis.curves.interpolation import linear_interp
            from trellis.core.differentiable import get_numpy
            np_ = get_numpy()
            pv = np_.array(0.0)
            for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                coupon = 2.25  # 4.5% semi-annual on 100 face
                rate = linear_interp(t, curve.tenors, r)
                pv = pv + coupon * np_.exp(-rate * t)
            # Add principal at maturity
            rate_5 = linear_interp(5.0, curve.tenors, r)
            pv = pv + 100 * np_.exp(-rate_5 * 5.0)
            return pv

        greeks = compute_greeks(price_fn, rates, tenors=[1.0, 5.0, 10.0])

        # Finite difference check for DV01
        bump = 0.0001
        p_up = float(price_fn(rates + bump))
        p_dn = float(price_fn(rates - bump))
        fd_dv01 = -(p_up - p_dn) / 2  # for a 1bp shift
        p = float(price_fn(rates))
        fd_convexity = (p_up + p_dn - 2 * p) / (p * bump ** 2)

        assert greeks["dv01"] == pytest.approx(fd_dv01, rel=0.05)
        assert greeks["duration"] > 0
        assert greeks["convexity"] == pytest.approx(fd_convexity, rel=0.05)
