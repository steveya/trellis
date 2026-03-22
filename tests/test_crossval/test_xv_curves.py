"""XV6: Yield curve cross-validation — discount factors and forward rates."""

from datetime import date

import numpy as raw_np
import pytest

# --- Trellis ---
from trellis.curves.yield_curve import YieldCurve
from trellis.curves.forward_curve import ForwardCurve

SETTLE = date(2024, 11, 15)
RATE = 0.05


def trellis_discount(rate, t):
    return float(YieldCurve.flat(rate).discount(t))


def trellis_forward(rate, t1, t2):
    fc = ForwardCurve(YieldCurve.flat(rate))
    return fc.forward_rate(t1, t2, compounding="simple")


def quantlib_discount(rate, t):
    import QuantLib as ql
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today
    curve = ql.FlatForward(today, rate, ql.Actual365Fixed(), ql.Continuous)
    return curve.discount(t)


def quantlib_forward(rate, t1, t2):
    import QuantLib as ql
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today
    curve = ql.FlatForward(today, rate, ql.Actual365Fixed(), ql.Continuous)
    # QL forwardRate with Time args: (t1, t2, Compounding, Frequency)
    return curve.forwardRate(t1, t2, ql.Simple, ql.Annual).rate()


class TestDiscountFactorCrossVal:

    def test_flat_curve_vs_quantlib(self):
        """Trellis flat curve discount factors match QuantLib."""
        for t in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
            trellis_df = trellis_discount(RATE, t)
            ql_df = quantlib_discount(RATE, t)
            assert trellis_df == pytest.approx(ql_df, rel=1e-6), (
                f"t={t}: Trellis={trellis_df:.8f}, QL={ql_df:.8f}"
            )

    def test_discount_at_zero(self):
        """df(0) = 1.0 for both."""
        assert trellis_discount(RATE, 0.0) == pytest.approx(1.0, rel=1e-10)
        assert quantlib_discount(RATE, 0.0) == pytest.approx(1.0, rel=1e-10)


class TestForwardRateCrossVal:

    def test_forward_rate_vs_quantlib(self):
        """Trellis simple forward rate matches QuantLib."""
        for t1, t2 in [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0)]:
            trellis_fwd = trellis_forward(RATE, t1, t2)
            ql_fwd = quantlib_forward(RATE, t1, t2)
            assert trellis_fwd == pytest.approx(ql_fwd, rel=0.01), (
                f"[{t1},{t2}]: Trellis={trellis_fwd:.6f}, QL={ql_fwd:.6f}"
            )

    def test_flat_curve_forward_equals_spot(self):
        """On flat CC curve, forward ≈ spot rate (both libs)."""
        trellis_fwd = trellis_forward(RATE, 1.0, 2.0)
        ql_fwd = quantlib_forward(RATE, 1.0, 2.0)
        # Simple forward on flat CC curve ≈ exp(r)-1 per year
        expected = raw_np.exp(RATE) - 1
        assert trellis_fwd == pytest.approx(expected, rel=0.01)
        assert ql_fwd == pytest.approx(expected, rel=0.01)
