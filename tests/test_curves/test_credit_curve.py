"""Tests for CreditCurve."""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve


class TestCreditCurve:
    def test_rejects_unsorted_tenors(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            CreditCurve([5.0, 1.0, 10.0], [0.02, 0.01, 0.03])

    def test_flat_survival(self):
        cc = CreditCurve.flat(0.02)
        assert cc.survival_probability(5) == pytest.approx(np.exp(-0.02 * 5))

    def test_flat_hazard(self):
        cc = CreditCurve.flat(0.02)
        assert cc.hazard_rate(3.0) == pytest.approx(0.02)

    def test_risky_discount(self):
        cc = CreditCurve.flat(0.02)
        yc = YieldCurve.flat(0.05)
        rd = cc.risky_discount(5.0, yc)
        expected = np.exp(-0.02 * 5) * np.exp(-0.05 * 5)
        assert rd == pytest.approx(expected, rel=1e-10)

    def test_from_spreads(self):
        cc = CreditCurve.from_spreads({5.0: 0.012}, recovery=0.4)
        # lambda = 0.012 / 0.6 = 0.02
        assert cc.hazard_rate(5.0) == pytest.approx(0.02)
        assert cc.survival_probability(5) == pytest.approx(np.exp(-0.02 * 5))

    def test_interpolation(self):
        cc = CreditCurve([1.0, 5.0, 10.0], [0.01, 0.02, 0.03])
        h3 = cc.hazard_rate(3.0)
        assert 0.01 < h3 < 0.02  # interpolated

    def test_shift(self):
        cc = CreditCurve.flat(0.02)
        cc2 = cc.shift(+50)
        assert cc2.hazard_rate(5.0) == pytest.approx(0.02 + 0.005)

    def test_capability(self):
        ms = MarketState(
            as_of=date(2024, 11, 15),
            settlement=date(2024, 11, 15),
            discount=YieldCurve.flat(0.05),
            credit_curve=CreditCurve.flat(0.02),
        )
        assert "credit_curve" in ms.available_capabilities
