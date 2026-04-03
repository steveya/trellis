"""Tests for FX spot, forwards, and cross-currency payoff."""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.instruments.fx import FXForward, FXForwardPayoff, FXRate


SETTLE = date(2024, 11, 15)


def _fx_rate():
    return FXRate(spot=1.10, domestic="USD", foreign="EUR")


class TestFXForward:

    def test_cip_identity(self):
        """F(t) = S * df_foreign / df_domestic."""
        dom = YieldCurve.flat(0.05)
        fgn = YieldCurve.flat(0.03)
        fwd = FXForward(_fx_rate(), dom, fgn)
        F1 = fwd.forward(1.0)
        expected = 1.10 * np.exp(-0.03) / np.exp(-0.05)
        assert F1 == pytest.approx(expected, rel=1e-10)

    def test_equal_rates_flat(self):
        """If domestic = foreign rates, forward = spot."""
        dom = YieldCurve.flat(0.04)
        fgn = YieldCurve.flat(0.04)
        fwd = FXForward(_fx_rate(), dom, fgn)
        assert fwd.forward(5.0) == pytest.approx(1.10, rel=1e-10)

    def test_forward_points(self):
        dom = YieldCurve.flat(0.05)
        fgn = YieldCurve.flat(0.03)
        fwd = FXForward(_fx_rate(), dom, fgn)
        pts = fwd.forward_points(1.0)
        assert pts == pytest.approx(fwd.forward(1.0) - 1.10, rel=1e-10)
        assert pts > 0  # domestic rate higher → forward > spot


class TestFXForwardPayoff:

    def test_foreign_zcb_conversion(self):
        """Foreign ZCB paying 100 EUR at 5Y, converted to USD."""
        bond = Bond(
            face=100, coupon=0.0,
            maturity_date=date(2029, 11, 15),
            maturity=5, frequency=2,
            issue_date=SETTLE,
        )
        inner = DeterministicCashflowPayoff(bond)

        dom = YieldCurve.flat(0.05)
        fgn = YieldCurve.flat(0.03)
        fx = _fx_rate()

        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=dom,
            forecast_curves={"EUR-DISC": fgn},
            fx_rates={"EURUSD": fx},
        )

        fx_payoff = FXForwardPayoff(inner, "EURUSD", "EUR-DISC")
        pv = price_payoff(fx_payoff, ms)

        # Inner payoff discounts at domestic rate (ms.discount = dom = 5%)
        # FXForwardPayoff multiplies by spot: PV = 100*exp(-0.05*5) * 1.10
        inner_pv = 100 * np.exp(-0.05 * 5)
        expected = inner_pv * 1.10
        assert pv == pytest.approx(expected, rel=0.02)

    def test_requirements(self):
        inner = DeterministicCashflowPayoff(
            Bond(face=100, coupon=0.0, maturity_date=date(2029, 11, 15),
                 maturity=5, frequency=2, issue_date=SETTLE)
        )
        fx_payoff = FXForwardPayoff(inner, "EURUSD", "EUR-DISC")
        assert "fx_rates" in fx_payoff.requirements
        assert "discount_curve" in fx_payoff.requirements
        assert "forward_curve" in fx_payoff.requirements
