"""Autograd regressions for public market objects."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.differentiable import get_numpy, gradient
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
from trellis.curves.forward_curve import ForwardCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.vol_surface import GridVolSurface


anp = get_numpy()
SETTLE = date(2024, 11, 15)
MID_DATE = date(2025, 5, 15)
END_DATE = date(2025, 11, 15)


def test_yield_curve_discount_supports_node_gradients():
    def discount_from_rates(rates):
        return YieldCurve((1.0, 2.0), rates).discount(1.5)

    rates = anp.array([0.04, 0.06])
    gradient_vector = gradient(discount_from_rates, 0)(rates)
    expected_discount = np.exp(-0.05 * 1.5)
    expected_gradient = np.array([-0.75 * expected_discount, -0.75 * expected_discount])

    assert np.allclose(gradient_vector, expected_gradient, atol=1e-12)


def test_credit_curve_survival_supports_node_gradients():
    def survival_from_hazards(hazards):
        return CreditCurve((1.0, 3.0), hazards).survival_probability(2.0)

    hazards = anp.array([0.01, 0.02])
    gradient_vector = gradient(survival_from_hazards, 0)(hazards)
    expected_survival = np.exp(-0.03)
    expected_gradient = np.array([-expected_survival, -expected_survival])

    assert np.allclose(gradient_vector, expected_gradient, atol=1e-12)


def test_grid_vol_surface_supports_node_gradients():
    def surface_vol(vol_nodes):
        surface = GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=vol_nodes,
        )
        return surface.black_vol(1.5, 100.0)

    vol_nodes = anp.array([[0.25, 0.22], [0.27, 0.24]])
    gradient_matrix = gradient(surface_vol, 0)(vol_nodes)

    assert np.allclose(gradient_matrix, np.full((2, 2), 0.25), atol=1e-12)


def test_date_aware_flat_curve_discount_date_preserves_trace():
    def dated_discount(flat_rate):
        curve = DateAwareFlatYieldCurve(value_date=SETTLE, flat_rate=flat_rate)
        return curve.discount_date(END_DATE)

    gradient_value = gradient(dated_discount, 0)(0.05)
    maturity = year_fraction(SETTLE, END_DATE, DayCountConvention.ACT_ACT_ISDA)
    expected = -maturity * np.exp(-0.05 * maturity)

    assert gradient_value == pytest.approx(expected, rel=1e-6)


def test_forward_curve_date_helpers_preserve_trace():
    def dated_forward(flat_rate):
        curve = DateAwareFlatYieldCurve(value_date=SETTLE, flat_rate=flat_rate)
        return ForwardCurve(curve).forward_rate_dates(
            MID_DATE,
            END_DATE,
            day_count=DayCountConvention.ACT_365,
            compounding="continuous",
        )

    gradient_value = gradient(dated_forward, 0)(0.05)

    assert gradient_value == pytest.approx(1.0, rel=1e-12)


def test_price_payoff_uses_public_yield_curve_for_autograd():
    class DiscountProbePayoff:
        @property
        def requirements(self):
            return {"discount_curve"}

        def evaluate(self, market_state):
            return market_state.discount.discount(1.5)

    def price_from_rates(rates):
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve((1.0, 2.0), rates),
        )
        return price_payoff(DiscountProbePayoff(), market_state)

    gradient_vector = gradient(price_from_rates, 0)(anp.array([0.04, 0.06]))
    expected_discount = np.exp(-0.05 * 1.5)
    expected_gradient = np.array([-0.75 * expected_discount, -0.75 * expected_discount])

    assert np.allclose(gradient_vector, expected_gradient, atol=1e-12)


def test_price_payoff_uses_public_credit_curve_for_autograd():
    class CreditProbePayoff:
        @property
        def requirements(self):
            return {"credit_curve"}

        def evaluate(self, market_state):
            return market_state.credit_curve.survival_probability(2.0)

    def price_from_hazards(hazards):
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            credit_curve=CreditCurve((1.0, 3.0), hazards),
        )
        return price_payoff(CreditProbePayoff(), market_state)

    gradient_vector = gradient(price_from_hazards, 0)(anp.array([0.01, 0.02]))
    expected_survival = np.exp(-0.03)
    expected_gradient = np.array([-expected_survival, -expected_survival])

    assert np.allclose(gradient_vector, expected_gradient, atol=1e-12)


def test_price_payoff_uses_public_grid_vol_surface_for_autograd():
    class VolProbePayoff:
        @property
        def requirements(self):
            return {"black_vol_surface"}

        def evaluate(self, market_state):
            return market_state.vol_surface.black_vol(1.5, 100.0)

    def price_from_vol_nodes(vol_nodes):
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            vol_surface=GridVolSurface(
                expiries=(1.0, 2.0),
                strikes=(90.0, 110.0),
                vols=vol_nodes,
            ),
        )
        return price_payoff(VolProbePayoff(), market_state)

    gradient_matrix = gradient(
        price_from_vol_nodes,
        0,
    )(anp.array([[0.25, 0.22], [0.27, 0.24]]))

    assert np.allclose(gradient_matrix, np.full((2, 2), 0.25), atol=1e-12)
