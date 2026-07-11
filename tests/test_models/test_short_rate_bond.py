"""Tests for bounded affine short-rate zero-coupon bond helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.resolution.short_rate_claims import ShortRateComparisonRegime


SETTLE = date(2024, 11, 15)
MATURITY = date(2034, 11, 15)


@dataclass(frozen=True)
class ShortRateBondSpec:
    notional: float
    maturity_date: date
    day_count: DayCountConvention = DayCountConvention.ACT_365


@pytest.fixture()
def comparison_market_state():
    regime = ShortRateComparisonRegime(
        regime_name="t56_t57_short_rate_comparison",
        flat_discount_rate=0.05,
        flat_sigma=0.01,
        hull_white_mean_reversion=0.1,
    )
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=regime.build_discount_curve(max_tenor=20.0),
        vol_surface=regime.build_vol_surface(),
        market_provenance={"comparison_regime": regime.to_payload()},
    )


def test_vasicek_zero_coupon_analytical_reduces_to_flat_curve_when_vol_zero(comparison_market_state):
    from trellis.models.short_rate_bond import price_vasicek_zero_coupon_bond_analytical

    spec = ShortRateBondSpec(notional=100.0, maturity_date=MATURITY)
    price = price_vasicek_zero_coupon_bond_analytical(
        comparison_market_state,
        spec,
        sigma=0.0,
        long_term_rate=0.05,
    )

    assert price == pytest.approx(100.0 * comparison_market_state.discount.discount(10.0), rel=2e-3)


def test_vasicek_tree_matches_analytical_on_short_rate_regime(comparison_market_state):
    from trellis.models.short_rate_bond import (
        price_short_rate_zero_coupon_bond_tree,
        price_vasicek_zero_coupon_bond_analytical,
    )

    spec = ShortRateBondSpec(notional=100.0, maturity_date=MATURITY)

    analytical = price_vasicek_zero_coupon_bond_analytical(comparison_market_state, spec)
    tree = price_short_rate_zero_coupon_bond_tree(
        comparison_market_state,
        spec,
        model="vasicek",
        n_steps=360,
    )

    assert tree == pytest.approx(analytical, rel=0.015)


def test_cir_tree_matches_analytical_on_short_rate_regime(comparison_market_state):
    from trellis.models.short_rate_bond import (
        price_cir_zero_coupon_bond_analytical,
        price_short_rate_zero_coupon_bond_tree,
    )

    spec = ShortRateBondSpec(notional=100.0, maturity_date=MATURITY)

    analytical = price_cir_zero_coupon_bond_analytical(comparison_market_state, spec)
    tree = price_short_rate_zero_coupon_bond_tree(
        comparison_market_state,
        spec,
        model="cir",
        n_steps=360,
    )

    assert tree == pytest.approx(analytical, rel=0.02)


def test_short_rate_bond_resolves_named_model_parameter_set(comparison_market_state):
    from trellis.models.short_rate_bond import (
        price_short_rate_zero_coupon_bond_analytical,
        resolve_short_rate_bond_inputs,
    )

    spec = ShortRateBondSpec(notional=100.0, maturity_date=MATURITY)
    parameter_state = replace(
        comparison_market_state,
        model_parameters={
            "vasicek": {
                "a": 0.07,
                "b": 0.045,
                "sigma": 0.012,
                "r0": 0.052,
            }
        },
    )

    resolved = resolve_short_rate_bond_inputs(parameter_state, spec, model="vasicek")
    explicit_price = price_short_rate_zero_coupon_bond_analytical(
        comparison_market_state,
        spec,
        model="vasicek",
        initial_rate=0.052,
        mean_reversion=0.07,
        long_term_rate=0.045,
        sigma=0.012,
    )

    assert resolved.initial_rate == pytest.approx(0.052)
    assert resolved.mean_reversion == pytest.approx(0.07)
    assert resolved.long_term_rate == pytest.approx(0.045)
    assert price_short_rate_zero_coupon_bond_analytical(parameter_state, spec, model="vasicek") == pytest.approx(
        explicit_price,
        rel=1e-12,
    )


def test_short_rate_bond_does_not_treat_heston_parameters_as_rate_model():
    from trellis.models.short_rate_bond import resolve_short_rate_bond_inputs

    spec = ShortRateBondSpec(notional=100.0, maturity_date=MATURITY)
    heston_shaped_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=20.0),
        model_parameters={
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.30,
            "rho": -0.70,
            "v0": 0.04,
        },
    )

    with pytest.raises(ValueError, match="short-rate bond pricing requires model sigma"):
        resolve_short_rate_bond_inputs(heston_shaped_state, spec, model="vasicek")

    resolved = resolve_short_rate_bond_inputs(
        heston_shaped_state,
        spec,
        model="vasicek",
        allow_benchmark_defaults=True,
    )

    assert resolved.parameter_source == "benchmark_defaults"
    assert resolved.mean_reversion == pytest.approx(0.1)
    assert resolved.long_term_rate == pytest.approx(resolved.initial_rate)
    assert resolved.sigma == pytest.approx(0.01)
