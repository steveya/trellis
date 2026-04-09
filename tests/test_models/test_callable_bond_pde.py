"""Tests for reusable callable-bond PDE helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.callable_bond import CallableBondSpec
from trellis.models.callable_bond_tree import price_callable_bond_tree, straight_bond_present_value
from trellis.models.vol_surface import FlatVol


def _spec() -> CallableBondSpec:
    return CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=date(2025, 1, 15),
        end_date=date(2035, 1, 15),
        call_dates=[date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15)],
        call_price=100.0,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def test_price_callable_bond_pde_stays_below_straight_bond():
    from trellis.models.callable_bond_pde import price_callable_bond_pde

    market_state = _market_state()
    spec = _spec()

    callable_price = price_callable_bond_pde(market_state, spec, n_r=121, n_t=160)
    straight_price = straight_bond_present_value(
        market_state,
        spec,
        settlement=market_state.settlement,
    )

    assert callable_price <= straight_price + 0.1
    assert 85.0 < callable_price < 102.0


def test_price_callable_bond_pde_tracks_tree_on_flat_curve():
    from trellis.models.callable_bond_pde import price_callable_bond_pde

    market_state = _market_state()
    spec = _spec()

    pde_price = price_callable_bond_pde(market_state, spec, n_r=121, n_t=160)
    tree_price = price_callable_bond_tree(
        market_state,
        spec,
        model="hull_white",
        n_steps=120,
    )

    assert abs(pde_price - tree_price) / max(abs(tree_price), 1.0) < 0.03


def test_price_callable_bond_pde_uses_market_state_hull_white_params():
    from trellis.models.callable_bond_pde import price_callable_bond_pde

    market_state = _market_state()
    calibrated_state = replace(
        market_state,
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
            "sigma": 0.004,
        },
    )

    via_market_state = price_callable_bond_pde(
        calibrated_state,
        _spec(),
        n_r=101,
        n_t=120,
    )
    via_explicit = price_callable_bond_pde(
        market_state,
        _spec(),
        mean_reversion=0.03,
        sigma=0.004,
        n_r=101,
        n_t=120,
    )

    assert via_market_state == pytest.approx(via_explicit, rel=1e-10)


def test_price_callable_bond_pde_accepts_flat_keyword_spec_arguments():
    from trellis.models.callable_bond_pde import price_callable_bond_pde

    market_state = _market_state()
    spec = _spec()

    via_spec = price_callable_bond_pde(
        market_state,
        spec,
        mean_reversion=0.05,
        sigma=0.01,
        n_r=101,
        n_t=120,
    )
    via_keywords = price_callable_bond_pde(
        market_state,
        notional=spec.notional,
        coupon=spec.coupon,
        start_date=spec.start_date,
        end_date=spec.end_date,
        call_dates=tuple(spec.call_dates),
        call_price=spec.call_price,
        frequency=spec.frequency,
        day_count=spec.day_count,
        mean_reversion=0.05,
        sigma=0.01,
        n_r=101,
        n_t=120,
    )

    assert via_keywords == pytest.approx(via_spec, rel=1e-10)
