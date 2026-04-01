"""Tests for reusable callable-bond tree helpers."""

from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.callable_bond import CallableBondSpec
from trellis.models.callable_bond_tree import (
    build_callable_bond_lattice,
    price_callable_bond_tree,
)
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


def _market_state(vol: float) -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def test_price_callable_bond_tree_hull_white_has_negative_vega():
    low = price_callable_bond_tree(_market_state(0.05), _spec(), model="hull_white")
    high = price_callable_bond_tree(_market_state(0.40), _spec(), model="hull_white")

    assert high < low


def test_build_callable_bond_lattice_supports_bdt_model():
    lattice = build_callable_bond_lattice(
        _market_state(0.20),
        _spec(),
        model="bdt",
        mean_reversion=0.05,
        n_steps=50,
    )

    assert lattice.n_steps == 50


def test_price_callable_bond_tree_bdt_and_hull_white_are_consistent_on_flat_curve():
    bdt = price_callable_bond_tree(_market_state(0.20), _spec(), model="bdt")
    hw = price_callable_bond_tree(_market_state(0.20), _spec(), model="hull_white")

    assert abs(bdt - hw) / max(abs(hw), 1.0) < 0.05
