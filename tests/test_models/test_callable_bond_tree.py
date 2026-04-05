"""Tests for reusable callable-bond tree helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.callable_bond import CallableBondSpec
from trellis.models.callable_bond_tree import (
    build_callable_bond_lattice,
    compile_callable_bond_contract_spec,
    price_callable_bond_tree,
    price_callable_bond_on_lattice,
    straight_bond_present_value,
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


class _PuttableSpec:
    def __init__(self) -> None:
        self.notional = 100.0
        self.coupon = 0.05
        self.start_date = date(2025, 1, 15)
        self.end_date = date(2035, 1, 15)
        self.put_dates = (date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15))
        self.put_price = 100.0
        self.frequency = Frequency.SEMI_ANNUAL
        self.day_count = DayCountConvention.ACT_365


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


def test_compile_callable_bond_contract_spec_matches_helper_price():
    market_state = _market_state(0.20)
    spec = _spec()
    lattice = build_callable_bond_lattice(
        market_state,
        spec,
        model="hull_white",
        n_steps=80,
    )
    contract = compile_callable_bond_contract_spec(
        spec,
        settlement=market_state.settlement,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
    )

    assert price_callable_bond_on_lattice(
        lattice,
        spec=spec,
        settlement=market_state.settlement,
    ) == price_callable_bond_on_lattice(
        lattice,
        contract_spec=contract,
    )


def test_price_callable_bond_tree_supports_puttable_specs_and_floors_straight_value():
    market_state = _market_state(0.20)
    spec = _PuttableSpec()

    puttable_price = price_callable_bond_tree(market_state, spec, model="hull_white")
    straight_price = straight_bond_present_value(
        market_state,
        spec,
        settlement=market_state.settlement,
    )

    assert puttable_price >= straight_price


def test_price_callable_bond_tree_uses_market_state_hull_white_params():
    market_state = _market_state(0.20)
    calibrated_state = replace(
        market_state,
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
            "sigma": 0.004,
        },
    )

    via_market_state = price_callable_bond_tree(
        calibrated_state,
        _spec(),
        model="hull_white",
    )
    via_explicit = price_callable_bond_tree(
        market_state,
        _spec(),
        model="hull_white",
        mean_reversion=0.03,
        sigma=0.004,
    )

    assert via_market_state == pytest.approx(via_explicit, rel=1e-10)
