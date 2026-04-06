"""Tests for stable zero-coupon-bond option helper surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pytest

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.jamshidian import zcb_option_hw
from trellis.models.vol_surface import FlatVol
from trellis.models.zcb_option import (
    normalize_zcb_option_strike,
    price_zcb_option_jamshidian,
)
from trellis.models.zcb_option_tree import price_zcb_option_tree


SETTLE = date(2024, 11, 15)
EXPIRY = date(2027, 11, 15)
BOND_MATURITY = date(2033, 11, 15)


@dataclass(frozen=True)
class ZCBOptionSpec:
    notional: float
    strike: float
    expiry_date: date
    bond_maturity_date: date
    option_type: str = "call"


@pytest.fixture()
def market_state():
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=20.0),
        vol_surface=FlatVol(0.01),
    )


def test_normalize_zcb_option_strike_converts_face_quote_to_unit_face():
    assert normalize_zcb_option_strike(63.0, 100.0) == pytest.approx(0.63)
    assert normalize_zcb_option_strike(0.63, 100.0) == pytest.approx(0.63)


def test_price_zcb_option_jamshidian_matches_closed_form_with_face_quote(market_state):
    spec = ZCBOptionSpec(
        notional=100.0,
        strike=63.0,
        expiry_date=EXPIRY,
        bond_maturity_date=BOND_MATURITY,
        option_type="call",
    )

    helper_price = price_zcb_option_jamshidian(market_state, spec, mean_reversion=0.1)
    t_exp = year_fraction(SETTLE, EXPIRY, DayCountConvention.ACT_365)
    t_bond = year_fraction(SETTLE, BOND_MATURITY, DayCountConvention.ACT_365)
    reference = zcb_option_hw(
        market_state.discount,
        0.63,
        t_exp,
        t_bond,
        0.01,
        0.1,
    )["call"] * 100.0

    assert helper_price == pytest.approx(reference, rel=1e-6)


def test_price_zcb_option_tree_matches_jamshidian_within_tolerance(market_state):
    spec = ZCBOptionSpec(
        notional=100.0,
        strike=63.0,
        expiry_date=EXPIRY,
        bond_maturity_date=BOND_MATURITY,
        option_type="call",
    )

    tree_price = price_zcb_option_tree(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=0.1,
        n_steps=200,
    )
    analytical = price_zcb_option_jamshidian(market_state, spec, mean_reversion=0.1)

    assert tree_price == pytest.approx(analytical, rel=0.02)


def test_price_zcb_option_jamshidian_uses_market_state_hull_white_mean_reversion(market_state):
    spec = ZCBOptionSpec(
        notional=100.0,
        strike=63.0,
        expiry_date=EXPIRY,
        bond_maturity_date=BOND_MATURITY,
        option_type="call",
    )
    calibrated_state = replace(
        market_state,
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
        },
    )

    via_market_state = price_zcb_option_jamshidian(calibrated_state, spec)
    via_explicit = price_zcb_option_jamshidian(market_state, spec, mean_reversion=0.03)

    assert via_market_state == pytest.approx(via_explicit, rel=1e-12)


def test_price_zcb_option_tree_uses_market_state_hull_white_params(market_state):
    spec = ZCBOptionSpec(
        notional=100.0,
        strike=63.0,
        expiry_date=EXPIRY,
        bond_maturity_date=BOND_MATURITY,
        option_type="call",
    )
    calibrated_state = replace(
        market_state,
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
            "sigma": 0.004,
        },
    )

    via_market_state = price_zcb_option_tree(
        calibrated_state,
        spec,
        model="hull_white",
        n_steps=200,
    )
    via_explicit = price_zcb_option_tree(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=0.03,
        sigma=0.004,
        n_steps=200,
    )

    assert via_market_state == pytest.approx(via_explicit, rel=1e-10)
