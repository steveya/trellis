"""Focused tests for rates calibration shared term vocabulary and tolerances."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.instruments._agent.swaption import SwaptionSpec
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.calibration.rates import calibrate_cap_floor_black_vol, calibrate_swaption_black_vol
from trellis.models.rate_style_swaption import price_swaption_black76
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _multi_curve_market_state():
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="test",
        discount_curves={"usd_ois": YieldCurve.flat(0.050)},
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.052)},
        provenance={
            "source": "test",
            "source_kind": "explicit_input",
            "source_ref": "_multi_curve_market_state",
        },
    )
    return snapshot.to_market_state(
        settlement=SETTLE,
        discount_curve="usd_ois",
        forecast_curve="USD-SOFR-3M",
    )


@pytest.mark.parametrize("kind,payoff_cls", [("cap", CapPayoff), ("floor", FloorPayoff)])
def test_cap_floor_calibration_reports_shared_terms_and_residual_policy(kind, payoff_cls):
    market_state = _multi_curve_market_state()
    true_vol = 0.215
    spec = CapFloorSpec(
        notional=1_000_000.0,
        strike=0.05,
        start_date=date(2025, 2, 15),
        end_date=date(2027, 2, 15),
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
    )
    target_state = replace(market_state, vol_surface=FlatVol(true_vol))
    target_price = payoff_cls(spec).evaluate(target_state)

    result = calibrate_cap_floor_black_vol(spec, market_state, target_price, kind=kind)

    assert result.summary["period_count"] > 0
    assert result.summary["payment_count"] == result.summary["period_count"]
    assert result.summary["annuity"] > 0.0
    assert result.summary["forward_rate"] > 0.0
    assert result.summary["expiry_years"] > 0.0
    assert result.summary["residual_tolerance_abs"] > 0.0
    assert abs(result.residual) <= result.summary["residual_tolerance_abs"]
    assert result.summary["residual_within_tolerance"] is True


def test_swaption_calibration_reports_shared_residual_policy():
    market_state = _multi_curve_market_state()
    true_vol = 0.18
    spec = SwaptionSpec(
        notional=5_000_000.0,
        strike=0.05,
        expiry_date=date(2026, 2, 15),
        swap_start=date(2026, 2, 15),
        swap_end=date(2031, 2, 15),
        swap_frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
        is_payer=True,
    )
    target_state = replace(market_state, vol_surface=FlatVol(true_vol))
    target_price = price_swaption_black76(target_state, spec)

    result = calibrate_swaption_black_vol(spec, market_state, target_price)

    assert result.summary["annuity"] > 0.0
    assert result.summary["forward_swap_rate"] > 0.0
    assert result.summary["residual_tolerance_abs"] > 0.0
    assert abs(result.residual) <= result.summary["residual_tolerance_abs"]
    assert result.summary["residual_within_tolerance"] is True
