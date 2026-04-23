"""Tests for rates-vol market-object reconstruction and staged SABR fits."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.swaption import SwaptionSpec
from trellis.instruments.cap import CapFloorSpec, CapPayoff
from trellis.models.calibration.rates import swaption_terms
from trellis.models.calibration.rates_vol_surface import (
    CapletStripQuote,
    CapletVolStripAuthorityResult,
    SwaptionCubeQuote,
    SwaptionCubeStageComparisonResult,
    SwaptionVolCubeAuthorityResult,
    calibrate_caplet_vol_strip_workflow,
    calibrate_swaption_vol_cube_workflow,
    compare_sabr_to_swaption_cube_workflow,
)
from trellis.models.processes.sabr import SABRProcess
from trellis.models.rate_style_swaption import price_swaption_black76
from trellis.models.vol_surface import FlatVol, GridVolSurface


SETTLE = date(2024, 11, 15)


def _rates_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.040),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.0415)},
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        },
        market_provenance={
            "source": "test",
            "source_kind": "explicit_input",
            "source_ref": "_rates_market_state",
        },
    )


def test_caplet_strip_workflow_bootstraps_surface_and_materializes():
    market_state = _rates_market_state()
    target_surface = GridVolSurface(
        expiries=(0.2493150684931507, 0.4986301369863014, 0.7506849315068493, 1.0),
        strikes=(0.04, 0.05),
        vols=(
            (0.185, 0.195),
            (0.188, 0.198),
            (0.191, 0.201),
            (0.194, 0.204),
        ),
    )
    target_state = replace(market_state, vol_surface=target_surface)
    start_date = date(2025, 2, 15)
    end_dates = (
        date(2025, 5, 15),
        date(2025, 8, 15),
        date(2025, 11, 15),
        date(2026, 2, 15),
    )

    quotes: list[CapletStripQuote] = []
    for strike in target_surface.strikes:
        for end_date in end_dates:
            spec = CapFloorSpec(
                notional=1_000_000.0,
                strike=float(strike),
                start_date=start_date,
                end_date=end_date,
                frequency=Frequency.QUARTERLY,
                day_count=DayCountConvention.ACT_360,
                rate_index="USD-SOFR-3M",
            )
            quotes.append(
                CapletStripQuote(
                    spec=spec,
                    quote=CapPayoff(spec).evaluate(target_state),
                    quote_kind="price",
                    kind="cap",
                )
            )

    result = calibrate_caplet_vol_strip_workflow(
        quotes,
        market_state,
        surface_name="usd_caplet_strip",
    )

    assert isinstance(result, CapletVolStripAuthorityResult)
    assert result.summary["quote_count"] == len(quotes)
    assert result.summary["strike_count"] == len(target_surface.strikes)
    assert result.summary["expiry_count"] == len(target_surface.expiries)

    for expiry in result.expiries:
        for strike in result.strikes:
            assert result.vol_surface.black_vol(expiry, strike) == pytest.approx(
                target_surface.black_vol(expiry, strike),
                abs=1e-6,
            )

    enriched_state = result.apply_to_market_state(market_state)
    record = enriched_state.materialized_calibrated_object(object_kind="black_vol_surface")
    assert record is not None
    assert record["object_name"] == "usd_caplet_strip"
    assert record["metadata"]["surface_model_family"] == "caplet_vol_strip"

    repriced_spec = CapFloorSpec(
        notional=1_000_000.0,
        strike=0.05,
        start_date=start_date,
        end_date=end_dates[-1],
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
    )
    assert CapPayoff(repriced_spec).evaluate(enriched_state) == pytest.approx(
        CapPayoff(repriced_spec).evaluate(target_state),
        rel=1e-6,
    )


def test_swaption_cube_workflow_normalizes_quotes_and_is_tenor_aware_at_runtime():
    market_state = _rates_market_state()
    expiries = (date(2025, 11, 15), date(2026, 11, 15))
    swap_tenors = (5, 10)
    strike_grid = (0.03, 0.04, 0.05)
    quotes: list[SwaptionCubeQuote] = []

    for expiry_date in expiries:
        for tenor_years in swap_tenors:
            tenor_scale = 1.0 + 0.08 * ((tenor_years - swap_tenors[0]) / 5.0)
            swap_end = date(expiry_date.year + tenor_years, expiry_date.month, expiry_date.day)
            atm_spec = SwaptionSpec(
                notional=5_000_000.0,
                strike=0.04,
                expiry_date=expiry_date,
                swap_start=expiry_date,
                swap_end=swap_end,
                swap_frequency=Frequency.SEMI_ANNUAL,
                day_count=DayCountConvention.ACT_360,
                rate_index="USD-SOFR-3M",
                is_payer=True,
            )
            expiry_years, _annuity, forward_swap_rate, _payment_count = swaption_terms(atm_spec, market_state)
            sabr = SABRProcess(
                alpha=0.030 * tenor_scale,
                beta=0.5,
                rho=-0.20 - 0.05 * ((tenor_years - swap_tenors[0]) / 5.0),
                nu=0.35 + 0.05 * ((tenor_years - swap_tenors[0]) / 5.0),
            )
            for index, strike in enumerate(strike_grid):
                spec = replace(atm_spec, strike=float(strike))
                market_vol = float(sabr.implied_vol(forward_swap_rate, float(strike), expiry_years))
                target_price = price_swaption_black76(replace(market_state, vol_surface=FlatVol(market_vol)), spec)
                quotes.append(
                    SwaptionCubeQuote(
                        spec=spec,
                        quote=target_price,
                        quote_kind="price",
                        label=f"{expiry_date.isoformat()}_{tenor_years}Y_{index}",
                    )
                )

    result = calibrate_swaption_vol_cube_workflow(
        quotes,
        market_state,
        surface_name="usd_swaption_cube",
    )

    assert isinstance(result, SwaptionVolCubeAuthorityResult)
    assert result.summary["quote_count"] == len(quotes)
    assert result.summary["slice_count"] == len(expiries) * len(swap_tenors)
    assert len(result.tenors) == len(swap_tenors)
    assert len(result.expiries) == len(expiries)

    first_expiry = result.expiries[0]
    first_strike = strike_grid[1]
    assert result.vol_cube.swaption_black_vol(first_expiry, first_strike, result.tenors[0]) != pytest.approx(
        result.vol_cube.swaption_black_vol(first_expiry, first_strike, result.tenors[1]),
        abs=1e-10,
    )

    enriched_state = result.apply_to_market_state(market_state)
    first_slice_spec = quotes[0].spec
    long_tenor_spec = next(quote.spec for quote in quotes if quote.spec.swap_end.year - quote.spec.swap_start.year == 10)
    first_slice_price = price_swaption_black76(enriched_state, first_slice_spec)
    long_tenor_price = price_swaption_black76(enriched_state, long_tenor_spec)
    assert first_slice_price != pytest.approx(long_tenor_price, abs=1e-8)

    record = enriched_state.materialized_calibrated_object(object_kind="black_vol_surface")
    assert record is not None
    assert record["object_name"] == "usd_swaption_cube"
    assert record["metadata"]["surface_model_family"] == "swaption_vol_cube"


def test_compare_sabr_to_swaption_cube_reports_slice_diagnostics():
    market_state = _rates_market_state()
    expiries = (date(2025, 11, 15), date(2026, 11, 15))
    swap_tenors = (5, 10)
    strike_grid = (0.03, 0.04, 0.05)
    quotes: list[SwaptionCubeQuote] = []

    for expiry_date in expiries:
        for tenor_years in swap_tenors:
            swap_end = date(expiry_date.year + tenor_years, expiry_date.month, expiry_date.day)
            atm_spec = SwaptionSpec(
                notional=5_000_000.0,
                strike=0.04,
                expiry_date=expiry_date,
                swap_start=expiry_date,
                swap_end=swap_end,
                swap_frequency=Frequency.SEMI_ANNUAL,
                day_count=DayCountConvention.ACT_360,
                rate_index="USD-SOFR-3M",
                is_payer=True,
            )
            expiry_years, _annuity, forward_swap_rate, _payment_count = swaption_terms(atm_spec, market_state)
            sabr = SABRProcess(
                alpha=0.028 + 0.001 * tenor_years,
                beta=0.5,
                rho=-0.30,
                nu=0.40,
            )
            for strike in strike_grid:
                market_vol = float(sabr.implied_vol(forward_swap_rate, float(strike), expiry_years))
                quotes.append(
                    SwaptionCubeQuote(
                        spec=replace(atm_spec, strike=float(strike)),
                        quote=market_vol,
                        quote_kind="black_vol",
                    )
                )

    authority = calibrate_swaption_vol_cube_workflow(
        quotes,
        market_state,
        surface_name="usd_swaption_cube",
    )
    comparison = compare_sabr_to_swaption_cube_workflow(authority, beta=0.5)

    assert isinstance(comparison, SwaptionCubeStageComparisonResult)
    assert comparison.summary["slice_count"] == len(expiries) * len(swap_tenors)
    assert len(comparison.sabr_slice_results) == len(expiries) * len(swap_tenors)
    assert comparison.max_abs_vol_error < 2e-3
    assert comparison.rms_vol_error < 1e-3
