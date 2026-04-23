"""Tests for dated multi-curve bootstrapping and dependency graphs."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.schedule import StubType
from trellis.core.date_utils import build_period_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import Frequency
from trellis.curves.bootstrap import (
    BootstrapConventionBundle,
    DatedBootstrapCurveInputBundle,
    DatedBootstrapInstrument,
    MultiCurveBootstrapProgram,
    MultiCurveBootstrapResult,
    bootstrap_dated_curve_result,
    bootstrap_multi_curve_program,
)
from trellis.curves.yield_curve import YieldCurve


SETTLE = date(2024, 11, 15)


def _deposit_quote(curve: YieldCurve, *, start_date: date, end_date: date, day_count: DayCountConvention) -> float:
    start_years = year_fraction(SETTLE, start_date, day_count)
    end_years = year_fraction(SETTLE, end_date, day_count)
    accrual = year_fraction(start_date, end_date, day_count)
    return (float(curve.discount(start_years)) / float(curve.discount(end_years)) - 1.0) / accrual


def _future_quote(curve: YieldCurve, *, start_date: date, end_date: date, day_count: DayCountConvention) -> float:
    start_years = year_fraction(SETTLE, start_date, day_count)
    end_years = year_fraction(SETTLE, end_date, day_count)
    accrual = year_fraction(start_date, end_date, day_count)
    forward_rate = (float(curve.discount(start_years)) / float(curve.discount(end_years)) - 1.0) / accrual
    return 100.0 - forward_rate * 100.0


def _swap_quote(
    *,
    discount_curve: YieldCurve,
    forecast_curve: YieldCurve,
    start_date: date,
    end_date: date,
    fixed_frequency: Frequency,
    fixed_day_count: DayCountConvention,
    float_frequency: Frequency,
    float_day_count: DayCountConvention,
    stub_type: StubType = StubType.SHORT_LAST,
) -> float:
    fixed_schedule = build_period_schedule(
        start_date,
        end_date,
        fixed_frequency,
        day_count=fixed_day_count,
        time_origin=SETTLE,
        stub=stub_type,
    )
    float_schedule = build_period_schedule(
        start_date,
        end_date,
        float_frequency,
        day_count=float_day_count,
        time_origin=SETTLE,
        stub=stub_type,
    )
    annuity = sum(
        float(period.accrual_fraction) * float(discount_curve.discount(float(period.t_payment)))
        for period in fixed_schedule.periods
    )
    float_pv = 0.0
    for period in float_schedule.periods:
        start_years = max(float(period.t_start), 0.0)
        end_years = float(period.t_end)
        accrual = float(period.accrual_fraction)
        forward_rate = (
            float(forecast_curve.discount(start_years))
            / float(forecast_curve.discount(end_years))
            - 1.0
        ) / accrual
        float_pv += forward_rate * accrual * float(discount_curve.discount(float(period.t_payment)))
    return float_pv / annuity


def test_dated_multi_curve_program_bootstraps_discount_then_forecast_with_dependency_graph():
    ois_true = YieldCurve.flat(0.040)
    sofr_true = YieldCurve.flat(0.042)
    swap_fixed_frequency = Frequency.ANNUAL
    swap_float_frequency = Frequency.QUARTERLY
    swap_fixed_day_count = DayCountConvention.THIRTY_360_US
    swap_float_day_count = DayCountConvention.ACT_360

    ois_bundle = DatedBootstrapCurveInputBundle(
        curve_name="usd_ois_dated",
        currency="USD",
        rate_index="USD-OIS",
        curve_role="discount_curve",
        conventions=BootstrapConventionBundle(
            future_quote_style="price",
            swap_fixed_frequency=swap_fixed_frequency,
            swap_fixed_day_count=swap_fixed_day_count,
            swap_float_frequency=swap_float_frequency,
            swap_float_day_count=swap_float_day_count,
        ),
        instruments=(
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2025, 2, 15),
                quote=_deposit_quote(
                    ois_true,
                    start_date=SETTLE,
                    end_date=date(2025, 2, 15),
                    day_count=DayCountConvention.ACT_360,
                ),
                instrument_type="deposit",
                day_count=DayCountConvention.ACT_360,
                label="OIS_DEP3M",
            ),
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2026, 11, 15),
                quote=_swap_quote(
                    discount_curve=ois_true,
                    forecast_curve=ois_true,
                    start_date=SETTLE,
                    end_date=date(2026, 11, 15),
                    fixed_frequency=swap_fixed_frequency,
                    fixed_day_count=swap_fixed_day_count,
                    float_frequency=swap_float_frequency,
                    float_day_count=swap_float_day_count,
                ),
                instrument_type="swap",
                label="OIS_SWAP2Y",
            ),
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2029, 11, 15),
                quote=_swap_quote(
                    discount_curve=ois_true,
                    forecast_curve=ois_true,
                    start_date=SETTLE,
                    end_date=date(2029, 11, 15),
                    fixed_frequency=swap_fixed_frequency,
                    fixed_day_count=swap_fixed_day_count,
                    float_frequency=swap_float_frequency,
                    float_day_count=swap_float_day_count,
                ),
                instrument_type="swap",
                label="OIS_SWAP5Y",
            ),
        ),
    )

    sofr_bundle = DatedBootstrapCurveInputBundle(
        curve_name="USD-SOFR-3M_dated",
        currency="USD",
        rate_index="USD-SOFR-3M",
        curve_role="forecast_curve",
        dependency_names={"discount_curve": "usd_ois_dated"},
        conventions=BootstrapConventionBundle(
            future_quote_style="price",
            swap_fixed_frequency=swap_fixed_frequency,
            swap_fixed_day_count=swap_fixed_day_count,
            swap_float_frequency=swap_float_frequency,
            swap_float_day_count=swap_float_day_count,
        ),
        instruments=(
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2025, 2, 15),
                quote=_deposit_quote(
                    sofr_true,
                    start_date=SETTLE,
                    end_date=date(2025, 2, 15),
                    day_count=DayCountConvention.ACT_360,
                ),
                instrument_type="deposit",
                day_count=DayCountConvention.ACT_360,
                label="SOFR_DEP3M",
            ),
            DatedBootstrapInstrument(
                start_date=date(2025, 2, 15),
                end_date=date(2025, 5, 15),
                quote=_future_quote(
                    sofr_true,
                    start_date=date(2025, 2, 15),
                    end_date=date(2025, 5, 15),
                    day_count=DayCountConvention.ACT_360,
                ),
                instrument_type="future",
                day_count=DayCountConvention.ACT_360,
                label="SOFR_FUT_1",
            ),
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2027, 1, 20),
                quote=_swap_quote(
                    discount_curve=ois_true,
                    forecast_curve=sofr_true,
                    start_date=SETTLE,
                    end_date=date(2027, 1, 20),
                    fixed_frequency=swap_fixed_frequency,
                    fixed_day_count=swap_fixed_day_count,
                    float_frequency=swap_float_frequency,
                    float_day_count=swap_float_day_count,
                    stub_type=StubType.SHORT_LAST,
                ),
                instrument_type="swap",
                stub_type=StubType.SHORT_LAST,
                label="SOFR_SWAP_STUB",
            ),
        ),
    )

    program = MultiCurveBootstrapProgram(
        settlement_date=SETTLE,
        curve_inputs=(ois_bundle, sofr_bundle),
    )
    result = bootstrap_multi_curve_program(program, max_iter=75, tol=1e-12)

    assert isinstance(result, MultiCurveBootstrapResult)
    assert result.dependency_order == ("usd_ois_dated", "USD-SOFR-3M_dated")
    assert result.dependency_graph["USD-SOFR-3M_dated"]["discount_curve"] == "usd_ois_dated"
    assert result.node_results["usd_ois_dated"].diagnostics.max_abs_residual < 1e-8
    assert result.node_results["USD-SOFR-3M_dated"].diagnostics.max_abs_residual < 1e-8

    market_state = MarketState(as_of=SETTLE, settlement=SETTLE)
    enriched_state = result.apply_to_market_state(
        market_state,
        discount_curve_name="usd_ois_dated",
        forecast_curve_name="USD-SOFR-3M_dated",
    )

    assert enriched_state.selected_curve_names["discount_curve"] == "usd_ois_dated"
    assert enriched_state.selected_curve_names["forecast_curve"] == "USD-SOFR-3M_dated"
    assert enriched_state.discount is result.node_results["usd_ois_dated"].curve
    assert enriched_state.forecast_curves["USD-SOFR-3M_dated"] is result.node_results["USD-SOFR-3M_dated"].curve
    assert enriched_state.market_provenance["bootstrap_dependency_order"] == ["usd_ois_dated", "USD-SOFR-3M_dated"]


def test_dated_bootstrap_curve_result_handles_stubbed_swap_schedule():
    discount_curve = YieldCurve.flat(0.041)
    bundle = DatedBootstrapCurveInputBundle(
        curve_name="usd_ois_stubbed",
        currency="USD",
        rate_index="USD-OIS",
        curve_role="discount_curve",
        conventions=BootstrapConventionBundle(
            swap_fixed_frequency=Frequency.ANNUAL,
            swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
            swap_float_frequency=Frequency.QUARTERLY,
            swap_float_day_count=DayCountConvention.ACT_360,
        ),
        instruments=(
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2025, 2, 15),
                quote=_deposit_quote(
                    discount_curve,
                    start_date=SETTLE,
                    end_date=date(2025, 2, 15),
                    day_count=DayCountConvention.ACT_360,
                ),
                instrument_type="deposit",
                day_count=DayCountConvention.ACT_360,
                label="DEP3M",
            ),
            DatedBootstrapInstrument(
                start_date=SETTLE,
                end_date=date(2027, 1, 20),
                quote=_swap_quote(
                    discount_curve=discount_curve,
                    forecast_curve=discount_curve,
                    start_date=SETTLE,
                    end_date=date(2027, 1, 20),
                    fixed_frequency=Frequency.ANNUAL,
                    fixed_day_count=DayCountConvention.THIRTY_360_US,
                    float_frequency=Frequency.QUARTERLY,
                    float_day_count=DayCountConvention.ACT_360,
                    stub_type=StubType.SHORT_LAST,
                ),
                instrument_type="swap",
                stub_type=StubType.SHORT_LAST,
                label="SWAP_STUB",
            ),
        ),
    )

    result = bootstrap_dated_curve_result(bundle, settlement_date=SETTLE, max_iter=75, tol=1e-12)

    assert result.solve_request.request_id == "rates_dated_bootstrap_least_squares"
    assert result.diagnostics.max_abs_residual < 1e-8
    assert result.input_bundle.to_payload()["instruments"][1]["stub_type"] == "SHORT_LAST"
