"""Tests for bounded quanto-correlation calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.calibration.materialization import materialize_black_vol_surface
from trellis.models.calibration.quanto import (
    QuantoCorrelationCalibrationQuote,
    calibrate_quanto_correlation_workflow,
)
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


@dataclass(frozen=True)
class _QuantoSpec:
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _market_state(*, correlation: float | None = 0.35) -> MarketState:
    state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters=(
            None if correlation is None else {"quanto_correlation": correlation}
        ),
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
        },
        market_provenance={"source_kind": "explicit_input", "source_ref": "unit_test"},
    )
    return materialize_black_vol_surface(
        state,
        surface_name="quanto_flat_vol",
        vol_surface=FlatVol(0.20),
        source_kind="calibrated_surface",
        source_ref="calibrate_equity_vol_surface_workflow",
        selected_curve_roles={
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
        },
        metadata={"instrument_family": "equity_fx_quanto"},
    )


def _quote_from_spec(
    market_state: MarketState,
    spec: _QuantoSpec,
    *,
    label: str,
    weight: float = 1.0,
) -> QuantoCorrelationCalibrationQuote:
    market_price = price_quanto_option_analytical_from_market_state(market_state, spec)
    return QuantoCorrelationCalibrationQuote(
        market_price=market_price,
        notional=spec.notional,
        strike=spec.strike,
        expiry_date=spec.expiry_date,
        fx_pair=spec.fx_pair,
        underlier_currency=spec.underlier_currency,
        domestic_currency=spec.domestic_currency,
        option_type=spec.option_type,
        day_count=spec.day_count,
        quanto_correlation_key=spec.quanto_correlation_key,
        label=label,
        weight=weight,
    )


def test_quanto_correlation_quote_normalizes_and_rejects_invalid_inputs():
    quote = QuantoCorrelationCalibrationQuote(
        market_price=12.5,
        notional=1_000_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair=" EURUSD ",
        label=" atm ",
    )

    assert quote.market_price == pytest.approx(12.5)
    assert quote.fx_pair == "EURUSD"
    assert quote.label == "atm"
    assert quote.option_type == "call"
    assert quote.resolved_label(3) == "atm"

    with pytest.raises(ValueError, match="market_price"):
        QuantoCorrelationCalibrationQuote(
            market_price=-0.01,
            notional=1_000_000,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
        )
    with pytest.raises(ValueError, match="option_type"):
        QuantoCorrelationCalibrationQuote(
            market_price=12.5,
            notional=1_000_000,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
            option_type="digital",
        )


def test_calibrates_quanto_correlation_and_materializes_runtime_parameter_set():
    true_state = _market_state(correlation=0.35)
    calibration_state = replace(true_state, model_parameters=None, model_parameter_sets=None)
    specs = (
        _QuantoSpec(
            notional=1_000_000.0,
            strike=95.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
        ),
        _QuantoSpec(
            notional=1_000_000.0,
            strike=105.0,
            expiry_date=date(2026, 5, 15),
            fx_pair="EURUSD",
        ),
    )
    quotes = tuple(
        _quote_from_spec(true_state, spec, label=f"q{index}")
        for index, spec in enumerate(specs)
    )

    result = calibrate_quanto_correlation_workflow(
        quotes,
        calibration_state,
        parameter_set_name="quanto_rho_fit",
    )

    assert result.correlation == pytest.approx(0.35, abs=5e-7)
    assert result.diagnostics.max_abs_price_residual == pytest.approx(0.0, abs=1e-2)
    assert result.summary["support_boundary"] == "bounded_quanto_correlation"
    assert result.summary["quote_count"] == 2
    assert result.provenance["dependency_graph"]["workflow_id"] == (
        "quanto_correlation_calibration"
    )
    assert "quanto_correlation" in result.provenance["dependency_order"]
    assert (
        result.provenance["upstream_materializations"]["black_vol_surface"]["object_name"]
        == "quanto_flat_vol"
    )
    assert result.solve_provenance.backend["resolved_derivative_method"] == (
        "scipy_2point_residual_jacobian"
    )

    enriched = result.apply_to_market_state(calibration_state)
    record = enriched.materialized_calibrated_object(object_kind="model_parameter_set")
    assert record is not None
    assert record["object_name"] == "quanto_rho_fit"
    assert record["metadata"]["instrument_family"] == "hybrid_quanto"
    assert record["metadata"]["max_abs_price_residual"] == pytest.approx(0.0, abs=1e-2)
    assert enriched.model_parameter_sets["quanto_rho_fit"]["quanto_correlation"]["value"] == (
        pytest.approx(0.35, abs=5e-7)
    )

    repriced = [
        price_quanto_option_analytical_from_market_state(enriched, spec)
        for spec in specs
    ]
    assert repriced == pytest.approx([quote.market_price for quote in quotes], abs=1e-2)


def test_missing_hybrid_input_fails_with_actionable_diagnostics():
    state = replace(_market_state(correlation=None), forecast_curves={})
    quote = QuantoCorrelationCalibrationQuote(
        market_price=1.0,
        notional=1_000_000.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
    )

    with pytest.raises(ValueError, match="foreign carry/discount curve"):
        calibrate_quanto_correlation_workflow((quote,), state)


def test_calibration_updates_quote_specific_correlation_keys():
    true_state = replace(
        _market_state(correlation=None),
        model_parameters={"EURUSD_corr": 0.35},
    )
    spec = _QuantoSpec(
        notional=1_000_000.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        quanto_correlation_key="EURUSD_corr",
    )
    quote = _quote_from_spec(true_state, spec, label="keyed")
    calibration_state = replace(
        true_state,
        model_parameters={"EURUSD_corr": -0.20},
        model_parameter_sets=None,
    )

    result = calibrate_quanto_correlation_workflow(
        (quote,),
        calibration_state,
        parameter_set_name="quanto_rho_keyed",
    )

    assert result.correlation == pytest.approx(0.35, abs=5e-7)
    assert result.summary["correlation_keys"] == ["EURUSD_corr"]

    enriched = result.apply_to_market_state(calibration_state)
    keyed_descriptor = enriched.model_parameter_sets["quanto_rho_keyed"]["EURUSD_corr"]
    assert keyed_descriptor["value"] == pytest.approx(0.35, abs=5e-7)
    assert price_quanto_option_analytical_from_market_state(enriched, spec) == pytest.approx(
        quote.market_price,
        abs=1e-2,
    )
