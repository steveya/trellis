from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

import pytest

from trellis.analytics.hybrid_ad import (
    HybridCorrelationStructureRequest,
    HybridDerivativeRequest,
    differentiate_quanto_scalar_correlation,
    differentiate_quanto_scalar_inputs,
    fail_closed_correlation_structure_derivative,
)
from trellis.core.differentiable import get_backend_capabilities
from trellis.analytics.risk_factors import RiskFactorId
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.analytical.quanto import price_quanto_option_raw
from trellis.models.resolution.quanto import resolve_quanto_inputs
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)


@dataclass(frozen=True)
class _QuantoSpec:
    notional: float = 2_000_000.0
    strike: float = 100.0
    expiry_date: date = date(2025, 11, 15)
    fx_pair: str = "EURUSD"
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    quanto_correlation_key: str | None = "sx5e_eurusd"


def _market_state(corr: float = 0.25) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"sx5e_eurusd": {"kind": "explicit", "value": corr}},
        selected_curve_names={"discount_curve": "USD-OIS", "forecast_curve": "EUR-DISC"},
    )


def _resolved(corr: float = 0.25):
    return resolve_quanto_inputs(
        _market_state(corr),
        _QuantoSpec(),
        include_hybrid_factor_graph=True,
    )


def test_quanto_scalar_correlation_vjp_matches_constrained_finite_difference():
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    bump = 1.0e-5

    result = differentiate_quanto_scalar_correlation(spec, resolved)
    factor = tuple(result.risk_vector)[0]
    finite_difference = (
        price_quanto_option_raw(spec, replace(resolved, corr=0.25 + bump))
        - price_quanto_option_raw(spec, replace(resolved, corr=0.25 - bump))
    ) / (2.0 * bump)

    assert result.support_status == "supported"
    assert result.value == pytest.approx(price_quanto_option_raw(spec, resolved))
    assert result.method_metadata["resolved_derivative_method"] == "hybrid_scalar_vjp"
    assert result.method_metadata["coordinate_space"] == "constrained"
    assert result.method_metadata["hybrid_factor_graph_id"] == "quanto:EUR:USD:EURUSD"
    assert result.risk_vector[factor] == pytest.approx(
        finite_difference,
        rel=5.0e-6,
        abs=1.0e-6,
    )


def _factor_by(
    result,
    *,
    object_type: str,
    coordinate_type: str,
    object_name: str | None = None,
    currency: str | None = None,
    tenor_years: float | None = None,
):
    for factor in result.risk_vector:
        if factor.object_type != object_type:
            continue
        if factor.coordinate_type != coordinate_type:
            continue
        if object_name is not None and factor.object_name != object_name:
            continue
        if currency is not None and factor.currency != currency:
            continue
        axes = dict(factor.axes)
        if tenor_years is not None and axes.get("tenor_years") != format(tenor_years, ".12g"):
            continue
        return factor
    raise AssertionError(
        f"missing factor {object_type=} {coordinate_type=} {object_name=} {tenor_years=}"
    )


def test_quanto_scalar_inputs_vjp_matches_finite_differences_for_graph_factors():
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    result = differentiate_quanto_scalar_inputs(spec, resolved)
    bump = 1.0e-5

    assert result.support_status == "supported"
    assert result.method_metadata["resolved_derivative_method"] == "hybrid_scalar_vector_vjp"
    assert result.method_metadata["factor_count"] >= 6

    spot_factor = _factor_by(result, object_type="spot", coordinate_type="spot", object_name="EUR")
    spot_fd = (
        price_quanto_option_raw(spec, replace(resolved, spot=resolved.spot + bump))
        - price_quanto_option_raw(spec, replace(resolved, spot=resolved.spot - bump))
    ) / (2.0 * bump)
    assert result.risk_vector[spot_factor] == pytest.approx(spot_fd, rel=5.0e-6)

    domestic_curve_factor = _factor_by(
        result,
        object_type="curve",
        coordinate_type="zero_rate",
        object_name="USD-OIS",
        tenor_years=0.0,
    )
    domestic_up = YieldCurve((0.0, 30.0), (0.05 + bump, 0.05)).discount(resolved.T)
    domestic_down = YieldCurve((0.0, 30.0), (0.05 - bump, 0.05)).discount(resolved.T)
    domestic_fd = (
        price_quanto_option_raw(spec, replace(resolved, domestic_df=domestic_up))
        - price_quanto_option_raw(spec, replace(resolved, domestic_df=domestic_down))
    ) / (2.0 * bump)
    assert result.risk_vector[domestic_curve_factor] == pytest.approx(
        domestic_fd,
        rel=5.0e-6,
        abs=1.0e-5,
    )

    foreign_curve_factor = _factor_by(
        result,
        object_type="curve",
        coordinate_type="zero_rate",
        object_name="EUR-DISC",
        tenor_years=0.0,
    )
    foreign_up = YieldCurve((0.0, 30.0), (0.03 + bump, 0.03)).discount(resolved.T)
    foreign_down = YieldCurve((0.0, 30.0), (0.03 - bump, 0.03)).discount(resolved.T)
    foreign_fd = (
        price_quanto_option_raw(spec, replace(resolved, foreign_df=foreign_up))
        - price_quanto_option_raw(spec, replace(resolved, foreign_df=foreign_down))
    ) / (2.0 * bump)
    assert result.risk_vector[foreign_curve_factor] == pytest.approx(
        foreign_fd,
        rel=5.0e-6,
        abs=1.0e-5,
    )

    underlier_vol_factor = _factor_by(
        result,
        object_type="vol_surface",
        coordinate_type="flat_vol",
        object_name="vol_surface",
        currency="EUR",
    )
    fx_vol_factor = _factor_by(
        result,
        object_type="vol_surface",
        coordinate_type="flat_vol",
        object_name="vol_surface",
        currency="USD",
    )
    vol_fd = (
        price_quanto_option_raw(
            spec,
            replace(
                resolved,
                sigma_underlier=resolved.sigma_underlier + bump,
                sigma_fx=resolved.sigma_fx + bump,
            ),
        )
        - price_quanto_option_raw(
            spec,
            replace(
                resolved,
                sigma_underlier=resolved.sigma_underlier - bump,
                sigma_fx=resolved.sigma_fx - bump,
            ),
        )
    ) / (2.0 * bump)
    assert (
        result.risk_vector[underlier_vol_factor] + result.risk_vector[fx_vol_factor]
    ) == pytest.approx(vol_fd, rel=5.0e-6)

    corr_factor = _factor_by(
        result,
        object_type="model_parameter",
        coordinate_type="correlation",
        object_name="sx5e_eurusd",
    )
    corr_fd = (
        price_quanto_option_raw(spec, replace(resolved, corr=resolved.corr + bump))
        - price_quanto_option_raw(spec, replace(resolved, corr=resolved.corr - bump))
    ) / (2.0 * bump)
    assert result.risk_vector[corr_factor] == pytest.approx(corr_fd, rel=5.0e-6)


def test_quanto_unconstrained_tanh_coordinate_obeys_chain_rule():
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    constrained = differentiate_quanto_scalar_correlation(spec, resolved)
    unconstrained = differentiate_quanto_scalar_correlation(
        spec,
        resolved,
        HybridDerivativeRequest(coordinate_space="unconstrained"),
    )
    factor = tuple(constrained.risk_vector)[0]
    chart_derivative = unconstrained.method_metadata["chart_derivative"]

    x_value = math.atanh(0.25)
    bump = 1.0e-5
    finite_difference_x = (
        price_quanto_option_raw(spec, replace(resolved, corr=math.tanh(x_value + bump)))
        - price_quanto_option_raw(spec, replace(resolved, corr=math.tanh(x_value - bump)))
    ) / (2.0 * bump)

    assert unconstrained.method_metadata["coordinate_space"] == "unconstrained"
    assert unconstrained.risk_vector[factor] == pytest.approx(
        constrained.risk_vector[factor] * chart_derivative,
        rel=1.0e-10,
        abs=1.0e-8,
    )
    assert unconstrained.risk_vector[factor] == pytest.approx(
        finite_difference_x,
        rel=5.0e-6,
        abs=1.0e-6,
    )


def test_quanto_scalar_correlation_vjp_filters_unknown_selected_factor():
    spec = _QuantoSpec()
    selected = RiskFactorId(
        object_type="model_parameter",
        object_name="missing",
        coordinate_type="correlation",
    )

    result = differentiate_quanto_scalar_correlation(
        spec,
        _resolved(0.25),
        HybridDerivativeRequest(selected_factors=(selected,)),
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "selected_factors_unavailable"


def test_quanto_scalar_correlation_jvp_fails_closed_with_backend_reason():
    spec = _QuantoSpec()
    result = differentiate_quanto_scalar_correlation(
        spec,
        _resolved(0.25),
        HybridDerivativeRequest(derivative_method="jvp"),
    )

    assert get_backend_capabilities().supports("jvp") is False
    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "hybrid_jvp_backend_unsupported"
    assert result.diagnostics[0]["backend_id"] == "autograd"
    assert result.method_metadata["backend_operator"] == "jvp"
    assert result.method_metadata["fallback_reason"]["code"] == (
        "hybrid_jvp_backend_unsupported"
    )


def test_correlation_matrix_derivative_request_fails_closed_without_psd_chart():
    result = fail_closed_correlation_structure_derivative(
        HybridCorrelationStructureRequest(
            object_name="sx5e_eurusd_matrix",
            structure_type="correlation_matrix",
            factors=("SX5E", "EURUSD"),
            requested_derivative_method="vjp",
        )
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "correlation_matrix_chart_not_implemented"
    assert result.unsupported_dependencies[0].reason == (
        "correlation_matrix_chart_not_implemented"
    )
    assert result.method_metadata["derivative_method_support"] == "unsupported"
