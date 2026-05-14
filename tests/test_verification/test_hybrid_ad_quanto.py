from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

import pytest

from trellis.analytics.hybrid_ad import (
    HybridCorrelationStructureRequest,
    HybridDerivativeRequest,
    differentiate_quanto_scalar_correlation,
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
