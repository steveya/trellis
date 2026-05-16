from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pytest

from trellis.analytics import (
    HybridCorrelationStructureRequest,
    HybridDerivativeRequest,
    differentiate_quanto_correlation_matrix,
)
from trellis.analytics.risk_factors import RiskFactorId, SparseRiskVector
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


def _matrix_request(corr: float = 0.25) -> HybridCorrelationStructureRequest:
    return HybridCorrelationStructureRequest(
        object_name="cross_asset_correlation",
        structure_type="correlation_matrix",
        factors=("EUR", "EURUSD", "USD-OIS"),
        requested_derivative_method="vjp",
        correlation_matrix=(
            (1.0, corr, 0.10),
            (corr, 1.0, -0.20),
            (0.10, -0.20, 1.0),
        ),
    )


def _matrix_factor_by_pair(result, factor_a: str, factor_b: str) -> RiskFactorId:
    target = frozenset((factor_a, factor_b))
    for coordinate in result.graph.coordinates:
        factor = coordinate.factor_id
        axes = dict(factor.axes)
        if factor.object_type == "correlation_matrix" and frozenset(
            (axes.get("row", ""), axes.get("column", ""))
        ) == target:
            return factor
    raise AssertionError(f"missing matrix factor for {factor_a}/{factor_b}")


def test_quanto_correlation_matrix_vjp_matches_active_entry_finite_difference() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    result = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
    )
    active_factor = _matrix_factor_by_pair(result, "EUR", "EURUSD")
    bump = 1.0e-5
    finite_difference = (
        price_quanto_option_raw(spec, replace(resolved, corr=resolved.corr + bump))
        - price_quanto_option_raw(spec, replace(resolved, corr=resolved.corr - bump))
    ) / (2.0 * bump)

    assert result.support_status == "supported"
    assert result.value == pytest.approx(price_quanto_option_raw(spec, resolved))
    assert result.diagnostics == ()
    assert result.method_metadata["resolved_derivative_method"] == "hybrid_matrix_vector_vjp"
    assert result.method_metadata["coordinate_space"] == "matrix"
    assert result.method_metadata["matrix_coordinate_count"] == 3
    assert result.method_metadata["factor_count"] == 3
    assert result.method_metadata["active_factor_key"] == active_factor.key
    assert result.graph.nodes[0].coordinate_chart.support_status == "supported"
    assert result.risk_vector[active_factor] == pytest.approx(
        finite_difference,
        rel=5.0e-6,
        abs=1.0e-6,
    )


def test_quanto_correlation_matrix_vjp_selected_factors_keep_full_metadata() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    full = differentiate_quanto_correlation_matrix(spec, resolved, _matrix_request(0.25))
    active_factor = _matrix_factor_by_pair(full, "EUR", "EURUSD")
    zero_factor = _matrix_factor_by_pair(full, "EUR", "USD-OIS")

    selected_zero = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(selected_factors=(zero_factor,)),
    )
    selected_active = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(selected_factors=(active_factor,)),
    )

    assert selected_zero.support_status == "supported"
    assert len(selected_zero.risk_vector) == 0
    assert selected_zero.diagnostics == ()
    assert selected_zero.method_metadata["factor_count"] == full.method_metadata["factor_count"]
    assert selected_zero.method_metadata["matrix_coordinate_count"] == 3

    assert selected_active.support_status == "supported"
    assert set(selected_active.risk_vector) == {active_factor}
    assert selected_active.risk_vector[active_factor] == full.risk_vector[active_factor]
    assert selected_active.method_metadata["factor_count"] == 3


def test_quanto_correlation_matrix_vjp_missing_selected_factor_is_partial() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    full = differentiate_quanto_correlation_matrix(spec, resolved, _matrix_request(0.25))
    active_factor = _matrix_factor_by_pair(full, "EUR", "EURUSD")
    missing_factor = RiskFactorId(
        object_type="correlation_matrix",
        object_name="missing",
        coordinate_type="correlation",
        axes={"row": "EUR", "column": "GBPUSD"},
        provenance_namespace="hybrid_ad",
    )

    partial = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(selected_factors=(active_factor, missing_factor)),
    )

    assert partial.support_status == "partial"
    assert set(partial.risk_vector) == {active_factor}
    assert partial.diagnostics[0]["code"] == "selected_factors_unavailable"
    assert partial.diagnostics[0]["missing_factor_keys"] == [missing_factor.key]


def test_quanto_correlation_matrix_vjp_fails_closed_near_psd_boundary() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.999999)
    request = HybridCorrelationStructureRequest(
        object_name="cross_asset_correlation",
        structure_type="correlation_matrix",
        factors=("EUR", "EURUSD"),
        requested_derivative_method="vjp",
        correlation_matrix=(
            (1.0, 0.999999),
            (0.999999, 1.0),
        ),
    )

    result = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        request,
        min_eigenvalue_floor=1e-4,
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "correlation_matrix_near_psd_boundary"
    assert result.method_metadata["fallback_reason"]["code"] == (
        "correlation_matrix_near_psd_boundary"
    )


def test_quanto_correlation_matrix_jvp_fails_closed() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.25)

    result = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(derivative_method="jvp"),
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "hybrid_jvp_backend_unsupported"


def test_quanto_correlation_matrix_hvp_matches_finite_difference_of_vjp() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    base_vjp = differentiate_quanto_correlation_matrix(spec, resolved, _matrix_request(0.25))
    active_factor = _matrix_factor_by_pair(base_vjp, "EUR", "EURUSD")
    direction = SparseRiskVector.from_items(((active_factor, 1.0),))

    hvp = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(
            derivative_method="hvp",
            hvp_direction=direction,
        ),
    )
    bump = 1.0e-4
    up_vjp = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25 + bump),
    )
    down_vjp = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25 - bump),
    )
    finite_difference = (
        up_vjp.risk_vector[active_factor] - down_vjp.risk_vector[active_factor]
    ) / (2.0 * bump)

    assert hvp.support_status == "supported"
    assert hvp.method_metadata["resolved_derivative_method"] == "hybrid_matrix_vector_hvp"
    assert hvp.method_metadata["backend_operator"] == "hessian_vector_product"
    assert hvp.method_metadata["matrix_coordinate_count"] == 3
    assert hvp.method_metadata["hvp_direction_factor_count"] == 1
    assert hvp.risk_vector[active_factor] == pytest.approx(
        finite_difference,
        rel=5.0e-5,
        abs=1.0e-5,
    )


def test_quanto_correlation_matrix_hvp_direction_failures_are_typed() -> None:
    spec = _QuantoSpec()
    resolved = _resolved(0.25)
    full = differentiate_quanto_correlation_matrix(spec, resolved, _matrix_request(0.25))
    missing_factor = RiskFactorId(
        object_type="correlation_matrix",
        object_name="missing",
        coordinate_type="correlation",
        axes={"row": "EUR", "column": "GBPUSD"},
        provenance_namespace="hybrid_ad",
    )

    empty = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(derivative_method="hvp"),
    )
    unknown = differentiate_quanto_correlation_matrix(
        spec,
        resolved,
        _matrix_request(0.25),
        HybridDerivativeRequest(
            derivative_method="hvp",
            hvp_direction=SparseRiskVector.from_items(((missing_factor, 1.0),)),
        ),
    )

    assert full.support_status == "supported"
    assert empty.support_status == "unsupported"
    assert len(empty.risk_vector) == 0
    assert empty.diagnostics[0]["code"] == "hvp_direction_required"

    assert unknown.support_status == "unsupported"
    assert len(unknown.risk_vector) == 0
    assert unknown.diagnostics[0]["code"] == "hvp_direction_factors_unavailable"
    assert unknown.diagnostics[0]["missing_factor_keys"] == [missing_factor.key]
