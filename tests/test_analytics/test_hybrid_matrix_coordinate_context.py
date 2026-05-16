from __future__ import annotations

import json

import pytest

from trellis.analytics import (
    HybridCorrelationStructureRequest,
    HybridMatrixCoordinateContext,
    build_correlation_matrix_coordinate_context,
)


def _matrix_request(
    *,
    matrix: tuple[tuple[float, ...], ...] = (
        (1.0, 0.35, 0.10),
        (0.35, 1.0, -0.20),
        (0.10, -0.20, 1.0),
    ),
    factors: tuple[str, ...] = ("EUR", "EURUSD", "USD-OIS"),
) -> HybridCorrelationStructureRequest:
    return HybridCorrelationStructureRequest(
        object_name="cross_asset_correlation",
        structure_type="correlation_matrix",
        factors=factors,
        requested_derivative_method="vjp",
        correlation_matrix=matrix,
    )


def test_matrix_coordinate_context_marks_well_conditioned_chart_executable() -> None:
    context = build_correlation_matrix_coordinate_context(
        _matrix_request(),
        active_factor_pair=("EUR", "EURUSD"),
        min_eigenvalue_floor=1e-4,
    )

    assert context.support_status == "supported"
    assert context.chart.support_status == "supported"
    assert context.chart.metadata["chart_policy_status"] == "validated_executable"
    assert context.coordinate_count == 3
    assert context.min_eigenvalue >= 1e-4
    assert context.min_eigenvalue_floor == 1e-4
    assert context.factor_labels == ("EUR", "EURUSD", "USD-OIS")

    assert context.active_factor_id is not None
    active_axes = dict(context.active_factor_id.axes)
    assert active_axes["row"] == "EUR"
    assert active_axes["column"] == "EURUSD"
    assert context.coordinate_index_for_pair("EURUSD", "EUR") == context.coordinate_index_for_pair(
        "EUR", "EURUSD"
    )


def test_matrix_coordinate_context_payload_round_trips() -> None:
    context = build_correlation_matrix_coordinate_context(
        _matrix_request(),
        active_factor_pair=("EUR", "EURUSD"),
        min_eigenvalue_floor=1e-4,
    )

    payload = context.to_payload()
    json_payload = json.loads(json.dumps(payload))

    assert HybridMatrixCoordinateContext.from_payload(json_payload) == context


def test_matrix_coordinate_context_fails_closed_near_psd_boundary() -> None:
    near_boundary = (
        (1.0, 0.999999),
        (0.999999, 1.0),
    )

    with pytest.raises(ValueError, match="correlation_matrix_near_psd_boundary"):
        build_correlation_matrix_coordinate_context(
            _matrix_request(matrix=near_boundary, factors=("EUR", "EURUSD")),
            active_factor_pair=("EUR", "EURUSD"),
            min_eigenvalue_floor=1e-4,
        )


def test_matrix_coordinate_context_rejects_missing_active_pair() -> None:
    with pytest.raises(ValueError, match="active_factor_pair_unavailable"):
        build_correlation_matrix_coordinate_context(
            _matrix_request(),
            active_factor_pair=("EUR", "GBPUSD"),
            min_eigenvalue_floor=1e-4,
        )
