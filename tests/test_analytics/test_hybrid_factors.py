from __future__ import annotations

import math

import pytest

from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.risk_factors import RiskFactorCoordinate, RiskFactorId


def _spot_coordinate() -> RiskFactorCoordinate:
    return RiskFactorCoordinate(
        factor_id=RiskFactorId(
            object_type="spot",
            object_name="SX5E",
            coordinate_type="spot",
            currency="EUR",
            provenance_namespace="hybrid_ad",
        ),
        object_path="market_state.underlier_spots.SX5E",
        display_name="SX5E spot",
        unit="price",
        transform="identity",
        support_status="supported",
        reporting_buckets={
            "risk_class": "equity",
            "currency": "EUR",
            "object_name": "SX5E",
        },
    )


def _correlation_coordinate() -> RiskFactorCoordinate:
    return RiskFactorCoordinate(
        factor_id=RiskFactorId(
            object_type="model_parameter",
            object_name="sx5e_eurusd",
            coordinate_type="correlation",
            currency="EUR",
            axes={"factor_a": "SX5E", "factor_b": "EURUSD"},
            provenance_namespace="hybrid_ad",
        ),
        object_path="model_parameters.sx5e_eurusd.correlation",
        display_name="SX5E/EURUSD scalar correlation",
        unit="correlation",
        transform="tanh",
        support_status="supported",
        reporting_buckets={
            "risk_class": "hybrid",
            "currency": "EUR",
            "object_name": "sx5e_eurusd",
            "factor_a": "SX5E",
            "factor_b": "EURUSD",
        },
    )


def test_scalar_correlation_chart_round_trips_tanh_coordinate():
    chart = MarketObjectCoordinateChart.scalar_correlation(
        coordinate=_correlation_coordinate(),
        correlation=0.25,
        chart_id="chart:quanto:rho",
    )

    assert chart.chart_type == "tanh_scalar_correlation"
    assert chart.differentiability_class == "smooth"
    assert chart.constrained_value == pytest.approx(0.25)
    assert chart.unconstrained_value == pytest.approx(math.atanh(0.25))
    assert chart.constrained_from_unconstrained(chart.unconstrained_value) == pytest.approx(
        0.25
    )
    assert chart.unconstrained_from_constrained(0.25) == pytest.approx(
        chart.unconstrained_value
    )
    assert chart.derivative_constrained_wrt_unconstrained == pytest.approx(1.0 - 0.25**2)

    payload = chart.to_payload()
    rebuilt = MarketObjectCoordinateChart.from_payload(payload)

    assert rebuilt == chart
    assert payload["coordinate_keys"] == [chart.coordinates[0].factor_id.key]
    assert payload["constraints"]["lower"] == -1.0
    assert payload["constraints"]["upper"] == 1.0


def test_scalar_correlation_chart_rejects_boundary_values():
    with pytest.raises(ValueError, match="strictly inside"):
        MarketObjectCoordinateChart.scalar_correlation(
            coordinate=_correlation_coordinate(),
            correlation=1.0,
        )


def test_correlation_matrix_policy_chart_payload_round_trips():
    chart = MarketObjectCoordinateChart.correlation_matrix_policy(
        object_name="cross_asset_correlation",
        factor_labels=("SX5E", "EURUSD", "USD-OIS"),
        correlation_matrix=(
            (1.0, 0.25, -0.10),
            (0.25, 1.0, 0.35),
            (-0.10, 0.35, 1.0),
        ),
        chart_id="chart:correlation_matrix:cross_asset",
    )

    assert chart.chart_type == "correlation_matrix_psd_policy"
    assert chart.coordinate_space == "matrix"
    assert chart.differentiability_class == "smooth"
    assert chart.support_status == "unsupported"
    assert chart.coordinate_values["dimension"] == 3
    assert chart.coordinate_values["factor_labels"] == ("SX5E", "EURUSD", "USD-OIS")
    assert chart.coordinate_values["correlation_matrix"] == (
        (1.0, 0.25, -0.10),
        (0.25, 1.0, 0.35),
        (-0.10, 0.35, 1.0),
    )
    assert chart.constraints["psd"] is True
    assert chart.constraints["projection_policy"] == "unsupported_no_smoothing_or_projection"
    assert chart.metadata["chart_family"] == "correlation_matrix"
    assert chart.metadata["coordinate_count"] == 3
    assert chart.metadata["min_eigenvalue"] > 0.0
    assert [coordinate.factor_id.axes for coordinate in chart.coordinates] == [
        (("column", "EURUSD"), ("column_index", "1"), ("row", "SX5E"), ("row_index", "0")),
        (("column", "USD-OIS"), ("column_index", "2"), ("row", "EURUSD"), ("row_index", "1")),
        (("column", "USD-OIS"), ("column_index", "2"), ("row", "SX5E"), ("row_index", "0")),
    ]

    payload = chart.to_payload()
    rebuilt = MarketObjectCoordinateChart.from_payload(payload)

    assert rebuilt == chart
    assert payload["coordinate_keys"] == [
        coordinate.factor_id.key for coordinate in chart.coordinates
    ]


def test_correlation_matrix_policy_chart_json_round_trips():
    import json

    chart = MarketObjectCoordinateChart.correlation_matrix_policy(
        object_name="cross_asset_correlation",
        factor_labels=("SX5E", "EURUSD", "USD-OIS"),
        correlation_matrix=(
            (1.0, 0.25, -0.10),
            (0.25, 1.0, 0.35),
            (-0.10, 0.35, 1.0),
        ),
        chart_id="chart:correlation_matrix:cross_asset",
    )

    rebuilt = MarketObjectCoordinateChart.from_payload(json.loads(json.dumps(chart.to_payload())))
    assert rebuilt == chart
    assert rebuilt.coordinate_values["factor_labels"] == ("SX5E", "EURUSD", "USD-OIS")
    assert rebuilt.coordinate_values["correlation_matrix"] == (
        (1.0, 0.25, -0.10),
        (0.25, 1.0, 0.35),
        (-0.10, 0.35, 1.0),
    )
    assert rebuilt.constraints["bounds"] == (-1.0, 1.0)


def test_correlation_matrix_policy_chart_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="unique"):
        MarketObjectCoordinateChart.correlation_matrix_policy(
            object_name="cross_asset_correlation",
            factor_labels=("SX5E", "SX5E"),
            correlation_matrix=((1.0, 0.25), (0.25, 1.0)),
        )


@pytest.mark.parametrize(
    ("matrix", "match"),
    (
        (((1.0, 0.25, 0.10), (0.25, 1.0, 0.20)), "square"),
        (((1.0, 0.25), (0.25, 0.99)), "unit diagonal"),
        (((1.0, 0.25), (0.30, 1.0)), "symmetric"),
        (((1.0, 1.20), (1.20, 1.0)), "inside \\[-1, 1\\]"),
        (((1.0, 0.95, 0.95), (0.95, 1.0, -0.95), (0.95, -0.95, 1.0)), "positive semidefinite"),
    ),
)
def test_correlation_matrix_policy_chart_rejects_invalid_matrices(matrix, match):
    labels = tuple(f"factor_{index}" for index in range(len(matrix)))
    with pytest.raises(ValueError, match=match):
        MarketObjectCoordinateChart.correlation_matrix_policy(
            object_name="cross_asset_correlation",
            factor_labels=labels,
            correlation_matrix=matrix,
        )


def test_hybrid_factor_graph_collects_coordinates_and_dependencies():
    spot_chart = MarketObjectCoordinateChart.identity(
        chart_id="chart:spot:SX5E",
        object_type="spot",
        object_name="SX5E",
        coordinates=(_spot_coordinate(),),
        coordinate_values={"spot": 100.0},
    )
    corr_chart = MarketObjectCoordinateChart.scalar_correlation(
        coordinate=_correlation_coordinate(),
        correlation=0.25,
        chart_id="chart:correlation:sx5e_eurusd",
    )
    spot_node = HybridDependencyNode(
        node_id="node:spot:SX5E",
        node_type="spot",
        object_name="SX5E",
        coordinate_chart=spot_chart,
        derivative_method="vjp",
    )
    corr_node = HybridDependencyNode(
        node_id="node:correlation:sx5e_eurusd",
        node_type="correlation",
        object_name="sx5e_eurusd",
        coordinate_chart=corr_chart,
        upstream_node_ids=("node:spot:SX5E",),
        derivative_method="vjp",
    )
    unsupported = HybridUnsupportedDependency(
        dependency_id="node:correlation:matrix",
        node_type="correlation_matrix",
        object_name="cross_asset_matrix",
        reason="correlation_matrix_chart_not_implemented",
    )

    graph = HybridFactorGraph(
        graph_id="quanto:SX5E:EURUSD",
        nodes=(corr_node, spot_node),
        unsupported_dependencies=(unsupported,),
        metadata={"route": "bounded_quanto"},
    )

    assert graph.node_ids == ("node:correlation:sx5e_eurusd", "node:spot:SX5E")
    assert graph.node_by_id("node:spot:SX5E") == spot_node
    assert graph.coordinates == (corr_chart.coordinates[0], spot_chart.coordinates[0])
    assert graph.coordinate_keys == tuple(
        coordinate.factor_id.key for coordinate in graph.coordinates
    )
    assert graph.unsupported_reasons == ("correlation_matrix_chart_not_implemented",)

    payload = graph.to_payload()
    rebuilt = HybridFactorGraph.from_payload(payload)

    assert rebuilt == graph
    assert payload["metadata"] == {"route": "bounded_quanto"}
    assert payload["unsupported_dependencies"][0]["support_status"] == "unsupported"


def test_hybrid_dependency_node_rejects_unknown_upstream_reference():
    graph = HybridFactorGraph(
        graph_id="bad",
        nodes=(
            HybridDependencyNode(
                node_id="node:correlation:sx5e_eurusd",
                node_type="correlation",
                object_name="sx5e_eurusd",
                upstream_node_ids=("node:missing",),
            ),
        ),
    )

    with pytest.raises(KeyError, match="node:missing"):
        graph.validate()
