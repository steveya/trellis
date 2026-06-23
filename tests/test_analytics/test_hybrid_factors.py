from __future__ import annotations

import json
import math

import pytest

from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    RiskFactorRegistry,
)
from trellis.models.vol_surface import GridVolSurface


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


def _grid_vol_coordinates() -> tuple[RiskFactorCoordinate, ...]:
    surface = GridVolSurface(
        expiries=(0.5, 1.0),
        strikes=(90.0, 110.0),
        vols=((0.21, 0.22), (0.23, 0.24)),
    )
    return RiskFactorRegistry().discover_grid_vol_surface(
        surface,
        object_name="spx_grid",
        currency="USD",
        provenance_namespace="hybrid_ad",
        support_status="discovery_only",
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
    assert chart.constrained_from_unconstrained(
        chart.unconstrained_value
    ) == pytest.approx(0.25)
    assert chart.unconstrained_from_constrained(0.25) == pytest.approx(
        chart.unconstrained_value
    )
    assert chart.derivative_constrained_wrt_unconstrained == pytest.approx(
        1.0 - 0.25**2
    )

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
    assert (
        chart.constraints["projection_policy"]
        == "unsupported_no_smoothing_or_projection"
    )
    assert chart.metadata["chart_family"] == "correlation_matrix"
    assert chart.metadata["coordinate_count"] == 3
    assert chart.metadata["min_eigenvalue"] > 0.0
    assert [coordinate.factor_id.axes for coordinate in chart.coordinates] == [
        (
            ("column", "EURUSD"),
            ("column_index", "1"),
            ("row", "SX5E"),
            ("row_index", "0"),
        ),
        (
            ("column", "USD-OIS"),
            ("column_index", "2"),
            ("row", "EURUSD"),
            ("row_index", "1"),
        ),
        (
            ("column", "USD-OIS"),
            ("column_index", "2"),
            ("row", "SX5E"),
            ("row_index", "0"),
        ),
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

    rebuilt = MarketObjectCoordinateChart.from_payload(
        json.loads(json.dumps(chart.to_payload()))
    )
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
        (
            ((1.0, 0.95, 0.95), (0.95, 1.0, -0.95), (0.95, -0.95, 1.0)),
            "positive semidefinite",
        ),
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


def test_correlation_surface_policy_chart_payload_round_trips():
    chart = MarketObjectCoordinateChart.correlation_surface_policy(
        object_name="base_correlation_surface",
        factor_labels=("HY-IG", "HY-XO", "IG9"),
        surface_axes={
            "expiry": ("1Y", "3Y"),
            "detachment": ("3%", "7%"),
        },
        interpolation_basis="exact_node_surface",
        locality_policy="full_surface_node_vector",
        selected_factor_policy="filter_known_fail_closed_unknown",
        metadata={"source": "unit-test"},
    )

    payload = chart.to_payload()
    rebuilt = MarketObjectCoordinateChart.from_payload(json.loads(json.dumps(payload)))

    assert rebuilt == chart
    assert chart.chart_type == "correlation_surface_policy"
    assert chart.coordinate_space == "surface_nodes"
    assert chart.differentiability_class == "piecewise"
    assert chart.support_status == "discovery_only"
    assert chart.coordinate_values["parameterization"] == "surface_node_correlations"
    assert chart.coordinate_values["factor_labels"] == ("HY-IG", "HY-XO", "IG9")
    assert chart.coordinate_values["surface_axes"] == {
        "detachment": ("3%", "7%"),
        "expiry": ("1Y", "3Y"),
    }
    assert chart.coordinate_values["surface_axis_names"] == ("detachment", "expiry")
    assert chart.coordinate_values["factor_pair_count"] == 3
    assert chart.coordinate_values["active_node_count"] == 12
    assert chart.coordinate_values["active_node_keys"] == chart.coordinate_keys
    assert chart.constraints["interpolation_basis"] == "exact_node_surface"
    assert chart.constraints["locality_policy"] == "full_surface_node_vector"
    assert chart.constraints["selected_factor_policy"] == (
        "filter_known_fail_closed_unknown"
    )
    assert chart.constraints["unsupported_selected_factor_reason"] == (
        "unsupported_selected_correlation_surface_factors"
    )
    assert chart.constraints["projection_policy"] == (
        "unsupported_no_smoothing_or_projection"
    )
    assert chart.metadata["chart_family"] == "correlation_surface"
    assert chart.metadata["coordinate_count"] == 12
    assert chart.metadata["source"] == "unit-test"
    assert payload["coordinate_keys"] == list(chart.coordinate_keys)


def test_correlation_surface_policy_chart_coordinates_are_stable():
    chart = MarketObjectCoordinateChart.correlation_surface_policy(
        object_name="base_correlation_surface",
        factor_labels=("underlier", "fx"),
        surface_axes={"expiry": ("1Y", "3Y"), "strike": (90.0, 110.0)},
    )

    assert len(chart.coordinates) == 4
    assert chart.coordinate_keys == tuple(sorted(chart.coordinate_keys))
    coordinate = chart.coordinates[0]

    assert coordinate.factor_id.object_type == "correlation_surface"
    assert coordinate.factor_id.object_name == "base_correlation_surface"
    assert coordinate.factor_id.coordinate_type == "correlation"
    assert coordinate.support_status == "discovery_only"
    assert coordinate.unit == "correlation"
    assert coordinate.transform == "identity"
    assert dict(coordinate.factor_id.axes) == {
        "expiry": "1Y",
        "expiry_index": "0",
        "factor_a": "underlier",
        "factor_a_index": "0",
        "factor_b": "fx",
        "factor_b_index": "1",
        "strike": "110",
        "strike_index": "1",
    }
    assert dict(coordinate.reporting_buckets)["risk_class"] == "hybrid"
    assert dict(coordinate.metadata)["surface_axis_names"] == "expiry,strike"


def test_correlation_surface_policy_rejects_missing_axes():
    with pytest.raises(ValueError, match="surface_axes must contain at least one axis"):
        MarketObjectCoordinateChart.correlation_surface_policy(
            object_name="base_correlation_surface",
            factor_labels=("underlier", "fx"),
            surface_axes={},
        )


def test_correlation_surface_policy_unsupported_dependency_is_typed():
    dependency = HybridUnsupportedDependency.correlation_surface_policy(
        object_name="base_correlation_surface",
        reason="correlation_surface_derivative_not_implemented",
        metadata={"chart_id": "chart:correlation_surface:base_correlation_surface"},
    )

    assert dependency.dependency_id == (
        "unsupported:correlation_surface:base_correlation_surface:"
        "correlation_surface_derivative_not_implemented"
    )
    assert dependency.node_type == "correlation_surface"
    assert dependency.reason == "correlation_surface_derivative_not_implemented"
    assert dependency.differentiability_class == "piecewise"
    assert dependency.metadata["chart_family"] == "correlation_surface"
    assert dependency.metadata["parameterization"] == "surface_node_correlations"
    assert dependency.metadata["chart_id"] == (
        "chart:correlation_surface:base_correlation_surface"
    )


def test_grid_vol_state_control_policy_round_trips_and_orders_nodes():
    coordinates = _grid_vol_coordinates()
    chart = MarketObjectCoordinateChart.grid_vol_state_control_policy(
        object_name="spx_grid",
        lane_family="path_summary",
        coordinates=reversed(coordinates),
        interpolation_basis="bilinear_black_vol",
        locality_policy="full_grid_node_vector",
        selected_factor_policy="filter_known_fail_closed_unknown",
        metadata={"semantic_state_kind": "smooth_path_summary"},
    )

    payload = chart.to_payload()
    rebuilt = MarketObjectCoordinateChart.from_payload(json.loads(json.dumps(payload)))

    assert rebuilt == chart
    assert chart.chart_type == "grid_vol_state_control_policy"
    assert chart.coordinate_space == "grid_nodes"
    assert chart.differentiability_class == "piecewise"
    assert chart.support_status == "discovery_only"
    assert chart.coordinate_keys == tuple(
        sorted(coordinate.factor_id.key for coordinate in coordinates)
    )
    assert chart.coordinate_values["parameterization"] == "grid_node_vols"
    assert chart.coordinate_values["active_node_keys"] == chart.coordinate_keys
    assert chart.constraints["interpolation_basis"] == "bilinear_black_vol"
    assert chart.constraints["locality_policy"] == "full_grid_node_vector"
    assert chart.constraints["selected_factor_policy"] == (
        "filter_known_fail_closed_unknown"
    )
    assert chart.constraints["unsupported_selected_factor_reason"] == (
        "unsupported_selected_grid_vol_factors"
    )
    assert chart.metadata["lane_family"] == "path_summary"
    assert chart.metadata["active_node_count"] == 4
    assert payload["coordinate_keys"] == list(chart.coordinate_keys)


def test_grid_vol_state_control_policy_rejects_non_grid_vol_coordinates():
    with pytest.raises(ValueError, match="grid-vol black-vol coordinates"):
        MarketObjectCoordinateChart.grid_vol_state_control_policy(
            object_name="spx_grid",
            lane_family="path_summary",
            coordinates=(_spot_coordinate(),),
        )


def test_grid_vol_state_control_unsupported_dependency_reasons_are_typed():
    dependencies = tuple(
        HybridUnsupportedDependency.grid_vol_state_control_policy(
            object_name="spx_grid",
            lane_family="early_exercise_control",
            reason=reason,
        )
        for reason in (
            "missing_grid_vol_surface",
            "unsupported_grid_vol_interpolation",
            "unsupported_selected_grid_vol_factors",
            "unsupported_discontinuous_event_monitor",
            "early_exercise_boundary_kink",
        )
    )

    graph = HybridFactorGraph(
        graph_id="grid-vol-policy",
        unsupported_dependencies=reversed(dependencies),
    )
    payload = graph.to_payload()

    assert graph.unsupported_reasons == (
        "early_exercise_boundary_kink",
        "missing_grid_vol_surface",
        "unsupported_discontinuous_event_monitor",
        "unsupported_grid_vol_interpolation",
        "unsupported_selected_grid_vol_factors",
    )
    assert payload["unsupported_dependencies"][0]["dependency_id"] == (
        "unsupported:grid_vol_state_control:spx_grid:" "early_exercise_boundary_kink"
    )
    assert payload["unsupported_dependencies"][0]["metadata"]["lane_family"] == (
        "early_exercise_control"
    )
    assert payload["unsupported_dependencies"][0]["node_type"] == "vol_surface"


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
