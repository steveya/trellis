"""Tests for generic calibration dependency graphs."""

from __future__ import annotations

import pytest

from trellis.models.calibration.dependency_graph import (
    CalibrationDependencyCycleError,
    CalibrationDependencyGraph,
    CalibrationDependencyNode,
    DuplicateCalibrationDependencyNodeError,
    MissingCalibrationDependencyNodeError,
)


def test_dependency_graph_orders_dependencies_before_dependents():
    graph = CalibrationDependencyGraph(
        workflow_id="demo_workflow",
        nodes=(
            CalibrationDependencyNode(
                node_id="discount_curve",
                object_kind="curve",
                object_name="USD OIS",
                source_ref="market_snapshot",
            ),
            CalibrationDependencyNode(
                node_id="vol_surface",
                object_kind="surface",
                object_name="EQ Smile",
                depends_on=("discount_curve",),
                source_ref="market_snapshot",
            ),
            CalibrationDependencyNode(
                node_id="final_report",
                object_kind="report",
                object_name="Calibration Summary",
                depends_on=("vol_surface", "discount_curve"),
                source_ref="workflow_state",
            ),
        ),
        edges=(("final_report", "vol_surface"),),
    )

    assert graph.dependency_order() == ("discount_curve", "vol_surface", "final_report")
    assert [node.node_id for node in graph.topological_order] == [
        "discount_curve",
        "vol_surface",
        "final_report",
    ]
    assert graph.detect_cycle() is None
    assert graph.to_payload() == {
        "workflow_id": "demo_workflow",
        "nodes": [
            {
                "node_id": "discount_curve",
                "object_kind": "curve",
                "object_name": "USD OIS",
                "source_ref": "market_snapshot",
                "required": True,
                "depends_on": [],
                "description": "",
                "metadata": {},
            },
            {
                "node_id": "vol_surface",
                "object_kind": "surface",
                "object_name": "EQ Smile",
                "source_ref": "market_snapshot",
                "required": True,
                "depends_on": ["discount_curve"],
                "description": "",
                "metadata": {},
            },
            {
                "node_id": "final_report",
                "object_kind": "report",
                "object_name": "Calibration Summary",
                "source_ref": "workflow_state",
                "required": True,
                "depends_on": ["vol_surface", "discount_curve"],
                "description": "",
                "metadata": {},
            },
        ],
        "edges": [
            ["final_report", "vol_surface"],
            ["vol_surface", "discount_curve"],
            ["final_report", "discount_curve"],
        ],
        "dependency_order": ["discount_curve", "vol_surface", "final_report"],
    }


def test_dependency_node_payload_is_json_friendly_and_normalized():
    node = CalibrationDependencyNode(
        node_id="  report  ",
        object_kind="summary",
        object_name="  final check  ",
        source_ref="  source-123  ",
        required=False,
        depends_on=("  alpha  ", "", "beta"),
        description="  generated payload  ",
        metadata={"level": 3},
    )

    assert node.to_payload() == {
        "node_id": "report",
        "object_kind": "summary",
        "object_name": "final check",
        "source_ref": "source-123",
        "required": False,
        "depends_on": ["alpha", "beta"],
        "description": "generated payload",
        "metadata": {"level": 3},
    }


def test_dependency_node_rejects_null_identifier():
    with pytest.raises(ValueError, match="node_id"):
        CalibrationDependencyNode(
            node_id=None,  # type: ignore[arg-type]
            object_kind="summary",
            object_name="final check",
        )


def test_dependency_graph_rejects_missing_edge_nodes():
    with pytest.raises(MissingCalibrationDependencyNodeError, match="missing source 'missing_source'"):
        CalibrationDependencyGraph(
            workflow_id="missing_source_demo",
            nodes=(
                CalibrationDependencyNode(
                    node_id="discount_curve",
                    object_kind="curve",
                    object_name="USD OIS",
                ),
                CalibrationDependencyNode(
                    node_id="vol_surface",
                    object_kind="surface",
                    object_name="EQ Smile",
                ),
            ),
            edges=(("missing_source", "discount_curve"),),
        )

    with pytest.raises(MissingCalibrationDependencyNodeError, match="missing target 'missing_target'"):
        CalibrationDependencyGraph(
            workflow_id="missing_target_demo",
            nodes=(
                CalibrationDependencyNode(
                    node_id="discount_curve",
                    object_kind="curve",
                    object_name="USD OIS",
                ),
                CalibrationDependencyNode(
                    node_id="vol_surface",
                    object_kind="surface",
                    object_name="EQ Smile",
                ),
            ),
            edges=(("vol_surface", "missing_target"),),
        )


def test_dependency_graph_rejects_duplicate_nodes():
    with pytest.raises(DuplicateCalibrationDependencyNodeError, match="duplicate node_id\\(s\\): 'discount_curve'"):
        CalibrationDependencyGraph(
            workflow_id="duplicate_demo",
            nodes=(
                CalibrationDependencyNode(
                    node_id="discount_curve",
                    object_kind="curve",
                    object_name="USD OIS",
                ),
                CalibrationDependencyNode(
                    node_id="discount_curve",
                    object_kind="curve",
                    object_name="USD SOFR",
                ),
            ),
            edges=(),
        )


def test_dependency_graph_rejects_cycles():
    with pytest.raises(CalibrationDependencyCycleError, match="contains a cycle"):
        CalibrationDependencyGraph(
            workflow_id="cycle_demo",
            nodes=(
                CalibrationDependencyNode(
                    node_id="discount_curve",
                    object_kind="curve",
                    object_name="USD OIS",
                    depends_on=("vol_surface",),
                ),
                CalibrationDependencyNode(
                    node_id="vol_surface",
                    object_kind="surface",
                    object_name="EQ Smile",
                    depends_on=("discount_curve",),
                ),
            ),
            edges=(),
        )
