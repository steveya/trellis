"""Tests for generic calibration dependency graphs."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.credit import (
    CreditHazardCalibrationQuote,
    build_single_name_credit_calibration_problem_ir,
)
from trellis.models.calibration.dependency_graph import (
    CalibrationDependencyCycleError,
    CalibrationDependencyGraph,
    CalibrationDependencyNode,
    compile_calibration_problem_dependency_graph,
    DuplicateCalibrationDependencyNodeError,
    MissingCalibrationDependencyNodeError,
)
from trellis.models.calibration.problem_ir import (
    CalibrationDependencySpec,
    CalibrationMaterializationSpec,
    CalibrationObjectiveSpec,
    CalibrationProblemIR,
    CalibrationTargetSpec,
    CalibrationVariableSpec,
)


SETTLE = date(2024, 11, 15)


def _credit_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.03),
        selected_curve_names={"discount_curve": "usd_ois"},
        market_provenance={"source_kind": "explicit_input", "source_ref": "dependency_graph_unit_test"},
    )


def _credit_problem():
    return build_single_name_credit_calibration_problem_ir(
        (
            CreditHazardCalibrationQuote(1.0, 120.0, "spread", label="spread_1y"),
            CreditHazardCalibrationQuote(5.0, 180.0, "spread", label="spread_5y"),
        ),
        _credit_market_state(),
        recovery=0.4,
        curve_name="benchmark_single_name_credit",
    )


def _dependent_basket_problem(
    *,
    dependency_source_ref: str = "",
    dependency_object_name: str = "benchmark_single_name_credit",
) -> CalibrationProblemIR:
    return CalibrationProblemIR(
        problem_id="basket_credit_tranche_correlation_problem",
        workflow_id="basket_credit_problem_ir_graph_fixture",
        family_id="basket_credit",
        variables=(
            CalibrationVariableSpec(
                name="base_correlation",
                coordinate_chart="bounded_correlation",
                initial_value=0.3,
                lower_bound=0.0,
                upper_bound=0.999,
            ),
        ),
        targets=(
            CalibrationTargetSpec(
                target_id="tranche_0_3",
                instrument_id="basket_credit:tranche_0_3",
                quote_family="spread",
                quote_convention="tranche_running_spread",
                quote_value=100.0,
                validation_tags=("basket_credit", "problem_ir_graph_fixture"),
            ),
        ),
        objective=CalibrationObjectiveSpec(
            objective_kind="least_squares",
            loss_function="weighted_sum_of_squares",
            residual_count=1,
            derivative_method="scipy_2point_residual_jacobian",
            solve_request_id="basket_credit_graph_fixture",
        ),
        dependencies=(
            CalibrationDependencySpec(
                dependency_id="representative_credit_curve",
                object_kind="credit_curve",
                object_name=dependency_object_name,
                source_ref=dependency_source_ref,
            ),
        ),
        materialization=CalibrationMaterializationSpec(
            object_kind="correlation_surface",
            object_name="basket_base_correlation",
            destination_field="model_parameter_sets",
            source_ref="basket_credit_graph_fixture",
        ),
        solve_request_payload={"request_id": "basket_credit_graph_fixture"},
        metadata={"adapter_id": "basket_credit_graph_fixture", "engine_backed": False},
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


def test_problem_ir_dependency_graph_infers_materialized_upstream_edges():
    credit_problem = _credit_problem()
    basket_problem = _dependent_basket_problem()

    compiled = compile_calibration_problem_dependency_graph(
        "basket_credit_problem_ir_program",
        problems=(basket_problem, credit_problem),
    )
    payload = compiled.to_payload()

    assert compiled.problem_ids == (
        "basket_credit_tranche_correlation_problem",
        "single_name_credit_cds_par_spread_least_squares",
    )
    assert compiled.dependency_order() == (
        "single_name_credit_cds_par_spread_least_squares",
        "basket_credit_tranche_correlation_problem",
    )
    assert [problem.problem_id for problem in compiled.topological_problems] == list(compiled.dependency_order())
    assert payload["dependency_graph"]["edges"] == [
        [
            "basket_credit_tranche_correlation_problem",
            "single_name_credit_cds_par_spread_least_squares",
        ],
    ]
    assert payload["problems"][0]["problem_id"] == "basket_credit_tranche_correlation_problem"
    assert payload["problems"][1]["materialization"]["object_kind"] == "credit_curve"


def test_problem_ir_dependency_graph_rejects_missing_problem_source_refs():
    with pytest.raises(MissingCalibrationDependencyNodeError, match="missing target 'missing_credit_problem'"):
        compile_calibration_problem_dependency_graph(
            "missing_problem_ref",
            problems=(
                _dependent_basket_problem(
                    dependency_source_ref="problem:missing_credit_problem",
                    dependency_object_name="missing_credit_curve",
                ),
            ),
        )
