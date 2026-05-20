from __future__ import annotations

import pytest

from trellis.analytics.hybrid_ad import HybridDerivativeResult
from trellis.analytics.hybrid_ad_admission import HybridADLaneAdmission
from trellis.analytics.hybrid_ad_multi_product import (
    HybridADMultiProductLaneResult,
    HybridADMultiProductRequest,
    HybridADMultiProductResult,
    aggregate_hybrid_ad_lane_results,
)
from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    SparseRiskVector,
)


def _factor(object_name: str, coordinate_type: str = "flat_vol") -> RiskFactorId:
    return RiskFactorId(
        object_type="vol_surface",
        object_name=object_name,
        coordinate_type=coordinate_type,
        currency="USD",
        provenance_namespace="hybrid_ad",
    )


def _coordinate(object_name: str) -> RiskFactorCoordinate:
    return RiskFactorCoordinate(
        factor_id=_factor(object_name),
        object_path=f"market_state.vol_surface.{object_name}",
        display_name=f"{object_name} flat vol",
        unit="vol",
        reporting_buckets={"risk_class": "equity_vol", "object_name": object_name},
    )


def _supported_graph(object_name: str) -> HybridFactorGraph:
    coordinate = _coordinate(object_name)
    chart = MarketObjectCoordinateChart.identity(
        chart_id=f"chart:vol:{object_name}",
        object_type="vol_surface",
        object_name=object_name,
        coordinates=(coordinate,),
        coordinate_values={"vol": 0.20},
        metadata={"chart_family": "flat_vol"},
    )
    return HybridFactorGraph(
        graph_id=f"hybrid:test:{object_name}",
        nodes=(
            HybridDependencyNode(
                node_id=f"node:vol:{object_name}",
                node_type="vol_surface",
                object_name=object_name,
                coordinate_chart=chart,
                derivative_method="vjp",
                support_status="supported",
            ),
        ),
        metadata={"fixture": "multi_product"},
    )


def _supported_result(object_name: str, value: float, sensitivity: float) -> HybridDerivativeResult:
    graph = _supported_graph(object_name)
    factor = graph.coordinates[0].factor_id
    return HybridDerivativeResult(
        value=value,
        risk_vector=SparseRiskVector.from_items(((factor, sensitivity),)),
        graph=graph,
        support_status="supported",
        method_metadata={
            "resolved_derivative_method": "hybrid_path_summary_vjp",
            "backend_operator": "vjp",
            "hybrid_factor_graph_id": graph.graph_id,
        },
    )


def _unsupported_result() -> HybridDerivativeResult:
    dependency = HybridUnsupportedDependency(
        dependency_id="node:event:barrier",
        node_type="event_monitor",
        object_name="barrier_monitor",
        reason="discontinuous_event_monitor_unsupported",
        metadata={"event_type": "barrier"},
    )
    graph = HybridFactorGraph(
        graph_id="hybrid:test:unsupported",
        unsupported_dependencies=(dependency,),
        metadata={"fixture": "unsupported"},
    )
    return HybridDerivativeResult(
        value=None,
        risk_vector=SparseRiskVector(),
        graph=graph,
        support_status="unsupported",
        method_metadata={
            "resolved_derivative_method": "unsupported_hybrid_structure",
            "fallback_reason": {"code": "discontinuous_event_monitor_unsupported"},
        },
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=(
            {
                "code": "discontinuous_event_monitor_unsupported",
                "severity": "warning",
            },
        ),
    )


def _admission(lane_id: str, product_family: str) -> HybridADLaneAdmission:
    return HybridADLaneAdmission(
        admitted=True,
        lane_id=lane_id,
        support_status="supported",
        reason="supported",
        semantic_contract_type="ContractIR",
        product_family=product_family,
        contract_shape=product_family,
        derivative_methods=("vjp",),
    )


def test_multi_product_lane_result_preserves_supported_lane_payload_round_trip():
    result = _supported_result("spx_flat", value=12.5, sensitivity=4.25)
    admission = _admission("arithmetic_asian_path_summary_vjp", "arithmetic_asian_option")
    lane = HybridADMultiProductLaneResult(
        lane_id="lane:asian",
        position_name="asian_call",
        product_family="arithmetic_asian_option",
        requested_derivative_method="vjp",
        quantity=3.0,
        derivative_result=result,
        semantic_admission=admission,
        metadata={"desk": "equity"},
    )

    assert lane.support_status == "supported"
    assert lane.value_contribution == pytest.approx(37.5)
    assert lane.risk_vector_contribution[result.graph.coordinates[0].factor_id] == pytest.approx(
        12.75
    )

    payload = lane.to_payload()
    rebuilt = HybridADMultiProductLaneResult.from_payload(payload)

    assert rebuilt == lane
    assert payload["semantic_admission"]["lane_id"] == "arithmetic_asian_path_summary_vjp"
    assert payload["derivative_result"]["risk_vector"]["values"][0]["sensitivity"] == 4.25


def test_multi_product_result_records_supported_and_unsupported_lanes():
    supported = HybridADMultiProductLaneResult(
        lane_id="lane:early_exercise",
        position_name="american_put",
        product_family="american_vanilla_option",
        requested_derivative_method="vjp",
        derivative_result=_supported_result("spx_early", value=8.0, sensitivity=-1.5),
        semantic_admission=_admission(
            "early_exercise_smooth_interior_vjp",
            "american_vanilla_option",
        ),
    )
    unsupported = HybridADMultiProductLaneResult(
        lane_id="lane:barrier",
        position_name="barrier_event",
        product_family="barrier_option",
        requested_derivative_method="vjp",
        derivative_result=_unsupported_result(),
        metadata={"reason": "event_monitor"},
    )
    request = HybridADMultiProductRequest(
        request_id="mixed_fixture",
        derivative_method="vjp",
        unsupported_lane_policy="collect_supported",
    )
    multi = HybridADMultiProductResult(
        request=request,
        lane_results=(unsupported, supported),
        metadata={"fixture": "mixed"},
    )

    assert multi.support_status == "partial"
    assert multi.supported_lane_count == 1
    assert multi.unsupported_lane_count == 1
    assert multi.lane_ids == ("lane:barrier", "lane:early_exercise")
    assert multi.diagnostic_codes == ("discontinuous_event_monitor_unsupported",)
    assert multi.value_contribution == pytest.approx(8.0)

    payload = multi.to_payload()
    rebuilt = HybridADMultiProductResult.from_payload(payload)

    assert rebuilt == multi
    assert payload["lane_results"][0]["support_status"] == "unsupported"
    assert payload["unsupported_lane_count"] == 1


def test_multi_product_result_aggregates_supported_lane_values_and_sparse_risk():
    shared_factor = _factor("shared_flat")
    lane_a_result = _supported_result("shared_flat", value=10.0, sensitivity=2.0)
    lane_b_graph = _supported_graph("shared_flat")
    lane_b_result = HybridDerivativeResult(
        value=7.0,
        risk_vector=SparseRiskVector.from_items(((shared_factor, 5.0),)),
        graph=lane_b_graph,
        support_status="supported",
        method_metadata={
            "resolved_derivative_method": "hybrid_early_exercise_vjp",
            "backend_operator": "vjp",
            "hybrid_factor_graph_id": lane_b_graph.graph_id,
        },
    )
    lane_c_result = _supported_result("quanto_corr", value=3.0, sensitivity=-4.0)
    lane_a = HybridADMultiProductLaneResult(
        lane_id="lane:a",
        position_name="asian_call",
        product_family="arithmetic_asian_option",
        requested_derivative_method="vjp",
        quantity=2.0,
        derivative_result=lane_a_result,
    )
    lane_b = HybridADMultiProductLaneResult(
        lane_id="lane:b",
        position_name="american_put",
        product_family="american_vanilla_option",
        requested_derivative_method="vjp",
        quantity=3.0,
        derivative_result=lane_b_result,
    )
    lane_c = HybridADMultiProductLaneResult(
        lane_id="lane:c",
        position_name="quanto_call",
        product_family="terminal_quanto_option",
        requested_derivative_method="vjp",
        quantity=-1.0,
        derivative_result=lane_c_result,
    )

    result = aggregate_hybrid_ad_lane_results(
        HybridADMultiProductRequest(request_id="supported_mix"),
        (lane_c, lane_a, lane_b),
        metadata={"fixture": "supported"},
    )

    assert isinstance(result, HybridADMultiProductResult)
    assert result.support_status == "supported"
    assert result.value_contribution == pytest.approx(38.0)
    assert result.risk_vector[shared_factor] == pytest.approx(19.0)
    assert result.risk_vector[_factor("quanto_corr")] == pytest.approx(4.0)
    assert result.risk_factor_keys == tuple(factor.key for factor in result.risk_vector)
    assert result.to_payload()["risk_vector"] == result.risk_vector.to_payload()
    assert result.to_payload()["lane_results"][0]["lane_id"] == "lane:a"


def test_multi_product_result_fail_closed_policy_suppresses_aggregate_risk():
    supported = HybridADMultiProductLaneResult(
        lane_id="lane:supported",
        position_name="asian_call",
        product_family="arithmetic_asian_option",
        requested_derivative_method="vjp",
        derivative_result=_supported_result("spx_flat", value=10.0, sensitivity=2.0),
    )
    unsupported = HybridADMultiProductLaneResult(
        lane_id="lane:unsupported",
        position_name="event_monitor",
        product_family="barrier_option",
        requested_derivative_method="vjp",
        derivative_result=_unsupported_result(),
    )

    result = aggregate_hybrid_ad_lane_results(
        HybridADMultiProductRequest(
            request_id="strict_mix",
            unsupported_lane_policy="fail_closed",
        ),
        (supported, unsupported),
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.value_contribution is None
    assert result.to_payload()["risk_vector"]["values"] == []


def test_multi_product_request_validates_policy_and_selected_factor_payloads():
    factor = _factor("spx_flat")
    request = HybridADMultiProductRequest(
        request_id="selected_fixture",
        derivative_method="vjp",
        unsupported_lane_policy="fail_closed",
        selected_factors=(factor.to_payload(),),
    )

    assert request.selected_factors == (factor,)
    assert request.to_payload()["selected_factors"][0]["factor_key"] == factor.key

    with pytest.raises(ValueError, match="unsupported_lane_policy"):
        HybridADMultiProductRequest(
            request_id="bad",
            unsupported_lane_policy="ignore_unsupported",
        )
