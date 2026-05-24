from __future__ import annotations

import pytest

from trellis.analytics.hybrid_ad import (
    HybridDerivativeRequest,
    differentiate_arithmetic_asian_path_summary,
    differentiate_quanto_scalar_inputs,
    differentiate_vanilla_early_exercise,
)
from trellis.analytics.hybrid_ad_admission import admit_hybrid_ad_lane
from trellis.analytics.hybrid_ad_multi_product import (
    HybridADMultiProductLaneResult,
    HybridADMultiProductRequest,
    aggregate_hybrid_ad_lane_results,
)
from trellis.analytics.risk_factors import SparseRiskVector
from tests.test_verification.test_hybrid_ad_early_exercise import (
    _american_put_spec,
    _early_exercise_ir,
    _market_state as _early_exercise_market_state,
)
from tests.test_verification.test_hybrid_ad_path_summary import (
    _asian_spec,
    _market_state as _path_summary_market_state,
    _path_summary_contract_ir,
)
from tests.test_verification.test_hybrid_ad_quanto import (
    _QuantoSpec,
    _resolved,
    _terminal_quanto_contract_ir,
)


def _sum_vectors(*vectors: SparseRiskVector) -> SparseRiskVector:
    total = SparseRiskVector()
    for vector in vectors:
        total = total + vector
    return total


def test_hybrid_ad_multi_product_fixture_aggregates_executable_lane_outputs():
    quanto_admission = admit_hybrid_ad_lane(
        _terminal_quanto_contract_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
    )
    quanto_result = differentiate_quanto_scalar_inputs(
        _QuantoSpec(),
        _resolved(),
        HybridDerivativeRequest(semantic_admission=quanto_admission),
    )
    path_admission = admit_hybrid_ad_lane(
        _path_summary_contract_ir("call"),
        product_family="arithmetic_asian_option",
        derivative_method="vjp",
    )
    path_result = differentiate_arithmetic_asian_path_summary(
        _asian_spec("call"),
        _path_summary_market_state(0.20),
        HybridDerivativeRequest(semantic_admission=path_admission),
        position_name="asian_call",
        vol_surface_name="spx_path_flat",
        currency="USD",
    )
    early_admission = admit_hybrid_ad_lane(
        _early_exercise_ir("american"),
        product_family="american_vanilla_option",
        derivative_method="vjp",
    )
    early_result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _early_exercise_market_state(0.20),
        HybridDerivativeRequest(semantic_admission=early_admission),
        position_name="american_put",
        vol_surface_name="spx_early_flat",
        currency="USD",
    )

    quanto_lane = HybridADMultiProductLaneResult(
        lane_id="lane:terminal_quanto",
        position_name="quanto_call",
        product_family="terminal_quanto_option",
        requested_derivative_method="vjp",
        quantity=1.0,
        derivative_result=quanto_result,
        semantic_admission=quanto_admission,
    )
    path_lane = HybridADMultiProductLaneResult(
        lane_id="lane:path_summary",
        position_name="asian_call",
        product_family="arithmetic_asian_option",
        requested_derivative_method="vjp",
        quantity=2.0,
        derivative_result=path_result,
        semantic_admission=path_admission,
    )
    early_lane = HybridADMultiProductLaneResult(
        lane_id="lane:early_exercise",
        position_name="american_put",
        product_family="american_vanilla_option",
        requested_derivative_method="vjp",
        quantity=-1.0,
        derivative_result=early_result,
        semantic_admission=early_admission,
    )

    aggregate = aggregate_hybrid_ad_lane_results(
        HybridADMultiProductRequest(request_id="executable_multi_product_vjp"),
        (path_lane, early_lane, quanto_lane),
        metadata={"fixture": "terminal_quanto_path_summary_early_exercise"},
    )
    expected_risk = _sum_vectors(
        quanto_result.risk_vector,
        path_result.risk_vector.scale(2.0),
        early_result.risk_vector.scale(-1.0),
    )
    expected_value = (
        quanto_result.value
        + 2.0 * path_result.value
        - early_result.value
    )

    assert aggregate.support_status == "supported"
    assert aggregate.supported_lane_count == 3
    assert aggregate.unsupported_lane_count == 0
    assert aggregate.lane_ids == (
        "lane:early_exercise",
        "lane:path_summary",
        "lane:terminal_quanto",
    )
    assert aggregate.value_contribution == pytest.approx(expected_value)
    assert tuple(aggregate.risk_vector) == tuple(expected_risk)
    for factor, value in expected_risk.items():
        assert aggregate.risk_vector[factor] == pytest.approx(value)
    assert aggregate.to_payload()["risk_vector"] == expected_risk.to_payload()
    assert aggregate.to_payload()["lane_results"][0]["semantic_admission"][
        "support_status"
    ] == "supported"
