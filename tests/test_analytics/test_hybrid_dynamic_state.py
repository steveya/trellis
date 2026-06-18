"""Tests for DynamicContractIR hybrid AD fail-closed runtime lane."""

from __future__ import annotations

from datetime import date

import trellis.analytics as analytics
from trellis.agent.contract_ir import (
    Constant,
    ContinuousInterval,
    ContractIR,
    EquitySpot,
    Exercise,
    Max,
    Observation,
    Singleton,
    Spot,
    Strike,
    Sub,
    Underlying,
)
from trellis.agent.dynamic_contract_ir import DynamicContractIR
from trellis.analytics import (
    HybridDerivativeRequest,
    admit_hybrid_ad_lane,
    fail_closed_dynamic_state_derivative,
)
from trellis.analytics.risk_factors import RiskFactorId


EXPIRY = date(2026, 11, 15)
OBSERVATIONS = (
    date(2026, 2, 15),
    date(2026, 5, 15),
    date(2026, 8, 15),
    EXPIRY,
)


def _terminal_call_ir() -> ContractIR:
    schedule = Singleton(EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Spot("SPX"), Strike(100.0)), Constant(0.0))),
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


def _dynamic_hybrid_ir() -> DynamicContractIR:
    return DynamicContractIR(
        base_contract=_terminal_call_ir(),
        semantic_family="autocallable_note",
        base_track="payoff_expression",
    )


def _american_call_ir() -> ContractIR:
    observation_schedule = Singleton(EXPIRY)
    exercise_schedule = ContinuousInterval(OBSERVATIONS[0], EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Spot("SPX"), Strike(100.0)), Constant(0.0))),
        exercise=Exercise("american", exercise_schedule),
        observation=Observation("terminal", observation_schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


def _dynamic_early_exercise_ir() -> DynamicContractIR:
    return DynamicContractIR(
        base_contract=_american_call_ir(),
        semantic_family="american_vanilla_option",
        base_track="payoff_expression",
    )


def test_public_api_exports_dynamic_state_fail_closed_helper() -> None:
    assert analytics.fail_closed_dynamic_state_derivative is (
        fail_closed_dynamic_state_derivative
    )
    assert "fail_closed_dynamic_state_derivative" in analytics.__all__


def test_dynamic_state_vjp_returns_typed_fail_closed_result() -> None:
    result = fail_closed_dynamic_state_derivative(
        _dynamic_hybrid_ir(),
        value=12.5,
        position_name="autocallable_probe",
        product_family="autocallable_note",
    )

    chart = result.graph.nodes[0].coordinate_chart

    assert result.support_status == "unsupported"
    assert result.value == 12.5
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "dynamic_hybrid_state_admission_pending"
    assert result.graph.graph_id == "hybrid:dynamic_state:autocallable_probe"
    assert result.graph.metadata["semantic_contract_type"] == "DynamicContractIR"
    assert result.graph.metadata["semantic_family"] == "autocallable_note"
    assert result.graph.metadata["base_track"] == "payoff_expression"
    assert result.graph.unsupported_reasons == (
        "dynamic_hybrid_state_admission_pending",
    )
    assert result.unsupported_dependencies[0].reason == (
        "dynamic_hybrid_state_admission_pending"
    )
    assert chart is not None
    assert chart.chart_type == "dynamic_state_policy"
    assert chart.support_status == "discovery_only"
    assert chart.metadata["state_kind"] == "dynamic_state"
    assert result.graph.coordinate_keys == (chart.coordinates[0].factor_id.key,)
    assert result.method_metadata["resolved_derivative_method"] == (
        "unsupported_hybrid_structure"
    )
    assert result.method_metadata["semantic_state_kind"] == "dynamic_state"
    assert result.method_metadata["semantic_state_event_policy"] == (
        "stateful_event_program"
    )
    assert result.method_metadata["semantic_state_control_policy"] == (
        "dynamic_control_fail_closed"
    )
    assert result.method_metadata["semantic_state_fail_closed"] is True


def test_dynamic_state_jvp_fails_closed_with_backend_payload() -> None:
    result = fail_closed_dynamic_state_derivative(
        _dynamic_hybrid_ir(),
        request=HybridDerivativeRequest(derivative_method="jvp"),
        position_name="autocallable_probe",
        product_family="autocallable_note",
    )

    assert result.support_status == "unsupported"
    assert result.diagnostics[0]["code"] == "hybrid_jvp_backend_unsupported"
    assert result.method_metadata["resolved_derivative_method"] == (
        "unsupported_hybrid_jvp"
    )
    assert result.method_metadata["requested_backend_operator"] == "jvp"
    assert result.method_metadata["backend_support"]["supported"] is False
    assert result.method_metadata["semantic_state_kind"] == "dynamic_state"
    assert "backend_operator" not in result.method_metadata


def test_dynamic_state_runtime_uses_supplied_semantic_admission() -> None:
    admission = admit_hybrid_ad_lane(
        _dynamic_hybrid_ir(),
        product_family="structured_note",
        derivative_method="vjp",
    )
    result = fail_closed_dynamic_state_derivative(
        _dynamic_hybrid_ir(),
        request=HybridDerivativeRequest(semantic_admission=admission),
        position_name="autocallable_probe",
        product_family="autocallable_note",
    )

    assert result.method_metadata["semantic_admission"]["product_family"] == (
        "structured_note"
    )
    assert result.graph.metadata["semantic_family"] == "autocallable_note"
    assert result.diagnostics[0]["semantic_admission_reason"] == (
        "dynamic_hybrid_state_admission_pending"
    )


def test_dynamic_state_runtime_rejects_wrong_lane_semantic_admission() -> None:
    wrong_admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
    )

    result = fail_closed_dynamic_state_derivative(
        _dynamic_hybrid_ir(),
        request=HybridDerivativeRequest(semantic_admission=wrong_admission),
        position_name="autocallable_probe",
        product_family="autocallable_note",
    )

    assert result.support_status == "unsupported"
    assert result.diagnostics[0]["code"] == "semantic_admission_lane_unavailable"
    assert result.method_metadata["semantic_admission"]["reason"] == (
        "semantic_admission_lane_unavailable"
    )
    assert result.method_metadata["semantic_admission"]["metadata"][
        "supplied_semantic_admission"
    ]["reason"] == "supported_quanto_scalar_graph_vjp"
    assert result.graph.nodes[0].coordinate_chart.support_status == "unsupported"


def test_dynamic_state_runtime_rejects_other_dynamic_state_control_lane() -> None:
    result = fail_closed_dynamic_state_derivative(
        _dynamic_early_exercise_ir(),
        position_name="american_grid_probe",
        product_family="american_vanilla_option",
        market_parameterization="grid_vol",
    )

    assert result.support_status == "unsupported"
    assert result.diagnostics[0]["code"] == "semantic_admission_lane_unavailable"
    assert result.diagnostics[0]["actual_contract_shape"] == (
        "dynamic_early_exercise_grid_vol_control"
    )
    assert result.method_metadata["semantic_admission"]["contract_shape"] == (
        "dynamic_early_exercise_grid_vol_control"
    )
    assert result.method_metadata["semantic_state_kind"] == "early_exercise_control"
    assert result.graph.nodes == ()
    assert result.graph.metadata["expected_contract_shape"] == "dynamic_hybrid_state"
    assert result.graph.metadata["actual_contract_shape"] == (
        "dynamic_early_exercise_grid_vol_control"
    )


def test_dynamic_state_selected_factor_fail_closed_policy_is_reported() -> None:
    missing_factor = RiskFactorId(
        object_type="dynamic_contract",
        object_name="other_autocallable",
        coordinate_type="dynamic_state_policy",
        provenance_namespace="hybrid_ad",
    )
    request = HybridDerivativeRequest(
        selected_factors=(missing_factor,),
        unsupported_selected_factor_policy="fail_closed",
    )

    result = fail_closed_dynamic_state_derivative(
        _dynamic_hybrid_ir(),
        request=request,
        position_name="autocallable_probe",
        product_family="autocallable_note",
    )

    diagnostic_codes = tuple(diagnostic["code"] for diagnostic in result.diagnostics)

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert "selected_factors_unavailable" in diagnostic_codes
    assert result.diagnostics[-1]["missing_factor_keys"] == [missing_factor.key]
    assert result.diagnostics[-1]["unsupported_selected_factor_policy"] == "fail_closed"
