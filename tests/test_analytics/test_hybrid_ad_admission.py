"""Tests for semantic hybrid-AD lane admission."""

from __future__ import annotations

import json
from datetime import date

import trellis.analytics as analytics
from trellis.agent.contract_ir import (
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContinuousInterval,
    ContractIR,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    Gt,
    Indicator,
    Max,
    Observation,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    Underlying,
)
from trellis.agent.dynamic_contract_ir import DynamicContractIR
from trellis.analytics import (
    HybridADFactorRequirement,
    HybridADLaneAdmission,
    HybridADStatePolicy,
    admit_hybrid_ad_lane,
)


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
        payoff=Max((Sub(Spot("EUR"), Strike(100.0)), Constant(0.0))),
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(EquitySpot("EUR", "gbm")),
    )


def _composite_underlying_ir() -> ContractIR:
    schedule = Singleton(EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Spot("EUR"), Strike(100.0)), Constant(0.0))),
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(
            CompositeUnderlying(
                (
                    EquitySpot("EUR", "gbm"),
                    EquitySpot("EURUSD", "gbm"),
                )
            )
        ),
    )


def _path_dependent_ir() -> ContractIR:
    averaging_schedule = FiniteSchedule(OBSERVATIONS)
    return ContractIR(
        payoff=Max(
            (
                Sub(ArithmeticMean(Spot("EUR"), averaging_schedule), Strike(100.0)),
                Constant(0.0),
            )
        ),
        exercise=Exercise("european", Singleton(EXPIRY)),
        observation=Observation("path_dependent", averaging_schedule),
        underlying=Underlying(EquitySpot("EUR", "gbm")),
    )


def _american_call_ir() -> ContractIR:
    observation_schedule = Singleton(EXPIRY)
    exercise_schedule = ContinuousInterval(OBSERVATIONS[0], EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Spot("EUR"), Strike(100.0)), Constant(0.0))),
        exercise=Exercise("american", exercise_schedule),
        observation=Observation("terminal", observation_schedule),
        underlying=Underlying(EquitySpot("EUR", "gbm")),
    )


def _dynamic_hybrid_ir() -> DynamicContractIR:
    return DynamicContractIR(
        base_contract=_terminal_call_ir(),
        semantic_family="autocallable_note",
        base_track="payoff_expression",
    )


def _discontinuous_event_ir() -> ContractIR:
    schedule = Singleton(EXPIRY)
    payoff = Scaled(
        Indicator(Gt(Spot("EUR"), Constant(80.0))),
        Max((Sub(Spot("EUR"), Strike(100.0)), Constant(0.0))),
    )
    return ContractIR(
        payoff=payoff,
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(EquitySpot("EUR", "gbm")),
    )


def test_public_api_exports_hybrid_ad_admission_symbols():
    assert analytics.HybridADFactorRequirement is HybridADFactorRequirement
    assert analytics.HybridADLaneAdmission is HybridADLaneAdmission
    assert analytics.HybridADStatePolicy is HybridADStatePolicy
    assert analytics.admit_hybrid_ad_lane is admit_hybrid_ad_lane
    assert "HybridADFactorRequirement" in analytics.__all__
    assert "HybridADLaneAdmission" in analytics.__all__
    assert "HybridADStatePolicy" in analytics.__all__
    assert "admit_hybrid_ad_lane" in analytics.__all__


def test_state_policy_payload_round_trips_and_validates_members():
    policy = HybridADStatePolicy(
        state_kind="smooth_path_summary",
        support_status="planned",
        differentiability_class="smooth",
        reason="path_dependent_hybrid_state_pending",
        event_policy="sampled_path_summary",
        control_policy="none",
        state_variable_roles=("arithmetic_mean", "underlier_path"),
        metadata={"observation_kind": "path_dependent"},
        diagnostics=(
            {
                "code": "path_dependent_hybrid_state_pending",
                "severity": "warning",
            },
        ),
    )

    payload = policy.to_payload()

    assert policy.supported is False
    assert payload["state_kind"] == "smooth_path_summary"
    assert payload["state_variable_roles"] == ["arithmetic_mean", "underlier_path"]
    assert json.loads(json.dumps(payload)) == payload
    assert HybridADStatePolicy.from_payload(payload) == policy

    try:
        HybridADStatePolicy(
            state_kind="unclassified_path",
            support_status="planned",
            differentiability_class="smooth",
            reason="path_dependent_hybrid_state_pending",
        )
    except ValueError as exc:
        assert "state_kind" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid state_kind should fail validation")


def test_supported_quanto_vjp_admission_payload_round_trips():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
    )

    payload = admission.to_payload()

    assert admission.admitted is True
    assert admission.supported is True
    assert admission.support_status == "supported"
    assert admission.lane_id == "quanto_scalar_graph_vjp"
    assert admission.reason == "supported_quanto_scalar_graph_vjp"
    assert admission.derivative_methods == ("vjp", "hvp")
    assert payload["derivative_method_category"] == "hybrid_ad"
    assert payload["derivative_method_support"] == "supported"
    assert json.loads(json.dumps(payload)) == payload
    assert HybridADLaneAdmission.from_payload(payload) == admission


def test_supported_quanto_hvp_reports_factor_coordinate_requirements():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="single_name_quanto_option",
        derivative_method="hvp",
    )

    requirements = {
        requirement.semantic_role: requirement
        for requirement in admission.factor_requirements
    }

    assert admission.admitted is True
    assert admission.lane_id == "quanto_scalar_graph_hvp"
    assert admission.reason == "supported_quanto_scalar_graph_hvp"
    assert requirements["underlier_spot"].object_type == "spot"
    assert requirements["fx_spot"].object_type == "fx_rate"
    assert requirements["domestic_curve"].coordinate_type == "zero_rate"
    assert requirements["foreign_curve"].coordinate_type == "zero_rate"
    assert requirements["underlier_vol"].object_type == "vol_surface"
    assert requirements["fx_vol"].object_type == "vol_surface"
    assert requirements["scalar_correlation"].coordinate_type == "correlation"
    assert requirements["scalar_correlation"].graph_role == "correlation"


def test_jvp_request_fails_closed_at_admission():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="jvp",
    )

    assert admission.admitted is False
    assert admission.supported is False
    assert admission.support_status == "unsupported"
    assert admission.reason == "hybrid_jvp_backend_unsupported"
    assert admission.diagnostics[0]["code"] == "hybrid_jvp_backend_unsupported"
    assert admission.metadata["requested_derivative_method"] == "jvp"


def test_dynamic_contract_jvp_request_fails_closed_as_backend_unsupported():
    admission = admit_hybrid_ad_lane(
        _dynamic_hybrid_ir(),
        product_family="autocallable_note",
        derivative_method="jvp",
    )

    assert admission.admitted is False
    assert admission.supported is False
    assert admission.support_status == "unsupported"
    assert admission.reason == "hybrid_jvp_backend_unsupported"
    assert admission.semantic_contract_type == "DynamicContractIR"
    assert admission.contract_shape == "dynamic_hybrid_state"
    assert admission.diagnostics[0]["code"] == "hybrid_jvp_backend_unsupported"
    assert admission.metadata["requested_derivative_method"] == "jvp"


def test_unknown_derivative_method_fails_closed_at_admission():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="reverse_mode",
    )

    payload = admission.to_payload()

    assert admission.admitted is False
    assert admission.support_status == "unsupported"
    assert admission.reason == "hybrid_derivative_method_unsupported"
    assert admission.diagnostics[0]["code"] == "hybrid_derivative_method_unsupported"
    assert json.loads(json.dumps(payload)) == payload


def test_matrix_correlation_structure_is_supported_for_terminal_quanto_vjp():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
        correlation_structure="correlation_matrix",
    )

    requirements = {
        requirement.semantic_role: requirement
        for requirement in admission.factor_requirements
    }

    assert admission.admitted is True
    assert admission.supported is True
    assert admission.support_status == "supported"
    assert admission.lane_id == "quanto_matrix_graph_vjp"
    assert admission.reason == "supported_quanto_matrix_graph_vjp"
    assert requirements["cross_factor_dependence"].object_type == "correlation_matrix"
    assert requirements["cross_factor_dependence"].parameterization == (
        "correlation_matrix_psd_policy"
    )
    assert requirements["cross_factor_dependence"].graph_role == "correlation_matrix"
    assert admission.metadata["chart_policy_status"] == "validated_executable"
    assert admission.metadata["runtime_helper"] == (
        "trellis.analytics.hybrid_ad.differentiate_quanto_correlation_matrix"
    )


def test_matrix_correlation_structure_is_supported_for_terminal_quanto_hvp():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="hvp",
        correlation_structure="correlation_matrix",
    )

    assert admission.admitted is True
    assert admission.supported is True
    assert admission.lane_id == "quanto_matrix_graph_hvp"
    assert admission.reason == "supported_quanto_matrix_graph_hvp"
    assert admission.metadata["coordinate_space"] == "matrix"
    assert admission.metadata["projection_policy"] == "unsupported_no_smoothing_or_projection"


def test_unknown_correlation_structure_fails_closed():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
        correlation_structure="cholesky_angle",
    )

    assert admission.admitted is False
    assert admission.support_status == "unsupported"
    assert admission.reason == "unsupported_correlation_structure"
    assert admission.diagnostics[0]["code"] == "unsupported_correlation_structure"
    assert admission.metadata["correlation_structure"] == "cholesky_angle"


def test_surface_correlation_structure_stays_planned_and_fail_closed():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="hvp",
        correlation_structure="correlation_surface",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "correlation_surface_chart_not_implemented"
    assert admission.factor_requirements[0].parameterization == "correlation_surface_policy"
    assert admission.diagnostics[0]["code"] == "correlation_surface_chart_not_implemented"


def test_discontinuous_event_monitor_is_unsupported():
    admission = admit_hybrid_ad_lane(
        _discontinuous_event_ir(),
        product_family="quanto_option",
    )

    state_policy = admission.metadata["state_policy"]

    assert admission.admitted is False
    assert admission.support_status == "unsupported"
    assert admission.reason == "unsupported_discontinuous_event_monitor"
    assert admission.contract_shape == "discontinuous_event_monitor"
    assert state_policy["state_kind"] == "discontinuous_event_monitor"
    assert state_policy["support_status"] == "unsupported"
    assert state_policy["differentiability_class"] == "discontinuous"
    assert state_policy["event_policy"] == "fail_closed_no_smoothing"
    assert state_policy["fail_closed"] is True
    assert state_policy["state_variable_roles"] == [
        "indicator_event",
        "underlier_terminal_state",
    ]


def test_composite_underlying_is_classified_as_planned_not_admitted():
    admission = admit_hybrid_ad_lane(
        _composite_underlying_ir(),
        product_family="quanto_option",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "hybrid_factor_graph_admission_pending"
    assert admission.contract_shape == "hybrid_composite_underlying"
    assert admission.factor_requirements[0].semantic_role == "cross_factor_dependence"


def test_early_exercise_contract_shape_is_classified_as_planned():
    admission = admit_hybrid_ad_lane(
        _american_call_ir(),
        product_family="quanto_option",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "early_exercise_hybrid_state_pending"
    assert admission.contract_shape == "early_exercise_hybrid_state"


def test_path_dependent_contract_shape_is_classified_as_planned():
    admission = admit_hybrid_ad_lane(
        _path_dependent_ir(),
        product_family="quanto_option",
    )

    state_policy = admission.metadata["state_policy"]

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "path_dependent_hybrid_state_pending"
    assert admission.contract_shape == "path_dependent_hybrid_state"
    assert state_policy["state_kind"] == "smooth_path_summary"
    assert state_policy["support_status"] == "planned"
    assert state_policy["differentiability_class"] == "smooth"
    assert state_policy["event_policy"] == "sampled_path_summary"
    assert state_policy["control_policy"] == "none"
    assert state_policy["state_variable_roles"] == [
        "arithmetic_mean",
        "underlier_path",
    ]
    assert state_policy["metadata"]["observation_kind"] == "path_dependent"


def test_factor_requirement_payload_round_trips_directly():
    requirement = HybridADFactorRequirement(
        object_type="model_parameter",
        coordinate_type="correlation",
        risk_class="hybrid",
        parameterization="scalar_correlation",
        semantic_role="scalar_correlation",
        graph_role="correlation",
    )

    assert HybridADFactorRequirement.from_payload(requirement.to_payload()) == requirement
