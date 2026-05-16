"""Tests for semantic hybrid-AD lane admission."""

from __future__ import annotations

import json
from datetime import date

from trellis.agent.contract_ir import (
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContractIR,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    Max,
    Observation,
    Singleton,
    Spot,
    Strike,
    Sub,
    Underlying,
)
from trellis.analytics import (
    HybridADFactorRequirement,
    HybridADLaneAdmission,
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


def test_matrix_correlation_structure_stays_planned_and_fail_closed():
    admission = admit_hybrid_ad_lane(
        _terminal_call_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
        correlation_structure="correlation_matrix",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "correlation_matrix_derivative_not_implemented"
    assert admission.factor_requirements[0].parameterization == "correlation_matrix_psd_policy"
    assert admission.diagnostics[0]["code"] == "correlation_matrix_derivative_not_implemented"
    assert admission.metadata["chart_policy_status"] == "validated_fail_closed"


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


def test_path_dependent_contract_shape_is_classified_as_planned():
    admission = admit_hybrid_ad_lane(
        _path_dependent_ir(),
        product_family="quanto_option",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "path_dependent_hybrid_state_pending"
    assert admission.contract_shape == "path_dependent_hybrid_state"


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
