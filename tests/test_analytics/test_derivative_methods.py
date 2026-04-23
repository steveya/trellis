"""Tests for runtime derivative-method reporting taxonomy."""

from __future__ import annotations

import pytest


def test_derivative_method_taxonomy_covers_runtime_and_matrix_methods():
    from trellis.analytics.derivative_methods import DERIVATIVE_METHODS, derivative_method_payload

    required_methods = {
        "autodiff_scalar_gradient",
        "autodiff_vector_jacobian",
        "finite_difference_vector_jacobian",
        "autodiff_public_curve",
        "autodiff_flat_vol",
        "surface_bucket_bump",
        "surface_parallel_bucket_bump",
        "flat_surface_expanded_bucket_bump",
        "representative_flat_vol_bump",
        "parallel_curve_bump",
        "curve_bucket_bump",
        "bootstrap_quote_bump_rebuild",
        "spot_central_bump",
        "calendar_roll_down_bump",
        "portfolio_aad_vjp",
        "autodiff_pathwise",
        "forward_price_only",
        "unsupported_discontinuous_pathwise",
        "finite_difference_bump_reprice",
        "vol_surface_unavailable",
        "not_applicable_root_scalar",
        "provided_scalar_gradient",
        "provided_vector_jacobian",
        "scipy_internal_finite_difference_gradient",
        "scipy_2point_residual_jacobian",
    }
    assert required_methods <= set(DERIVATIVE_METHODS)
    assert {spec.method_id for spec in DERIVATIVE_METHODS.values()} == set(DERIVATIVE_METHODS)

    allowed_categories = {
        "analytical_autograd",
        "autograd",
        "finite_difference_bump",
        "portfolio_aad",
        "forward",
        "unsupported",
        "unavailable",
        "not_applicable",
        "provided",
    }
    allowed_support = {"supported", "partial", "fallback", "unsupported", "not_applicable"}
    allowed_backend_operators = {None, "grad", "jacobian", "vjp", "hessian_vector_product"}
    for spec in DERIVATIVE_METHODS.values():
        assert spec.category in allowed_categories
        assert spec.support_status in allowed_support
        assert spec.backend_operator in allowed_backend_operators
        if spec.fallback_derivative_method is not None:
            assert spec.fallback_derivative_method in DERIVATIVE_METHODS

    analytical_payload = derivative_method_payload("autodiff_scalar_gradient")
    assert analytical_payload["derivative_method_category"] == "analytical_autograd"
    assert analytical_payload["backend_operator"] == "grad"

    autograd_payload = derivative_method_payload("autodiff_public_curve")
    assert autograd_payload["derivative_method_category"] == "autograd"
    assert autograd_payload["derivative_method_support"] == "supported"

    bump_payload = derivative_method_payload(
        "parallel_curve_bump",
        parameterization="parallel_zero_rate_shift",
        bump_bps=1.0,
        fallback_reason={"code": "autodiff_public_curve_unavailable"},
    )

    assert bump_payload["resolved_derivative_method"] == "parallel_curve_bump"
    assert bump_payload["derivative_method_category"] == "finite_difference_bump"
    assert bump_payload["derivative_method_support"] == "fallback"
    assert bump_payload["fallback_reason"]["code"] == "autodiff_public_curve_unavailable"
    assert bump_payload["warnings"][0]["code"] == "autodiff_public_curve_unavailable"
    assert bump_payload["parameterization"] == "parallel_zero_rate_shift"
    assert bump_payload["bump_bps"] == pytest.approx(1.0)

    unsupported_payload = derivative_method_payload("unsupported_discontinuous_pathwise")
    assert unsupported_payload["derivative_method_category"] == "unsupported"
    assert unsupported_payload["fallback_derivative_method"] == "finite_difference_bump_reprice"

    hybrid_calibration_payload = derivative_method_payload("scipy_2point_residual_jacobian")
    assert hybrid_calibration_payload["derivative_method_category"] == "finite_difference_bump"
    assert hybrid_calibration_payload["derivative_method_support"] == "fallback"
    assert hybrid_calibration_payload["fallback_derivative_method"] == (
        "scipy_2point_residual_jacobian"
    )
    assert "backend_operator" not in hybrid_calibration_payload


def test_derivative_method_payload_rejects_unknown_method_ids():
    from trellis.analytics.derivative_methods import derivative_method_payload

    with pytest.raises(ValueError, match="unknown derivative method"):
        derivative_method_payload("new_route_local_string")
