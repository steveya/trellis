"""Shared derivative-method taxonomy for runtime analytics metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class DerivativeMethodSpec:
    """One documented derivative method reported by runtime or calibration paths."""

    method_id: str
    category: str
    support_status: str
    backend_operator: str | None = None
    fallback_derivative_method: str | None = None
    description: str = ""


_METHOD_SPECS = {
    "autodiff_scalar_gradient": DerivativeMethodSpec(
        "autodiff_scalar_gradient",
        category="analytical_autograd",
        support_status="supported",
        backend_operator="grad",
        description="Scalar gradient through a smooth analytical or calibration objective.",
    ),
    "autodiff_vector_jacobian": DerivativeMethodSpec(
        "autodiff_vector_jacobian",
        category="autograd",
        support_status="supported",
        backend_operator="jacobian",
        description="Dense Jacobian through a smooth vector repricing map.",
    ),
    "finite_difference_vector_jacobian": DerivativeMethodSpec(
        "finite_difference_vector_jacobian",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="finite_difference_vector_jacobian",
        description="Explicit finite-difference vector Jacobian for non-autograd-safe calibration stacks.",
    ),
    "autodiff_public_curve": DerivativeMethodSpec(
        "autodiff_public_curve",
        category="autograd",
        support_status="supported",
        backend_operator="grad",
        description="Public curve node sensitivities through traced curve inputs.",
    ),
    "autodiff_flat_vol": DerivativeMethodSpec(
        "autodiff_flat_vol",
        category="autograd",
        support_status="supported",
        backend_operator="grad",
        description="Flat volatility scalar vega through traced flat-vol input.",
    ),
    "surface_bucket_bump": DerivativeMethodSpec(
        "surface_bucket_bump",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="surface_bucket_bump",
        description="Explicit expiry/strike bucket bump for grid volatility surfaces.",
    ),
    "surface_parallel_bucket_bump": DerivativeMethodSpec(
        "surface_parallel_bucket_bump",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="surface_parallel_bucket_bump",
        description="Parallel grid-vol node bump for scalar vega.",
    ),
    "flat_surface_expanded_bucket_bump": DerivativeMethodSpec(
        "flat_surface_expanded_bucket_bump",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="flat_surface_expanded_bucket_bump",
        description="Bucketed vega expanded from one flat-vol value.",
    ),
    "representative_flat_vol_bump": DerivativeMethodSpec(
        "representative_flat_vol_bump",
        category="finite_difference_bump",
        support_status="fallback",
        fallback_derivative_method="representative_flat_vol_bump",
        description="Representative flat-vol bump used when no explicit surface derivative contract exists.",
    ),
    "parallel_curve_bump": DerivativeMethodSpec(
        "parallel_curve_bump",
        category="finite_difference_bump",
        support_status="fallback",
        fallback_derivative_method="parallel_curve_bump",
        description="Parallel curve bump/reprice fallback for scalar rate risk.",
    ),
    "curve_bucket_bump": DerivativeMethodSpec(
        "curve_bucket_bump",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="curve_bucket_bump",
        description="Bucketed curve bump/reprice for key-rate and scenario risk.",
    ),
    "bootstrap_quote_bump_rebuild": DerivativeMethodSpec(
        "bootstrap_quote_bump_rebuild",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="bootstrap_quote_bump_rebuild",
        description="Bootstrap quote bump with curve rebuild.",
    ),
    "spot_central_bump": DerivativeMethodSpec(
        "spot_central_bump",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="spot_central_bump",
        description="Central spot bump/reprice for spot delta and gamma.",
    ),
    "calendar_roll_down_bump": DerivativeMethodSpec(
        "calendar_roll_down_bump",
        category="finite_difference_bump",
        support_status="supported",
        fallback_derivative_method="calendar_roll_down_bump",
        description="Calendar roll-down repricing for theta.",
    ),
    "portfolio_aad_vjp": DerivativeMethodSpec(
        "portfolio_aad_vjp",
        category="portfolio_aad",
        support_status="partial",
        backend_operator="vjp",
        description="Bounded book-level reverse-mode curve risk using the VJP backend operator.",
    ),
    "autodiff_pathwise": DerivativeMethodSpec(
        "autodiff_pathwise",
        category="autograd",
        support_status="supported",
        backend_operator="grad",
        description="Pathwise Monte Carlo derivative through deterministic explicit shocks.",
    ),
    "forward_price_only": DerivativeMethodSpec(
        "forward_price_only",
        category="forward",
        support_status="unsupported",
        fallback_derivative_method="finite_difference_bump_reprice",
        description="Forward price available, but the requested derivative path is unsupported.",
    ),
    "unsupported_discontinuous_pathwise": DerivativeMethodSpec(
        "unsupported_discontinuous_pathwise",
        category="unsupported",
        support_status="unsupported",
        fallback_derivative_method="finite_difference_bump_reprice",
        description="Pathwise AD is fail-closed because the payoff or event state is discontinuous.",
    ),
    "finite_difference_bump_reprice": DerivativeMethodSpec(
        "finite_difference_bump_reprice",
        category="finite_difference_bump",
        support_status="fallback",
        fallback_derivative_method="finite_difference_bump_reprice",
        description="Declared finite-difference bump/reprice fallback for unsupported runtime derivatives.",
    ),
    "vol_surface_unavailable": DerivativeMethodSpec(
        "vol_surface_unavailable",
        category="unavailable",
        support_status="unsupported",
        description="No volatility surface is available for the requested volatility risk.",
    ),
    "not_applicable_root_scalar": DerivativeMethodSpec(
        "not_applicable_root_scalar",
        category="not_applicable",
        support_status="not_applicable",
        description="Root-scalar solve path with no derivative method.",
    ),
    "provided_scalar_gradient": DerivativeMethodSpec(
        "provided_scalar_gradient",
        category="provided",
        support_status="supported",
        description="Caller-supplied scalar gradient.",
    ),
    "provided_vector_jacobian": DerivativeMethodSpec(
        "provided_vector_jacobian",
        category="provided",
        support_status="supported",
        description="Caller-supplied vector residual Jacobian.",
    ),
    "scipy_internal_finite_difference_gradient": DerivativeMethodSpec(
        "scipy_internal_finite_difference_gradient",
        category="finite_difference_bump",
        support_status="fallback",
        description="SciPy internal finite-difference gradient.",
    ),
    "scipy_2point_residual_jacobian": DerivativeMethodSpec(
        "scipy_2point_residual_jacobian",
        category="finite_difference_bump",
        support_status="fallback",
        fallback_derivative_method="scipy_2point_residual_jacobian",
        description="SciPy two-point finite-difference residual Jacobian.",
    ),
}

DERIVATIVE_METHODS = MappingProxyType(_METHOD_SPECS)


def get_derivative_method(method_id: str) -> DerivativeMethodSpec:
    """Return the registered derivative method or fail closed on string drift."""
    normalized = str(method_id)
    try:
        return DERIVATIVE_METHODS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown derivative method: {normalized!r}") from exc


def derivative_method_payload(
    method_id: str,
    *,
    method_support: str | None = None,
    backend_operator: str | None = None,
    fallback_derivative_method: str | None = None,
    fallback_reason: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return normalized derivative-method metadata while preserving legacy keys."""
    spec = get_derivative_method(method_id)
    resolved_fallback = fallback_derivative_method or spec.fallback_derivative_method
    resolved_warnings = [dict(warning) for warning in (warnings or ())]
    if fallback_reason is not None:
        resolved_warnings.append(dict(fallback_reason))

    payload: dict[str, Any] = {
        "resolved_derivative_method": spec.method_id,
        "derivative_method_category": spec.category,
        "derivative_method_support": str(method_support or spec.support_status),
        "warnings": resolved_warnings,
        "fallback_reason": None if fallback_reason is None else dict(fallback_reason),
    }
    resolved_backend_operator = backend_operator or spec.backend_operator
    if resolved_backend_operator is not None:
        payload["backend_operator"] = str(resolved_backend_operator)
    if resolved_fallback is not None:
        payload["fallback_derivative_method"] = str(resolved_fallback)

    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


__all__ = [
    "DERIVATIVE_METHODS",
    "DerivativeMethodSpec",
    "derivative_method_payload",
    "get_derivative_method",
]
