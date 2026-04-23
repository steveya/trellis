"""Tests for the autograd-backed differentiability wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from autograd.scipy.stats import norm

from trellis.core.differentiable import (
    get_backend_capabilities,
    get_numpy,
    gradient,
    hessian,
    hessian_vector_product,
    jacobian,
    jvp,
    require_capability,
    supports_capability,
    vjp,
)


def test_get_numpy_returns_autograd_module():
    np = get_numpy()

    values = np.array([1.0, 2.0, 3.0])

    assert values.shape == (3,)
    assert float(np.sum(values)) == pytest.approx(6.0)


def test_gradient_wrapper_returns_scalar_derivative():
    grad = gradient(lambda x: x**3 + 2.0 * x, 0)

    assert grad(3.0) == pytest.approx(29.0)


def test_hessian_and_jacobian_wrappers_preserve_autograd_surface():
    hess = hessian(lambda x: x**3 + x**2, 0)
    jac = jacobian(lambda vector: get_numpy().array([vector[0] ** 2, vector[0] + vector[1]]), 0)

    assert hess(2.0) == pytest.approx(14.0)
    assert np.allclose(
        jac(get_numpy().array([3.0, 4.0])),
        np.array([[6.0, 0.0], [1.0, 1.0]]),
    )


def test_backend_capabilities_describe_current_and_future_surface():
    capabilities = get_backend_capabilities()

    assert capabilities.backend_id == "autograd"
    assert capabilities.array_namespace == "autograd.numpy"
    assert capabilities.supports("grad") is True
    assert capabilities.supports("jacobian") is True
    assert capabilities.supports("hessian") is True
    assert capabilities.supports("jvp") is False
    assert capabilities.supports("vjp") is True
    assert capabilities.supports("hessian_vector_product") is True
    assert capabilities.supports("portfolio_aad") is False
    assert capabilities.to_payload()["operators"]["grad"] is True
    assert "portfolio_aad" in capabilities.unsupported_operators
    assert "vjp" in capabilities.supported_operators
    assert "hessian_vector_product" in capabilities.supported_operators


def test_require_capability_accepts_current_operators_and_rejects_future_hooks():
    require_capability("grad")
    require_capability("jacobian")
    require_capability("hessian")
    require_capability("vjp")
    require_capability("hessian_vector_product")

    assert supports_capability("grad") is True
    assert supports_capability("vjp") is True
    assert supports_capability("portfolio_aad") is False

    with pytest.raises(NotImplementedError, match="portfolio_aad"):
        require_capability("portfolio_aad")
    with pytest.raises(ValueError, match="unknown differentiable backend capability"):
        supports_capability("custom_adjoint")


def test_vjp_returns_value_and_pullback_matching_dense_jacobian_transpose():
    np_backend = get_numpy()

    def vector_function(x):
        return np_backend.array(
            [
                x[0] * x[1],
                x[0] ** 2 + np_backend.sin(x[1]),
                np_backend.exp(x[0] - 0.5 * x[1]),
            ]
        )

    x = np_backend.array([1.2, -0.7])
    cotangent = np_backend.array([0.25, -1.5, 0.75])

    value, pullback = vjp(vector_function, x)
    dense_jacobian = jacobian(vector_function)(x)

    assert np.allclose(value, vector_function(x))
    assert np.allclose(pullback(cotangent), np.asarray(dense_jacobian).T @ cotangent)


def test_vjp_preserves_unary_tuple_primal_inputs():
    np_backend = get_numpy()

    def tuple_function(pair):
        x, y = pair
        return np_backend.array([x * y, x + y**2])

    value, pullback = vjp(tuple_function, (2.0, 3.0))

    assert np.allclose(value, np_backend.array([6.0, 11.0]))
    assert pullback(np_backend.array([1.0, 2.0])) == (
        pytest.approx(5.0),
        pytest.approx(14.0),
    )


def test_vjp_supports_explicit_n_ary_primal_unpacking():
    np_backend = get_numpy()

    def two_arg_function(x, y):
        return np_backend.array([x * y, x + y**2])

    value, pullback = vjp(
        two_arg_function,
        (2.0, 3.0),
        argnum=1,
        unpack_primals=True,
    )

    assert np.allclose(value, np_backend.array([6.0, 11.0]))
    assert pullback(np_backend.array([1.0, 2.0])) == pytest.approx(14.0)


def test_hessian_vector_product_matches_dense_hessian_vector_multiplication():
    np_backend = get_numpy()

    def scalar_objective(x):
        return (
            np_backend.sum(x**4)
            + x[0] * np_backend.sin(x[1])
            + 0.5 * x[2] * x[0] ** 2
        )

    x = np_backend.array([0.8, -0.3, 1.4])
    vector = np_backend.array([0.25, -1.5, 0.75])

    hvp = hessian_vector_product(scalar_objective, x, vector)
    dense_hessian = hessian(scalar_objective)(x)

    assert np.allclose(hvp, np.asarray(dense_hessian) @ vector)


def test_hessian_vector_product_preserves_unary_tuple_primal_inputs():
    def tuple_objective(pair):
        x, y = pair
        return x**2 * y + y**3

    hvp = hessian_vector_product(tuple_objective, (2.0, 3.0), (0.5, -0.25))

    assert hvp == (pytest.approx(2.0), pytest.approx(-2.5))


def test_hessian_vector_product_supports_explicit_n_ary_primal_unpacking():
    def two_arg_objective(x, y):
        return x * y + y**3

    hvp = hessian_vector_product(
        two_arg_objective,
        (2.0, 3.0),
        0.5,
        argnum=1,
        unpack_primals=True,
    )

    assert hvp == pytest.approx(9.0)


def test_jvp_fails_closed_with_stock_autograd_normal_cdf_gap_reason():
    with pytest.raises(NotImplementedError, match="norm.cdf"):
        jvp(lambda x: norm.cdf(x), 1.0, 0.25)
