"""Tests for the autograd-backed differentiability wrapper."""

from __future__ import annotations

import numpy as np
import pytest

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
    assert capabilities.supports("vjp") is False
    assert capabilities.supports("hessian_vector_product") is False
    assert capabilities.supports("portfolio_aad") is False
    assert capabilities.to_payload()["operators"]["grad"] is True
    assert "portfolio_aad" in capabilities.unsupported_operators


def test_require_capability_accepts_current_operators_and_rejects_future_hooks():
    require_capability("grad")
    require_capability("jacobian")
    require_capability("hessian")

    assert supports_capability("grad") is True
    assert supports_capability("vjp") is False

    with pytest.raises(NotImplementedError, match="vjp"):
        require_capability("vjp")
    with pytest.raises(ValueError, match="unknown differentiable backend capability"):
        supports_capability("custom_adjoint")


def test_future_aad_hooks_fail_closed_with_capability_messages():
    with pytest.raises(NotImplementedError, match="jvp"):
        jvp(lambda x: x * x, 1.0, 1.0)
    with pytest.raises(NotImplementedError, match="vjp"):
        vjp(lambda x: x * x, 1.0)
    with pytest.raises(NotImplementedError, match="hessian_vector_product"):
        hessian_vector_product(lambda x: x * x, 1.0, 1.0)
