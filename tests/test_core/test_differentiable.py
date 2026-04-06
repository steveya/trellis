"""Tests for the autograd-backed differentiability wrapper."""

from __future__ import annotations

import pytest
import numpy as np

from trellis.core.differentiable import get_numpy, gradient, hessian, jacobian


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
