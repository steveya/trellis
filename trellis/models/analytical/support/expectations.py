"""Reusable bounded numerical expectations for analytical route assembly."""

from __future__ import annotations

from itertools import product
from math import pi, sqrt
from typing import Callable

import numpy as raw_np
from scipy.special import roots_hermitenorm


def gauss_hermite_product_expectation(
    integrand: Callable[[raw_np.ndarray], object],
    *,
    dimension: int,
    order: int = 21,
    max_nodes: int = 2_000_000,
):
    """Integrate a function of independent standard normal variables.

    The node budget makes the tensor-product growth explicit and fail-closed.
    """
    dimensions = int(dimension)
    quadrature_order = int(order)
    node_budget = int(max_nodes)
    if dimensions <= 0:
        raise ValueError("dimension must be positive")
    if quadrature_order <= 0:
        raise ValueError("order must be positive")
    if node_budget <= 0:
        raise ValueError("max_nodes must be positive")
    node_count = quadrature_order ** dimensions
    if node_count > node_budget:
        raise ValueError(
            f"Gauss-Hermite product node budget exceeded: {node_count} > {node_budget}"
        )

    nodes, weights = roots_hermitenorm(quadrature_order)
    probability_weights = raw_np.asarray(weights, dtype=float) / sqrt(2.0 * pi)
    expectation = None
    for indices in product(range(quadrature_order), repeat=dimensions):
        normals = raw_np.asarray([nodes[index] for index in indices], dtype=float)
        probability = float(raw_np.prod(probability_weights[list(indices)]))
        term = probability * integrand(normals)
        expectation = term if expectation is None else expectation + term
    return expectation


__all__ = ["gauss_hermite_product_expectation"]
