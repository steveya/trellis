"""Reusable correlation-matrix construction primitives."""

from __future__ import annotations

from operator import index

from trellis.core.differentiable import get_numpy


np = get_numpy()


def equicorrelation_matrix(dimension: int, correlation: float):
    """Return a positive-definite non-negative equicorrelation matrix.

    This bounded constructor matches Trellis' homogeneous one-factor copula
    contract: ``dimension`` must be a positive integer and ``correlation`` must
    satisfy ``0 <= correlation < 1``.
    """
    if isinstance(dimension, bool):
        raise ValueError("dimension must be a positive integer")
    try:
        count = index(dimension)
    except TypeError:
        raise ValueError("dimension must be a positive integer") from None
    if count <= 0:
        raise ValueError("dimension must be a positive integer")

    try:
        rho = float(correlation)
    except (TypeError, ValueError):
        raise ValueError("correlation must satisfy 0 <= correlation < 1") from None
    if not bool(np.isfinite(rho)) or not 0.0 <= rho < 1.0:
        raise ValueError("correlation must satisfy 0 <= correlation < 1")

    identity = np.eye(count)
    return identity + rho * (np.ones((count, count)) - identity)


__all__ = ["equicorrelation_matrix"]
