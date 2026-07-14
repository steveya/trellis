"""Scalar Gaussian probability primitives for analytical composition."""

from __future__ import annotations

from math import isfinite

from scipy.special import ndtr
from scipy.stats import multivariate_normal


def _finite_float(value: float, *, name: str) -> float:
    """Return ``value`` as a finite float or fail closed."""
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def standard_normal_cdf(value: float) -> float:
    """Return the scalar standard-normal cumulative probability at ``value``.

    This SciPy-backed scalar kernel is intended for analytical composition. It
    is not an automatic-differentiation primitive.
    """
    normalized = _finite_float(value, name="value")
    return float(ndtr(normalized))


def bivariate_standard_normal_cdf(
    x: float,
    y: float,
    correlation: float,
) -> float:
    """Return ``P[X <= x, Y <= y]`` for correlated standard normals.

    The correlation must lie in the closed interval ``[-1, 1]``. The exact
    singular boundaries are evaluated analytically; interior correlations use
    SciPy's bivariate normal integration.
    """
    normalized_x = _finite_float(x, name="x")
    normalized_y = _finite_float(y, name="y")
    normalized_correlation = _finite_float(correlation, name="correlation")
    if not -1.0 <= normalized_correlation <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")

    if normalized_correlation == 1.0:
        return standard_normal_cdf(min(normalized_x, normalized_y))
    if normalized_correlation == -1.0:
        return max(
            standard_normal_cdf(normalized_x)
            - standard_normal_cdf(-normalized_y),
            0.0,
        )
    if normalized_correlation == 0.0:
        return standard_normal_cdf(normalized_x) * standard_normal_cdf(normalized_y)

    probability = float(
        multivariate_normal.cdf(
            [normalized_x, normalized_y],
            mean=[0.0, 0.0],
            cov=[
                [1.0, normalized_correlation],
                [normalized_correlation, 1.0],
            ],
            maxpts=2_000_000,
            abseps=1e-12,
            releps=1e-12,
        )
    )
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("bivariate normal integration returned an invalid probability")
    return probability


__all__ = [
    "bivariate_standard_normal_cdf",
    "standard_normal_cdf",
]
