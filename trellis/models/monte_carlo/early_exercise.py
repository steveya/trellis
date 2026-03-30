"""Contracts for early-exercise pricing in Monte Carlo simulations.

American and Bermudan options can be exercised before expiry. Pricing
them via Monte Carlo requires estimating whether it is better to exercise
now or continue holding. This module defines the shared interfaces and
helpers used by early-exercise policies (e.g. Longstaff-Schwartz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as raw_np


@dataclass(frozen=True)
class EarlyExerciseDiagnostics:
    """Diagnostics emitted by an early-exercise Monte Carlo policy."""

    policy_class: str
    exercise_dates_count: int
    exercised_paths_fraction: float
    regression_failures: int = 0
    estimator_name: str | None = None


@dataclass(frozen=True)
class EarlyExercisePolicyResult:
    """Generic result bundle for early-exercise Monte Carlo policies."""

    policy_class: str
    price_lower: float
    price_upper: float | None = None
    diagnostics: EarlyExerciseDiagnostics | None = None

    @property
    def price(self) -> float:
        """Backward-compatible alias for the lower-bound price."""
        return self.price_lower


class ContinuationEstimator(Protocol):
    """Protocol for estimating the value of continuing to hold (not exercising).

    Used at each exercise date: given the current states and future cashflows,
    estimate what each path is worth if the holder does NOT exercise now.
    """

    def fit_predict(
        self,
        states: raw_np.ndarray,
        discounted_cashflows: raw_np.ndarray,
    ) -> tuple[raw_np.ndarray, bool]:
        """Estimate continuation value for each path.

        Returns (estimates, failed) where failed is True if the regression
        could not be solved (e.g. singular matrix).
        """
        ...

    @property
    def name(self) -> str:
        """Stable estimator name for diagnostics."""
        ...


def polynomial_basis(states: raw_np.ndarray) -> raw_np.ndarray:
    """Default polynomial basis: 1, S, S^2."""
    return raw_np.column_stack([raw_np.ones_like(states), states, states ** 2])


class _ThreeBasisRegressionWorkspace:
    """Reusable scratch space for three-column continuation regressions."""

    def __init__(self):
        self.capacity = 0
        self.b1 = raw_np.empty(0, dtype=float)
        self.b2 = raw_np.empty(0, dtype=float)
        self.out = raw_np.empty(0, dtype=float)
        self.xtx = raw_np.empty((3, 3), dtype=float)
        self.xty = raw_np.empty(3, dtype=float)

    def ensure_capacity(self, n: int) -> None:
        """Grow reusable 1D buffers if the ITM path count increases."""
        if n <= self.capacity:
            return
        self.capacity = n
        self.b1 = raw_np.empty(n, dtype=float)
        self.b2 = raw_np.empty(n, dtype=float)
        self.out = raw_np.empty(n, dtype=float)


def _fit_three_basis_regression(
    basis_1: raw_np.ndarray,
    basis_2: raw_np.ndarray,
    discounted_cashflows: raw_np.ndarray,
    workspace: _ThreeBasisRegressionWorkspace,
) -> tuple[raw_np.ndarray, bool]:
    """Regress discounted future cashflows on [1, basis_1, basis_2] and return fitted values.

    Used to estimate continuation value: the constant + two basis functions
    (typically S and S^2) approximate what the option is worth if not exercised.
    """
    y = raw_np.asarray(discounted_cashflows, dtype=float)
    n = len(y)
    if n == 0:
        return raw_np.empty(0, dtype=float), False

    ws = workspace
    xtx = ws.xtx
    xty = ws.xty

    xtx[0, 0] = float(n)
    xtx[0, 1] = xtx[1, 0] = float(raw_np.sum(basis_1))
    xtx[0, 2] = xtx[2, 0] = float(raw_np.sum(basis_2))
    xtx[1, 1] = float(raw_np.dot(basis_1, basis_1))
    xtx[1, 2] = xtx[2, 1] = float(raw_np.dot(basis_1, basis_2))
    xtx[2, 2] = float(raw_np.dot(basis_2, basis_2))

    xty[0] = float(raw_np.sum(y))
    xty[1] = float(raw_np.dot(basis_1, y))
    xty[2] = float(raw_np.dot(basis_2, y))

    try:
        coeffs = raw_np.linalg.solve(xtx, xty)
        failed = False
    except raw_np.linalg.LinAlgError:
        X = raw_np.column_stack(
            [
                raw_np.ones(n, dtype=float),
                raw_np.asarray(basis_1, dtype=float),
                raw_np.asarray(basis_2, dtype=float),
            ]
        )
        try:
            coeffs = raw_np.linalg.lstsq(X, y, rcond=None)[0]
            failed = False
        except raw_np.linalg.LinAlgError:
            return y.copy(), True

    out = ws.out[:n]
    out[:] = coeffs[0]
    out += coeffs[1] * basis_1
    out += coeffs[2] * basis_2
    return out.copy(), failed


@dataclass(frozen=True)
class FastPolynomialContinuationEstimator:
    """Specialized continuation estimator for the default quadratic basis."""

    name: str = "least_squares_regression_polynomial_fast"
    _workspace: _ThreeBasisRegressionWorkspace = field(
        default_factory=_ThreeBasisRegressionWorkspace, repr=False, compare=False,
    )

    def fit_predict(
        self,
        states: raw_np.ndarray,
        discounted_cashflows: raw_np.ndarray,
    ) -> tuple[raw_np.ndarray, bool]:
        """Fit the default [1, S, S^2] continuation regression without forming X explicitly."""
        x = raw_np.asarray(states, dtype=float)
        ws = self._workspace
        ws.ensure_capacity(len(x))
        basis_1 = ws.b1[:len(x)]
        basis_2 = ws.b2[:len(x)]
        basis_1[:] = x
        raw_np.multiply(x, x, out=basis_2)
        return _fit_three_basis_regression(
            basis_1,
            basis_2,
            discounted_cashflows,
            ws,
        )


@dataclass(frozen=True)
class FastLaguerreContinuationEstimator:
    """Specialized continuation estimator for the legacy three-column Laguerre basis."""

    name: str = "least_squares_regression_laguerre_fast"
    _workspace: _ThreeBasisRegressionWorkspace = field(
        default_factory=_ThreeBasisRegressionWorkspace, repr=False, compare=False,
    )

    def fit_predict(
        self,
        states: raw_np.ndarray,
        discounted_cashflows: raw_np.ndarray,
    ) -> tuple[raw_np.ndarray, bool]:
        """Fit the legacy [1, 1-S, 0.5*(S^2-4S+2)] continuation regression efficiently."""
        x = raw_np.asarray(states, dtype=float)
        ws = self._workspace
        ws.ensure_capacity(len(x))
        basis_1 = ws.b1[:len(x)]
        basis_2 = ws.b2[:len(x)]
        basis_1[:] = 1.0 - x
        raw_np.multiply(x, x, out=basis_2)
        basis_2 -= 4.0 * x
        basis_2 += 2.0
        basis_2 *= 0.5
        return _fit_three_basis_regression(
            basis_1,
            basis_2,
            discounted_cashflows,
            ws,
        )


@dataclass(frozen=True)
class LeastSquaresContinuationEstimator:
    """Least-squares continuation estimator used by LSM-like methods."""

    basis_fn: Callable[[raw_np.ndarray], raw_np.ndarray] = polynomial_basis
    name: str = "least_squares_regression"
    def fit_predict(
        self,
        states: raw_np.ndarray,
        discounted_cashflows: raw_np.ndarray,
    ) -> tuple[raw_np.ndarray, bool]:
        """Fit continuation regression and predict on the same states."""
        X = self.basis_fn(states)
        try:
            coeffs = raw_np.linalg.lstsq(X, discounted_cashflows, rcond=None)[0]
            return X @ coeffs, False
        except raw_np.linalg.LinAlgError:
            return discounted_cashflows, True


def default_continuation_estimator(
    basis_fn: Callable[[raw_np.ndarray], raw_np.ndarray] | None = None,
) -> ContinuationEstimator:
    """Return the default estimator for a basis, using fast paths when recognized."""
    resolved_basis = basis_fn or polynomial_basis
    basis_name = getattr(resolved_basis, "name", None)
    func_name = getattr(resolved_basis, "__name__", None)

    if resolved_basis is polynomial_basis or func_name == "polynomial_basis" or basis_name == "polynomial_deg2":
        return FastPolynomialContinuationEstimator()
    if func_name == "laguerre_basis":
        return FastLaguerreContinuationEstimator()

    return LeastSquaresContinuationEstimator(basis_fn=resolved_basis)
