"""Primal lower bound plus optimistic upper-bound diagnostic for early exercise.

This module exposes a lightweight bound pair for early-exercise Monte Carlo.
The lower bound is the admissible Longstaff-Schwartz stopping policy. The upper
bound is a pathwise perfect-information diagnostic: on each path, it chooses the
largest discounted intrinsic payoff across the exercise dates. That upper bound
is intentionally simple and auditable; it is not a full Andersen-Broadie nested
dual construction.
"""

from __future__ import annotations

import numpy as raw_np

from trellis.models.monte_carlo.early_exercise import (
    EarlyExerciseDiagnostics,
    EarlyExercisePolicyResult,
)
from trellis.models.monte_carlo.lsm import longstaff_schwartz_result


def primal_dual_mc_result(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    *,
    basis_fn=None,
    continuation_estimator=None,
) -> EarlyExercisePolicyResult:
    """Return a lower/upper bound pair for early exercise on one path set."""
    lower_result = longstaff_schwartz_result(
        paths,
        exercise_dates,
        payoff_fn,
        discount_rate,
        dt,
        basis_fn=basis_fn,
        continuation_estimator=continuation_estimator,
    )
    df_step = raw_np.exp(-discount_rate * dt)
    discounted_intrinsic = raw_np.column_stack(
        [
            payoff_fn(paths[:, step]) * (df_step ** step)
            for step in sorted(set(exercise_dates))
        ]
    )
    upper = float(raw_np.mean(raw_np.max(discounted_intrinsic, axis=1)))
    diagnostics = EarlyExerciseDiagnostics(
        policy_class="primal_dual_mc",
        exercise_dates_count=len(exercise_dates),
        exercised_paths_fraction=(
            lower_result.diagnostics.exercised_paths_fraction
            if lower_result.diagnostics is not None
            else 0.0
        ),
        regression_failures=(
            lower_result.diagnostics.regression_failures
            if lower_result.diagnostics is not None
            else 0
        ),
        estimator_name=(
            getattr(lower_result.diagnostics, "estimator_name", None)
            or "least_squares_regression + perfect_information_upper_bound"
        ),
    )
    return EarlyExercisePolicyResult(
        policy_class="primal_dual_mc",
        price_lower=lower_result.price_lower,
        price_upper=max(upper, lower_result.price_lower),
        diagnostics=diagnostics,
    )


def primal_dual_mc(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    *,
    basis_fn=None,
    continuation_estimator=None,
) -> float:
    """Return the primal lower bound for backward-compatible call sites."""
    return primal_dual_mc_result(
        paths,
        exercise_dates,
        payoff_fn,
        discount_rate,
        dt,
        basis_fn=basis_fn,
        continuation_estimator=continuation_estimator,
    ).price_lower
