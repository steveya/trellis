"""Tsitsiklis-Van Roy continuation-regression policy for early exercise."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.monte_carlo.early_exercise import (
    EarlyExerciseDiagnostics,
    EarlyExercisePolicyResult,
    default_continuation_estimator,
    normalize_exercise_steps,
    polynomial_basis,
)


def tsitsiklis_van_roy_result(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    basis_fn=None,
    continuation_estimator=None,
) -> EarlyExercisePolicyResult:
    """Return the continuation-regression ADP result bundle.

    This implements a fitted continuation-value recursion over the supplied
    exercise dates. Compared with Longstaff-Schwartz, it regresses the next-step
    option value recursion directly instead of regressing realized future
    cashflows from currently in-the-money paths only.
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    df_step = raw_np.exp(-discount_rate * dt)
    discount_powers = df_step ** raw_np.arange(n_steps + 1)

    if continuation_estimator is None:
        continuation_estimator = default_continuation_estimator(
            basis_fn=basis_fn or polynomial_basis
        )

    effective_exercise_steps = normalize_exercise_steps(exercise_dates, n_steps)
    exercise_set = set(effective_exercise_steps)
    values = payoff_fn(paths[:, -1]).astype(float)
    exercise_time = raw_np.full(n_paths, n_steps, dtype=int)
    regression_failures = 0

    next_step = n_steps
    for step in sorted(exercise_set - {n_steps}, reverse=True):
        S = paths[:, step]
        intrinsic = payoff_fn(S).astype(float)
        continuation_targets = values * discount_powers[next_step - step]
        continuation, failed = continuation_estimator.fit_predict(S, continuation_targets)
        regression_failures += int(failed)

        exercise_now = intrinsic > continuation
        values = raw_np.where(exercise_now, intrinsic, continuation)
        exercise_time = raw_np.where(exercise_now, step, exercise_time)
        next_step = step

    price = float(raw_np.mean(values * discount_powers[next_step]))
    diagnostics = EarlyExerciseDiagnostics(
        policy_class="tsitsiklis_van_roy",
        exercise_dates_count=len(effective_exercise_steps),
        exercised_paths_fraction=float(raw_np.mean(exercise_time < n_steps)),
        regression_failures=regression_failures,
        estimator_name=getattr(continuation_estimator, "name", None),
    )
    return EarlyExercisePolicyResult(
        policy_class="tsitsiklis_van_roy",
        price_lower=price,
        price_upper=None,
        diagnostics=diagnostics,
    )


def tsitsiklis_van_roy(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    basis_fn=None,
    continuation_estimator=None,
) -> float:
    """Backward-compatible scalar price for the continuation-regression policy."""
    return tsitsiklis_van_roy_result(
        paths,
        exercise_dates,
        payoff_fn,
        discount_rate,
        dt,
        basis_fn=basis_fn,
        continuation_estimator=continuation_estimator,
    ).price_lower
