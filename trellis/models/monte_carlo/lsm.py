"""Longstaff-Schwartz early-exercise policy for American/Bermudan MC pricing."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.monte_carlo.early_exercise import (
    EarlyExerciseDiagnostics,
    EarlyExercisePolicyResult,
    default_continuation_estimator,
    polynomial_basis,
)


def longstaff_schwartz_result(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    basis_fn=None,
    continuation_estimator=None,
) -> EarlyExercisePolicyResult:
    """Run Longstaff-Schwartz and return the shared policy-result bundle."""
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    df_step = raw_np.exp(-discount_rate * dt)
    discount_powers = df_step ** raw_np.arange(n_steps + 1)
    exercise_steps = sorted(
        {
            int(step) for step in exercise_dates
            if 0 < int(step) <= n_steps
        },
        reverse=True,
    )

    if continuation_estimator is None:
        continuation_estimator = default_continuation_estimator(
            basis_fn=basis_fn or polynomial_basis
        )

    cashflows = raw_np.zeros(n_paths)
    cashflow_time = raw_np.full(n_paths, n_steps, dtype=int)
    regression_failures = 0

    if n_steps in exercise_steps:
        cashflows = raw_np.asarray(payoff_fn(paths[:, -1]), dtype=float)
        cashflow_time[:] = n_steps

    for step in exercise_steps:
        if step >= n_steps:
            continue

        S = paths[:, step]
        exercise = raw_np.asarray(payoff_fn(S), dtype=float)

        itm_indices = raw_np.flatnonzero(exercise > 0.0)
        if itm_indices.size == 0:
            continue

        S_itm = S[itm_indices]
        exercise_itm = exercise[itm_indices]
        steps_ahead = cashflow_time[itm_indices] - step
        discounted_cf = cashflows[itm_indices] * discount_powers[steps_ahead]

        continuation, failed = continuation_estimator.fit_predict(S_itm, discounted_cf)
        regression_failures += int(failed)

        exercise_now = exercise_itm > continuation
        if not raw_np.any(exercise_now):
            continue
        ex_indices = itm_indices[exercise_now]

        cashflows[ex_indices] = exercise_itm[exercise_now]
        cashflow_time[ex_indices] = step

    total_discount = discount_powers[cashflow_time]
    price = float(raw_np.mean(cashflows * total_discount))
    diagnostics = EarlyExerciseDiagnostics(
        policy_class="longstaff_schwartz",
        exercise_dates_count=len(exercise_steps),
        exercised_paths_fraction=float(raw_np.mean(cashflow_time < n_steps)),
        regression_failures=regression_failures,
        estimator_name=getattr(continuation_estimator, "name", None),
    )
    return EarlyExercisePolicyResult(
        policy_class="longstaff_schwartz",
        price_lower=price,
        price_upper=None,
        diagnostics=diagnostics,
    )


def longstaff_schwartz(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    basis_fn=None,
    continuation_estimator=None,
) -> float:
    """Longstaff-Schwartz least-squares Monte Carlo lower-bound price."""
    return longstaff_schwartz_result(
        paths,
        exercise_dates,
        payoff_fn,
        discount_rate,
        dt,
        basis_fn=basis_fn,
        continuation_estimator=continuation_estimator,
    ).price_lower


def laguerre_basis(S: raw_np.ndarray) -> raw_np.ndarray:
    """Laguerre polynomial basis (better for option pricing)."""
    x = S
    L0 = raw_np.ones_like(x)
    L1 = 1 - x
    L2 = 0.5 * (x ** 2 - 4 * x + 2)
    return raw_np.column_stack([L0, L1, L2])
