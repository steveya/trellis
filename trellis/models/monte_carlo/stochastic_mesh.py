"""Stochastic-mesh continuation weighting for early-exercise Monte Carlo."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.monte_carlo.early_exercise import (
    EarlyExerciseDiagnostics,
    EarlyExercisePolicyResult,
    normalize_exercise_steps,
)


def stochastic_mesh_result(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    *,
    bandwidth_scale: float = 0.35,
) -> EarlyExercisePolicyResult:
    """Estimate an early-exercise lower bound with mesh-style continuation weights."""
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    df_step = raw_np.exp(-discount_rate * dt)

    effective_exercise_steps = normalize_exercise_steps(exercise_dates, n_steps)
    cashflows = raw_np.zeros(n_paths)
    cashflow_time = raw_np.full(n_paths, n_steps)
    weighting_failures = 0

    if n_steps in effective_exercise_steps:
        cashflows = payoff_fn(paths[:, -1])
        cashflow_time[:] = n_steps

    for step in sorted(effective_exercise_steps, reverse=True):
        if step >= n_steps:
            continue

        states = paths[:, step]
        exercise = payoff_fn(states)
        itm = exercise > 0
        if not raw_np.any(itm):
            continue

        discounted_future = cashflows * (df_step ** (cashflow_time - step))
        scale = max(float(raw_np.std(states)), 1e-8)
        bandwidth = max(scale * bandwidth_scale, 1e-8)

        mesh_states = states[itm]
        deltas = (mesh_states[:, None] - states[None, :]) / bandwidth
        weights = raw_np.exp(-0.5 * deltas ** 2)
        denom = raw_np.sum(weights, axis=1)
        fallback = denom <= 1e-12
        weighting_failures += int(raw_np.sum(fallback))
        denom = raw_np.where(fallback, 1.0, denom)
        continuation = weights @ discounted_future / denom
        if raw_np.any(fallback):
            continuation = raw_np.where(
                fallback,
                float(raw_np.mean(discounted_future)),
                continuation,
            )

        exercise_now = exercise[itm] > continuation
        itm_indices = raw_np.where(itm)[0]
        ex_indices = itm_indices[exercise_now]
        cashflows[ex_indices] = exercise[ex_indices]
        cashflow_time[ex_indices] = step

    price = float(raw_np.mean(cashflows * (df_step ** cashflow_time)))
    diagnostics = EarlyExerciseDiagnostics(
        policy_class="stochastic_mesh",
        exercise_dates_count=len(effective_exercise_steps),
        exercised_paths_fraction=float(raw_np.mean(cashflow_time < n_steps)),
        regression_failures=weighting_failures,
        estimator_name="gaussian_mesh_weights",
    )
    return EarlyExercisePolicyResult(
        policy_class="stochastic_mesh",
        price_lower=price,
        price_upper=None,
        diagnostics=diagnostics,
    )


def stochastic_mesh(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    *,
    bandwidth_scale: float = 0.35,
) -> float:
    """Return the stochastic-mesh lower-bound price."""
    return stochastic_mesh_result(
        paths,
        exercise_dates,
        payoff_fn,
        discount_rate,
        dt,
        bandwidth_scale=bandwidth_scale,
    ).price_lower
