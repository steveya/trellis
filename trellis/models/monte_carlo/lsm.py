"""Longstaff-Schwartz (LSM) method for American/Bermudan option pricing via MC."""

from __future__ import annotations

import numpy as raw_np


def longstaff_schwartz(
    paths: raw_np.ndarray,
    exercise_dates: list[int],
    payoff_fn,
    discount_rate: float,
    dt: float,
    basis_fn=None,
) -> float:
    """Longstaff-Schwartz least-squares Monte Carlo.

    Parameters
    ----------
    paths : ndarray of shape (n_paths, n_steps + 1)
        Simulated paths of the underlying.
    exercise_dates : list[int]
        Step indices where exercise is allowed.
    payoff_fn : callable(S: ndarray) -> ndarray
        Maps spot values to exercise payoffs. Shape: (n_paths,) -> (n_paths,).
    discount_rate : float
        Continuously compounded risk-free rate.
    dt : float
        Time step size.
    basis_fn : callable(S: ndarray) -> ndarray, optional
        Basis functions for regression. Default: polynomial (1, S, S^2).
        Should return (n_paths, n_basis) array.

    Returns
    -------
    float
        Option price.
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    df_step = raw_np.exp(-discount_rate * dt)

    if basis_fn is None:
        basis_fn = _polynomial_basis

    # Cash flows and their timing
    cashflows = raw_np.zeros(n_paths)
    cashflow_time = raw_np.full(n_paths, n_steps)  # step index of cashflow

    # Initialize with terminal payoff
    if n_steps in exercise_dates:
        cashflows = payoff_fn(paths[:, -1])
        cashflow_time[:] = n_steps

    # Backward iteration
    for step in sorted(exercise_dates, reverse=True):
        if step >= n_steps:
            continue

        S = paths[:, step]
        exercise = payoff_fn(S)

        # Only consider paths that are in-the-money
        itm = exercise > 0
        if not raw_np.any(itm):
            continue

        # Discount future cashflows to this step
        steps_ahead = cashflow_time[itm] - step
        discounted_cf = cashflows[itm] * df_step ** steps_ahead

        # Regression: E[continuation | S] using basis functions
        X = basis_fn(S[itm])
        try:
            coeffs = raw_np.linalg.lstsq(X, discounted_cf, rcond=None)[0]
            continuation = X @ coeffs
        except raw_np.linalg.LinAlgError:
            continuation = discounted_cf

        # Exercise if immediate payoff > continuation value
        exercise_now = exercise[itm] > continuation
        itm_indices = raw_np.where(itm)[0]
        ex_indices = itm_indices[exercise_now]

        cashflows[ex_indices] = exercise[ex_indices]
        cashflow_time[ex_indices] = step

    # Discount all cashflows to t=0
    total_discount = df_step ** cashflow_time
    price = float(raw_np.mean(cashflows * total_discount))
    return price


def _polynomial_basis(S: raw_np.ndarray) -> raw_np.ndarray:
    """Default polynomial basis: 1, S, S^2."""
    return raw_np.column_stack([raw_np.ones_like(S), S, S ** 2])


def laguerre_basis(S: raw_np.ndarray) -> raw_np.ndarray:
    """Laguerre polynomial basis (better for option pricing)."""
    x = S
    L0 = raw_np.ones_like(x)
    L1 = 1 - x
    L2 = 0.5 * (x ** 2 - 4 * x + 2)
    return raw_np.column_stack([L0, L1, L2])
