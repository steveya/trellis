"""Projected SOR (Successive Over-Relaxation) for American options."""

from __future__ import annotations

import numpy as raw_np


def psor_1d(
    grid,
    sigma_fn,
    r_fn,
    terminal_condition: raw_np.ndarray,
    exercise_values: raw_np.ndarray,
    omega: float = 1.2,
    max_iter: int = 1000,
    tol: float = 1e-8,
    lower_bc_fn=None,
    upper_bc_fn=None,
) -> raw_np.ndarray:
    """PSOR method for American option pricing via PDE.

    At each time step, solves the linear complementarity problem:
    V >= g (exercise value), (A*V - rhs) >= 0, complementarity.

    Parameters
    ----------
    grid : Grid
    sigma_fn, r_fn : callables
    terminal_condition : ndarray
    exercise_values : ndarray of shape (n_x,)
        Intrinsic / exercise values at each spatial point.
    omega : float
        SOR relaxation parameter (1 < omega < 2 for over-relaxation).
    max_iter : int
    tol : float
    """
    S = grid.x
    n_x = grid.n_x
    dt = grid.dt
    n_t = grid.n_t

    V = terminal_condition.copy()

    for step in range(n_t - 1, -1, -1):
        t = step * dt
        r = r_fn(t)
        rhs = V.copy()

        # Build implicit system coefficients
        a_vec = raw_np.zeros(n_x)
        b_vec = raw_np.zeros(n_x)
        c_vec = raw_np.zeros(n_x)

        for i in range(1, n_x - 1):
            sig = sigma_fn(S[i], t)
            dS = S[i + 1] - S[i - 1]
            dS_plus = S[i + 1] - S[i]
            dS_minus = S[i] - S[i - 1]

            alpha = 0.5 * sig ** 2 * S[i] ** 2
            beta_coef = r * S[i]

            a_vec[i] = dt * (alpha / (dS_minus * dS / 2) - beta_coef / dS)
            b_vec[i] = 1 + dt * (2 * alpha / (dS_plus * dS_minus) + r)
            c_vec[i] = dt * (alpha / (dS_plus * dS / 2) + beta_coef / dS)

        b_vec[0] = 1.0
        b_vec[-1] = 1.0
        rhs[0] = lower_bc_fn(t) if lower_bc_fn else 0.0
        rhs[-1] = upper_bc_fn(t) if upper_bc_fn else V[-1]

        # PSOR iterations
        V_new = V.copy()
        for iteration in range(max_iter):
            max_change = 0.0
            for i in range(1, n_x - 1):
                gs = (rhs[i] + a_vec[i] * V_new[i - 1] + c_vec[i] * V_new[i + 1]) / b_vec[i]
                V_sor = V_new[i] + omega * (gs - V_new[i])
                # Project: ensure V >= exercise value
                V_proj = max(V_sor, exercise_values[i])
                max_change = max(max_change, abs(V_proj - V_new[i]))
                V_new[i] = V_proj

            if max_change < tol:
                break

        V = V_new

    return V
