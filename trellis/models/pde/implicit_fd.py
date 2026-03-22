"""Fully implicit finite difference (backward Euler) for 1D Black-Scholes PDE."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.pde.thomas import thomas_solve


def implicit_fd_1d(
    grid,
    sigma_fn,
    r_fn,
    terminal_condition: raw_np.ndarray,
    lower_bc_fn=None,
    upper_bc_fn=None,
) -> raw_np.ndarray:
    """Fully implicit finite difference — unconditionally stable.

    Uses full implicit (theta=1) instead of CN (theta=0.5).
    """
    S = grid.x
    n_x = grid.n_x
    dt = grid.dt
    n_t = grid.n_t
    dS = S[1] - S[0]

    V = terminal_condition.copy().astype(float)

    for step in range(n_t - 1, -1, -1):
        t = step * dt
        r = r_fn(t)

        n_int = n_x - 2
        a_lower = raw_np.zeros(n_int - 1)
        b_main = raw_np.zeros(n_int)
        c_upper = raw_np.zeros(n_int - 1)
        rhs = raw_np.zeros(n_int)

        for idx in range(n_int):
            i = idx + 1
            sig = sigma_fn(S[i], t)

            alpha_i = dt * (sig ** 2 * S[i] ** 2 / dS ** 2 - r * S[i] / (2 * dS))
            beta_i = -dt * (sig ** 2 * S[i] ** 2 / dS ** 2 + r)
            gamma_i = dt * (sig ** 2 * S[i] ** 2 / dS ** 2 + r * S[i] / (2 * dS))

            b_main[idx] = 1 - beta_i
            if idx > 0:
                a_lower[idx - 1] = -alpha_i
            if idx < n_int - 1:
                c_upper[idx] = -gamma_i

            rhs[idx] = V[i]

        V_lower = lower_bc_fn(t) if lower_bc_fn else 0.0
        V_upper = upper_bc_fn(t) if upper_bc_fn else V[-1]

        sig0 = sigma_fn(S[1], t)
        alpha_0 = dt * (sig0 ** 2 * S[1] ** 2 / dS ** 2 - r * S[1] / (2 * dS))
        rhs[0] += alpha_0 * V_lower

        sig_last = sigma_fn(S[n_x - 2], t)
        gamma_last = dt * (sig_last ** 2 * S[n_x - 2] ** 2 / dS ** 2 + r * S[n_x - 2] / (2 * dS))
        rhs[-1] += gamma_last * V_upper

        V_int = thomas_solve(a_lower, b_main, c_upper, rhs)

        V[0] = V_lower
        V[1:n_x - 1] = V_int
        V[-1] = V_upper

    return V
