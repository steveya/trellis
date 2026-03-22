"""Crank-Nicolson finite difference scheme for 1D Black-Scholes PDE."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.pde.thomas import thomas_solve


def crank_nicolson_1d(
    grid,
    sigma_fn,
    r_fn,
    terminal_condition: raw_np.ndarray,
    lower_bc_fn=None,
    upper_bc_fn=None,
) -> raw_np.ndarray:
    """Solve the Black-Scholes PDE backward in time via Crank-Nicolson.

    PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

    Uses uniform grid with standard central-difference coefficients.
    """
    S = grid.x
    n_x = grid.n_x
    dt = grid.dt
    n_t = grid.n_t
    dS = S[1] - S[0]  # uniform spacing

    V = terminal_condition.copy().astype(float)

    for step in range(n_t - 1, -1, -1):
        t = step * dt
        r = r_fn(t)

        # Build coefficient arrays for interior points (indices 1..n_x-2)
        n_int = n_x - 2  # number of interior points

        # Tridiagonal for the implicit side: (I - 0.5*dt*L) V^n = rhs
        a_lower = raw_np.zeros(n_int - 1)  # sub-diagonal
        b_main = raw_np.zeros(n_int)       # main diagonal
        c_upper = raw_np.zeros(n_int - 1)  # super-diagonal
        rhs = raw_np.zeros(n_int)

        for idx in range(n_int):
            i = idx + 1  # grid index
            sig = sigma_fn(S[i], t)

            # Standard FD coefficients for BS PDE on uniform grid
            alpha_i = 0.5 * dt * (sig ** 2 * S[i] ** 2 / dS ** 2 - r * S[i] / (2 * dS))
            beta_i = -0.5 * dt * (sig ** 2 * S[i] ** 2 / dS ** 2 + r)
            gamma_i = 0.5 * dt * (sig ** 2 * S[i] ** 2 / dS ** 2 + r * S[i] / (2 * dS))

            # Implicit side: (I - L/2) at index idx
            b_main[idx] = 1 - beta_i
            if idx > 0:
                a_lower[idx - 1] = -alpha_i
            if idx < n_int - 1:
                c_upper[idx] = -gamma_i

            # Explicit side: (I + L/2) * V^{n+1}
            rhs[idx] = alpha_i * V[i - 1] + (1 + beta_i) * V[i] + gamma_i * V[i + 1]

        # Boundary conditions
        V_lower = lower_bc_fn(t) if lower_bc_fn else 0.0
        V_upper = upper_bc_fn(t) if upper_bc_fn else V[-1]

        # Adjust RHS for boundary values
        # First interior point (idx=0, i=1): a * V[0] is known
        sig0 = sigma_fn(S[1], t)
        alpha_0 = 0.5 * dt * (sig0 ** 2 * S[1] ** 2 / dS ** 2 - r * S[1] / (2 * dS))
        rhs[0] += alpha_0 * V_lower  # implicit side contribution

        # Last interior point (idx=n_int-1, i=n_x-2): c * V[n_x-1] is known
        sig_last = sigma_fn(S[n_x - 2], t)
        gamma_last = 0.5 * dt * (sig_last ** 2 * S[n_x - 2] ** 2 / dS ** 2 + r * S[n_x - 2] / (2 * dS))
        rhs[-1] += gamma_last * V_upper  # implicit side contribution

        # Solve tridiagonal system for interior points
        V_int = thomas_solve(a_lower, b_main, c_upper, rhs)

        # Reconstruct full solution
        V[0] = V_lower
        V[1:n_x - 1] = V_int
        V[-1] = V_upper

    return V
