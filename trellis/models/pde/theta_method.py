r"""Unified theta-method PDE solver — replaces Crank-Nicolson and implicit FD.

The theta-method discretizes the backward PDE  dV/dt + L[V] = 0  as:

.. math::
    (I - \theta\,L\,\Delta t)\,V^n = (I + (1-\theta)\,L\,\Delta t)\,V^{n+1}

where L is a tridiagonal spatial operator provided by a :class:`PDEOperator`.

Special cases:
- θ = 0.0 — explicit Euler (conditionally stable, CFL constraint)
- θ = 0.5 — Crank-Nicolson (second-order in time, unconditionally stable)
- θ = 1.0 — fully implicit Euler (first-order, unconditionally stable)

For American/Bermudan options, pass ``exercise_values`` to activate
projected SOR at each time step (linear complementarity problem).
"""

from __future__ import annotations

import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit
from trellis.models.pde.thomas import thomas_solve


@maybe_njit(cache=False)
def _psor_sweep_max(
    V_new: raw_np.ndarray,
    rhs: raw_np.ndarray,
    a_full: raw_np.ndarray,
    b_full: raw_np.ndarray,
    c_full: raw_np.ndarray,
    exercise_values: raw_np.ndarray,
    omega: float,
    max_iter: int,
    tol: float,
) -> raw_np.ndarray:
    """Projected SOR sweep specialized for max-based exercise."""
    n = len(V_new)
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(1, n - 1):
            if b_full[i] == 0.0:
                continue
            gs = (rhs[i] + a_full[i] * V_new[i - 1] + c_full[i] * V_new[i + 1]) / b_full[i]
            v_sor = V_new[i] + omega * (gs - V_new[i])
            v_proj = v_sor if v_sor > exercise_values[i] else exercise_values[i]
            change = abs(v_proj - V_new[i])
            if change > max_change:
                max_change = change
            V_new[i] = v_proj
        if max_change < tol:
            break
    return V_new


@maybe_njit(cache=False)
def _psor_sweep_min(
    V_new: raw_np.ndarray,
    rhs: raw_np.ndarray,
    a_full: raw_np.ndarray,
    b_full: raw_np.ndarray,
    c_full: raw_np.ndarray,
    exercise_values: raw_np.ndarray,
    omega: float,
    max_iter: int,
    tol: float,
) -> raw_np.ndarray:
    """Projected SOR sweep specialized for min-based exercise."""
    n = len(V_new)
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(1, n - 1):
            if b_full[i] == 0.0:
                continue
            gs = (rhs[i] + a_full[i] * V_new[i - 1] + c_full[i] * V_new[i + 1]) / b_full[i]
            v_sor = V_new[i] + omega * (gs - V_new[i])
            v_proj = v_sor if v_sor < exercise_values[i] else exercise_values[i]
            change = abs(v_proj - V_new[i])
            if change > max_change:
                max_change = change
            V_new[i] = v_proj
        if max_change < tol:
            break
    return V_new


def theta_method_1d(
    grid,
    operator,
    terminal_condition: raw_np.ndarray,
    theta: float = 0.5,
    lower_bc_fn=None,
    upper_bc_fn=None,
    exercise_values: raw_np.ndarray | None = None,
    exercise_fn=None,
    omega: float = 1.2,
    psor_max_iter: int = 1000,
    psor_tol: float = 1e-8,
) -> raw_np.ndarray:
    r"""Solve a 1D PDE backward in time using the theta-method.

    Parameters
    ----------
    grid : Grid
        Spatial-temporal grid.
    operator : PDEOperator
        Provides tridiagonal coefficients ``(a, b, c)`` via
        ``operator.coefficients(S, t, dt)``.
    terminal_condition : ndarray of shape (n_x,)
        Option value at maturity.
    theta : float
        Implicitness parameter. 0.5 = Crank-Nicolson, 1.0 = fully implicit.
    lower_bc_fn, upper_bc_fn : callable(t) -> float, optional
        Boundary conditions at S_min and S_max.
    exercise_values : ndarray of shape (n_x,) or None
        If provided, activates PSOR for American/free-boundary problems.
        At each time step, the solution is projected: V = exercise_fn(V, g).
    exercise_fn : callable or None
        How to combine continuation and exercise. Default: ``max``
        (American option holder maximizes). Use ``min`` for callable bonds.
    omega : float
        SOR relaxation parameter (only used when exercise_values is set).
    psor_max_iter, psor_tol : int, float
        PSOR convergence parameters.

    Returns
    -------
    V : ndarray of shape (n_x,)
        Solution at t=0.
    """
    if exercise_fn is None:
        exercise_fn = max

    S = grid.x
    n_x = grid.n_x
    dt = grid.dt
    n_t = grid.n_t
    n_int = n_x - 2

    V = terminal_condition.copy().astype(float)

    for step in range(n_t - 1, -1, -1):
        t = step * dt

        # Get operator coefficients: L*dt tridiagonal
        a, b, c = operator.coefficients(S, t, dt)

        # Boundary values at this time
        V_lower = lower_bc_fn(t) if lower_bc_fn else 0.0
        V_upper = upper_bc_fn(t) if upper_bc_fn else V[-1]

        if exercise_values is not None:
            # PSOR: solve LCP via projected SOR
            V = _psor_step(
                V, a, b, c, theta, V_lower, V_upper, n_int,
                exercise_values, exercise_fn, omega, psor_max_iter, psor_tol,
            )
        else:
            # Standard theta-method: solve tridiagonal system
            V = _theta_step(V, a, b, c, theta, V_lower, V_upper, n_int)

    return V


def _theta_step(
    V: raw_np.ndarray,
    a: raw_np.ndarray,
    b: raw_np.ndarray,
    c: raw_np.ndarray,
    theta: float,
    V_lower: float,
    V_upper: float,
    n_int: int,
) -> raw_np.ndarray:
    """One time step of the theta-method (tridiagonal solve)."""
    # Implicit side: (I - theta * L*dt)
    b_impl = 1.0 - theta * b
    a_impl = -theta * a[1:]
    c_impl = -theta * c[:-1]

    # Explicit side: (I + (1-theta) * L*dt) * V^{n+1}
    one_m_theta = 1.0 - theta
    rhs = (
        one_m_theta * a * V[:-2]
        + (1.0 + one_m_theta * b) * V[1:-1]
        + one_m_theta * c * V[2:]
    )

    # Boundary adjustments
    rhs[0] += theta * a[0] * V_lower
    rhs[-1] += theta * c[-1] * V_upper

    # Solve
    V_int = thomas_solve(a_impl, b_impl, c_impl, rhs)

    V_new = V.copy()
    V_new[0] = V_lower
    V_new[1:1 + n_int] = V_int
    V_new[-1] = V_upper
    return V_new


def _psor_step(
    V: raw_np.ndarray,
    a: raw_np.ndarray,
    b: raw_np.ndarray,
    c: raw_np.ndarray,
    theta: float,
    V_lower: float,
    V_upper: float,
    n_int: int,
    exercise_values: raw_np.ndarray,
    exercise_fn,
    omega: float,
    max_iter: int,
    tol: float,
) -> raw_np.ndarray:
    """One time step of projected SOR for free-boundary problems."""
    # Build full implicit system coefficients for interior points
    a_full = raw_np.zeros_like(V)
    b_full = raw_np.ones_like(V)
    c_full = raw_np.zeros_like(V)

    a_full[1:-1] = theta * a
    b_full[1:-1] = 1.0 - theta * b
    c_full[1:-1] = theta * c

    # RHS from explicit side
    rhs = V.copy()
    one_m_theta = 1.0 - theta
    rhs[1:-1] = (
        one_m_theta * a * V[:-2]
        + (1.0 + one_m_theta * b) * V[1:-1]
        + one_m_theta * c * V[2:]
    )
    rhs[0] = V_lower
    rhs[-1] = V_upper

    # PSOR iteration
    V_new = V.copy()
    V_new[0] = V_lower
    V_new[-1] = V_upper

    if NUMBA_AVAILABLE and exercise_fn is max:
        return _psor_sweep_max(
            V_new, rhs, a_full, b_full, c_full, exercise_values, omega, max_iter, tol,
        )
    if NUMBA_AVAILABLE and exercise_fn is min:
        return _psor_sweep_min(
            V_new, rhs, a_full, b_full, c_full, exercise_values, omega, max_iter, tol,
        )

    for _ in range(max_iter):
        max_change = 0.0
        for i in range(1, len(V) - 1):
            if b_full[i] == 0.0:
                continue
            gs = (rhs[i] + a_full[i] * V_new[i - 1] + c_full[i] * V_new[i + 1]) / b_full[i]
            v_sor = V_new[i] + omega * (gs - V_new[i])
            v_proj = exercise_fn(v_sor, exercise_values[i])
            change = abs(v_proj - V_new[i])
            if change > max_change:
                max_change = change
            V_new[i] = v_proj
        if max_change < tol:
            break

    return V_new
