"""Hull-White short-rate PDE operator on a uniform grid in r.

The Hull-White PDE for a derivative V(r, t) on the short rate r:

    dV/dt + 0.5*sigma^2*V_rr + a*(theta(t) - r)*V_r - r*V = 0

This is a 1D PDE in r (the short rate), discretised using finite differences
on a uniform grid with spacing dr.

The operator uses an adaptive scheme: central differences when the local
Peclet number is small (diffusion-dominated), and upwind differences when
the Peclet number is large (convection-dominated).  This ensures
monotonicity and prevents spurious oscillations in the solution.

The operator produces tridiagonal coefficients (a, b, c) compatible with
the theta_method_1d solver.
"""

from __future__ import annotations

import numpy as raw_np


class HullWhitePDEOperator:
    """HW short-rate PDE operator on a uniform grid in r.

    L[V] = 0.5*sigma^2*V_rr + a*(theta - r)*V_r - r*V

    Parameters
    ----------
    sigma : float
        Hull-White rate volatility (absolute, normal vol).
    a : float
        Mean reversion speed.
    theta_fn : callable(t) -> float, optional
        Time-dependent long-run mean alpha(t).  The drift in the PDE is
        ``a * (alpha(t) - r)``.  For the HW model calibrated to a flat
        curve at rate r0, alpha(t) = r0 + sigma^2/(2*a^2)*(1-exp(-2*a*t)).
        If *None*, uses constant ``r0`` (drift = a*(r0 - r)).
    r0 : float
        Initial short rate (only used when theta_fn is None).
    """

    def __init__(self, sigma: float, a: float, theta_fn=None, r0: float = 0.05):
        """Store volatility, mean reversion, and either functional or flat theta input."""
        self.sigma = sigma
        self.a = a
        if theta_fn is not None:
            self._theta_fn = theta_fn
        else:
            # Constant long-run mean = r0
            self._theta_fn = lambda _t: r0

    def coefficients(self, r_grid, t, dt):
        """Compute tridiagonal coefficients (a, b, c) scaled by dt.

        Uses an adaptive central/upwind scheme to handle convection-dominated
        regimes (high Peclet number) without spurious oscillations.

        Parameters
        ----------
        r_grid : ndarray of shape (n_x,)
            Spatial grid points (short rate values).
        t : float
            Current time.
        dt : float
            Time step size.

        Returns
        -------
        a, b, c : ndarrays of shape (n_interior,)
            Sub-diagonal, diagonal, super-diagonal of L*dt.
        """
        r_int = r_grid[1:-1]
        dr = r_grid[1] - r_grid[0]  # uniform grid

        sigma2 = self.sigma ** 2
        D = 0.5 * sigma2  # diffusion coefficient
        theta = self._theta_fn(t)

        diff = D / dr**2
        mu = self.a * (theta - r_int)
        pos = mu >= 0.0

        a_coeff = raw_np.full(r_int.shape, dt * diff, dtype=float)
        b_coeff = dt * (-2.0 * diff - r_int)
        c_coeff = raw_np.full(r_int.shape, dt * diff, dtype=float)

        if raw_np.any(pos):
            mu_pos = mu[pos] / dr
            a_coeff[pos] = dt * (diff - mu_pos)
            b_coeff[pos] = dt * (-2.0 * diff + mu_pos - r_int[pos])

        if raw_np.any(~pos):
            mu_neg = mu[~pos] / dr
            b_coeff[~pos] = dt * (-2.0 * diff - mu_neg - r_int[~pos])
            c_coeff[~pos] = dt * (diff + mu_neg)

        return a_coeff, b_coeff, c_coeff
