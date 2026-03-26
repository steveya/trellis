"""PDE operator abstraction — separates the financial model from the numerical method.

A PDE operator defines the spatial differential operator L such that:

    dV/dt + L[V] = 0    (backward PDE, e.g., Black-Scholes)

The operator produces tridiagonal coefficients (a, b, c) at each grid point,
where the discrete operator is:

    L[V]_i ≈ a_i * V_{i-1} + b_i * V_i + c_i * V_{i+1}

Different financial models define different operators:
- Black-Scholes: L = 0.5*σ²*S²*∂²/∂S² + r*S*∂/∂S - r
- CEV:           L = 0.5*σ²*S^{2β}*∂²/∂S² + r*S*∂/∂S - r
- Local vol:     L = 0.5*σ(S,t)²*S²*∂²/∂S² + r*S*∂/∂S - r
- Heat equation:  L = D*∂²/∂x²

The numerical scheme (theta-method, PSOR, etc.) is independent of the operator.
"""

from __future__ import annotations

from typing import Protocol

import numpy as raw_np


def _evaluate_spatial_callable(fn, x: raw_np.ndarray, t: float) -> raw_np.ndarray:
    """Evaluate a scalar-or-vector callable over a spatial slice."""
    try:
        values = raw_np.asarray(fn(x, t))
    except Exception:
        values = None
    else:
        if values.shape == x.shape:
            return values
        if values.ndim == 0:
            dtype = raw_np.result_type(values, x, raw_np.float64)
            return raw_np.full(x.shape, values, dtype=dtype)

    return raw_np.fromiter((fn(xi, t) for xi in x), dtype=raw_np.float64, count=len(x))


class PDEOperator(Protocol):
    """Protocol for 1D PDE operators.

    An operator computes tridiagonal coefficients at interior grid points.
    """

    def coefficients(
        self, S: raw_np.ndarray, t: float, dt: float,
    ) -> tuple[raw_np.ndarray, raw_np.ndarray, raw_np.ndarray]:
        """Compute tridiagonal coefficients (a, b, c) scaled by dt.

        Parameters
        ----------
        S : ndarray of shape (n_x,)
            Spatial grid points.
        t : float
            Current time.
        dt : float
            Time step size.

        Returns
        -------
        a : ndarray of shape (n_interior,)
            Sub-diagonal: coefficient of V_{i-1}.
        b : ndarray of shape (n_interior,)
            Main diagonal: coefficient of V_i.
        c : ndarray of shape (n_interior,)
            Super-diagonal: coefficient of V_{i+1}.

        Convention: these are the L*dt coefficients, so the theta-method uses
            implicit: (I - theta * L*dt) V^n = (I + (1-theta) * L*dt) V^{n+1}
        """
        ...


class BlackScholesOperator:
    """Black-Scholes PDE operator on a uniform grid.

    L[V] = 0.5*σ²*S²*V_SS + r*S*V_S - r*V

    Central differences on uniform grid with spacing dS:
        V_SS ≈ (V_{i-1} - 2V_i + V_{i+1}) / dS²
        V_S  ≈ (V_{i+1} - V_{i-1}) / (2*dS)
    """

    def __init__(self, sigma_fn, r_fn):
        """
        Parameters
        ----------
        sigma_fn : callable(S, t) -> float
            Local volatility at spot S and time t.
        r_fn : callable(t) -> float
            Risk-free rate at time t.
        """
        self.sigma_fn = sigma_fn
        self.r_fn = r_fn

    def coefficients(self, S, t, dt):
        """Return the finite-difference Black-Scholes operator coefficients scaled by ``dt``."""
        S_int = S[1:-1]
        dS = S[1] - S[0]  # uniform grid
        r = self.r_fn(t)
        sig = _evaluate_spatial_callable(self.sigma_fn, S_int, t)

        inv_dS = 1.0 / dS
        inv_dS2 = inv_dS * inv_dS
        diff = 0.5 * raw_np.square(sig) * raw_np.square(S_int) * inv_dS2
        drift = 0.5 * r * S_int * inv_dS

        a = dt * (diff - drift)
        b = dt * (-2.0 * diff - r)
        c = dt * (diff + drift)

        return a, b, c


class CEVOperator:
    """Constant Elasticity of Variance PDE operator.

    L[V] = 0.5*σ²*S^{2β}*V_SS + r*S*V_S - r*V

    When β=1, reduces to Black-Scholes. When β=0.5, this is the CIR-like model.
    """

    def __init__(self, sigma_fn, r_fn, beta: float = 1.0):
        """Store local-vol, rate, and elasticity inputs for the CEV PDE operator."""
        self.sigma_fn = sigma_fn
        self.r_fn = r_fn
        self.beta = beta

    def coefficients(self, S, t, dt):
        """Return the finite-difference CEV operator coefficients scaled by ``dt``."""
        S_int = S[1:-1]
        dS = S[1] - S[0]
        r = self.r_fn(t)
        sig = _evaluate_spatial_callable(self.sigma_fn, S_int, t)

        inv_dS = 1.0 / dS
        inv_dS2 = inv_dS * inv_dS
        diff = 0.5 * raw_np.square(sig) * raw_np.power(S_int, 2 * self.beta) * inv_dS2
        drift = 0.5 * r * S_int * inv_dS

        a = dt * (diff - drift)
        b = dt * (-2.0 * diff - r)
        c = dt * (diff + drift)

        return a, b, c


class HeatOperator:
    """Heat equation operator: L[V] = D * V_xx.

    Useful for testing and for log-transformed coordinates.
    """

    def __init__(self, diffusivity: float = 1.0):
        """Store the constant diffusivity used by the heat-equation operator."""
        self.D = diffusivity

    def coefficients(self, S, t, dt):
        """Return uniform tridiagonal coefficients for the 1D heat equation."""
        n_x = len(S)
        n_int = n_x - 2
        dx = S[1] - S[0]

        coeff = self.D * dt / dx**2
        a = raw_np.full(n_int, coeff)
        b = raw_np.full(n_int, -2 * coeff)
        c = raw_np.full(n_int, coeff)
        return a, b, c
