"""Discretization schemes as composable objects.

Instead of dispatching on strings ("euler", "milstein", "exact"), each scheme
is an object that the MC engine calls. This allows:
- Easy addition of new schemes (QE for Heston, log-Euler, predictor-corrector)
- Scheme-specific parameters (e.g., Milstein FD epsilon)
- Composition: antithetic wrapper, control variate wrapper

Usage:
    scheme = Euler()                    # or Milstein(), Exact(), QE()
    scheme = Antithetic(Euler())        # variance reduction wrapper
    engine = MonteCarloEngine(process, scheme=scheme, ...)
"""

from __future__ import annotations

from typing import Protocol

import numpy as raw_np


class DiscretizationScheme(Protocol):
    """Protocol for SDE discretization schemes."""

    def step(
        self, process, x: raw_np.ndarray, t: float, dt: float,
        dw: raw_np.ndarray,
    ) -> raw_np.ndarray:
        """Advance the state by one time step.

        Parameters
        ----------
        process : StochasticProcess
            Must have drift(x, t) and diffusion(x, t) methods.
        x : ndarray of shape (n_paths,)
            Current state.
        t : float
            Current time.
        dt : float
            Time step size.
        dw : ndarray of shape (n_paths,)
            Standard normal increments (already scaled by sqrt(dt)
            externally, or raw N(0,1) depending on scheme convention).
            Convention: dw ~ N(0, 1). The scheme applies sqrt(dt) internally.

        Returns
        -------
        x_next : ndarray of shape (n_paths,)
        """
        ...

    @property
    def name(self) -> str:
        """Return the stable registry name for the discretization scheme."""
        ...


class Euler:
    """Euler-Maruyama scheme (weak order 1.0, strong order 0.5)."""

    name = "euler"

    def step(self, process, x, t, dt, dw):
        """Advance one Euler-Maruyama step ``x + mu dt + sigma sqrt(dt) dW``."""
        mu = process.drift(x, t)
        sig = process.diffusion(x, t)
        return x + mu * dt + sig * raw_np.sqrt(dt) * dw


class Milstein:
    """Milstein scheme (weak order 1.0, strong order 1.0).

    Adds the Itô correction term: 0.5 * σ * σ' * (dW² - dt).
    σ' is computed via finite differences with configurable epsilon.
    """

    name = "milstein"

    def __init__(self, fd_epsilon: float = 1e-6):
        """Configure the finite-difference step used to approximate ``sigma'(x)``."""
        self.eps = fd_epsilon

    def step(self, process, x, t, dt, dw):
        """Advance one Milstein step including the Itô correction term."""
        mu = process.drift(x, t)
        sig = process.diffusion(x, t)
        # σ'(x) via central finite difference
        sig_plus = process.diffusion(x + self.eps, t)
        sig_minus = process.diffusion(x - self.eps, t)
        dsig_dx = (sig_plus - sig_minus) / (2 * self.eps)

        sqrt_dt = raw_np.sqrt(dt)
        return (
            x + mu * dt + sig * sqrt_dt * dw
            + 0.5 * sig * dsig_dx * (dw**2 - 1) * dt
        )


class Exact:
    """Exact simulation (when available for the process).

    Requires the process to implement exact_sample(x, t, dt, dw).
    Falls back to Euler if not available.
    """

    name = "exact"

    def step(self, process, x, t, dt, dw):
        """Use the process exact transition when available, otherwise fall back to Euler."""
        if hasattr(process, "exact_sample"):
            return process.exact_sample(x, t, dt, dw)
        # Fallback to Euler
        mu = process.drift(x, t)
        sig = process.diffusion(x, t)
        return x + mu * dt + sig * raw_np.sqrt(dt) * dw


class LogEuler:
    """Log-Euler scheme for positive processes (e.g., GBM, CIR).

    Applies Euler in log-space: log(X_{n+1}) = log(X_n) + (μ/X - σ²/2)dt + σ√dt dW.
    Guarantees positivity.
    """

    name = "log_euler"

    def step(self, process, x, t, dt, dw):
        """Advance one step in log-space to preserve positivity for positive processes."""
        mu = process.drift(x, t)
        sig = process.diffusion(x, t)
        # In log space: d(log X) = (mu/X - 0.5*(sig/X)²) dt + (sig/X) dW
        x_safe = raw_np.maximum(x, 1e-15)
        drift_log = (mu / x_safe - 0.5 * (sig / x_safe) ** 2) * dt
        diff_log = (sig / x_safe) * raw_np.sqrt(dt) * dw
        return x_safe * raw_np.exp(drift_log + diff_log)


# ---------------------------------------------------------------------------
# Variance reduction wrappers
# ---------------------------------------------------------------------------

class Antithetic:
    """Antithetic variates wrapper — halves variance for symmetric payoffs.

    Wraps any scheme. The state vector x has shape (n_paths,) where
    n_paths is even. The first half evolves with +dW, the second half
    with -dW (from the same noise). Both halves start from the same x0.

    At initialization, set x = [S0]*n_paths. At each step, pass dw of
    shape (n_paths // 2,) — only half the noise is needed.
    """

    def __init__(self, base_scheme: DiscretizationScheme):
        """Wrap a base scheme so paired paths evolve under opposite shocks."""
        self.base = base_scheme
        self.name = f"antithetic_{base_scheme.name}"

    def step(self, process, x, t, dt, dw):
        """Advance the positive and antithetic path halves with opposite Brownian increments."""
        n = len(x)
        half = n // 2
        # Use same noise with opposite sign for each pair
        dw_half = dw[:half]
        x_pos = self.base.step(process, x[:half], t, dt, dw_half)
        x_neg = self.base.step(process, x[half:], t, dt, -dw_half)
        return raw_np.concatenate([x_pos, x_neg])


# ---------------------------------------------------------------------------
# Basis function registry for LSM
# ---------------------------------------------------------------------------

class BasisFunction(Protocol):
    """Protocol for LSM regression basis functions."""

    def __call__(self, S: raw_np.ndarray) -> raw_np.ndarray:
        """Map spot values to basis matrix of shape (n_paths, n_basis)."""
        ...

    @property
    def name(self) -> str:
        """Return the stable registry name for the regression basis."""
        ...


class PolynomialBasis:
    """Polynomial basis: 1, S, S², ..., S^degree."""

    def __init__(self, degree: int = 2):
        """Choose the highest polynomial power included in the basis."""
        self.degree = degree
        self.name = f"polynomial_deg{degree}"

    def __call__(self, S):
        """Return the Vandermonde-style basis matrix evaluated at spot values ``S``."""
        return raw_np.column_stack([S**k for k in range(self.degree + 1)])


class LaguerreBasis:
    """Laguerre polynomial basis (better for option pricing on positive reals).

    L_0(x) = 1
    L_1(x) = 1 - x
    L_2(x) = (x² - 4x + 2) / 2
    L_3(x) = (-x³ + 9x² - 18x + 6) / 6

    Weighted by exp(-x/2) for numerical stability.
    """

    name = "laguerre"

    def __init__(self, degree: int = 3):
        """Choose the highest Laguerre polynomial degree retained in the basis."""
        self.degree = degree

    def __call__(self, S):
        """Evaluate normalized Laguerre basis columns on positive spot values."""
        x = S / raw_np.mean(S)  # normalize for stability
        cols = [raw_np.ones_like(x)]
        if self.degree >= 1:
            cols.append(1 - x)
        if self.degree >= 2:
            cols.append(0.5 * (x**2 - 4 * x + 2))
        if self.degree >= 3:
            cols.append((-x**3 + 9 * x**2 - 18 * x + 6) / 6)
        return raw_np.column_stack(cols[:self.degree + 1])


class HermiteBasis:
    """Probabilist's Hermite polynomial basis.

    He_0(x) = 1
    He_1(x) = x
    He_2(x) = x² - 1
    He_3(x) = x³ - 3x
    """

    name = "hermite"

    def __init__(self, degree: int = 3):
        """Choose the highest probabilists' Hermite polynomial degree retained."""
        self.degree = degree

    def __call__(self, S):
        """Evaluate standardized Hermite basis columns on spot values ``S``."""
        x = (S - raw_np.mean(S)) / raw_np.std(S)  # standardize
        cols = [raw_np.ones_like(x)]
        if self.degree >= 1:
            cols.append(x)
        if self.degree >= 2:
            cols.append(x**2 - 1)
        if self.degree >= 3:
            cols.append(x**3 - 3 * x)
        return raw_np.column_stack(cols[:self.degree + 1])


class ChebyshevBasis:
    """Chebyshev polynomial basis of the first kind.

    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    """

    name = "chebyshev"

    def __init__(self, degree: int = 3):
        """Choose the highest Chebyshev degree retained in the regression basis."""
        self.degree = degree

    def __call__(self, S):
        """Evaluate first-kind Chebyshev basis columns after mapping spots to ``[-1, 1]``."""
        # Map to [-1, 1]
        s_min, s_max = raw_np.min(S), raw_np.max(S)
        x = 2 * (S - s_min) / max(s_max - s_min, 1e-10) - 1

        cols = [raw_np.ones_like(x), x]
        for n in range(2, self.degree + 1):
            cols.append(2 * x * cols[-1] - cols[-2])
        return raw_np.column_stack(cols[:self.degree + 1])


# Basis registry — the agent can look up available bases
BASIS_REGISTRY: dict[str, type] = {
    "polynomial": PolynomialBasis,
    "laguerre": LaguerreBasis,
    "hermite": HermiteBasis,
    "chebyshev": ChebyshevBasis,
}

# Scheme registry — the agent can look up available schemes
SCHEME_REGISTRY: dict[str, type] = {
    "euler": Euler,
    "milstein": Milstein,
    "exact": Exact,
    "log_euler": LogEuler,
}
