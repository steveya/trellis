"""SDE discretization schemes."""

from __future__ import annotations

import numpy as raw_np


def euler_maruyama(
    process,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
) -> raw_np.ndarray:
    """Euler-Maruyama discretization of a 1D SDE.

    Parameters
    ----------
    process : StochasticProcess
        Must have drift(x, t) and diffusion(x, t) methods.
    x0 : float
        Initial value.
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of sample paths.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Simulated paths including the initial value.
    """
    if rng is None:
        rng = raw_np.random.default_rng()

    dt = T / n_steps
    sqrt_dt = raw_np.sqrt(dt)
    paths = raw_np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for i in range(n_steps):
        t = i * dt
        x = paths[:, i]
        dw = rng.standard_normal(n_paths)
        mu = raw_np.vectorize(lambda xi: process.drift(xi, t))(x)
        sig = raw_np.vectorize(lambda xi: process.diffusion(xi, t))(x)
        paths[:, i + 1] = x + mu * dt + sig * sqrt_dt * dw

    return paths


def milstein(
    process,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
) -> raw_np.ndarray:
    """Milstein scheme (higher-order for scalar diffusion).

    Requires process to have a diffusion_derivative method or uses
    finite-difference approximation.
    """
    if rng is None:
        rng = raw_np.random.default_rng()

    dt = T / n_steps
    sqrt_dt = raw_np.sqrt(dt)
    paths = raw_np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    eps = 1e-6  # for finite-difference derivative of diffusion

    for i in range(n_steps):
        t = i * dt
        x = paths[:, i]
        dw = rng.standard_normal(n_paths)

        mu = raw_np.vectorize(lambda xi: process.drift(xi, t))(x)
        sig = raw_np.vectorize(lambda xi: process.diffusion(xi, t))(x)
        # Finite-difference approximation of d(sigma)/dx
        sig_up = raw_np.vectorize(lambda xi: process.diffusion(xi + eps, t))(x)
        dsig_dx = (sig_up - sig) / eps

        paths[:, i + 1] = (
            x + mu * dt + sig * sqrt_dt * dw
            + 0.5 * sig * dsig_dx * (dw ** 2 - 1) * dt
        )

    return paths


def exact_simulation(
    process,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
) -> raw_np.ndarray:
    """Exact simulation using process.exact_sample if available."""
    if rng is None:
        rng = raw_np.random.default_rng()

    dt = T / n_steps
    paths = raw_np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for i in range(n_steps):
        t = i * dt
        dw = rng.standard_normal(n_paths)
        for p in range(n_paths):
            paths[p, i + 1] = process.exact_sample(paths[p, i], t, dt, dw[p])

    return paths
