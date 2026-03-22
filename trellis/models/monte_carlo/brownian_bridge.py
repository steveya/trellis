"""Brownian bridge construction for path generation."""

from __future__ import annotations

import numpy as raw_np


def brownian_bridge(
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
    end_values: raw_np.ndarray | None = None,
) -> raw_np.ndarray:
    """Generate Brownian motion paths via bridge construction.

    Bridge construction fills in intermediate points conditional on
    endpoints, improving stratification and convergence.

    Parameters
    ----------
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    rng : numpy.random.Generator, optional
    end_values : ndarray of shape (n_paths,), optional
        Terminal W(T) values. If None, sampled from N(0, T).

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Brownian motion paths with W(0) = 0.
    """
    if rng is None:
        rng = raw_np.random.default_rng()

    dt = T / n_steps
    W = raw_np.zeros((n_paths, n_steps + 1))

    if end_values is None:
        W[:, -1] = rng.normal(0, raw_np.sqrt(T), size=n_paths)
    else:
        W[:, -1] = end_values

    # Recursive bisection: fill midpoints
    _fill_bridge(W, 0, n_steps, T, rng)

    return W


def _fill_bridge(W, i_start, i_end, T, rng):
    """Recursively fill the bridge between indices i_start and i_end."""
    if i_end - i_start <= 1:
        return

    n_steps_total = W.shape[1] - 1
    dt = T / n_steps_total

    i_mid = (i_start + i_end) // 2
    t_start = i_start * dt
    t_mid = i_mid * dt
    t_end = i_end * dt

    tau = t_end - t_start
    if tau == 0:
        return

    # Bridge mean and variance
    alpha = (t_mid - t_start) / tau
    bridge_var = (t_mid - t_start) * (t_end - t_mid) / tau

    n_paths = W.shape[0]
    W[:, i_mid] = (
        W[:, i_start] * (1 - alpha) + W[:, i_end] * alpha
        + rng.normal(0, raw_np.sqrt(bridge_var), size=n_paths)
    )

    _fill_bridge(W, i_start, i_mid, T, rng)
    _fill_bridge(W, i_mid, i_end, T, rng)
