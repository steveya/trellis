"""Brownian bridge construction for path generation."""

from __future__ import annotations

from functools import lru_cache

import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit


@maybe_njit(cache=False)
def _apply_bridge_schedule_numba(
    W: raw_np.ndarray,
    start_indices: raw_np.ndarray,
    mid_indices: raw_np.ndarray,
    end_indices: raw_np.ndarray,
    alphas: raw_np.ndarray,
    bridge_sigmas: raw_np.ndarray,
    shocks: raw_np.ndarray,
) -> None:
    """Fill bridge midpoints using a precomputed schedule."""
    n_fills = len(mid_indices)
    n_paths = W.shape[0]
    for k in range(n_fills):
        i_start = start_indices[k]
        i_mid = mid_indices[k]
        i_end = end_indices[k]
        alpha = alphas[k]
        sigma = bridge_sigmas[k]
        one_minus_alpha = 1.0 - alpha
        for path_idx in range(n_paths):
            W[path_idx, i_mid] = (
                one_minus_alpha * W[path_idx, i_start]
                + alpha * W[path_idx, i_end]
                + sigma * shocks[k, path_idx]
            )


@lru_cache(maxsize=None)
def _bridge_schedule(T: float, n_steps: int) -> tuple[raw_np.ndarray, ...]:
    """Return the midpoint-fill schedule matching recursive bridge order."""
    if n_steps <= 1:
        empty_int = raw_np.empty(0, dtype=raw_np.int64)
        empty_float = raw_np.empty(0, dtype=float)
        return empty_int, empty_int, empty_int, empty_float, empty_float

    dt = T / n_steps
    schedule: list[tuple[int, int, int, float, float]] = []

    def visit(i_start: int, i_end: int) -> None:
        if i_end - i_start <= 1:
            return

        i_mid = (i_start + i_end) // 2
        t_start = i_start * dt
        t_mid = i_mid * dt
        t_end = i_end * dt
        tau = t_end - t_start
        if tau <= 0.0:
            return

        alpha = (t_mid - t_start) / tau
        bridge_sigma = raw_np.sqrt((t_mid - t_start) * (t_end - t_mid) / tau)
        schedule.append((i_start, i_mid, i_end, alpha, bridge_sigma))

        visit(i_start, i_mid)
        visit(i_mid, i_end)

    visit(0, n_steps)

    start_indices = raw_np.asarray([item[0] for item in schedule], dtype=raw_np.int64)
    mid_indices = raw_np.asarray([item[1] for item in schedule], dtype=raw_np.int64)
    end_indices = raw_np.asarray([item[2] for item in schedule], dtype=raw_np.int64)
    alphas = raw_np.asarray([item[3] for item in schedule], dtype=float)
    bridge_sigmas = raw_np.asarray([item[4] for item in schedule], dtype=float)
    return start_indices, mid_indices, end_indices, alphas, bridge_sigmas


def brownian_bridge(
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
    end_values: raw_np.ndarray | None = None,
    bridge_shocks: raw_np.ndarray | None = None,
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
    bridge_shocks : ndarray of shape (n_paths, n_steps), optional
        Standard-normal shocks in bridge order. Column 0 sets the terminal
        draw, remaining columns fill recursive midpoints in schedule order.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Brownian motion paths with W(0) = 0.
    """
    if rng is None:
        rng = raw_np.random.default_rng()

    W = raw_np.zeros((n_paths, n_steps + 1), dtype=float)

    shocks = None
    if bridge_shocks is not None:
        shocks = raw_np.asarray(bridge_shocks, dtype=float)
        if shocks.shape != (n_paths, n_steps):
            raise ValueError(f"bridge_shocks must have shape ({n_paths}, {n_steps})")

    if end_values is None:
        if shocks is not None:
            W[:, -1] = raw_np.sqrt(T) * shocks[:, 0]
        else:
            W[:, -1] = rng.normal(0.0, raw_np.sqrt(T), size=n_paths)
    else:
        W[:, -1] = raw_np.asarray(end_values, dtype=float)

    start_indices, mid_indices, end_indices, alphas, bridge_sigmas = _bridge_schedule(T, n_steps)
    if len(mid_indices) == 0:
        return W

    if NUMBA_AVAILABLE:
        bridge_mid_shocks = (
            shocks[:, 1:].T.copy()
            if shocks is not None
            else rng.standard_normal((len(mid_indices), n_paths))
        )
        _apply_bridge_schedule_numba(
            W,
            start_indices,
            mid_indices,
            end_indices,
            alphas,
            bridge_sigmas,
            bridge_mid_shocks,
        )
        return W

    for idx, (i_start, i_mid, i_end, alpha, bridge_sigma) in enumerate(
        zip(
            start_indices,
            mid_indices,
            end_indices,
            alphas,
            bridge_sigmas,
        )
    ):
        W[:, i_mid] = (
            (1.0 - alpha) * W[:, i_start]
            + alpha * W[:, i_end]
            + bridge_sigma * (
                shocks[:, idx + 1] if shocks is not None else rng.standard_normal(n_paths)
            )
        )

    return W
