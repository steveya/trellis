"""Grid construction for finite difference methods."""

from __future__ import annotations

import numpy as raw_np


class Grid:
    """1D spatial grid for PDE solvers.

    Parameters
    ----------
    x_min, x_max : float
        Domain boundaries.
    n_x : int
        Number of spatial points.
    T : float
        Time horizon.
    n_t : int
        Number of time steps.
    log_spacing : bool
        If True, use log-spaced grid (better for lognormal processes).
    """

    def __init__(self, x_min: float, x_max: float, n_x: int,
                 T: float, n_t: int, log_spacing: bool = False):
        self.x_min = x_min
        self.x_max = x_max
        self.n_x = n_x
        self.T = T
        self.n_t = n_t
        self.dt = T / n_t
        self.log_spacing = log_spacing

        if log_spacing:
            self.x = raw_np.exp(raw_np.linspace(raw_np.log(max(x_min, 1e-10)),
                                                 raw_np.log(x_max), n_x))
        else:
            self.x = raw_np.linspace(x_min, x_max, n_x)

        self.dx = self.x[1] - self.x[0] if not log_spacing else None
        self.t = raw_np.linspace(0, T, n_t + 1)
