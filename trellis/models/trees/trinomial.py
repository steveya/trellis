"""Trinomial tree construction."""

from __future__ import annotations

import numpy as raw_np

from trellis.core.differentiable import get_numpy

np = get_numpy()


class TrinomialTree:
    """Recombining trinomial tree.

    Parameters
    ----------
    S0 : float
        Initial value.
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    u : float
        Up factor.
    d : float
        Down factor.
    pu : float
        Probability of up move.
    pm : float
        Probability of middle (stay).
    pd : float
        Probability of down move.
    """

    def __init__(self, S0: float, T: float, n_steps: int,
                 u: float, d: float, pu: float, pm: float, pd: float):
        """Store trinomial parameters and prebuild the recombining lattice values."""
        self.S0 = S0
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.u = u
        self.d = d
        self.pu = pu
        self.pm = pm
        self.pd = pd
        self._values = self._build_tree()

    def _build_tree(self) -> raw_np.ndarray:
        """Build (n+1) x (2n+1) array. Node (i, j) where j is centered."""
        n = self.n_steps
        steps = np.arange(n + 1)[:, None]
        offsets = np.arange(-n, n + 1)[None, :]
        values = self.S0 * (self.u ** np.maximum(offsets, 0)) * (self.d ** np.maximum(-offsets, 0))
        return np.where(np.abs(offsets) <= steps, values, 0.0)

    def value_at(self, step: int, node_offset: int) -> float:
        """Return value at (step, offset from center)."""
        return self._values[step, self.n_steps + node_offset]

    @classmethod
    def standard(cls, S0: float, T: float, n_steps: int,
                 r: float, sigma: float) -> TrinomialTree:
        """Standard trinomial parameterization.

        u = exp(sigma * sqrt(2*dt)), d = 1/u, middle = 1.
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1.0 / u
        # Risk-neutral probabilities
        erddt = np.exp(r * dt / 2)
        esvdt = np.exp(sigma * np.sqrt(dt / 2))
        pu = ((erddt - 1 / esvdt) / (esvdt - 1 / esvdt)) ** 2
        pd = ((esvdt - erddt) / (esvdt - 1 / esvdt)) ** 2
        pm = 1.0 - pu - pd
        return cls(S0, T, n_steps, u, d, pu, pm, pd)
