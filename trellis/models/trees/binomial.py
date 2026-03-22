"""Binomial tree construction (CRR, Jarrow-Rudd, Leisen-Reimer)."""

from __future__ import annotations

import numpy as raw_np

from trellis.core.differentiable import get_numpy

np = get_numpy()


class BinomialTree:
    """Recombining binomial tree for a 1D process.

    Parameters
    ----------
    S0 : float
        Initial value.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    u : float
        Up factor per step.
    d : float
        Down factor per step.
    p : float
        Risk-neutral probability of up move.
    dt : float
        Time step size.
    """

    def __init__(self, S0: float, T: float, n_steps: int,
                 u: float, d: float, p: float):
        self.S0 = S0
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.u = u
        self.d = d
        self.p = p

        # Build the tree: node (i, j) has value S0 * u^j * d^(i-j)
        # where i = time step, j = number of up moves
        self._values = self._build_tree()

    def _build_tree(self) -> raw_np.ndarray:
        """Build (n+1) x (n+1) array of node values."""
        n = self.n_steps
        tree = raw_np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                tree[i, j] = self.S0 * self.u ** j * self.d ** (i - j)
        return tree

    def value_at(self, step: int, node: int) -> float:
        """Return the process value at tree node (step, node)."""
        return self._values[step, node]

    def terminal_values(self) -> raw_np.ndarray:
        """Values at maturity: shape (n_steps + 1,)."""
        n = self.n_steps
        return self._values[n, :n + 1]

    @classmethod
    def crr(cls, S0: float, T: float, n_steps: int,
            r: float, sigma: float) -> BinomialTree:
        """Cox-Ross-Rubinstein parameterization.

        u = exp(sigma * sqrt(dt)), d = 1/u, p = (exp(r*dt) - d) / (u - d).
        """
        dt = T / n_steps
        u = raw_np.exp(sigma * raw_np.sqrt(dt))
        d = 1.0 / u
        p = (raw_np.exp(r * dt) - d) / (u - d)
        return cls(S0, T, n_steps, u, d, p)

    @classmethod
    def jarrow_rudd(cls, S0: float, T: float, n_steps: int,
                    r: float, sigma: float) -> BinomialTree:
        """Jarrow-Rudd (equal probability) parameterization.

        p = 0.5, u and d chosen to match first two moments.
        """
        dt = T / n_steps
        u = raw_np.exp((r - 0.5 * sigma ** 2) * dt + sigma * raw_np.sqrt(dt))
        d = raw_np.exp((r - 0.5 * sigma ** 2) * dt - sigma * raw_np.sqrt(dt))
        return cls(S0, T, n_steps, u, d, 0.5)

    @classmethod
    def from_rate_vol(cls, r0: float, T: float, n_steps: int,
                      sigma: float) -> BinomialTree:
        """Build a short-rate tree (BDT-style, simplified).

        For short rates: u = exp(sigma*sqrt(dt)), d = 1/u applied to rate.
        """
        return cls.crr(r0, T, n_steps, r0, sigma)
