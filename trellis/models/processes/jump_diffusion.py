"""Merton jump-diffusion: dS/S = (mu - lambda*k)dt + sigma*dW + J*dN."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


class MertonJumpDiffusion(StochasticProcess):
    """Merton jump-diffusion model.

    Parameters
    ----------
    mu : float
        Drift (before jump compensation).
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity (expected jumps per year).
    jump_mean : float
        Mean of log-jump size.
    jump_vol : float
        Std dev of log-jump size.
    """

    def __init__(self, mu: float, sigma: float, lam: float,
                 jump_mean: float, jump_vol: float):
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.jump_mean = jump_mean
        self.jump_vol = jump_vol
        # Compensator: k = E[e^J - 1]
        self.k = np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1

    def drift(self, x, t):
        return (self.mu - self.lam * self.k) * x

    def diffusion(self, x, t):
        return self.sigma * x

    def sample_jump(self, dt: float, rng=None) -> float:
        """Sample the jump component over interval dt.

        Returns the multiplicative jump factor (1.0 = no jump).
        """
        if rng is None:
            import numpy as raw_np
            rng = raw_np.random.default_rng()

        n_jumps = rng.poisson(self.lam * dt)
        if n_jumps == 0:
            return 1.0
        log_jumps = rng.normal(self.jump_mean, self.jump_vol, size=n_jumps)
        return float(np.exp(np.sum(log_jumps)))
