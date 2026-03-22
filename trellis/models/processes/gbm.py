"""Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


class GBM(StochasticProcess):
    """Geometric Brownian Motion.

    Parameters
    ----------
    mu : float
        Drift rate (risk-neutral: r - q).
    sigma : float
        Volatility.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def drift(self, x, t):
        return self.mu * x

    def diffusion(self, x, t):
        return self.sigma * x

    def exact_sample(self, x, t, dt, dw):
        """Exact log-normal transition."""
        return x * np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * dw
        )

    def exact_mean(self, x, t, dt):
        return x * np.exp(self.mu * dt)

    def exact_variance(self, x, t, dt):
        return x ** 2 * np.exp(2 * self.mu * dt) * (np.exp(self.sigma ** 2 * dt) - 1)
