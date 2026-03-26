"""Vasicek (Ornstein-Uhlenbeck) process: dr = a(b - r)dt + sigma*dW."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


class Vasicek(StochasticProcess):
    """Vasicek / Ornstein-Uhlenbeck mean-reverting process.

    Parameters
    ----------
    a : float
        Mean reversion speed.
    b : float
        Long-term mean level.
    sigma : float
        Volatility.
    """

    def __init__(self, a: float, b: float, sigma: float):
        """Store the Ornstein-Uhlenbeck mean-reversion parameters."""
        self.a = a
        self.b = b
        self.sigma = sigma

    def drift(self, x, t):
        """Return the mean-reverting drift ``a (b - x)``."""
        return self.a * (self.b - x)

    def diffusion(self, x, t):
        """Return the constant Vasicek diffusion coefficient ``sigma``."""
        return self.sigma

    def exact_sample(self, x, t, dt, dw):
        """Sample from the exact Gaussian transition using its closed-form moments."""
        mean = self.exact_mean(x, t, dt)
        std = np.sqrt(self.exact_variance(x, t, dt))
        return mean + std * dw

    def exact_mean(self, x, t, dt):
        """Return the conditional Vasicek mean over horizon ``dt``."""
        return x * np.exp(-self.a * dt) + self.b * (1 - np.exp(-self.a * dt))

    def exact_variance(self, x, t, dt):
        """Return the conditional Vasicek variance over horizon ``dt``."""
        return (self.sigma ** 2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * dt))
