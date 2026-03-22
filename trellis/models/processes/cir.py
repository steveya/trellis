"""CIR (Cox-Ingersoll-Ross) process: dr = a(b - r)dt + sigma*sqrt(r)*dW."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


class CIR(StochasticProcess):
    """Cox-Ingersoll-Ross square-root diffusion.

    Parameters
    ----------
    a : float
        Mean reversion speed.
    b : float
        Long-term mean level.
    sigma : float
        Vol-of-vol parameter.

    Note: Feller condition 2*a*b > sigma^2 ensures positivity.
    """

    def __init__(self, a: float, b: float, sigma: float):
        self.a = a
        self.b = b
        self.sigma = sigma

    def drift(self, x, t):
        return self.a * (self.b - x)

    def diffusion(self, x, t):
        return self.sigma * np.sqrt(np.maximum(x, 0.0))

    def exact_mean(self, x, t, dt):
        return x * np.exp(-self.a * dt) + self.b * (1 - np.exp(-self.a * dt))

    def exact_variance(self, x, t, dt):
        e = np.exp(-self.a * dt)
        return (x * self.sigma ** 2 * e / self.a * (1 - e)
                + self.b * self.sigma ** 2 / (2 * self.a) * (1 - e) ** 2)
