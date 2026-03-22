"""Base class for stochastic processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trellis.core.differentiable import get_numpy

np = get_numpy()


class StochasticProcess(ABC):
    """Base class for 1D stochastic processes: dX = mu(X,t)dt + sigma(X,t)dW."""

    @abstractmethod
    def drift(self, x: float, t: float) -> float:
        """Return mu(x, t)."""
        ...

    @abstractmethod
    def diffusion(self, x: float, t: float) -> float:
        """Return sigma(x, t)."""
        ...

    def exact_sample(self, x: float, t: float, dt: float, dw: float) -> float:
        """Exact transition if available. Default: Euler-Maruyama."""
        return x + self.drift(x, t) * dt + self.diffusion(x, t) * dw * np.sqrt(dt)

    def exact_mean(self, x: float, t: float, dt: float) -> float:
        """Conditional mean E[X(t+dt) | X(t)=x] if known. Default: Euler."""
        return x + self.drift(x, t) * dt

    def exact_variance(self, x: float, t: float, dt: float) -> float:
        """Conditional variance if known. Default: Euler."""
        sigma = self.diffusion(x, t)
        return sigma ** 2 * dt
