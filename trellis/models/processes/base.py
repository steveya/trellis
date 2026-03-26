"""Base class for stochastic processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trellis.core.differentiable import get_numpy

np = get_numpy()


class StochasticProcess(ABC):
    """Base class for stochastic processes.

    Scalar processes use state vectors of shape ``(n_paths,)``. Vector-state
    processes use ``(n_paths, state_dim)`` and may expose ``factor_dim``
    independent Brownian drivers.
    """

    @property
    def state_dim(self) -> int:
        """Return the process state dimension."""
        return 1

    @property
    def factor_dim(self) -> int:
        """Return the number of independent Brownian factors."""
        return self.state_dim

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
