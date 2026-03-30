"""Base class for random processes used in Monte Carlo simulation.

A stochastic process defines how a price (or set of prices) evolves
randomly over time. Subclasses specify the drift (expected direction)
and diffusion (randomness magnitude) at each point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trellis.core.differentiable import get_numpy

np = get_numpy()


class StochasticProcess(ABC):
    """Base class for random processes driven by Brownian motion.

    A scalar process (e.g. GBM) tracks one value per path — state arrays
    have shape (n_paths,). A multi-dimensional process (e.g. Heston with
    price + variance) uses shape (n_paths, state_dim).

    Subclasses must implement drift(x, t) and diffusion(x, t). They may
    also override exact_sample() for processes with known closed-form
    transitions (faster and more accurate than discretization).
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
        """Expected rate of change at state x and time t."""
        ...

    @abstractmethod
    def diffusion(self, x: float, t: float) -> float:
        """Volatility (randomness magnitude) at state x and time t."""
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
