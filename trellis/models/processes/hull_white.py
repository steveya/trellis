"""Hull-White (extended Vasicek) process: dr = (theta(t) - a*r)dt + sigma*dW."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


class HullWhite(StochasticProcess):
    """Hull-White one-factor short-rate model.

    Parameters
    ----------
    a : float
        Mean reversion speed.
    sigma : float
        Volatility.
    theta_fn : callable(t) -> float, optional
        Time-dependent drift. If None, uses constant ``theta``.
    theta : float
        Constant theta (used when theta_fn is None).
    """

    def __init__(self, a: float, sigma: float,
                 theta_fn=None, theta: float = 0.0):
        """Store mean reversion, volatility, and either functional or constant theta."""
        self.a = a
        self.sigma = sigma
        self._theta_fn = theta_fn
        self._theta_const = theta

    def theta(self, t: float) -> float:
        """Return the time-dependent drift level ``theta(t)`` used by the model."""
        if self._theta_fn is not None:
            return self._theta_fn(t)
        return self._theta_const

    def drift(self, x, t):
        """Return the Hull-White drift ``theta(t) - a r_t``."""
        return self.theta(t) - self.a * x

    def diffusion(self, x, t):
        """Return the constant short-rate volatility ``sigma``."""
        return self.sigma

    def exact_mean(self, x, t, dt):
        """Return a one-step conditional mean using a frozen-``theta`` approximation.

        When ``theta`` is time-dependent, this implementation holds ``theta(t)``
        constant over ``[t, t + dt]`` rather than integrating it exactly.
        """
        e = np.exp(-self.a * dt)
        # Approximation for time-dependent theta: use theta(t) constant over dt
        th = self.theta(t)
        return x * e + (th / self.a) * (1 - e)

    def exact_variance(self, x, t, dt):
        """Return the conditional variance under the constant-vol Gaussian transition."""
        return (self.sigma ** 2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * dt))
