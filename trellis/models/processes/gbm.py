"""Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW."""

from __future__ import annotations

from bisect import bisect_right
from math import isfinite

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
        """Store the constant drift and volatility parameters of the diffusion."""
        self.mu = mu
        self.sigma = sigma

    def drift(self, x, t):
        """Return the GBM drift term ``\\mu S_t``."""
        return self.mu * x

    def diffusion(self, x, t):
        """Return the GBM diffusion loading ``\\sigma S_t``."""
        return self.sigma * x

    def exact_sample(self, x, t, dt, dw):
        """Exact log-normal transition."""
        return x * np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * dw
        )

    def exact_mean(self, x, t, dt):
        """Return ``E[S_{t+dt} | S_t = x]`` under the exact lognormal transition."""
        return x * np.exp(self.mu * dt)

    def exact_variance(self, x, t, dt):
        """Return ``Var[S_{t+dt} | S_t = x]`` for the exact GBM transition."""
        return x ** 2 * np.exp(2 * self.mu * dt) * (np.exp(self.sigma ** 2 * dt) - 1)


class PiecewiseConstantGBM(StochasticProcess):
    """GBM with deterministic drift and volatility on ordered time intervals.

    ``interval_ends[i]`` closes the interval governed by ``mus[i]`` and
    ``sigmas[i]``. Exact transitions integrate drift and variance across every
    crossed interval, so simulation steps do not need to align with regime
    boundaries.
    """

    def __init__(self, interval_ends, mus, sigmas):
        """Validate and store one drift/volatility pair per time interval."""
        ends = tuple(float(value) for value in interval_ends)
        drift_rates = tuple(float(value) for value in mus)
        volatilities = tuple(float(value) for value in sigmas)
        if not ends:
            raise ValueError("PiecewiseConstantGBM requires at least one interval")
        if len(ends) != len(drift_rates) or len(ends) != len(volatilities):
            raise ValueError(
                "PiecewiseConstantGBM interval_ends, mus, and sigmas must have the same length"
            )
        if any(not isfinite(value) or value <= 0.0 for value in ends):
            raise ValueError("PiecewiseConstantGBM interval ends must be finite and positive")
        if any(right <= left for left, right in zip(ends, ends[1:])):
            raise ValueError("PiecewiseConstantGBM interval ends must be strictly increasing")
        if any(not isfinite(value) for value in drift_rates):
            raise ValueError("PiecewiseConstantGBM drift rates must be finite")
        if any(not isfinite(value) or value < 0.0 for value in volatilities):
            raise ValueError("PiecewiseConstantGBM volatilities must be finite and non-negative")
        self.interval_ends = ends
        self.mus = drift_rates
        self.sigmas = volatilities

    def _parameter_index(self, t: float) -> int:
        return min(bisect_right(self.interval_ends, float(t)), len(self.interval_ends) - 1)

    def _integrated_moments(self, t: float, dt: float) -> tuple[float, float]:
        start = float(t)
        width = float(dt)
        if not isfinite(start) or start < 0.0:
            raise ValueError("PiecewiseConstantGBM transition time must be finite and non-negative")
        if not isfinite(width) or width < 0.0:
            raise ValueError("PiecewiseConstantGBM transition width must be finite and non-negative")
        end = start + width
        integrated_mu = 0.0
        integrated_variance = 0.0
        current = start
        while current < end:
            index = self._parameter_index(current)
            segment_end = min(end, self.interval_ends[index])
            if segment_end <= current:
                segment_end = end
            segment_width = segment_end - current
            integrated_mu += self.mus[index] * segment_width
            integrated_variance += self.sigmas[index] ** 2 * segment_width
            current = segment_end
        return integrated_mu, integrated_variance

    def drift(self, x, t):
        """Return the active interval drift term."""
        return self.mus[self._parameter_index(t)] * x

    def diffusion(self, x, t):
        """Return the active interval diffusion loading."""
        return self.sigmas[self._parameter_index(t)] * x

    def exact_sample(self, x, t, dt, dw):
        """Sample the exact log-normal transition over all crossed intervals."""
        integrated_mu, integrated_variance = self._integrated_moments(t, dt)
        return x * np.exp(
            integrated_mu
            - 0.5 * integrated_variance
            + np.sqrt(integrated_variance) * dw
        )

    def exact_mean(self, x, t, dt):
        """Return the exact conditional mean over a piecewise interval."""
        integrated_mu, _ = self._integrated_moments(t, dt)
        return x * np.exp(integrated_mu)

    def exact_variance(self, x, t, dt):
        """Return the exact conditional variance over a piecewise interval."""
        integrated_mu, integrated_variance = self._integrated_moments(t, dt)
        return x ** 2 * np.exp(2.0 * integrated_mu) * (
            np.exp(integrated_variance) - 1.0
        )
