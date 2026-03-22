"""Heston stochastic volatility model.

dS = mu*S*dt + sqrt(V)*S*dW_1
dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_2
corr(dW_1, dW_2) = rho
"""

from __future__ import annotations

from trellis.core.differentiable import get_numpy

np = get_numpy()


class Heston:
    """Heston stochastic volatility model (two-factor).

    Parameters
    ----------
    mu : float
        Drift of the spot process.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-term variance level.
    xi : float
        Vol-of-vol.
    rho : float
        Correlation between spot and variance Brownians.
    v0 : float
        Initial variance.
    """

    def __init__(self, mu: float, kappa: float, theta: float,
                 xi: float, rho: float, v0: float):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0

    def drift_s(self, s, v, t):
        return self.mu * s

    def diffusion_s(self, s, v, t):
        return np.sqrt(np.maximum(v, 0.0)) * s

    def drift_v(self, s, v, t):
        return self.kappa * (self.theta - v)

    def diffusion_v(self, s, v, t):
        return self.xi * np.sqrt(np.maximum(v, 0.0))

    def characteristic_function(self, u, t):
        """Heston characteristic function phi(u, t) for log-spot.

        Used by FFT and COS pricing methods.
        """
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho
        v0 = self.v0

        d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        C = (kappa * theta / xi ** 2) * (
            (kappa - rho * xi * 1j * u - d) * t
            - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        )
        D = ((kappa - rho * xi * 1j * u - d) / xi ** 2) * (
            (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        )

        return np.exp(C + D * v0 + 1j * u * self.mu * t)
