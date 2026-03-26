"""SABR stochastic volatility model.

dF = sigma * F^beta * dW_1
dsigma = nu * sigma * dW_2
corr(dW_1, dW_2) = rho
"""

from __future__ import annotations

from trellis.core.differentiable import get_numpy

np = get_numpy()


class SABRProcess:
    """SABR stochastic alpha-beta-rho model.

    Parameters
    ----------
    alpha : float
        Initial volatility.
    beta : float
        CEV exponent (0 = normal, 1 = lognormal).
    rho : float
        Correlation between forward and vol.
    nu : float
        Vol-of-vol.
    """

    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
        """Store the alpha, beta, rho, and nu parameters of the SABR model."""
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self, F: float, K: float, T: float) -> float:
        """Hagan et al. SABR implied vol approximation."""
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        if abs(F - K) < 1e-12:
            # ATM formula
            FK_mid = F
            logFK = 0.0
        else:
            FK_mid = np.sqrt(F * K)
            logFK = np.log(F / K)

        FK_beta = (F * K) ** ((1 - beta) / 2)

        # z and x(z)
        z = (nu / alpha) * FK_beta * logFK
        if abs(z) < 1e-12:
            xz = 1.0
        else:
            xz = z / np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # Numerator
        numer = alpha
        numer_corr = (1 + (
            ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2
            + (rho * beta * nu * alpha) / (4 * FK_beta)
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        ) * T)

        # Denominator
        denom = FK_beta * (1 + (1 - beta) ** 2 / 24 * logFK ** 2
                           + (1 - beta) ** 4 / 1920 * logFK ** 4)

        return (numer / denom) * xz * numer_corr
