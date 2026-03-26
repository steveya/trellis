"""SABR model calibration."""

from __future__ import annotations

import numpy as raw_np
from scipy.optimize import minimize

from trellis.models.processes.sabr import SABRProcess


def calibrate_sabr(
    F: float,
    T: float,
    strikes: list[float],
    market_vols: list[float],
    beta: float = 0.5,
) -> SABRProcess:
    """Calibrate SABR parameters (alpha, rho, nu) to market implied vols.

    Parameters
    ----------
    F : float
        Forward price.
    T : float
        Time to expiry.
    strikes : list[float]
        Strike prices.
    market_vols : list[float]
        Market implied volatilities at each strike.
    beta : float
        CEV exponent (typically fixed, not calibrated).

    Returns
    -------
    SABRProcess with calibrated parameters.
    """
    strikes = raw_np.asarray(strikes)
    market_vols = raw_np.asarray(market_vols)

    def objective(params):
        """Return the squared-error objective for one SABR parameter vector."""
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        sabr = SABRProcess(alpha, beta, rho, nu)
        model_vols = raw_np.array([sabr.implied_vol(F, K, T) for K in strikes])
        return float(raw_np.sum((model_vols - market_vols) ** 2))

    # Initial guess: ATM vol for alpha
    atm_idx = raw_np.argmin(raw_np.abs(strikes - F))
    alpha0 = market_vols[atm_idx] * F ** (1 - beta)

    result = minimize(
        objective,
        x0=[alpha0, 0.0, 0.3],
        bounds=[(1e-6, None), (-0.999, 0.999), (1e-6, None)],
        method="L-BFGS-B",
    )

    alpha, rho, nu = result.x
    return SABRProcess(alpha, beta, rho, nu)
