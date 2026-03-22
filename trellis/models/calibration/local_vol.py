"""Dupire local volatility surface construction."""

from __future__ import annotations

import numpy as raw_np


def dupire_local_vol(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    implied_vols: raw_np.ndarray,
    S0: float,
    r: float,
) -> callable:
    """Construct Dupire local vol function from an implied vol surface.

    Parameters
    ----------
    strikes : ndarray of shape (n_K,)
    expiries : ndarray of shape (n_T,)
    implied_vols : ndarray of shape (n_T, n_K)
        Market implied vols.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.

    Returns
    -------
    callable(S, t) -> float
        Local volatility function.
    """
    from scipy.interpolate import RectBivariateSpline

    # Fit a smooth surface to implied vols
    spline = RectBivariateSpline(expiries, strikes, implied_vols)

    def local_vol(S, t):
        t = max(t, 1e-6)
        K = S  # local vol evaluated at S=K

        sigma = float(spline(t, K, grid=False))
        dsigma_dT = float(spline(t, K, dx=1, grid=False))
        dsigma_dK = float(spline(t, K, dy=1, grid=False))
        d2sigma_dK2 = float(spline(t, K, dy=2, grid=False))

        d1 = (raw_np.log(S0 / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * raw_np.sqrt(t))

        # Dupire formula
        numer = sigma ** 2 + 2 * sigma * t * (dsigma_dT + r * K * dsigma_dK)
        denom = (1 + K * d1 * raw_np.sqrt(t) * dsigma_dK) ** 2 + \
                K ** 2 * t * sigma * (d2sigma_dK2 - d1 * raw_np.sqrt(t) * dsigma_dK ** 2)

        if denom <= 0:
            return sigma  # fallback
        return raw_np.sqrt(max(numer / denom, 0))

    return local_vol
