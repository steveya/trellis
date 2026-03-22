"""Black76 option pricing formulas — autograd-compatible."""

from __future__ import annotations

from scipy.stats import norm

from trellis.core.differentiable import get_numpy

np = get_numpy()


def _d1d2(F: float, K: float, sigma: float, T: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black76."""
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def black76_call(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 call price.

    Parameters
    ----------
    F : float
        Forward rate.
    K : float
        Strike rate.
    sigma : float
        Black (lognormal) volatility.
    T : float
        Time to expiry in years.

    Returns
    -------
    float
        Undiscounted call value: F*N(d1) - K*N(d2).
    """
    if sigma <= 0 or T <= 0:
        return float(np.maximum(F - K, 0.0))
    d1, d2 = _d1d2(F, K, sigma, T)
    return F * norm.cdf(float(d1)) - K * norm.cdf(float(d2))


def black76_put(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 put price.

    Returns
    -------
    float
        Undiscounted put value: K*N(-d2) - F*N(-d1).
    """
    if sigma <= 0 or T <= 0:
        return float(np.maximum(K - F, 0.0))
    d1, d2 = _d1d2(F, K, sigma, T)
    return K * norm.cdf(-float(d2)) - F * norm.cdf(-float(d1))
