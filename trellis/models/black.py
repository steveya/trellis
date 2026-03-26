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


def garman_kohlhagen_call(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    df_domestic: float,
    df_foreign: float,
) -> float:
    """Domestic-currency FX vanilla call under Garman-Kohlhagen.

    Parameters
    ----------
    spot
        Spot FX quote using the same convention as :class:`trellis.instruments.fx.FXRate`.
    strike
        Strike in domestic currency per unit of foreign currency.
    sigma
        Lognormal FX volatility.
    T
        Time to expiry in years.
    df_domestic
        Domestic discount factor to expiry.
    df_foreign
        Foreign discount factor to expiry.
    """
    if T <= 0:
        return float(np.maximum(spot - strike, 0.0))
    if sigma <= 0:
        return float(np.maximum(spot * df_foreign - strike * df_domestic, 0.0))
    forward = spot * df_foreign / df_domestic
    return float(df_domestic) * black76_call(float(forward), float(strike), sigma, T)


def garman_kohlhagen_put(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    df_domestic: float,
    df_foreign: float,
) -> float:
    """Domestic-currency FX vanilla put under Garman-Kohlhagen."""
    if T <= 0:
        return float(np.maximum(strike - spot, 0.0))
    if sigma <= 0:
        return float(np.maximum(strike * df_domestic - spot * df_foreign, 0.0))
    forward = spot * df_foreign / df_domestic
    return float(df_domestic) * black76_put(float(forward), float(strike), sigma, T)
