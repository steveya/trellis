"""Implied volatility solvers."""

from __future__ import annotations

import numpy as raw_np
from scipy.optimize import brentq
from scipy.stats import norm


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> float:
    """Find Black-Scholes implied vol via Brent's method.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S, K, T, r : float
        Spot, strike, time, rate.
    option_type : str
        ``"call"`` or ``"put"``.
    """
    def objective(sigma):
        return _bs_price(S, K, T, r, sigma, option_type) - market_price

    return brentq(objective, 1e-6, 5.0, xtol=tol)


def implied_vol_jaeckel(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Implied vol via rational approximation (Jaeckel-inspired).

    Uses Brent as fallback for edge cases.
    """
    # Normalized price
    F = S * raw_np.exp(r * T)
    df = raw_np.exp(-r * T)

    if option_type == "call":
        intrinsic = max(F - K, 0) * df
    else:
        intrinsic = max(K - F, 0) * df

    if market_price <= intrinsic + 1e-12:
        return 0.0

    # Initial guess from Brenner-Subrahmanyam
    sigma_init = raw_np.sqrt(2 * raw_np.pi / T) * market_price / S

    # Newton refinement
    sigma = max(sigma_init, 0.01)
    for _ in range(20):
        price = _bs_price(S, K, T, r, sigma, option_type)
        vega = _bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        sigma -= (price - market_price) / vega
        sigma = max(sigma, 1e-6)
        if abs(price - market_price) < 1e-10:
            return sigma

    # Fallback
    return implied_vol(market_price, S, K, T, r, option_type)


def _bs_price(S, K, T, r, sigma, option_type):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * raw_np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_vega(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * raw_np.sqrt(T))
    return S * raw_np.sqrt(T) * norm.pdf(d1)
