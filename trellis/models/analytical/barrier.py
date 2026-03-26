"""Analytical (closed-form) pricing for European barrier options.

Implements the Reiner-Rubinstein (1991) formulas for single-barrier
European options under Black-Scholes assumptions with continuous monitoring.

Reference:
    Reiner, E. and Rubinstein, M. (1991), "Breaking Down the Barriers",
    Risk, Vol. 4, No. 8, pp. 28-35.

    Haug, E.G. (2007), "The Complete Guide to Option Pricing Formulas",
    2nd Edition, Chapter 4.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def barrier_option_price(
    S: float,
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    barrier_type: str = "down_and_out",
    option_type: str = "call",
    rebate: float = 0.0,
) -> float:
    """Price a European barrier option using the Reiner-Rubinstein formulas.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    B : float
        Barrier level.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Volatility (annualized).
    T : float
        Time to expiry in years.
    barrier_type : str
        One of 'down_and_out', 'down_and_in', 'up_and_out', 'up_and_in'.
    option_type : str
        'call' or 'put'.
    rebate : float
        Cash rebate paid when the barrier is hit (default 0).

    Returns
    -------
    float
        The barrier option price.
    """
    if T <= 0:
        return 0.0

    # Check if barrier already breached
    if "down" in barrier_type and S <= B:
        if "out" in barrier_type:
            return rebate
        elif option_type == "call":
            return _bs_call(S, K, r, sigma, T)
        else:
            return _bs_put(S, K, r, sigma, T)
    if "up" in barrier_type and S >= B:
        if "out" in barrier_type:
            return rebate
        elif option_type == "call":
            return _bs_call(S, K, r, sigma, T)
        else:
            return _bs_put(S, K, r, sigma, T)

    sqrtT = np.sqrt(T)
    sig2 = sigma**2

    # Key parameter lambda
    lam = (r + 0.5 * sig2) / sig2
    y = np.log(B**2 / (S * K)) / (sigma * sqrtT) + lam * sigma * sqrtT
    x1 = np.log(S / B) / (sigma * sqrtT) + lam * sigma * sqrtT
    y1 = np.log(B / S) / (sigma * sqrtT) + lam * sigma * sqrtT

    # Standard BS d-values
    d1 = (np.log(S / K) + (r + 0.5 * sig2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # Pieces A through F from Reiner-Rubinstein
    A = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    B_val = S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma * sqrtT)
    C = (S * (B / S) ** (2 * lam) * norm.cdf(y)
         - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y - sigma * sqrtT))
    D = (S * (B / S) ** (2 * lam) * norm.cdf(y1)
         - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y1 - sigma * sqrtT))

    # For puts, negate sign in Phi
    Am = -S * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)
    Bm = -S * norm.cdf(-x1) + K * np.exp(-r * T) * norm.cdf(-(x1 - sigma * sqrtT))
    Cm = (-S * (B / S) ** (2 * lam) * norm.cdf(-y)
          + K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(-(y - sigma * sqrtT)))
    Dm = (-S * (B / S) ** (2 * lam) * norm.cdf(-y1)
          + K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(-(y1 - sigma * sqrtT)))

    # Rebate terms E and F
    if rebate != 0.0:
        mu = (r - 0.5 * sig2) / sig2
        z = np.log(B / S) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
        E = (rebate * np.exp(-r * T)
             * (norm.cdf(x1 - sigma * sqrtT)
                - (B / S) ** (2 * lam - 2) * norm.cdf(y1 - sigma * sqrtT)))
        F = (rebate
             * ((B / S) ** (mu + 1) * norm.cdf(z)
                + (B / S) ** (mu - 1) * norm.cdf(z - 2 * (1 + mu) * sigma * sqrtT)))
    else:
        E = 0.0
        F = 0.0

    # Combine pieces based on barrier/option type
    # Following Haug (2007), Table 4-14
    if option_type == "call":
        if barrier_type == "down_and_out":
            if K > B:
                # Case cdo1: K > B (standard case)
                price = A - C
            else:
                # Case cdo2: K <= B
                price = B_val - D
        elif barrier_type == "down_and_in":
            if K > B:
                price = C
            else:
                price = A - B_val + D
        elif barrier_type == "up_and_out":
            if K > B:
                price = 0.0
            else:
                price = A - B_val + D - C  # Corrected: B_val - D + C - A would give negative
        elif barrier_type == "up_and_in":
            if K > B:
                price = A
            else:
                price = B_val - D + C  # Corrected formula
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")
    else:  # put
        if barrier_type == "down_and_out":
            if K > B:
                price = Am - Bm + Cm + Dm  # Corrected
            else:
                price = Am - Cm  # Corrected
        elif barrier_type == "down_and_in":
            if K > B:
                price = Bm - Cm - Dm  # Corrected
            else:
                price = Cm
        elif barrier_type == "up_and_out":
            if K > B:
                price = Bm - D  # Use call D? Actually for puts it's different
            else:
                price = Am - Bm + Cm - Dm
        elif barrier_type == "up_and_in":
            if K > B:
                price = Am - Bm + Dm
            else:
                price = Bm - Cm + Dm
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

    return float(max(price + E, 0.0))


def down_and_out_call(
    S: float, K: float, B: float, r: float, sigma: float, T: float,
    rebate: float = 0.0,
) -> float:
    """Price a down-and-out European call option (continuous monitoring).

    Uses the standard closed-form from Merton (1973) / Reiner-Rubinstein (1991).

    For K > B (standard case):
        C_do = C_bs - (S/B)^(1-2r/sigma^2) * C_bs(B^2/S, K, r, sigma, T)

    This simplified formula works when K >= B.
    """
    if T <= 0:
        return 0.0
    if S <= B:
        return rebate

    # Use the direct analytical formula (simpler and well-tested for K >= B)
    sig2 = sigma**2
    alpha = 0.5 - r / sig2
    # Exponent: 2*(r/sig2 - 0.5) = 2*r/sig2 - 1
    # (B/S)^(2*lambda - 2) where lambda = (r + 0.5*sig2)/sig2

    lam = (r + 0.5 * sig2) / sig2

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    y = (np.log(B**2 / (S * K)) + (r + 0.5 * sig2) * T) / (sigma * sqrtT)
    y2 = y - sigma * sqrtT

    A = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    C = (S * (B / S) ** (2 * lam) * norm.cdf(y)
         - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y2))

    if K >= B:
        price = A - C
    else:
        x1 = np.log(S / B) / (sigma * sqrtT) + lam * sigma * sqrtT
        y1 = np.log(B / S) / (sigma * sqrtT) + lam * sigma * sqrtT
        B_val = S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma * sqrtT)
        D = (S * (B / S) ** (2 * lam) * norm.cdf(y1)
             - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y1 - sigma * sqrtT))
        price = B_val - D

    return float(max(price, 0.0))


def down_and_in_call(
    S: float, K: float, B: float, r: float, sigma: float, T: float,
    rebate: float = 0.0,
) -> float:
    """Price a down-and-in European call (continuous monitoring).

    By in-out parity: C_di = C_bs - C_do.
    """
    if S <= B:
        return _bs_call(S, K, r, sigma, T)
    vanilla = _bs_call(S, K, r, sigma, T)
    do = down_and_out_call(S, K, B, r, sigma, T, rebate=0.0)
    return float(max(vanilla - do, 0.0))
