"""Closed-form pricing for European barrier options.

A barrier option is like a standard option but with an extra condition:
if the underlying price crosses a specified barrier level, the option
either activates ("knock-in") or is cancelled ("knock-out").

This module implements the Reiner-Rubinstein (1991) analytical formulas
for down-and-out / down-and-in calls and puts. The functions are split
into small building blocks (vanilla price, image term, rebate) that
support automatic differentiation.

Reference:
    Reiner, E. and Rubinstein, M. (1991), "Breaking Down the Barriers",
    Risk, Vol. 4, No. 8, pp. 28-35.

    Haug, E.G. (2007), "The Complete Guide to Option Pricing Formulas",
    2nd Edition, Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from autograd.scipy.stats import norm

from trellis.core.differentiable import get_numpy

np = get_numpy()


@dataclass(frozen=True)
class ResolvedBarrierInputs:
    """Pre-extracted market data needed for barrier option pricing."""

    spot: float
    strike: float
    barrier: float
    rate: float
    sigma: float
    T: float
    rebate: float = 0.0


def _bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European call price."""
    sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
    T_safe = np.where(T > 0.0, T, 1.0)
    sigma_sqrt_T = sigma_safe * np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    valid = (sigma > 0.0) & (T > 0.0)
    return np.where(valid, price, np.maximum(S - K, 0.0))


def _bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European put price."""
    sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
    T_safe = np.where(T > 0.0, T, 1.0)
    sigma_sqrt_T = sigma_safe * np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
    valid = (sigma > 0.0) & (T > 0.0)
    return np.where(valid, price, np.maximum(K - S, 0.0))


def vanilla_call_raw(resolved: ResolvedBarrierInputs) -> float:
    """Standard Black-Scholes call price (building block for barrier formulas)."""
    return _bs_call(
        resolved.spot,
        resolved.strike,
        resolved.rate,
        resolved.sigma,
        resolved.T,
    )


def barrier_image_raw(resolved: ResolvedBarrierInputs) -> float:
    """Reflection-principle correction term for the down-and-out barrier.

    This "image" term accounts for paths that would have crossed the barrier.
    Subtracting it from the vanilla price removes the value of those paths.
    """
    sqrt_T = np.sqrt(resolved.T)
    sig2 = resolved.sigma**2
    lam = (resolved.rate + 0.5 * sig2) / sig2
    y = (
        np.log(resolved.barrier**2 / (resolved.spot * resolved.strike))
        / (resolved.sigma * sqrt_T)
        + lam * resolved.sigma * sqrt_T
    )
    y2 = y - resolved.sigma * sqrt_T
    return (
        resolved.spot * (resolved.barrier / resolved.spot) ** (2 * lam) * norm.cdf(y)
        - resolved.strike
        * np.exp(-resolved.rate * resolved.T)
        * (resolved.barrier / resolved.spot) ** (2 * lam - 2) * norm.cdf(y2)
    )


def rebate_raw(resolved: ResolvedBarrierInputs) -> float:
    """Return the rebate paid if the barrier is hit.

    Currently returns the rebate field from resolved inputs (typically 0).
    Kept as a separate function for extensibility.
    """
    return resolved.rebate


def barrier_regime_selector_raw(resolved: ResolvedBarrierInputs) -> float:
    """Compute the down-and-out call price: vanilla - image + rebate.

    Only valid when strike > barrier and rebate is zero.
    """
    if resolved.rebate != 0.0:
        raise ValueError("The raw T09 barrier kernel only supports zero rebate.")
    if resolved.strike <= resolved.barrier:
        raise ValueError("The raw T09 barrier kernel only supports K > B.")
    return vanilla_call_raw(resolved) - barrier_image_raw(resolved) + rebate_raw(
        resolved
    )


def down_and_out_call_raw(resolved: ResolvedBarrierInputs) -> float:
    """Analytical price of a down-and-out call option."""
    return barrier_regime_selector_raw(resolved)


def down_and_in_call_raw(resolved: ResolvedBarrierInputs) -> float:
    """Analytical price of a down-and-in call (vanilla minus down-and-out)."""
    return barrier_image_raw(resolved) + rebate_raw(resolved)


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

    if barrier_type == "down_and_out" and option_type == "call":
        return down_and_out_call(S, K, B, r, sigma, T, rebate=rebate)
    if barrier_type == "down_and_in" and option_type == "call":
        return down_and_in_call(S, K, B, r, sigma, T, rebate=rebate)

    # Check if barrier already breached
    if "down" in barrier_type and S <= B:
        if "out" in barrier_type:
            return rebate
        elif option_type == "call":
            return float(_bs_call(S, K, r, sigma, T))
        else:
            return float(_bs_put(S, K, r, sigma, T))
    if "up" in barrier_type and S >= B:
        if "out" in barrier_type:
            return rebate
        elif option_type == "call":
            return float(_bs_call(S, K, r, sigma, T))
        else:
            return float(_bs_put(S, K, r, sigma, T))

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


def _legacy_down_and_out_call(
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


def _legacy_down_and_in_call(
    S: float, K: float, B: float, r: float, sigma: float, T: float,
    rebate: float = 0.0,
) -> float:
    """Price a down-and-in European call (continuous monitoring).

    By in-out parity: C_di = C_bs - C_do.
    """
    if S <= B:
        return float(_bs_call(S, K, r, sigma, T))
    vanilla = _bs_call(S, K, r, sigma, T)
    do = _legacy_down_and_out_call(S, K, B, r, sigma, T, rebate=0.0)
    return float(max(vanilla - do, 0.0))


def down_and_out_call(
    S: float, K: float, B: float, r: float, sigma: float, T: float,
    rebate: float = 0.0,
) -> float:
    """Price a down-and-out European call.

    The zero-rebate, K>B T09 branch is assembled from the route-local raw
    kernels. Other branches fall back to the legacy closed-form implementation.
    """
    if T <= 0:
        return 0.0
    if S <= B:
        return rebate
    if rebate == 0.0 and K > B:
        resolved = ResolvedBarrierInputs(
            spot=S,
            strike=K,
            barrier=B,
            rate=r,
            sigma=sigma,
            T=T,
            rebate=rebate,
        )
        return float(down_and_out_call_raw(resolved))
    return _legacy_down_and_out_call(S, K, B, r, sigma, T, rebate=rebate)


def down_and_in_call(
    S: float, K: float, B: float, r: float, sigma: float, T: float,
    rebate: float = 0.0,
) -> float:
    """Price a down-and-in European call.

    The T09 branch is the barrier image term from the route-local raw kernel
    pack. Other branches fall back to the legacy parity-based implementation.
    """
    if T <= 0:
        return 0.0
    if S <= B:
        return float(_bs_call(S, K, r, sigma, T))
    if rebate == 0.0 and K > B:
        resolved = ResolvedBarrierInputs(
            spot=S,
            strike=K,
            barrier=B,
            rate=r,
            sigma=sigma,
            T=T,
            rebate=rebate,
        )
        return float(down_and_in_call_raw(resolved))
    return _legacy_down_and_in_call(S, K, B, r, sigma, T, rebate=rebate)


__all__ = [
    "ResolvedBarrierInputs",
    "barrier_image_raw",
    "barrier_option_price",
    "barrier_regime_selector_raw",
    "down_and_in_call",
    "down_and_in_call_raw",
    "down_and_out_call",
    "down_and_out_call_raw",
    "rebate_raw",
    "vanilla_call_raw",
]
