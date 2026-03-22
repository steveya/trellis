"""FFT-based option pricing (Carr-Madan method)."""

from __future__ import annotations

import numpy as raw_np


def fft_price(
    char_fn,
    S0: float,
    K: float,
    T: float,
    r: float,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> float:
    """Price a European call via FFT (Carr-Madan 1999).

    Parameters
    ----------
    char_fn : callable(u) -> complex
        Characteristic function of log(S_T) under risk-neutral measure.
        phi(u) = E[exp(i*u*log(S_T))].
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry.
    r : float
        Risk-free rate.
    alpha : float
        Dampening parameter (must be > 0, typically 1.5).
    N : int
        Number of FFT points (power of 2).
    eta : float
        Grid spacing in frequency domain.

    Returns
    -------
    float
        European call price.
    """
    # Grid setup
    lam = 2 * raw_np.pi / (N * eta)  # log-strike spacing
    b = N * lam / 2  # upper bound for log-strike grid

    # Frequency grid
    v = raw_np.arange(N) * eta

    # Modified characteristic function
    def psi(v):
        cf = char_fn(v - (alpha + 1) * 1j)
        denom = alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v
        return raw_np.exp(-r * T) * cf / denom

    # Simpson's rule weights
    simpson = 3 + (-1) ** (raw_np.arange(N) + 1)
    simpson[0] = 1
    simpson = simpson / 3

    # FFT input
    x = raw_np.exp(1j * v * b) * psi(v) * eta * simpson

    # FFT
    fft_result = raw_np.fft.fft(x)

    # Log-strike grid
    k = -b + lam * raw_np.arange(N)

    # Call prices
    call_prices = raw_np.exp(-alpha * k) / raw_np.pi * raw_np.real(fft_result)

    # Interpolate to get price at desired strike
    log_K = raw_np.log(K)
    price = float(raw_np.interp(log_K, k, call_prices))

    return max(price, 0.0)
