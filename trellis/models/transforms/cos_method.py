"""COS method for option pricing (Fang-Oosterlee 2008)."""

from __future__ import annotations

import numpy as raw_np


def cos_price(
    char_fn,
    S0: float,
    K: float,
    T: float,
    r: float,
    N: int = 256,
    L: float = 10.0,
    option_type: str = "call",
) -> float:
    """Price a European option via the COS method.

    Parameters
    ----------
    char_fn : callable(u) -> complex
        Characteristic function of log(S_T / K) (or log(S_T/S0) shifted).
    S0, K, T, r : float
    N : int
        Number of cosine expansion terms.
    L : float
        Truncation parameter.
    option_type : str
        "call" or "put".
    """
    # Log-moneyness
    x = raw_np.log(S0 / K)

    # Truncation range for the log-return density
    # Use moments of the characteristic function for better range
    a = x - L * raw_np.sqrt(T)
    b = x + L * raw_np.sqrt(T)

    k = raw_np.arange(N)
    w = k * raw_np.pi / (b - a)

    # Characteristic function values
    cf_vals = raw_np.array([char_fn(w_i) for w_i in w])

    # Payoff coefficients
    if option_type == "call":
        # V_k for call: integrate (e^y - 1)^+ * cos(k*pi*(y-a)/(b-a)) dy from a to b
        # Split at y=0: integral from max(0,a) to b
        U_k = _chi(k, a, b, max(0, a), b) - _psi(k, a, b, max(0, a), b)
    else:
        # Put: integrate (1 - e^y)^+ from a to min(0,b)
        U_k = -_chi(k, a, b, a, min(0, b)) + _psi(k, a, b, a, min(0, b))

    # COS expansion
    summation = raw_np.real(
        cf_vals * raw_np.exp(1j * k * raw_np.pi * (x - a) / (b - a))
    ) * U_k
    summation[0] *= 0.5

    price = K * raw_np.exp(-r * T) * (2.0 / (b - a)) * raw_np.sum(summation)
    return max(float(price), 0.0)


def _chi(k, a, b, c, d):
    """Chi_k(c, d) = integral of e^y * cos(k*pi*(y-a)/(b-a)) dy from c to d."""
    w = k * raw_np.pi / (b - a)
    denom = 1 + w ** 2

    # Handle k=0 separately
    result = raw_np.where(
        k == 0,
        raw_np.exp(d) - raw_np.exp(c),
        (raw_np.exp(d) * (raw_np.cos(w * (d - a)) + w * raw_np.sin(w * (d - a)))
         - raw_np.exp(c) * (raw_np.cos(w * (c - a)) + w * raw_np.sin(w * (c - a))))
        / denom,
    )
    return result


def _psi(k, a, b, c, d):
    """Psi_k(c, d) = integral of cos(k*pi*(y-a)/(b-a)) dy from c to d."""
    return raw_np.where(
        k == 0,
        d - c,
        (raw_np.sin(k * raw_np.pi * (d - a) / (b - a))
         - raw_np.sin(k * raw_np.pi * (c - a) / (b - a)))
        * (b - a) / (k * raw_np.pi),
    )
