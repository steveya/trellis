"""Thomas algorithm for tridiagonal linear systems."""

from __future__ import annotations

import numpy as raw_np


def thomas_solve(a: raw_np.ndarray, b: raw_np.ndarray,
                 c: raw_np.ndarray, d: raw_np.ndarray) -> raw_np.ndarray:
    """Solve tridiagonal system Ax = d using the Thomas algorithm.

    Parameters
    ----------
    a : ndarray
        Lower diagonal (length n-1).
    b : ndarray
        Main diagonal (length n).
    c : ndarray
        Upper diagonal (length n-1).
    d : ndarray
        Right-hand side (length n).

    Returns
    -------
    ndarray
        Solution vector (length n).
    """
    n = len(b)
    na = len(a)  # may be n-1 or n-2 depending on caller
    nc = len(c)

    c_prime = raw_np.zeros(n)
    d_prime = raw_np.zeros(n)

    c_prime[0] = c[0] / b[0] if nc > 0 else 0.0
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        ai = a[i - 1] if (i - 1) < na else 0.0
        m = b[i] - ai * c_prime[i - 1]
        if i < nc:
            c_prime[i] = c[i] / m
        d_prime[i] = (d[i] - ai * d_prime[i - 1]) / m

    x = raw_np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x
