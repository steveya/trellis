"""Optional Numba helpers for hot numerical kernels."""

from __future__ import annotations

import os


def _identity_decorator(*args, **kwargs):
    """Return a decorator that leaves the wrapped function unchanged."""

    def decorator(func):
        return func

    return decorator


_DISABLE_NUMBA = os.getenv("TRELLIS_DISABLE_NUMBA", "").lower() in {"1", "true", "yes"}

if _DISABLE_NUMBA:
    NUMBA_AVAILABLE = False
    maybe_njit = _identity_decorator
else:
    try:
        from numba import njit as maybe_njit
    except Exception:  # pragma: no cover - exercised when numba is absent
        NUMBA_AVAILABLE = False
        maybe_njit = _identity_decorator
    else:
        NUMBA_AVAILABLE = True
