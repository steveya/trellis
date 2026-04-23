"""Autograd-compatible interpolation methods."""

from __future__ import annotations

import numpy as _np
from trellis.core.differentiable import get_numpy

np = get_numpy()


def to_backend_array(a):
    """Convert to array without breaking autograd tracing.

    ``np.asarray(box, dtype=float)`` triggers a VJP lookup that autograd
    doesn't define. Instead we only convert plain lists/tuples.
    """
    if isinstance(a, _np.ndarray):
        return a
    # Check for autograd Box (traced array) — don't re-wrap
    if hasattr(a, '_value'):
        return a
    return _np.asarray(a, dtype=float)


def validation_view(a):
    """Return a non-traced view suitable for shape and monotonicity checks."""
    return getattr(a, "_value", a)


def _to_array(a):
    """Backward-compatible alias for older internal call sites."""
    return to_backend_array(a)


def linear_interp(x: float, xs, ys):
    """Piecewise-linear interpolation (autograd-safe, no searchsorted).

    *xs* and *ys* are 1-D arrays of knots, assumed sorted ascending.
    Flat extrapolation outside the range.
    """
    xs = _to_array(xs)
    ys = _to_array(ys)
    n = len(xs)
    if n == 1:
        return ys[0]
    # Clamp
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # Linear scan for bracket (avoids np.searchsorted which autograd can't diff)
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] * (1 - t) + ys[i + 1] * t
    return ys[-1]


def log_linear_interp(x: float, xs, ys):
    """Log-linear interpolation on discount factors.

    Interpolates *log(ys)* linearly, then exponentiates.
    Useful for discount factor curves to preserve positivity.
    """
    xs = _to_array(xs)
    ys = _to_array(ys)
    log_ys = np.log(ys)
    log_val = linear_interp(x, xs, log_ys)
    return np.exp(log_val)
