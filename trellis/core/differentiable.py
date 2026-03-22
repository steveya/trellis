"""Autograd backend wrapper.

Provides ``get_numpy`` and ``gradient`` so the rest of the library
can be swapped to a different AD backend later.
"""

from __future__ import annotations

import autograd
import autograd.numpy as anp


def get_numpy():
    """Return the autograd-wrapped numpy module."""
    return anp


def gradient(fn, argnum: int = 0):
    """Return a function that computes the gradient of *fn* w.r.t. ``argnum``."""
    return autograd.grad(fn, argnum)
