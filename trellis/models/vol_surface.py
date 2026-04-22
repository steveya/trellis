"""Volatility surface protocol and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as _np

from trellis.core.differentiable import get_numpy
from trellis.curves.interpolation import to_backend_array, validation_view

np = get_numpy()


@runtime_checkable
class VolSurface(Protocol):
    """Protocol for anything that provides Black implied volatility."""

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return Black (lognormal) vol for given expiry and strike."""
        ...


@dataclass(frozen=True)
class FlatVol:
    """Constant volatility surface."""

    vol: float

    def __post_init__(self):
        """Validate the flat volatility quote."""
        if _is_negative_scalar(self.vol):
            raise ValueError("flat volatility must be non-negative")

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return the same volatility for every expiry/strike pair."""
        return self.vol


@dataclass(frozen=True)
class GridVolSurface:
    """Strike/expiry grid of Black implied vols with bilinear interpolation.

    The surface uses flat extrapolation outside the provided strike/expiry
    grid. It is differentiable in the stored node values and only piecewise
    differentiable in expiry/strike query coordinates away from knot
    boundaries.
    """

    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        """Validate that the supplied expiry/strike grid is rectangular and sorted."""
        object.__setattr__(self, "expiries", tuple(self.expiries))
        object.__setattr__(self, "strikes", tuple(self.strikes))
        object.__setattr__(self, "vols", to_backend_array(self.vols))
        vols_view = validation_view(self.vols)

        if not self.expiries or not self.strikes:
            raise ValueError("expiries and strikes must be non-empty")
        if getattr(vols_view, "ndim", None) != 2:
            raise ValueError("vol grid must be two-dimensional")
        if vols_view.shape[0] != len(self.expiries):
            raise ValueError("vol grid row count must match expiries")
        if vols_view.shape[1] != len(self.strikes):
            raise ValueError("each vol row must match strike count")
        if tuple(sorted(self.expiries)) != self.expiries:
            raise ValueError("expiries must be sorted ascending")
        if tuple(sorted(self.strikes)) != self.strikes:
            raise ValueError("strikes must be sorted ascending")
        if len(set(self.expiries)) != len(self.expiries):
            raise ValueError("expiries must be strictly increasing")
        if len(set(self.strikes)) != len(self.strikes):
            raise ValueError("strikes must be strictly increasing")
        if _has_negative_entries(vols_view):
            raise ValueError("vol nodes must be non-negative")

    def black_vol(self, expiry: float, strike: float) -> float:
        """Interpolate Black vol bilinearly with flat extrapolation beyond the grid.

        If the query lies between expiries ``t_0`` and ``t_1`` and strikes
        ``k_0`` and ``k_1``, the surface evaluates the standard bilinear blend
        of the four surrounding node volatilities. The resulting map is
        differentiable in node values and piecewise differentiable in
        ``expiry`` and ``strike`` away from knot boundaries.
        """
        i0, i1, w_expiry = _bracket_and_weight(expiry, self.expiries)
        j0, j1, w_strike = _bracket_and_weight(strike, self.strikes)

        v00 = self.vols[i0][j0]
        v01 = self.vols[i0][j1]
        v10 = self.vols[i1][j0]
        v11 = self.vols[i1][j1]

        lower = _lerp(v00, v01, w_strike)
        upper = _lerp(v10, v11, w_strike)
        return _lerp(lower, upper, w_expiry)


def _lerp(left: float, right: float, weight: float) -> float:
    """Return the linear interpolation ``(1-w) * left + w * right``."""
    return (1.0 - weight) * left + weight * right


def _bracket_and_weight(value: float, grid: tuple[float, ...]) -> tuple[int, int, float]:
    """Locate the surrounding grid nodes and normalized interpolation weight."""
    if len(grid) == 1:
        return 0, 0, 0.0
    if value <= grid[0]:
        return 0, 0, 0.0
    if value >= grid[-1]:
        last = len(grid) - 1
        return last, last, 0.0

    for lower in range(len(grid) - 1):
        upper = lower + 1
        left = grid[lower]
        right = grid[upper]
        if left <= value <= right:
            if right == left:
                return lower, upper, 0.0
            return lower, upper, (value - left) / (right - left)
    last = len(grid) - 1
    return last, last, 0.0


def _is_negative_scalar(value) -> bool:
    """Return whether ``value`` is strictly negative when it is safe to test."""
    try:
        return bool(value < 0.0)
    except (TypeError, ValueError):
        return False


def _has_negative_entries(values) -> bool:
    """Return whether an array-like contains negative entries."""
    return bool(_np.any(_np.asarray(values) < 0.0))
