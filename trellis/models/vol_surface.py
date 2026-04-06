"""Volatility surface protocol and implementations."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
        if self.vol < 0.0:
            raise ValueError("flat volatility must be non-negative")

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return the same volatility for every expiry/strike pair."""
        return self.vol


@dataclass(frozen=True)
class GridVolSurface:
    """Strike/expiry grid of Black implied vols with bilinear interpolation.

    The surface uses flat extrapolation outside the provided strike/expiry grid.
    """

    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        """Validate that the supplied expiry/strike grid is rectangular and sorted."""
        if not self.expiries or not self.strikes:
            raise ValueError("expiries and strikes must be non-empty")
        if len(self.vols) != len(self.expiries):
            raise ValueError("vol grid row count must match expiries")
        if any(len(row) != len(self.strikes) for row in self.vols):
            raise ValueError("each vol row must match strike count")
        if tuple(sorted(self.expiries)) != self.expiries:
            raise ValueError("expiries must be sorted ascending")
        if tuple(sorted(self.strikes)) != self.strikes:
            raise ValueError("strikes must be sorted ascending")
        if len(set(self.expiries)) != len(self.expiries):
            raise ValueError("expiries must be strictly increasing")
        if len(set(self.strikes)) != len(self.strikes):
            raise ValueError("strikes must be strictly increasing")
        if any(vol < 0.0 for row in self.vols for vol in row):
            raise ValueError("vol nodes must be non-negative")

    def black_vol(self, expiry: float, strike: float) -> float:
        """Interpolate Black vol bilinearly with flat extrapolation beyond the grid.

        If the query lies between expiries ``t_0`` and ``t_1`` and strikes
        ``k_0`` and ``k_1``, the surface evaluates the standard bilinear blend
        of the four surrounding node volatilities.
        """
        i0, i1, w_expiry = _bracket_and_weight(expiry, self.expiries)
        j0, j1, w_strike = _bracket_and_weight(strike, self.strikes)

        v00 = self.vols[i0][j0]
        v01 = self.vols[i0][j1]
        v10 = self.vols[i1][j0]
        v11 = self.vols[i1][j1]

        lower = _lerp(v00, v01, w_strike)
        upper = _lerp(v10, v11, w_strike)
        return float(_lerp(lower, upper, w_expiry))


def _lerp(left: float, right: float, weight: float) -> float:
    """Return the linear interpolation ``(1-w) * left + w * right``."""
    return (1.0 - weight) * left + weight * right


def _bracket_and_weight(value: float, grid: tuple[float, ...]) -> tuple[int, int, float]:
    """Locate the surrounding grid nodes and normalized interpolation weight."""
    if value <= grid[0]:
        return 0, 0, 0.0
    if value >= grid[-1]:
        last = len(grid) - 1
        return last, last, 0.0

    upper = bisect_right(grid, value)
    lower = upper - 1
    left = grid[lower]
    right = grid[upper]
    if right == left:
        return lower, upper, 0.0
    return lower, upper, (value - left) / (right - left)
