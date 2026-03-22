"""Volatility surface protocol and implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VolSurface(Protocol):
    """Protocol for anything that provides Black implied volatility."""

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return Black (lognormal) vol for given expiry and strike."""
        ...


class FlatVol:
    """Constant volatility surface."""

    def __init__(self, vol: float):
        self._vol = vol

    def black_vol(self, expiry: float, strike: float) -> float:
        return self._vol
