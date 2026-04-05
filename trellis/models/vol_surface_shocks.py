"""Reusable expiry/strike bucket shocks for volatility surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from trellis.core.differentiable import get_numpy
from trellis.models.vol_surface import FlatVol, GridVolSurface

np = get_numpy()


@dataclass(frozen=True)
class VolSurfaceShockWarning:
    """Explicit warning describing a volatility-surface bucket limitation."""

    code: str
    message: str
    expiry: float | None = None
    strike: float | None = None


@dataclass(frozen=True)
class VolSurfaceShockBucket:
    """Metadata describing one expiry/strike bucket on a vol surface."""

    expiry: float
    strike: float
    base_vol: float
    is_exact_surface_node: bool
    expiry_support: tuple[float | None, float | None]
    strike_support: tuple[float | None, float | None]
    warnings: tuple[VolSurfaceShockWarning, ...] = ()


@dataclass(frozen=True)
class VolSurfaceShockSurface:
    """Reusable bucket surface for later vega and vol-scenario workflows."""

    requested_expiries: tuple[float, ...]
    requested_strikes: tuple[float, ...]
    buckets: tuple[VolSurfaceShockBucket, ...]

    @property
    def warnings(self) -> tuple[VolSurfaceShockWarning, ...]:
        """Flatten all bucket warnings into one stable tuple."""
        return tuple(warning for bucket in self.buckets for warning in bucket.warnings)

    def bucketed_surface(self) -> GridVolSurface:
        """Return the base surface re-expressed on the configured bucket grid."""
        vol_rows = []
        for expiry in self.requested_expiries:
            row = []
            for strike in self.requested_strikes:
                bucket = self.bucket_for(expiry, strike)
                row.append(float(bucket.base_vol))
            vol_rows.append(tuple(row))
        surface = GridVolSurface(
            expiries=self.requested_expiries,
            strikes=self.requested_strikes,
            vols=tuple(vol_rows),
        )
        object.__setattr__(surface, "vol_surface_shock_surface", self)
        object.__setattr__(surface, "vol_surface_shock_warnings", self.warnings)
        return surface

    def bucket_for(self, expiry: float, strike: float) -> VolSurfaceShockBucket:
        """Return the configured bucket matching ``(expiry, strike)``."""
        expiry = float(expiry)
        strike = float(strike)
        for bucket in self.buckets:
            if np.isclose(bucket.expiry, expiry) and np.isclose(bucket.strike, strike):
                return bucket
        raise KeyError(f"No vol bucket configured for expiry={expiry}, strike={strike}.")

    def apply_bumps(self, bucket_bumps: Mapping[tuple[float, float], float]) -> GridVolSurface:
        """Return a bumped grid surface with bucket bumps applied in volatility bps."""
        base = self.bucketed_surface()
        vol_rows = [list(row) for row in base.vols]
        expiry_index = {float(expiry): index for index, expiry in enumerate(base.expiries)}
        strike_index = {float(strike): index for index, strike in enumerate(base.strikes)}

        for (expiry, strike), bump_bps in bucket_bumps.items():
            bucket = self.bucket_for(float(expiry), float(strike))
            i = expiry_index[float(bucket.expiry)]
            j = strike_index[float(bucket.strike)]
            vol_rows[i][j] = float(vol_rows[i][j]) + float(bump_bps) / 10_000.0

        surface = GridVolSurface(
            expiries=base.expiries,
            strikes=base.strikes,
            vols=tuple(tuple(float(vol) for vol in row) for row in vol_rows),
        )
        object.__setattr__(surface, "vol_surface_shock_surface", self)
        object.__setattr__(surface, "vol_surface_shock_warnings", self.warnings)
        object.__setattr__(
            surface,
            "vol_surface_shock_bumps",
            {(float(expiry), float(strike)): float(bump) for (expiry, strike), bump in bucket_bumps.items()},
        )
        return surface


def build_vol_surface_shock_surface(
    surface,
    *,
    expiries,
    strikes,
) -> VolSurfaceShockSurface:
    """Describe expiry/strike bucket support on top of a supported vol surface."""
    requested_expiries = _normalize_axis(expiries)
    requested_strikes = _normalize_axis(strikes)
    buckets = tuple(
        _build_bucket(surface, expiry, strike)
        for expiry in requested_expiries
        for strike in requested_strikes
    )
    return VolSurfaceShockSurface(
        requested_expiries=requested_expiries,
        requested_strikes=requested_strikes,
        buckets=buckets,
    )


def _normalize_axis(values) -> tuple[float, ...]:
    normalized = []
    for value in sorted(float(value) for value in values):
        if normalized and np.isclose(normalized[-1], value):
            continue
        normalized.append(value)
    if not normalized:
        raise ValueError("vol surface shock axes must be non-empty")
    return tuple(normalized)


def _build_bucket(surface, expiry: float, strike: float) -> VolSurfaceShockBucket:
    expiry = float(expiry)
    strike = float(strike)
    warnings: list[VolSurfaceShockWarning] = []

    if isinstance(surface, GridVolSurface):
        exact_node = any(np.isclose(expiry, node) for node in surface.expiries) and any(
            np.isclose(strike, node) for node in surface.strikes
        )
        expiry_support = _support_bracket(expiry, surface.expiries)
        strike_support = _support_bracket(strike, surface.strikes)
    elif isinstance(surface, FlatVol):
        exact_node = False
        expiry_support = (None, None)
        strike_support = (None, None)
        warnings.append(
            VolSurfaceShockWarning(
                code="flat_surface_expanded",
                message="Flat vol was expanded onto the requested expiry/strike bucket grid.",
                expiry=expiry,
                strike=strike,
            )
        )
    else:
        raise TypeError("vol surface shock substrate currently supports GridVolSurface and FlatVol only")

    return VolSurfaceShockBucket(
        expiry=expiry,
        strike=strike,
        base_vol=float(surface.black_vol(expiry, strike)),
        is_exact_surface_node=bool(exact_node),
        expiry_support=expiry_support,
        strike_support=strike_support,
        warnings=tuple(warnings),
    )


def _support_bracket(value: float, grid: tuple[float, ...]) -> tuple[float | None, float | None]:
    if not grid:
        return (None, None)
    if value <= grid[0]:
        return (None, float(grid[0]))
    if value >= grid[-1]:
        return (float(grid[-1]), None)

    upper_index = int(np.searchsorted(np.asarray(grid, dtype=float), value))
    lower_index = upper_index - 1
    return (float(grid[lower_index]), float(grid[upper_index]))


__all__ = [
    "VolSurfaceShockBucket",
    "VolSurfaceShockSurface",
    "VolSurfaceShockWarning",
    "build_vol_surface_shock_surface",
]
