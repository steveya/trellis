"""Mock data provider with embedded historical yield snapshots.

No network, no API keys, no disk I/O.  Ships with the core package
so ``pip install trellis`` is immediately productive.
"""

from __future__ import annotations

from datetime import date

from trellis.data.base import BaseDataProvider

# ---------------------------------------------------------------------------
# Embedded snapshots: {date: {tenor_years: yield_decimal}}
#
# Tenors match the 11-point grid used by FRED and Treasury.gov providers:
#   1mo, 3mo, 6mo, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y
#
# Yields are in decimal (0.045 = 4.5%), semi-annual BEY convention
# (same as what the real providers return).
# ---------------------------------------------------------------------------

_TENOR_GRID = (1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)

SNAPSHOTS: dict[date, dict[float, float]] = {
    # Pre-COVID normal curve — modest steepening, ~1.6-2.1%
    date(2019, 9, 15): dict(zip(_TENOR_GRID, [
        0.0193, 0.0188, 0.0190, 0.0175, 0.0163, 0.0157,
        0.0155, 0.0163, 0.0172, 0.0195, 0.0210,
    ])),

    # COVID crisis — near-zero front end, mild steepening
    date(2020, 3, 15): dict(zip(_TENOR_GRID, [
        0.0008, 0.0022, 0.0033, 0.0026, 0.0025, 0.0032,
        0.0037, 0.0052, 0.0073, 0.0112, 0.0129,
    ])),

    # Peak rates, inverted curve — 5.3% front end, 4.6-4.8% long end
    date(2023, 10, 15): dict(zip(_TENOR_GRID, [
        0.0533, 0.0530, 0.0527, 0.0507, 0.0500, 0.0487,
        0.0469, 0.0470, 0.0473, 0.0509, 0.0495,
    ])),

    # Easing cycle begins — moderate curve, ~4.2-4.6%
    date(2024, 11, 15): dict(zip(_TENOR_GRID, [
        0.0455, 0.0447, 0.0435, 0.0420, 0.0415, 0.0418,
        0.0425, 0.0432, 0.0438, 0.0462, 0.0458,
    ])),
}

# Sorted for binary-ish lookup
_SORTED_DATES = sorted(SNAPSHOTS.keys())


class MockDataProvider(BaseDataProvider):
    """In-memory data provider with embedded historical yield snapshots.

    Parameters
    ----------
    overrides : dict or None
        Additional ``{date: {tenor: yield}}`` entries that supplement
        (or shadow) the built-in snapshots.
    """

    def __init__(self, overrides: dict[date, dict[float, float]] | None = None):
        self._data = dict(SNAPSHOTS)
        if overrides:
            self._data.update(overrides)
        self._sorted_dates = sorted(self._data.keys())

    @classmethod
    def from_dict(cls, data: dict[date, dict[float, float]]) -> MockDataProvider:
        """Create a provider with *only* user-supplied data (no built-in snapshots)."""
        inst = cls.__new__(cls)
        inst._data = dict(data)
        inst._sorted_dates = sorted(inst._data.keys())
        return inst

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Return the snapshot closest to (but not after) *as_of*.

        If *as_of* is ``None``, returns the most recent snapshot.
        If *as_of* is before all snapshots, returns an empty dict.
        """
        if not self._sorted_dates:
            return {}

        if as_of is None:
            return dict(self._data[self._sorted_dates[-1]])

        # Find the latest date <= as_of
        best = None
        for d in self._sorted_dates:
            if d <= as_of:
                best = d
            else:
                break

        if best is None:
            return {}

        return dict(self._data[best])

    @property
    def available_dates(self) -> list[date]:
        """List of snapshot dates available in this provider."""
        return list(self._sorted_dates)
