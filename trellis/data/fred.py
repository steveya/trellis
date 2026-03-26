"""FRED data provider for Treasury constant-maturity yields (H.15)."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.data.base import BaseDataProvider
from trellis.data.cache import DiskCache

# FRED series IDs → tenor in years
TREASURY_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}


class FredDataProvider(BaseDataProvider):
    """Fetch Treasury yields via the FRED API (requires ``fredapi`` package).

    Parameters
    ----------
    api_key : str or None
        FRED API key.  If *None*, reads ``FRED_API_KEY`` from the environment.
    """

    def __init__(self, api_key: str | None = None):
        """Store the API key and initialize the on-disk response cache."""
        import os
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._cache = DiskCache()

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Fetch or cache Treasury constant-maturity yields from FRED."""
        as_of = as_of or date.today()
        cached = self._cache.get("fred_yields", as_of)
        if cached is not None:
            return cached

        from fredapi import Fred  # type: ignore[import-untyped]

        fred = Fred(api_key=self.api_key)
        start = as_of - timedelta(days=10)
        yields: dict[float, float] = {}
        for series_id, tenor in TREASURY_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=as_of)
                val = s.dropna().iloc[-1]
                yields[tenor] = float(val) / 100.0  # percent → decimal
            except Exception:
                continue

        if yields:
            self._cache.put("fred_yields", as_of, yields)
        return yields
