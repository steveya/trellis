"""Simple filesystem cache for market data."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path


def _default_cache_dir() -> Path:
    """Return the default on-disk cache directory under the user home directory."""
    return Path.home() / ".trellis" / "cache"


class DiskCache:
    """JSON-based filesystem cache keyed by (namespace, date)."""

    def __init__(self, cache_dir: Path | str | None = None):
        """Store the root directory used for cache reads and writes."""
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()

    def _path(self, namespace: str, as_of: date) -> Path:
        """Return the JSON file path for one namespace/date cache entry."""
        return self.cache_dir / namespace / f"{as_of.isoformat()}.json"

    def get(self, namespace: str, as_of: date):
        """Load one cached JSON payload, converting tenor keys back to floats."""
        p = self._path(namespace, as_of)
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            # Convert string keys back to floats
            return {float(k): v for k, v in raw.items()}
        return None

    def put(self, namespace: str, as_of: date, data: dict):
        """Persist one cache payload under the namespace/date key."""
        p = self._path(namespace, as_of)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({str(k): v for k, v in data.items()}, f)
