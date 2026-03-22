"""Simple filesystem cache for market data."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path


def _default_cache_dir() -> Path:
    return Path.home() / ".trellis" / "cache"


class DiskCache:
    """JSON-based filesystem cache keyed by (namespace, date)."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()

    def _path(self, namespace: str, as_of: date) -> Path:
        return self.cache_dir / namespace / f"{as_of.isoformat()}.json"

    def get(self, namespace: str, as_of: date):
        p = self._path(namespace, as_of)
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            # Convert string keys back to floats
            return {float(k): v for k, v in raw.items()}
        return None

    def put(self, namespace: str, as_of: date, data: dict):
        p = self._path(namespace, as_of)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({str(k): v for k, v in data.items()}, f)
