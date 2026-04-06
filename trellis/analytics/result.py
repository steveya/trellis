"""Analytics result container — structured output from analyze()."""

from __future__ import annotations

from typing import Any


class RiskMeasureOutput(dict):
    """Dictionary-like risk output with attached methodology metadata."""

    def __init__(self, values=None, *, metadata: dict[str, Any] | None = None):
        super().__init__(values or {})
        self.metadata = dict(metadata or {})

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-friendly payload with values and metadata."""
        return {
            "values": dict(self),
            "metadata": dict(self.metadata),
        }


class AnalyticsResult:
    """Container for computed analytics measures.

    Supports attribute access, dict access, and iteration.

    Example::

        result = s.analyze(payoff, measures=["price", "dv01", "vega"])
        result.price          # 96.83
        result["dv01"]        # 0.078
        result.to_dict()      # {"price": 96.83, "dv01": 0.078, "vega": 0.45}
    """

    def __init__(self, data: dict[str, Any]):
        """Store resolved measure values keyed by measure name."""
        self._data = data

    def __getattr__(self, name: str) -> Any:
        """Expose computed measures through attribute access."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"Measure '{name}' not computed. "
                f"Available: {sorted(self._data.keys())}"
            )

    def __getitem__(self, key: str) -> Any:
        """Expose computed measures through dictionary-style access."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Return whether a measure with the given name was computed."""
        return key in self._data

    def __iter__(self):
        """Iterate over computed measure names."""
        return iter(self._data)

    def __repr__(self):
        """Return a short inline representation for debugging and notebooks."""
        items = ", ".join(f"{k}={_short(v)}" for k, v in self._data.items())
        return f"AnalyticsResult({items})"

    def keys(self):
        """Return the measure names available in this result container."""
        return self._data.keys()

    def to_dict(self) -> dict[str, Any]:
        """Return all measures as a plain dict."""
        return dict(self._data)


class BookAnalyticsResult:
    """Aggregated analytics for a book of instruments.

    Supports per-position access and book-level aggregation.

    Example::

        result = s.analyze(book, measures=["price", "dv01"])
        result["10Y"].price       # single position
        result.total_mv           # book aggregate
        result.book_dv01          # notional-weighted DV01
        result.to_dataframe()     # one row per position
    """

    def __init__(
        self,
        positions: dict[str, AnalyticsResult],
        notionals: dict[str, float],
    ):
        """Store per-position analytics plus the notionals used for aggregation."""
        self._positions = positions
        self._notionals = notionals

    def __getitem__(self, key: str) -> AnalyticsResult:
        """Return analytics for a named book position."""
        return self._positions[key]

    def __iter__(self):
        """Iterate over position names."""
        return iter(self._positions)

    def __len__(self):
        """Return the number of positions represented in the result."""
        return len(self._positions)

    def __repr__(self):
        """Return a concise summary suitable for notebook display."""
        return f"BookAnalyticsResult({len(self._positions)} positions)"

    @property
    def total_mv(self) -> float:
        """Sum of price * notional across all positions."""
        total = 0.0
        for name, r in self._positions.items():
            if "price" in r:
                total += r["price"] * self._notionals.get(name, 1.0)
        return total

    @property
    def book_dv01(self) -> float:
        """Notional-weighted portfolio DV01."""
        total = 0.0
        for name, r in self._positions.items():
            if "dv01" in r:
                total += r["dv01"] * self._notionals.get(name, 1.0) / 100
        return total

    @property
    def book_duration(self) -> float:
        """MV-weighted average duration."""
        tmv = self.total_mv
        if tmv == 0:
            return 0.0
        weighted = 0.0
        for name, r in self._positions.items():
            if "duration" in r and "price" in r:
                mv = r["price"] * self._notionals.get(name, 1.0)
                weighted += r["duration"] * mv
        return weighted / tmv

    @property
    def book_krd(self) -> dict[float, float]:
        """MV-weighted aggregate key rate durations."""
        tmv = self.total_mv
        if tmv == 0:
            return {}
        agg: dict[object, float] = {}
        metadata = None
        for name, r in self._positions.items():
            if "key_rate_durations" not in r:
                continue
            mv = r.get("price", 0) * self._notionals.get(name, 1.0)
            weight = mv / tmv if tmv else 0
            krd_surface = r["key_rate_durations"]
            if isinstance(krd_surface, RiskMeasureOutput) and metadata is None:
                metadata = dict(krd_surface.metadata)
                metadata["aggregation"] = "market_value_weighted_book"
            for tenor, krd in krd_surface.items():
                agg[tenor] = agg.get(tenor, 0.0) + krd * weight
        if metadata is not None:
            return RiskMeasureOutput(agg, metadata=metadata)
        return agg

    def to_dict(self) -> dict:
        """Return book analytics as a JSON-serializable nested dictionary."""
        return {
            "positions": {name: r.to_dict() for name, r in self._positions.items()},
            "total_mv": self.total_mv,
            "book_dv01": self.book_dv01,
            "book_duration": self.book_duration,
        }

    def to_dataframe(self):
        """Convert per-position scalar measures into a tabular DataFrame."""
        import pandas as pd
        records = []
        for name, r in self._positions.items():
            rec = {"name": name, "notional": self._notionals.get(name, 1.0)}
            for k, v in r.to_dict().items():
                if not isinstance(v, (dict, list)):
                    rec[k] = v
            records.append(rec)
        return pd.DataFrame(records)


def _short(v):
    """Short repr for display."""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return f"{{{len(v)} items}}"
    return repr(v)
