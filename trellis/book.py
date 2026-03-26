"""Book (instrument collection) and BookResult (aggregated pricing)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

from trellis.core.types import Instrument, PricingResult


class Book:
    """A named collection of instruments with optional notionals.

    Parameters
    ----------
    instruments : list or dict
        If a list, names are auto-generated (``"inst_0"``, …).
        If a dict, keys are used as names.
    notionals : dict or None
        Per-name notional amounts. Defaults to 1.0 for each.
    """

    def __init__(
        self,
        instruments: list | dict[str, Instrument],
        notionals: dict[str, float] | None = None,
    ):
        """Store instruments and optional notionals under stable position names."""
        if isinstance(instruments, dict):
            self._instruments = dict(instruments)
        else:
            self._instruments = {
                f"inst_{i}": inst for i, inst in enumerate(instruments)
            }
        self._notionals = dict(notionals) if notionals else {}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Instrument:
        """Return the instrument stored under ``key``."""
        return self._instruments[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over position names in insertion order."""
        return iter(self._instruments)

    def __len__(self) -> int:
        """Return the number of positions in the book."""
        return len(self._instruments)

    @property
    def names(self) -> list[str]:
        """Return all position names as a list."""
        return list(self._instruments.keys())

    @property
    def instruments(self) -> dict[str, Instrument]:
        """Return a shallow copy of the underlying name-to-instrument mapping."""
        return dict(self._instruments)

    def notional(self, name: str) -> float:
        """Return the stored notional for a position, defaulting to ``1.0``."""
        return self._notionals.get(name, 1.0)

    # ------------------------------------------------------------------
    # Constructors from tabular data
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df) -> Book:
        """Build a Book from a pandas DataFrame.

        Expected columns: ``face``, ``coupon``, ``maturity_date``, ``frequency``.
        Optional: ``name`` (or index), ``notional``, ``maturity``.
        """
        from trellis.instruments.bond import Bond

        instruments: dict[str, Instrument] = {}
        notionals: dict[str, float] = {}

        for idx, row in df.iterrows():
            name = str(row["name"]) if "name" in df.columns else str(idx)
            mat_date = row["maturity_date"]
            if hasattr(mat_date, "date"):
                mat_date = mat_date.date()

            bond = Bond(
                face=float(row["face"]),
                coupon=float(row["coupon"]),
                maturity_date=mat_date,
                frequency=int(row.get("frequency", 2)),
                maturity=int(row["maturity"]) if "maturity" in df.columns else None,
            )
            instruments[name] = bond
            if "notional" in df.columns:
                notionals[name] = float(row["notional"])

        return cls(instruments, notionals if notionals else None)

    @classmethod
    def from_csv(cls, path: str) -> Book:
        """Build a Book from a CSV file. Delegates to :meth:`from_dataframe`."""
        import pandas as pd

        df = pd.read_csv(path, parse_dates=["maturity_date"])
        return cls.from_dataframe(df)


class BookResult:
    """Aggregated pricing results for a :class:`Book`.

    Parameters
    ----------
    results : dict
        ``{name: PricingResult}`` for every instrument in the book.
    book : Book
        The book that was priced (used for notional weighting).
    """

    def __init__(self, results: dict[str, PricingResult], book: Book):
        """Bind per-position pricing results to the originating book definition."""
        self._results = results
        self._book = book

    def __getitem__(self, key: str) -> PricingResult:
        """Return the pricing result for a named position."""
        return self._results[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over position names with available pricing results."""
        return iter(self._results)

    def __len__(self) -> int:
        """Return the number of priced positions."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Aggregated analytics
    # ------------------------------------------------------------------

    @property
    def total_mv(self) -> float:
        """Total market value: sum(dirty_price * notional)."""
        return sum(
            r.dirty_price * self._book.notional(name)
            for name, r in self._results.items()
        )

    @property
    def book_dv01(self) -> float:
        """Portfolio DV01: sum(dv01 * notional / 100)."""
        total = 0.0
        for name, r in self._results.items():
            dv01 = r.greeks.get("dv01", 0.0)
            total += dv01 * self._book.notional(name) / 100.0
        return total

    @property
    def book_duration(self) -> float:
        """Market-value-weighted average duration."""
        tmv = self.total_mv
        if tmv == 0:
            return 0.0
        weighted = sum(
            r.greeks.get("duration", 0.0)
            * r.dirty_price
            * self._book.notional(name)
            for name, r in self._results.items()
        )
        return weighted / tmv

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable dict representation."""
        rows = {}
        for name, r in self._results.items():
            row = {
                "clean_price": r.clean_price,
                "dirty_price": r.dirty_price,
                "accrued_interest": r.accrued_interest,
                "notional": self._book.notional(name),
            }
            row.update(r.greeks)
            rows[name] = row
        return {
            "positions": rows,
            "total_mv": self.total_mv,
            "book_dv01": self.book_dv01,
            "book_duration": self.book_duration,
        }

    def to_dataframe(self):
        """Return a pandas DataFrame with one row per position."""
        import pandas as pd

        records = []
        for name, r in self._results.items():
            rec = {
                "name": name,
                "clean_price": r.clean_price,
                "dirty_price": r.dirty_price,
                "accrued_interest": r.accrued_interest,
                "notional": self._book.notional(name),
            }
            for k, v in r.greeks.items():
                if not isinstance(v, dict):
                    rec[k] = v
            records.append(rec)
        return pd.DataFrame(records)
