"""Tests for trellis.book — Book and BookResult."""

import json
import tempfile
from datetime import date

import pytest

from trellis.book import Book, BookResult
from trellis.core.types import PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond


def _make_bond(**kwargs):
    defaults = dict(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                    maturity=10, frequency=2)
    defaults.update(kwargs)
    return Bond(**defaults)


class TestBook:
    """Book construction and accessors."""

    def test_from_dict(self):
        b = Book({"A": _make_bond(), "B": _make_bond(coupon=0.04)})
        assert len(b) == 2
        assert "A" in b.names
        assert "B" in b.names

    def test_from_list(self):
        b = Book([_make_bond(), _make_bond()])
        assert len(b) == 2
        assert b.names == ["inst_0", "inst_1"]

    def test_getitem(self):
        bond = _make_bond()
        b = Book({"X": bond})
        assert b["X"] is bond

    def test_iter(self):
        b = Book({"A": _make_bond(), "B": _make_bond()})
        assert list(b) == ["A", "B"]

    def test_notional_default(self):
        b = Book({"A": _make_bond()})
        assert b.notional("A") == 1.0

    def test_notional_custom(self):
        b = Book({"A": _make_bond()}, notionals={"A": 1_000_000})
        assert b.notional("A") == 1_000_000

    def test_empty_book(self):
        b = Book({})
        assert len(b) == 0
        assert b.names == []

    def test_single_instrument(self):
        b = Book([_make_bond()])
        assert len(b) == 1

    def test_instruments_property(self):
        bond = _make_bond()
        b = Book({"A": bond})
        assert b.instruments == {"A": bond}

    def test_from_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({
            "name": ["10Y"],
            "face": [100.0],
            "coupon": [0.05],
            "maturity_date": [date(2034, 11, 15)],
            "frequency": [2],
            "maturity": [10],
            "notional": [1_000_000.0],
        })
        b = Book.from_dataframe(df)
        assert len(b) == 1
        assert "10Y" in b.names
        assert b.notional("10Y") == 1_000_000.0
        inst = b["10Y"]
        assert inst.coupon_rate == 0.05

    def test_from_csv_roundtrip(self):
        import pandas as pd
        df = pd.DataFrame({
            "name": ["5Y", "10Y"],
            "face": [100.0, 100.0],
            "coupon": [0.04, 0.05],
            "maturity_date": ["2029-11-15", "2034-11-15"],
            "frequency": [2, 2],
            "maturity": [5, 10],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = f.name

        b = Book.from_csv(path)
        assert len(b) == 2
        assert "5Y" in b.names


class TestBookResult:
    """BookResult aggregation and export."""

    def _price_book(self, greeks="all"):
        curve = YieldCurve.flat(0.045)
        bond_a = _make_bond()
        bond_b = _make_bond(coupon=0.04)
        book = Book(
            {"A": bond_a, "B": bond_b},
            notionals={"A": 1_000_000, "B": 500_000},
        )
        results = {}
        for name in book:
            results[name] = price_instrument(
                book[name], curve, date(2024, 11, 15), greeks=greeks,
            )
        return BookResult(results, book), book

    def test_total_mv(self):
        br, _ = self._price_book()
        assert br.total_mv > 0

    def test_book_dv01(self):
        br, _ = self._price_book()
        assert br.book_dv01 > 0

    def test_book_duration(self):
        br, _ = self._price_book()
        assert 0 < br.book_duration < 15

    def test_to_dict_json_serializable(self):
        br, _ = self._price_book()
        d = br.to_dict()
        # Should not raise
        json.dumps(d)
        assert "total_mv" in d
        assert "positions" in d

    def test_to_dataframe_shape(self):
        br, book = self._price_book()
        df = br.to_dataframe()
        assert len(df) == len(book)
        assert "clean_price" in df.columns
        assert "notional" in df.columns

    def test_getitem(self):
        br, _ = self._price_book()
        assert isinstance(br["A"], PricingResult)

    def test_len_and_iter(self):
        br, book = self._price_book()
        assert len(br) == len(book)
        assert set(br) == set(book)

    def test_zero_notional(self):
        curve = YieldCurve.flat(0.045)
        bond = _make_bond()
        book = Book({"A": bond}, notionals={"A": 0.0})
        results = {"A": price_instrument(bond, curve, date(2024, 11, 15))}
        br = BookResult(results, book)
        assert br.total_mv == 0.0
        assert br.book_duration == 0.0
