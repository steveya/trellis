"""Tests for trellis.samples and trellis.quickstart."""

from datetime import date

from trellis.core.types import PricingResult
from trellis.book import BookResult
from trellis.samples import (
    sample_bond_2y, sample_bond_5y, sample_bond_10y, sample_bond_30y,
    sample_book, sample_curve, sample_session,
)
import trellis


class TestSampleBonds:

    def test_all_bonds_priceable(self):
        curve = sample_curve()
        settlement = date(2024, 11, 15)
        for bond_fn in [sample_bond_2y, sample_bond_5y, sample_bond_10y, sample_bond_30y]:
            bond = bond_fn()
            result = trellis.price(bond, curve, settlement)
            assert isinstance(result, PricingResult)
            assert result.clean_price > 0

    def test_bond_maturities_correct(self):
        assert sample_bond_2y().maturity == 2
        assert sample_bond_5y().maturity == 5
        assert sample_bond_10y().maturity == 10
        assert sample_bond_30y().maturity == 30


class TestSampleBook:

    def test_book_structure(self):
        book = sample_book()
        assert len(book) == 4
        assert set(book.names) == {"2Y", "5Y", "10Y", "30Y"}

    def test_notionals_set(self):
        book = sample_book()
        assert book.notional("10Y") == 25_000_000
        assert book.notional("2Y") == 5_000_000


class TestSampleSession:

    def test_session_prices_bond(self):
        s = sample_session()
        result = s.price(sample_bond_10y())
        assert isinstance(result, PricingResult)
        assert result.greeks["dv01"] > 0

    def test_session_prices_book(self):
        s = sample_session()
        br = s.price(sample_book())
        assert isinstance(br, BookResult)
        assert br.total_mv > 0


class TestQuickstart:

    def test_quickstart_returns_session(self):
        s = trellis.quickstart()
        result = s.price(trellis.sample_bond_10y())
        assert isinstance(result, PricingResult)
        assert result.clean_price > 0

    def test_quickstart_sample_bond_prices(self):
        s = trellis.quickstart()
        result = s.price(trellis.sample_bond_10y())
        assert result.clean_price > 0
