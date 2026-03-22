"""Tests for trellis.session — Session (immutable market snapshot)."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from trellis.book import Book, BookResult
from trellis.core.types import PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.bond import Bond
from trellis.session import Session


def _curve():
    return YieldCurve.flat(0.045)


def _bond():
    return Bond(face=100, coupon=0.045, maturity_date=date(2034, 11, 15),
                maturity=10, frequency=2)


SETTLE = date(2024, 11, 15)


class TestSessionPricing:

    def test_price_bond(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        r = s.price(_bond())
        assert isinstance(r, PricingResult)
        assert 80 < r.clean_price < 120

    def test_greeks_none_returns_empty(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        r = s.price(_bond(), greeks=None)
        assert r.greeks == {}

    def test_greeks_selective(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        r = s.price(_bond(), greeks=["dv01"])
        assert "dv01" in r.greeks
        assert "convexity" not in r.greeks

    def test_price_book(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        book = Book({"10Y": _bond()})
        br = s.price(book)
        assert isinstance(br, BookResult)
        assert br.book_dv01 > 0

    def test_greeks_method(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        g = s.greeks(_bond())
        assert "dv01" in g
        assert g["dv01"] > 0


class TestSessionScenarios:

    def test_with_curve_shift_returns_new_session(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        s2 = s.with_curve_shift(+100)
        assert s2 is not s
        # Original unchanged
        assert np.allclose(s.curve.rates, 0.045)
        # Shifted curve
        assert np.allclose(s2.curve.rates, 0.045 + 0.01)

    def test_shift_lowers_price(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        p_base = s.price(_bond()).clean_price
        s2 = s.with_curve_shift(+100)
        p_shifted = s2.price(_bond()).clean_price
        assert p_shifted < p_base

    def test_with_tenor_bumps(self):
        tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.04, 0.042, 0.045, 0.047, 0.05]
        curve = YieldCurve(tenors, rates)
        s = Session(curve=curve, settlement=SETTLE)
        s2 = s.with_tenor_bumps({10.0: +50})
        # 10Y rate bumped by 50bps
        idx = 3  # 10.0 is at index 3
        assert float(s2.curve.rates[idx]) == pytest.approx(0.047 + 0.005, abs=1e-9)
        # Other rates unchanged
        assert float(s2.curve.rates[0]) == pytest.approx(0.04, abs=1e-9)

    def test_with_curve_replaces(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        new_curve = YieldCurve.flat(0.06)
        s2 = s.with_curve(new_curve)
        assert np.allclose(s2.curve.rates, 0.06)


class TestSessionImmutability:

    def test_setattr_raises(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        with pytest.raises(AttributeError, match="immutable"):
            s.foo = "bar"


class TestSessionAgent:

    def test_agent_false_raises(self):
        """With agent=False (default), unsupported types propagate errors."""
        s = Session(curve=_curve(), settlement=SETTLE)
        with pytest.raises((NotImplementedError, TypeError, ValueError, AttributeError)):
            # Pass something that isn't an instrument
            s.price("not_a_bond")


class TestSessionMarketData:

    @patch("trellis.data.resolver.resolve_curve")
    def test_auto_resolution(self, mock_resolve):
        mock_resolve.return_value = _curve()
        s = Session(as_of="2024-11-15", data_source="treasury_gov")
        assert s.curve is not None
        mock_resolve.assert_called_once_with(as_of="2024-11-15", source="treasury_gov")

    def test_mock_source_no_mocking_needed(self):
        """Full offline path — no patches, no network."""
        s = Session(as_of="2024-11-15", data_source="mock")
        result = s.price(_bond())
        assert result.clean_price > 0


class TestSpreadToCurve:

    def test_spread_to_curve(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        bond = _bond()
        # Price at base curve
        base_price = s.price(bond, greeks=None).clean_price
        # Now shift curve +50bp and get that price
        s2 = s.with_curve_shift(50)
        target_price = s2.price(bond, greeks=None).clean_price
        # spread_to_curve should find ~50 bps
        spread = s.spread_to_curve(bond, target_price)
        assert spread == pytest.approx(50.0, abs=0.5)


class TestRiskReport:

    def test_risk_report_structure(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        book = Book({"10Y": _bond()})
        report = s.risk_report(book)
        assert "total_mv" in report
        assert "book_dv01" in report
        assert "book_duration" in report
        assert "book_krd" in report
        assert "positions" in report
        assert "10Y" in report["positions"]


class TestSessionPricePayoff:

    def test_price_payoff_basic(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        s = Session(curve=_curve(), settlement=SETTLE)
        adapter = DeterministicCashflowPayoff(_bond())
        pv = s.price_payoff(adapter)
        assert 80 < pv < 120

    def test_price_payoff_matches_price_instrument(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        s = Session(curve=_curve(), settlement=SETTLE)
        bond = _bond()
        result = s.price(bond, greeks=None)
        adapter = DeterministicCashflowPayoff(bond)
        pv = s.price_payoff(DeterministicCashflowPayoff(bond, day_count=bond.day_count))
        assert pv == pytest.approx(result.dirty_price, rel=1e-12)

    def test_price_payoff_with_scenario(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        s = Session(curve=_curve(), settlement=SETTLE)
        adapter = DeterministicCashflowPayoff(_bond())
        pv_base = s.price_payoff(adapter)
        s2 = s.with_curve_shift(+100)
        pv_shifted = s2.price_payoff(adapter)
        assert pv_shifted < pv_base

    def test_price_cap_payoff(self):
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency
        from trellis.models.vol_surface import FlatVol

        s = Session(curve=_curve(), settlement=SETTLE, vol_surface=FlatVol(0.20))
        spec = CapFloorSpec(
            notional=1_000_000, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )
        pv = s.price_payoff(CapPayoff(spec))
        assert pv > 0

    def test_with_vol_surface(self):
        from trellis.models.vol_surface import FlatVol
        s = Session(curve=_curve(), settlement=SETTLE)
        s2 = s.with_vol_surface(FlatVol(0.30))
        assert s2 is not s
        assert s2.vol_surface is not None
        assert s.vol_surface is None
