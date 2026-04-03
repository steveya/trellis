"""Tests for term sheet parsing and matching."""

from datetime import date
from unittest.mock import patch

import pytest

from trellis.agent.term_sheet import TermSheet, parse_term_sheet
from trellis.agent.ask import match_payoff, ask_session, AskResult


SETTLE = date(2024, 11, 15)


class TestTermSheet:

    def test_construction(self):
        ts = TermSheet(
            instrument_type="cap",
            notional=10_000_000,
            parameters={"strike": 0.04, "end_date": "2029-11-15"},
        )
        assert ts.instrument_type == "cap"
        assert ts.notional == 10_000_000
        assert ts.parameters["strike"] == 0.04

    def test_defaults(self):
        ts = TermSheet(instrument_type="bond")
        assert ts.notional == 100.0
        assert ts.currency == "USD"
        assert ts.parameters == {}


class TestMatchPayoff:

    def test_match_bond(self):
        ts = TermSheet(
            instrument_type="bond",
            notional=100,
            parameters={"coupon": 0.05, "maturity": 10},
        )
        result = match_payoff(ts, SETTLE)
        assert result is not None
        payoff, reqs = result
        assert "discount_curve" in reqs
        assert type(payoff).__name__ == "DeterministicCashflowPayoff"

    def test_match_cap(self):
        ts = TermSheet(
            instrument_type="cap",
            notional=1_000_000,
            parameters={
                "strike": 0.04,
                "end_date": "2029-11-15",
                "frequency": "quarterly",
            },
        )
        result = match_payoff(ts, SETTLE)
        assert result is not None
        payoff, reqs = result
        assert type(payoff).__name__ == "CapPayoff"
        assert "black_vol_surface" in reqs

    def test_match_floor(self):
        ts = TermSheet(
            instrument_type="floor",
            notional=1_000_000,
            parameters={"strike": 0.03, "end_date": "2029-11-15"},
        )
        result = match_payoff(ts, SETTLE)
        assert result is not None
        payoff, reqs = result
        assert type(payoff).__name__ == "FloorPayoff"

    def test_match_swap(self):
        ts = TermSheet(
            instrument_type="swap",
            notional=10_000_000,
            parameters={
                "fixed_rate": 0.045,
                "end_date": "2029-11-15",
            },
        )
        result = match_payoff(ts, SETTLE)
        assert result is not None
        payoff, reqs = result
        assert type(payoff).__name__ == "SwapPayoff"

    def test_unknown_returns_none(self):
        ts = TermSheet(
            instrument_type="variance_swap",
            notional=1_000_000,
            parameters={"strike_vol": 0.20},
        )
        result = match_payoff(ts, SETTLE)
        assert result is None  # triggers build


class TestParseMocked:
    """Test the parser with mocked LLM responses."""

    @patch("trellis.agent.config.llm_generate_json")
    def test_parse_cap(self, mock_llm):
        mock_llm.return_value = {
            "instrument_type": "cap",
            "notional": 10000000,
            "currency": "USD",
            "parameters": {
                "strike": 0.04,
                "end_date": "2029-11-15",
                "frequency": "quarterly",
                "rate_index": "SOFR_3M",
            },
        }
        ts = parse_term_sheet("5Y cap at 4% on $10M SOFR", SETTLE)
        assert ts.instrument_type == "cap"
        assert ts.notional == 10_000_000
        assert ts.parameters["strike"] == 0.04
        assert ts.parameters["rate_index"] == "SOFR_3M"

    @patch("trellis.agent.config.llm_generate_json")
    def test_parse_swaption(self, mock_llm):
        mock_llm.return_value = {
            "instrument_type": "swaption",
            "notional": 1000000,
            "currency": "USD",
            "parameters": {
                "strike": 0.045,
                "expiry_date": "2025-11-15",
                "swap_start": "2025-11-15",
                "swap_end": "2030-11-15",
                "is_payer": True,
            },
        }
        ts = parse_term_sheet("1Y into 5Y payer swaption at 4.5%", SETTLE)
        assert ts.instrument_type == "swaption"
        assert ts.parameters["expiry_date"] == "2025-11-15"

    @patch("trellis.agent.config.llm_generate_json")
    def test_parse_bond(self, mock_llm):
        mock_llm.return_value = {
            "instrument_type": "bond",
            "notional": 100,
            "currency": "USD",
            "parameters": {
                "coupon": 0.05,
                "maturity": 10,
                "end_date": "2034-11-15",
            },
        }
        ts = parse_term_sheet("10Y US Treasury bond 5% coupon", SETTLE)
        assert ts.instrument_type == "bond"


class TestAskSessionMocked:
    """Test the full ask pipeline with mocked LLM."""

    @patch("trellis.agent.config.llm_generate_json")
    def test_ask_bond(self, mock_llm):
        """Bond pricing should work without vol surface."""
        mock_llm.return_value = {
            "instrument_type": "bond",
            "notional": 100,
            "currency": "USD",
            "parameters": {
                "coupon": 0.05,
                "maturity": 10,
            },
        }
        from trellis.session import Session
        from trellis.curves.yield_curve import YieldCurve

        s = Session(curve=YieldCurve.flat(0.05), settlement=SETTLE)
        result = ask_session("10Y 5% bond", s)

        assert isinstance(result, AskResult)
        assert result.price > 0
        assert result.matched_existing is True
        assert result.payoff_class == "DeterministicCashflowPayoff"

    @patch("trellis.agent.config.llm_generate_json")
    def test_ask_cap(self, mock_llm):
        mock_llm.return_value = {
            "instrument_type": "cap",
            "notional": 1000000,
            "currency": "USD",
            "parameters": {
                "strike": 0.04,
                "end_date": "2029-11-15",
                "frequency": "quarterly",
            },
        }
        from trellis.session import Session
        from trellis.curves.yield_curve import YieldCurve
        from trellis.models.vol_surface import FlatVol

        s = Session(
            curve=YieldCurve.flat(0.05), settlement=SETTLE,
            vol_surface=FlatVol(0.20),
        )
        result = ask_session("5Y cap at 4% on $1M", s)

        assert result.price > 0
        assert result.matched_existing is True
        assert result.payoff_class == "CapPayoff"
