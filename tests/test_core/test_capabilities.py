"""Tests for capability inventory, gap analysis, and market data checking."""

from datetime import date

import pytest

from trellis.core.capabilities import (
    analyze_gap,
    capability_summary,
    check_market_data,
    discover_capabilities,
)
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class TestDiscoverCapabilities:

    def test_returns_market_data_and_methods(self):
        caps = discover_capabilities()
        assert "market_data" in caps
        assert "methods" in caps

    def test_market_data_includes_core(self):
        caps = discover_capabilities()
        names = {c.name for c in caps["market_data"]}
        assert {"discount", "forward_rate", "black_vol", "credit",
                "forecast_rate", "state_space", "fx"}.issubset(names)

    def test_methods_include_phase7(self):
        caps = discover_capabilities()
        names = {c.name for c in caps["methods"]}
        for name in ["rate_tree", "monte_carlo", "pde_solver",
                      "fft_pricing", "copula", "waterfall"]:
            assert name in names

    def test_each_market_data_has_how_to_provide(self):
        caps = discover_capabilities()
        for c in caps["market_data"]:
            assert c.how_to_provide


class TestAnalyzeGap:

    def test_market_data_satisfied(self):
        satisfied, missing = analyze_gap({"discount", "forward_rate"})
        assert satisfied == {"discount", "forward_rate"}
        assert missing == set()

    def test_methods_also_satisfied(self):
        satisfied, missing = analyze_gap({"discount", "rate_tree", "monte_carlo"})
        assert satisfied == {"discount", "rate_tree", "monte_carlo"}
        assert missing == set()

    def test_truly_unknown_is_missing(self):
        satisfied, missing = analyze_gap({"discount", "quantum_computing"})
        assert missing == {"quantum_computing"}

    def test_empty(self):
        satisfied, missing = analyze_gap(set())
        assert satisfied == set()
        assert missing == set()


class TestCheckMarketData:

    def test_all_present(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05), vol_surface=FlatVol(0.20))
        errors = check_market_data({"discount", "black_vol"}, ms)
        assert errors == []

    def test_missing_vol(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05))
        errors = check_market_data({"discount", "black_vol"}, ms)
        assert len(errors) == 1
        assert "black_vol" in errors[0]
        assert "FlatVol" in errors[0]

    def test_missing_discount(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE)
        errors = check_market_data({"discount"}, ms)
        assert len(errors) == 1
        assert "discount" in errors[0]
        assert "YieldCurve" in errors[0]

    def test_method_not_checked_as_market_data(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05))
        # rate_tree is a method, not market data — should not appear in errors
        errors = check_market_data({"discount", "rate_tree"}, ms)
        assert errors == []


class TestPricePayoffErrors:

    def test_missing_vol_gives_helpful_error(self):
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency
        from trellis.engine.payoff_pricer import price_payoff

        cap = CapPayoff(CapFloorSpec(
            notional=1e6, strike=0.04,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        ))
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05))
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(cap, ms)
        msg = str(exc_info.value)
        assert "black_vol" in msg
        assert "FlatVol" in msg

    def test_missing_discount_gives_helpful_error(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        from trellis.instruments.bond import Bond
        from trellis.engine.payoff_pricer import price_payoff

        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)
        ms = MarketState(as_of=SETTLE, settlement=SETTLE)
        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(payoff, ms)
        msg = str(exc_info.value)
        assert "discount" in msg
        assert "YieldCurve" in msg


class TestCapabilitySummary:

    def test_includes_both_sections(self):
        summary = capability_summary()
        assert "Market Data" in summary
        assert "Computational Methods" in summary

    def test_methods_show_required_market_data(self):
        summary = capability_summary()
        assert "Requires market data" in summary
