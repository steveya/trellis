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
        assert {
            "discount_curve",
            "forward_curve",
            "black_vol_surface",
            "credit_curve",
            "state_space",
            "fx_rates",
            "spot",
            "local_vol_surface",
            "jump_parameters",
            "model_parameters",
        }.issubset(names)

    def test_methods_include_phase7(self):
        caps = discover_capabilities()
        names = {c.name for c in caps["methods"]}
        for name in ["rate_tree", "monte_carlo", "qmc", "pde_solver",
                      "fft_pricing", "copula", "waterfall"]:
            assert name in names

    def test_each_market_data_has_how_to_provide(self):
        caps = discover_capabilities()
        for c in caps["market_data"]:
            assert c.how_to_provide


class TestAnalyzeGap:

    def test_market_data_satisfied(self):
        satisfied, missing = analyze_gap({"discount_curve", "forward_curve"})
        assert satisfied == {"discount_curve", "forward_curve"}
        assert missing == set()

    def test_methods_also_satisfied(self):
        satisfied, missing = analyze_gap({"discount_curve", "rate_tree", "monte_carlo"})
        assert satisfied == {"discount_curve", "rate_tree", "monte_carlo"}
        assert missing == set()

    def test_truly_unknown_is_missing(self):
        satisfied, missing = analyze_gap({"discount_curve", "quantum_computing"})
        assert missing == {"quantum_computing"}

    def test_empty(self):
        satisfied, missing = analyze_gap(set())
        assert satisfied == set()
        assert missing == set()

    def test_legacy_aliases_are_not_treated_as_known_capabilities(self):
        satisfied, missing = analyze_gap({
            "discount",
            "forward_rate",
            "black_vol",
            "credit",
            "forecast_rate",
            "fx",
            "discount_curve",
            "yield_curve",
            "risk_free_curve",
            "forward_rate_curve",
            "volatility_surface",
            "black_vol_surface",
            "spot",
            "local_vol_surface",
            "jump_parameters",
            "model_parameters",
        })
        assert satisfied == {
            "discount_curve",
            "black_vol_surface",
            "spot",
            "local_vol_surface",
            "jump_parameters",
            "model_parameters",
        }
        assert missing == {
            "discount",
            "forward_rate",
            "black_vol",
            "credit",
            "forecast_rate",
            "fx",
            "yield_curve",
            "risk_free_curve",
            "forward_rate_curve",
            "volatility_surface",
        }


class TestCheckMarketData:

    def test_all_present(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05), vol_surface=FlatVol(0.20))
        errors = check_market_data({"discount_curve", "black_vol_surface"}, ms)
        assert errors == []

    def test_missing_vol(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05))
        errors = check_market_data({"discount_curve", "black_vol_surface"}, ms)
        assert len(errors) == 1
        assert "black_vol_surface" in errors[0]
        assert "FlatVol" in errors[0]

    def test_missing_discount(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE)
        errors = check_market_data({"discount_curve"}, ms)
        assert len(errors) == 1
        assert "discount_curve" in errors[0]
        assert "YieldCurve" in errors[0]

    def test_method_not_checked_as_market_data(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                         discount=YieldCurve.flat(0.05))
        # rate_tree is a method, not market data — should not appear in errors
        errors = check_market_data({"discount_curve", "rate_tree"}, ms)
        assert errors == []

    def test_unknown_requirement_names_raise(self):
        ms = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            vol_surface=FlatVol(0.20),
            forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.051)},
        )
        with pytest.raises(ValueError, match="Unknown market-data requirements"):
            check_market_data(
                {
                    "discount_curve",
                    "yield_curve",
                    "risk_free_curve",
                    "forward_rate_curve",
                    "volatility_surface",
                    "black_vol_surface",
                },
                ms,
            )

    def test_market_state_exposes_valuation_date_alias(self):
        ms = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
        )

        assert ms.valuation_date == SETTLE


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
        assert "black_vol_surface" in msg
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
        assert "discount_curve" in msg
        assert "YieldCurve" in msg


class TestCapabilitySummary:

    def test_includes_both_sections(self):
        summary = capability_summary()
        assert "Market Data" in summary
        assert "Computational Methods" in summary

    def test_methods_show_required_market_data(self):
        summary = capability_summary()
        assert "Requires market data" in summary

    def test_qmc_summary_mentions_accelerator_role(self):
        summary = capability_summary()
        assert "qmc" in summary
        assert "Sobol" in summary or "low-discrepancy" in summary

    def test_pde_summary_uses_current_theta_method_signature(self):
        summary = capability_summary()
        # theta_method_1d is present with grid/op/terminal/theta args
        # (rendered as multiline in the capabilities summary)
        assert "theta_method_1d" in summary
        assert "theta=0.5" in summary
        assert "sigma_fn, r_fn, payoff" not in summary
