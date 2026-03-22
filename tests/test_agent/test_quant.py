"""Tests for the quant agent: method selection and data availability checking."""

from datetime import date

import pytest

from trellis.agent.quant import (
    PricingPlan,
    STATIC_PLANS,
    check_data_availability,
    select_pricing_method,
)
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class TestStaticPlans:

    def test_bond_is_analytical(self):
        plan = select_pricing_method("10Y Treasury bond", "bond")
        assert plan.method == "analytical"
        assert "discount" in plan.required_market_data

    def test_cap_is_analytical_black76(self):
        plan = select_pricing_method("5Y cap at 4%", "cap")
        assert plan.method == "analytical"
        assert "black_vol" in plan.required_market_data

    def test_callable_bond_uses_tree(self):
        plan = select_pricing_method("Callable bond with call schedule", "callable_bond")
        assert plan.method == "rate_tree"
        assert "trellis.models.trees.lattice" in plan.method_modules
        assert "discount" in plan.required_market_data
        assert "black_vol" in plan.required_market_data
        assert len(plan.modeling_requirements) > 0
        assert any("CALIBRATION" in r for r in plan.modeling_requirements)

    def test_barrier_option_uses_mc(self):
        plan = select_pricing_method("Down-and-out call option", "barrier_option")
        assert plan.method == "monte_carlo"

    def test_cdo_uses_copula(self):
        plan = select_pricing_method("CDO mezzanine tranche", "cdo")
        assert plan.method == "copula"
        assert "credit" in plan.required_market_data

    def test_mbs_uses_mc_with_waterfall(self):
        plan = select_pricing_method("Agency MBS passthrough", "mbs")
        assert plan.method == "monte_carlo"
        assert any("waterfall" in m for m in plan.method_modules)

    def test_swaption_is_analytical(self):
        plan = select_pricing_method("1Y into 5Y payer swaption", "swaption")
        assert plan.method == "analytical"

    def test_bermudan_swaption_uses_tree(self):
        plan = select_pricing_method("Bermudan swaption", "bermudan_swaption")
        assert plan.method == "rate_tree"


class TestMethodFromDescription:

    def test_callable_bond_extracted(self):
        plan = select_pricing_method("Price a callable bond with 3Y, 5Y, 7Y call dates")
        assert plan.method == "rate_tree"

    def test_cap_extracted(self):
        plan = select_pricing_method("Price a 5Y interest rate cap at 4%")
        assert plan.method == "analytical"

    def test_unknown_falls_back(self):
        """Unknown instrument type falls back to LLM or conservative default."""
        plan = select_pricing_method("Price a reverse convertible autocallable note")
        # Should get some plan (LLM or fallback), not crash
        assert plan.method in ("analytical", "rate_tree", "monte_carlo", "pde_solver",
                                "fft_pricing", "copula", "waterfall")


class TestDataAvailability:

    def test_all_data_present(self):
        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            vol_surface=FlatVol(0.20),
        )
        plan = PricingPlan(
            method="rate_tree",
            method_modules=[],
            required_market_data={"discount", "black_vol"},
            model_to_build=None,
            reasoning="test",
        )
        errors = check_data_availability(plan, ms)
        assert errors == []

    def test_missing_vol(self):
        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
        )
        plan = PricingPlan(
            method="rate_tree",
            method_modules=[],
            required_market_data={"discount", "black_vol"},
            model_to_build=None,
            reasoning="test",
        )
        errors = check_data_availability(plan, ms)
        assert len(errors) == 1
        assert "black_vol" in errors[0]

    def test_missing_discount(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE)
        plan = PricingPlan(
            method="analytical",
            method_modules=[],
            required_market_data={"discount"},
            model_to_build=None,
            reasoning="test",
        )
        errors = check_data_availability(plan, ms)
        assert len(errors) == 1
        assert "discount" in errors[0]

    def test_missing_credit_for_cdo(self):
        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
        )
        plan = select_pricing_method("CDO tranche", "cdo")
        errors = check_data_availability(plan, ms)
        assert any("credit" in e for e in errors)


class TestPricingPlan:

    def test_frozen(self):
        plan = STATIC_PLANS["bond"]
        with pytest.raises(AttributeError):
            plan.method = "changed"

    def test_has_reasoning(self):
        for name, plan in STATIC_PLANS.items():
            assert plan.reasoning, f"Plan for {name} missing reasoning"
