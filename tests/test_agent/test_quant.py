"""Tests for the quant agent: method selection and data availability checking."""

from datetime import date
from types import SimpleNamespace

import pytest

from trellis.agent.quant import (
    PricingPlan,
    STATIC_PLANS,
    check_data_availability,
    quant_challenger_packet_summary,
    select_pricing_method,
    select_pricing_method_for_product_ir,
)
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class TestStaticPlans:

    def test_bond_is_analytical(self):
        plan = select_pricing_method("10Y Treasury bond", "bond")
        assert plan.method == "analytical"
        assert "discount_curve" in plan.required_market_data

    def test_cap_is_analytical_black76(self):
        plan = select_pricing_method("5Y cap at 4%", "cap")
        assert plan.method == "analytical"
        assert "black_vol_surface" in plan.required_market_data

    def test_callable_bond_uses_tree(self):
        plan = select_pricing_method("Callable bond with call schedule", "callable_bond")
        assert plan.method == "rate_tree"
        assert "trellis.models.trees.lattice" in plan.method_modules
        assert "discount_curve" in plan.required_market_data
        assert "black_vol_surface" in plan.required_market_data
        assert len(plan.modeling_requirements) > 0
        assert any("CALIBRATION" in r for r in plan.modeling_requirements)

    def test_barrier_option_uses_mc(self):
        plan = select_pricing_method("Down-and-out call option", "barrier_option")
        assert plan.method == "monte_carlo"

    def test_cdo_uses_copula(self):
        plan = select_pricing_method("CDO mezzanine tranche", "cdo")
        assert plan.method == "copula"
        assert "credit_curve" in plan.required_market_data

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

    def test_fx_option_description_enriches_market_requirements(self):
        plan = select_pricing_method(
            "Price an FX option on EURUSD with Garman-Kohlhagen",
            "european_option",
        )
        assert plan.method == "analytical"
        assert plan.selection_reason.endswith("fx_context_override")
        assert "fx_cross_currency_context" in plan.assumption_summary
        assert "garman_kohlhagen_or_equivalent_context" in plan.assumption_summary
        assert "fx_rates" in plan.required_market_data
        assert "forward_curve" in plan.required_market_data
        assert "spot" in plan.required_market_data

    def test_local_vol_description_switches_to_surface_driven_monte_carlo(self):
        plan = select_pricing_method(
            "European equity call under local vol: PDE vs MC",
            "european_option",
        )

        assert plan.method == "monte_carlo"
        assert plan.selection_reason.endswith("local_vol_context_override")
        assert "local_vol_surface_driven_context" in plan.assumption_summary
        assert "simulation_based_valuation_route" in plan.assumption_summary
        assert "path_sampling_required" in plan.assumption_summary
        assert "closed_form_or_quasi_closed_form_route" not in plan.assumption_summary
        assert "local_vol_surface" in plan.required_market_data
        assert "spot" in plan.required_market_data
        assert "black_vol_surface" not in plan.required_market_data
        assert "trellis.models.monte_carlo.local_vol" in plan.method_modules

    def test_static_plans_loaded_from_canonical_decompositions(self):
        assert "heston_option" in STATIC_PLANS
        assert STATIC_PLANS["heston_option"].method == "fft_pricing"


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
            required_market_data={"discount_curve", "black_vol_surface"},
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
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build=None,
            reasoning="test",
        )
        errors = check_data_availability(plan, ms)
        assert len(errors) == 1
        assert "black_vol_surface" in errors[0]

    def test_missing_discount(self):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE)
        plan = PricingPlan(
            method="analytical",
            method_modules=[],
            required_market_data={"discount_curve"},
            model_to_build=None,
            reasoning="test",
        )
        errors = check_data_availability(plan, ms)
        assert len(errors) == 1
        assert "discount_curve" in errors[0]

    def test_missing_credit_for_cdo(self):
        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
        )
        plan = select_pricing_method("CDO tranche", "cdo")
        errors = check_data_availability(plan, ms)
        assert any("credit_curve" in e for e in errors)

    def test_legacy_market_data_names_raise(self):
        ms = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            vol_surface=FlatVol(0.20),
            forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.051)},
        )
        plan = PricingPlan(
            method="analytical",
            method_modules=[],
            required_market_data={
                "discount_curve",
                "yield_curve",
                "risk_free_curve",
                "forward_rate_curve",
                "volatility_surface",
                "black_vol_surface",
            },
            model_to_build=None,
            reasoning="test",
        )
        with pytest.raises(ValueError, match="Unknown market-data requirements"):
            check_data_availability(plan, ms)


class TestPricingPlan:

    def test_frozen(self):
        plan = STATIC_PLANS["bond"]
        with pytest.raises(AttributeError):
            plan.method = "changed"

    def test_has_reasoning(self):
        for name, plan in STATIC_PLANS.items():
            assert plan.reasoning, f"Plan for {name} missing reasoning"

    def test_product_ir_pricing_plan_includes_analytical_modeling_requirements(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)",
            instrument_type="european_option",
        )
        plan = select_pricing_method_for_product_ir(ir)

        assert plan.method == "analytical"
        assert "trellis.models.black" in plan.method_modules
        assert "discount_curve" in plan.required_market_data
        assert "black_vol_surface" in plan.required_market_data
        assert plan.modeling_requirements
        assert any("BLACK-SCHOLES" in requirement.upper() for requirement in plan.modeling_requirements)

    def test_product_ir_multiple_valid_candidates_prefers_simplest_valid_default(self):
        product_ir = SimpleNamespace(
            instrument="synthetic_option",
            required_market_data={"discount_curve", "black_vol_surface"},
            candidate_engine_families=("pde", "monte_carlo", "tree", "analytical"),
            multi_asset=False,
            schedule_dependence=False,
            state_dependence="static",
            exercise_style="",
        )

        plan = select_pricing_method_for_product_ir(product_ir)

        assert plan.method == "analytical"
        assert plan.selection_reason == "simplest_valid_default"
        assert plan.assumption_summary[:3] == (
            "simplest_valid_assumption_set",
            "closed_form_or_quasi_closed_form_route",
            "no_path_sampling_required",
        )
        assert "multiple_valid_methods_available" in plan.assumption_summary

    def test_product_ir_multiple_candidates_emits_challenger_packet(self):
        product_ir = SimpleNamespace(
            instrument="synthetic_option",
            required_market_data={"discount_curve", "black_vol_surface"},
            candidate_engine_families=("pde", "monte_carlo", "tree", "analytical"),
            multi_asset=False,
            schedule_dependence=False,
            state_dependence="static",
            exercise_style="",
        )

        plan = select_pricing_method_for_product_ir(product_ir)
        packet = plan.challenger_packet
        summary = quant_challenger_packet_summary(plan)

        assert packet is not None
        assert packet.selected_method == "analytical"
        assert summary["selected_method"] == "analytical"
        assert summary["method_identity"] == "analytical"
        assert summary["route_family"] == "analytical"
        assert summary["candidate_methods"][0]["method"] == "analytical"
        assert summary["candidate_methods"][0]["status"] == "selected"
        rejected = {
            item["method"]: item["rejection_reason"]
            for item in summary["candidate_methods"]
            if item["status"] == "rejected"
        }
        assert rejected == {
            "rate_tree": "higher_complexity_than_selected_default",
            "pde_solver": "higher_complexity_than_selected_default",
            "monte_carlo": "higher_complexity_than_selected_default",
        }
        assert summary["assumption_basis"] == list(plan.assumption_summary)
        assert summary["required_market_data"] == ["black_vol_surface", "discount_curve"]
        assert "market_data_capability_check" in summary["expected_executable_checks"]
        assert "deterministic_validation_bundle" in summary["expected_executable_checks"]
        assert "alternative_method_challenge" in summary["expected_executable_checks"]
        assert "quant:multiple_valid_methods_available" in summary["residual_risk_handoff"]

    def test_product_ir_sensitivity_request_prefers_rate_tree_for_callable_bond(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Build a pricer for: Callable bond with 5% coupon and issuer call schedule",
            instrument_type="callable_bond",
        )
        plan = select_pricing_method_for_product_ir(
            ir,
            requested_measures=["dv01", "duration"],
        )

        assert plan.method == "rate_tree"
        assert plan.sensitivity_support is not None
        assert plan.sensitivity_support.level == "bump_only"
        assert "dv01" in plan.sensitivity_support.supported_measures

    def test_product_ir_sensitivity_request_prefers_fft_over_monte_carlo_for_heston(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Build a pricer for: European call under Heston stochastic volatility",
            instrument_type="heston_option",
        )
        plan = select_pricing_method_for_product_ir(
            ir,
            requested_measures=["vega"],
        )

        assert plan.method == "fft_pricing"
        assert plan.sensitivity_support is not None
        assert plan.sensitivity_support.level == "bump_only"

    def test_explicit_preferred_method_still_wins_over_sensitivity_bias(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Build a pricer for: European call under Heston stochastic volatility",
            instrument_type="heston_option",
        )
        plan = select_pricing_method_for_product_ir(
            ir,
            preferred_method="monte_carlo",
            requested_measures=["vega"],
        )

        assert plan.method == "monte_carlo"
        assert plan.sensitivity_support is not None
        assert plan.sensitivity_support.level == "experimental"
