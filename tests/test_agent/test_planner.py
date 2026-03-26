"""Tests for agent build planner."""

import pytest

from trellis.agent.planner import (
    BuildPlan, BuildStep, FieldDef, SpecSchema, STATIC_SPECS, _plan_static, plan_build,
)


class TestSpecSchema:

    def test_swaption_static_spec(self):
        spec = STATIC_SPECS["swaption"]
        assert spec.class_name == "SwaptionPayoff"
        assert spec.spec_name == "SwaptionSpec"
        field_names = [f.name for f in spec.fields]
        assert "notional" in field_names
        assert "strike" in field_names
        assert "expiry_date" in field_names
        assert "swap_start" in field_names
        assert "swap_end" in field_names

    def test_cap_static_spec(self):
        spec = STATIC_SPECS["cap"]
        assert spec.class_name == "AgentCapPayoff"

    def test_frozen(self):
        spec = STATIC_SPECS["swaption"]
        with pytest.raises(AttributeError):
            spec.class_name = "changed"


class TestPlanStatic:

    def test_swaption_plan_has_spec(self):
        plan = _plan_static(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
            {"discount", "forward_rate", "black_vol"},
            set(),
        )
        assert plan.payoff_class_name == "SwaptionPayoff"
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "SwaptionPayoff"
        assert len(plan.steps) == 1
        assert "swaption" in plan.steps[0].module_path

    def test_unknown_instrument_no_spec(self):
        plan = _plan_static(
            "variance swap on VIX",
            {"discount", "black_vol"},
            {"discount", "black_vol"},
            set(),
        )
        assert plan.spec_schema is None  # no static spec → needs LLM Step 1

    def test_generic_class_name(self):
        plan = _plan_static(
            "variance swap on VIX",
            {"discount"},
            {"discount"},
            set(),
        )
        assert plan.payoff_class_name == "VarianceSwapPayoff"

    def test_barrier_option_has_spec(self):
        plan = _plan_static(
            "exotic barrier option",
            {"discount", "black_vol"},
            {"discount", "black_vol"},
            set(),
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "BarrierOptionPayoff"

    def test_basket_option_has_spec(self):
        plan = _plan_static(
            "worst-of basket option on two equities",
            {"discount", "spot"},
            {"discount", "spot"},
            set(),
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "BasketOptionPayoff"

    def test_fx_vanilla_description_gets_specialized_spec(self):
        plan = _plan_static(
            "Build a pricer for: FX option (EURUSD): GK analytical vs MC\n\nImplementation target: garman_kohlhagen\nPreferred method family: analytical",
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            set(),
            instrument_type="european_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "FXVanillaAnalyticalPayoff"
        assert plan.steps[0].module_path.endswith("fxvanillaanalytical.py")

    def test_fx_vanilla_description_honors_preferred_method_for_monte_carlo_spec(self):
        plan = _plan_static(
            "Build a pricer for: FX option (EURUSD): GK analytical vs MC",
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            set(),
            instrument_type="european_option",
            preferred_method="monte_carlo",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "FXVanillaMonteCarloPayoff"
        assert plan.steps[0].module_path.endswith("fxvanillamontecarlo.py")

    def test_european_option_description_gets_specialized_spec(self):
        plan = _plan_static(
            "European call option on equity",
            {"discount", "black_vol"},
            {"discount", "black_vol"},
            set(),
            instrument_type="european_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "EuropeanOptionAnalyticalPayoff"

    def test_local_vol_description_gets_deterministic_spec(self):
        plan = _plan_static(
            "Build a pricer for: European equity call under local vol: PDE vs MC",
            {"discount_curve", "spot", "local_vol_surface"},
            {"discount_curve", "spot", "local_vol_surface"},
            set(),
            instrument_type="european_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "EuropeanLocalVolMonteCarloPayoff"


class TestPlanBuild:

    def test_missing_capabilities_raises(self):
        with pytest.raises(NotImplementedError, match="missing capabilities"):
            plan_build("some exotic", {"discount", "simulated_paths"})

    def test_swaption_returns_plan_with_spec(self):
        plan = plan_build(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
        )
        assert isinstance(plan, BuildPlan)
        assert plan.spec_schema is not None
        assert plan.payoff_class_name == "SwaptionPayoff"

    def test_fx_vanilla_route_uses_deterministic_spec_schema(self):
        plan = plan_build(
            "Build a pricer for: FX option (EURUSD): GK analytical vs MC\n\nImplementation target: garman_kohlhagen\nPreferred method family: analytical",
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            instrument_type="european_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.spec_name == "FXVanillaOptionSpec"
        assert plan.payoff_class_name == "FXVanillaAnalyticalPayoff"

    def test_fx_vanilla_route_uses_preferred_method_for_monte_carlo(self):
        plan = plan_build(
            "Build a pricer for: FX option (EURUSD): GK analytical vs MC",
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            instrument_type="european_option",
            preferred_method="monte_carlo",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.spec_name == "FXVanillaOptionSpec"
        assert plan.payoff_class_name == "FXVanillaMonteCarloPayoff"

    def test_plan_gap_analysis(self):
        plan = plan_build(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
        )
        assert plan.satisfied == frozenset({"discount", "forward_rate", "black_vol"})
        assert plan.missing == frozenset()
