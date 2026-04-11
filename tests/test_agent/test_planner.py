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

    def test_schedule_dependent_static_specs_use_typed_date_tuples(self):
        callable_spec = STATIC_SPECS["callable_bond"]
        puttable_spec = STATIC_SPECS["puttable_bond"]
        bermudan_spec = STATIC_SPECS["bermudan_swaption"]

        assert next(f for f in callable_spec.fields if f.name == "call_dates").type == "tuple[date, ...]"
        assert next(f for f in puttable_spec.fields if f.name == "put_dates").type == "tuple[date, ...]"
        assert next(f for f in bermudan_spec.fields if f.name == "exercise_dates").type == "tuple[date, ...]"

    def test_frozen(self):
        spec = STATIC_SPECS["swaption"]
        with pytest.raises(AttributeError):
            spec.class_name = "changed"


class TestPlanStatic:

    def test_swaption_plan_has_spec(self):
        plan = _plan_static(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            {"discount_curve", "forward_curve", "black_vol_surface"},
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
            {"discount_curve", "black_vol_surface"},
            {"discount_curve", "black_vol_surface"},
            set(),
        )
        assert plan.spec_schema is None  # no static spec → needs LLM Step 1

    def test_generic_class_name(self):
        plan = _plan_static(
            "variance swap on VIX",
            {"discount_curve"},
            {"discount_curve"},
            set(),
        )
        assert plan.payoff_class_name == "VarianceSwapPayoff"

    def test_barrier_option_has_spec(self):
        plan = _plan_static(
            "exotic barrier option",
            {"discount_curve", "black_vol_surface"},
            {"discount_curve", "black_vol_surface"},
            set(),
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "BarrierOptionPayoff"

    def test_basket_option_has_spec(self):
        plan = _plan_static(
            "worst-of basket option on two equities",
            {"discount_curve", "spot"},
            {"discount_curve", "spot"},
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
            {"discount_curve", "black_vol_surface"},
            {"discount_curve", "black_vol_surface"},
            set(),
            instrument_type="european_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "EuropeanOptionAnalyticalPayoff"

    def test_explicit_zcb_option_instrument_beats_generic_european_option_specialization(self):
        plan = _plan_static(
            "European call option on a zero-coupon bond",
            {"discount_curve", "black_vol_surface"},
            {"discount_curve", "black_vol_surface"},
            set(),
            instrument_type="zcb_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "ZCBOptionPayoff"
        assert plan.spec_schema.spec_name == "ZCBOptionSpec"

    @pytest.mark.parametrize(
        ("description", "requirements", "instrument_type", "expected_class"),
        [
            (
                "European best-of basket option on two equities",
                {"discount_curve", "spot"},
                "basket_option",
                "BasketOptionPayoff",
            ),
            (
                "European down-and-out barrier option",
                {"discount_curve", "black_vol_surface"},
                "barrier_option",
                "BarrierOptionPayoff",
            ),
            (
                "European arithmetic asian option",
                {"discount_curve", "black_vol_surface"},
                "asian_option",
                "AsianOptionPayoff",
            ),
        ],
    )
    def test_explicit_family_specs_are_not_overridden_by_generic_european_specialization(
        self,
        description,
        requirements,
        instrument_type,
        expected_class,
    ):
        plan = _plan_static(
            description,
            requirements,
            requirements,
            set(),
            instrument_type=instrument_type,
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == expected_class

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

    def test_quanto_option_description_gets_specialized_spec(self):
        plan = _plan_static(
            "Build a pricer for: Quanto option on SAP settled in USD",
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"},
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"},
            set(),
            instrument_type="quanto_option",
            preferred_method="analytical",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "QuantoOptionAnalyticalPayoff"
        assert plan.steps[0].module_path.endswith("quantooptionanalytical.py")

    def test_cds_monte_carlo_specialization_still_refines_explicit_credit_family(self):
        plan = _plan_static(
            "Single-name CDS priced with Monte Carlo",
            {"discount_curve", "credit_curve"},
            {"discount_curve", "credit_curve"},
            set(),
            instrument_type="cds",
            preferred_method="monte_carlo",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "CDSPayoff"
        assert any(field.name == "n_paths" for field in plan.spec_schema.fields)

class TestPlanBuild:

    def test_missing_capabilities_raises(self):
        with pytest.raises(NotImplementedError, match="missing capabilities"):
            plan_build("some exotic", {"discount_curve", "simulated_paths"})

    def test_swaption_returns_plan_with_spec(self):
        plan = plan_build(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
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

    def test_american_put_tree_route_uses_deterministic_spec_schema(self):
        plan = plan_build(
            "American put: equity tree knowledge-light proving",
            {"discount_curve", "black_vol_surface"},
            instrument_type="american_put",
            preferred_method="rate_tree",
        )

        assert plan.spec_schema is not None
        assert plan.spec_schema.spec_name == "AmericanOptionSpec"
        assert plan.payoff_class_name == "AmericanOptionPayoff"
        assert [field.name for field in plan.spec_schema.fields] == [
            "spot",
            "strike",
            "expiry_date",
            "option_type",
            "exercise_style",
            "day_count",
        ]

    def test_plan_gap_analysis(self):
        plan = plan_build(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
        )
        assert plan.satisfied == frozenset({"discount_curve", "forward_curve", "black_vol_surface"})
        assert plan.missing == frozenset()
