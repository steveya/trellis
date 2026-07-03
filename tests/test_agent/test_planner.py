"""Tests for agent build planner."""

import pytest

from trellis.agent.planner import (
    BuildPlan, BuildStep, FieldDef, SpecSchema, STATIC_SPECS, _infer_method_hint, _plan_static, plan_build,
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

    def test_period_rate_option_strip_static_spec_supports_collar_aliases(self):
        spec = STATIC_SPECS["period_rate_option_strip"]
        assert spec.class_name == "PeriodRateOptionStripPayoff"
        field_names = [f.name for f in spec.fields]
        assert "cap_strike" in field_names
        assert "floor_strike" in field_names
        assert "exercise_dates" in field_names
        assert "accrual_dates" in field_names
        assert "coupon_dates" in field_names

    def test_heston_static_spec_avoids_llm_spec_design(self):
        spec = STATIC_SPECS["heston_option"]
        assert spec.class_name == "HestonOptionPayoff"
        assert spec.spec_name == "HestonOptionSpec"
        assert set(spec.requirements) == {"discount_curve", "model_parameters", "spot"}
        field_names = [f.name for f in spec.fields]
        assert "spot" in field_names
        assert "strike" in field_names
        assert "expiry_date" in field_names

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

    def test_period_rate_option_strip_plan_has_static_spec(self):
        plan = plan_build(
            "Price a callable cap/floor collar with a non-standard accrual schedule.",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            instrument_type="period_rate_option_strip",
            preferred_method="analytical",
        )

        assert plan.payoff_class_name == "PeriodRateOptionStripPayoff"
        assert plan.spec_schema is not None
        assert plan.spec_schema.spec_name == "PeriodRateOptionStripSpec"
        assert "periodrateoptionstrip" in plan.steps[0].module_path

    def test_unknown_instrument_no_spec(self):
        plan = _plan_static(
            "weather derivative on cumulative rainfall",
            {"discount_curve", "black_vol_surface"},
            {"discount_curve", "black_vol_surface"},
            set(),
        )
        assert plan.spec_schema is None  # no static spec → needs LLM Step 1

    def test_generic_class_name(self):
        plan = _plan_static(
            "weather derivative on cumulative rainfall",
            {"discount_curve"},
            {"discount_curve"},
            set(),
        )
        assert plan.payoff_class_name == "WeatherDerivativePayoff"

    def test_generic_module_path_uses_bounded_description_slug(self):
        plan = _plan_static(
            "Build a pricer for: Crank-Nicolson Rannacher smoothing for discontinuous payoffs\n\n"
            "Construct methods: pde_solver\n"
            "Comparison targets: cn_rannacher (pde_solver), cn_standard (pde_solver), "
            "black_scholes_digital (analytical)\n",
            {"discount_curve"},
            {"discount_curve"},
            set(),
        )

        module_path = plan.steps[0].module_path

        assert module_path.startswith("instruments/_agent/")
        assert "\n" not in module_path
        assert ":" not in module_path
        assert len(module_path) <= len("instruments/_agent/") + 72 + len(".py")
        assert module_path.endswith(".py")
        assert "crank_nicolson_rannacher" in module_path

        compact_plan = _plan_static(
            "BuildAPricerFor:CrankNicolsonRannacherSmoothingForDiscontinuousPayoffs\n\n"
            "ConstructMethods:PDESolver",
            {"discount_curve"},
            {"discount_curve"},
            set(),
        )

        assert "buildapricerfor" not in compact_plan.steps[0].module_path
        assert "cranknicolsonrannacher" in compact_plan.steps[0].module_path

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

    def test_explicit_cliquet_instrument_does_not_fall_back_to_floor_regex(self):
        plan = _plan_static(
            "Build a pricer for a cliquet option with local floor protection",
            {"discount_curve", "black_vol_surface"},
            {"discount_curve", "black_vol_surface"},
            set(),
            instrument_type="cliquet_option",
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "CliquetOptionPayoff"
        assert plan.steps[0].module_path.endswith("cliquetoption.py")
        assert plan.spec_schema.spec_name == "CliquetOptionSpec"
        field_names = {field.name for field in plan.spec_schema.fields}
        assert {
            "local_cap",
            "local_floor",
            "global_cap",
            "global_floor",
            "time_day_count",
            "quadrature_order",
            "max_quadrature_nodes",
        } <= field_names

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
        assert any(field.name == "valuation_date" for field in plan.spec_schema.fields)
        assert any(field.name == "pricing_method" for field in plan.spec_schema.fields)

    def test_credit_default_swap_analytical_uses_cds_static_spec(self):
        plan = _plan_static(
            "Single-name CDS priced analytically",
            {"discount_curve", "credit_curve"},
            {"discount_curve", "credit_curve"},
            set(),
            instrument_type="credit_default_swap",
            preferred_method="analytical",
        )
        assert plan.spec_schema is STATIC_SPECS["cds"]
        assert plan.payoff_class_name == "CDSPayoff"
        assert plan.steps[0].module_path.endswith("cds.py")

    def test_cds_analytical_static_spec_carries_time_origin_and_method_fields(self):
        spec = STATIC_SPECS["cds"]
        field_names = [field.name for field in spec.fields]

        assert "valuation_date" in field_names
        assert "pricing_method" in field_names

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
        assert plan.spec_schema.spec_name == "AmericanPutTreeSpec"
        assert plan.payoff_class_name == "AmericanPutTreePayoff"
        assert [field.name for field in plan.spec_schema.fields] == [
            "notional",
            "spot",
            "strike",
            "expiry_date",
            "option_type",
            "exercise_style",
            "day_count",
            "tree_steps",
            "n_paths",
            "n_steps",
            "seed",
            "n_x",
            "n_t",
        ]

    def test_american_put_pde_route_uses_deterministic_spec_schema(self):
        plan = plan_build(
            "American put: PSOR PDE vs LSM MC",
            {"discount_curve", "black_vol_surface"},
            instrument_type="american_put",
            preferred_method="pde_solver",
        )

        assert plan.spec_schema is not None
        assert plan.spec_schema.spec_name == "AmericanPutTreeSpec"
        assert plan.payoff_class_name == "AmericanPutTreePayoff"

    def test_cev_route_uses_deterministic_spec_schema(self):
        plan = plan_build(
            "CEV model: CEVOperator PDE vs CEV tree",
            {"discount_curve"},
            instrument_type="european_option",
            preferred_method="pde_solver",
        )

        assert plan.spec_schema is not None
        assert plan.spec_schema.spec_name == "CEVOptionSpec"
        assert plan.payoff_class_name == "CEVOptionPayoff"
        assert [field.name for field in plan.spec_schema.fields] == [
            "notional",
            "spot",
            "strike",
            "expiry_date",
            "option_type",
            "day_count",
            "cev_sigma",
            "cev_beta",
            "n_x",
            "n_t",
            "tree_steps",
            "tree_grid_size",
        ]

    def test_infer_method_hint_does_not_match_tree_substrings_inside_other_words(self):
        assert _infer_method_hint("main street spot-vol quote pack") is None
        assert _infer_method_hint("american put lattice benchmark") == "rate_tree"

    def test_plan_gap_analysis(self):
        plan = plan_build(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
        )
        assert plan.satisfied == frozenset({"discount_curve", "forward_curve", "black_vol_surface"})
        assert plan.missing == frozenset()
