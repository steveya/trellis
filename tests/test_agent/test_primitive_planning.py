"""Tests for deterministic primitive planning from ProductIR."""

from __future__ import annotations

from trellis.agent.quant import PricingPlan


def test_builds_primitive_plan_for_american_put():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "exercise_monte_carlo"
    assert plan.primitive_plan.engine_family == "exercise"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "GBM",
        "MonteCarloEngine",
        "longstaff_schwartz",
        "tsitsiklis_van_roy",
        "primal_dual_mc",
        "stochastic_mesh",
    } <= primitive_symbols
    assert "LaguerreBasis" not in primitive_symbols
    assert plan.primitive_plan.blockers == ()


def test_builds_primitive_plan_for_american_put_tree_route():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.trees",),
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "exercise_lattice"
    assert plan.primitive_plan.engine_family == "tree"
    assert plan.primitive_plan.route_family == "equity_tree"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_vanilla_equity_option_tree"}
    assert primitive_modules == {"trellis.models.equity_option_tree"}
    assert "build_rate_lattice" not in primitive_symbols
    assert any("price_vanilla_equity_option_tree" in note for note in plan.primitive_plan.notes)


def test_builds_pde_plan_for_barrier_option_uses_grid_and_operator():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.pde.theta_method"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="barrier_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="barrier_option",
        inspected_modules=(
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
        product_ir=decompose_to_ir(
            "Barrier call: PDE absorbing BC vs MC discrete monitoring",
            instrument_type="barrier_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "pde_theta_1d"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert {"Grid", "BlackScholesOperator", "theta_method_1d"} <= primitive_symbols
    assert {
        "trellis.models.pde.grid",
        "trellis.models.pde.operator",
        "trellis.models.pde.theta_method",
    } <= primitive_modules
    assert any("rannacher_timesteps" in note for note in plan.primitive_plan.notes)


def test_builds_pde_plan_for_european_option_uses_helper_route():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.equity_option_pde"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.equity_option_pde",),
        product_ir=decompose_to_ir(
            "European call: theta-method convergence order measurement",
            instrument_type="european_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "vanilla_equity_theta_pde"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_vanilla_equity_option_pde"}
    assert primitive_modules == {"trellis.models.equity_option_pde"}
    assert any("theta_0.5" in note for note in plan.primitive_plan.notes)


def test_builds_primitive_plan_for_swaption():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface"},
        model_to_build="swaption",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="swaption",
        inspected_modules=("trellis.models.black",),
        product_ir=decompose_to_ir("European payer swaption", instrument_type="swaption"),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_black76"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "resolve_swaption_black76_inputs",
        "price_swaption_black76_raw",
    } <= primitive_symbols
    assert "black76_call" not in primitive_symbols
    assert plan.primitive_plan.blockers == ()


def test_builds_cds_monte_carlo_plan_without_generic_mc_engine():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.core.date_utils", "trellis.core.differentiable"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="credit_default_swap",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="credit_default_swap",
        inspected_modules=("trellis.core.date_utils", "trellis.core.differentiable"),
        product_ir=decompose_to_ir(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="credit_default_swap",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "credit_default_swap_monte_carlo"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {"build_cds_schedule", "interval_default_probability", "price_cds_monte_carlo", "get_numpy"} <= primitive_symbols
    assert "MonteCarloEngine" not in primitive_symbols
    assert any("Do not import or instantiate `MonteCarloEngine`" in note for note in plan.primitive_plan.notes)


def test_builds_equity_analytical_plan_for_european_option():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        product_ir=decompose_to_ir(
            "Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)",
            instrument_type="european_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_black76"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {"black76_call", "black76_put", "year_fraction"} <= primitive_symbols
    assert "map_spot_discount_and_vol_to_forward_black76" in plan.primitive_plan.adapters
    assert "extract_forward_and_annuity_from_market_state" not in plan.primitive_plan.adapters
    assert plan.primitive_plan.blockers == ()


def test_builds_fx_analytical_plan_for_fx_option_context():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.quant import select_pricing_method

    pricing_plan = select_pricing_method(
        "FX option (EURUSD): GK analytical vs MC",
        instrument_type="european_option",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        product_ir=decompose_to_ir(
            "FX option (EURUSD): GK analytical vs MC",
            instrument_type="european_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_garman_kohlhagen"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "garman_kohlhagen_price_raw",
        "year_fraction",
    } <= primitive_symbols
    assert "map_fx_spot_and_curves_to_garman_kohlhagen_inputs" in plan.primitive_plan.adapters
    assert plan.primitive_plan.blockers == ()


def test_builds_quanto_analytical_plan_with_shared_resolution_and_black76():
    from types import SimpleNamespace

    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.quant import select_pricing_method_for_family_blueprint

    plan = build_generation_plan(
        pricing_plan=select_pricing_method_for_family_blueprint(
            SimpleNamespace(
                family_id="quanto_option",
                preferred_method="analytical",
                candidate_methods=("analytical",),
                instrument_type="quanto_option",
                product_ir=SimpleNamespace(
                    instrument="quanto_option",
                    required_market_data=(
                        "discount_curve",
                        "forward_curve",
                        "black_vol_surface",
                        "fx_rates",
                        "spot",
                        "model_parameters",
                    ),
                ),
                route_candidates=(
                    {
                        "method": "analytical",
                        "route": "quanto_adjustment_analytical",
                    },
                ),
            ),
            preferred_method="analytical",
        ),
        instrument_type="quanto_option",
        inspected_modules=("trellis.models.black", "trellis.models.resolution.quanto"),
        product_ir=decompose_to_ir(
            "Quanto option: quanto-adjusted BS vs MC cross-currency",
            instrument_type="quanto_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "quanto_adjustment_analytical"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {"black76_call", "black76_put", "resolve_quanto_inputs"} <= primitive_symbols
    assert "apply_quanto_adjustment_terms" in plan.primitive_plan.adapters


def test_builds_local_vol_monte_carlo_plan_for_local_vol_context():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.quant import select_pricing_method
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = select_pricing_method(
        "European equity call under local vol: PDE vs MC",
        instrument_type="european_option",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=tuple(pricing_plan.method_modules),
        product_ir=decompose_to_ir(
            "European equity call under local vol: PDE vs MC",
            instrument_type="european_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "local_vol_monte_carlo"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {"LocalVol", "MonteCarloEngine", "local_vol_european_vanilla_price"} <= primitive_symbols
    assert "map_market_state_local_vol_surface_spot_and_discount_into_local_vol_mc_inputs" in plan.primitive_plan.adapters
    assert plan.primitive_plan.blockers == ()


def test_builds_zcb_option_analytical_plan():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.zcb_option"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="zcb_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="zcb_option",
        inspected_modules=("trellis.models.zcb_option",),
        product_ir=decompose_to_ir(
            "ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical",
            instrument_type="zcb_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "zcb_option_analytical"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_zcb_option_jamshidian"}


def test_builds_zcb_option_rate_tree_plan():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.zcb_option_tree"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="zcb_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="zcb_option",
        inspected_modules=("trellis.models.zcb_option_tree",),
        product_ir=decompose_to_ir(
            "ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical",
            instrument_type="zcb_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "zcb_option_rate_tree"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_zcb_option_tree"}


def test_builds_exercise_lattice_plan_for_callable_bond():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="callable_bond",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="callable_bond",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "exercise_lattice"
    assert plan.primitive_plan.engine_family == "lattice"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_callable_bond_tree"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.blockers == ()


def test_builds_exercise_lattice_plan_for_puttable_bond():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="puttable_bond",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="puttable_bond",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=decompose_to_ir(
            "Puttable bond with semiannual coupon and put schedule",
            instrument_type="puttable_bond",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "exercise_lattice"
    assert plan.primitive_plan.engine_family == "lattice"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_callable_bond_tree"}
    assert "trellis.models.callable_bond_tree" in plan.approved_modules
    assert "tests/test_tasks/test_t05_puttable_bond.py" in plan.proposed_tests


def test_builds_exercise_lattice_plan_for_bermudan_swaption():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
        model_to_build="bermudan_swaption",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="bermudan_swaption",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "exercise_lattice"
    assert plan.primitive_plan.engine_family == "lattice"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_bermudan_swaption_tree"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.blockers == ()


def test_builds_analytical_plan_for_bermudan_swaption_lower_bound():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.rate_style_swaption"],
        required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
        model_to_build="bermudan_swaption",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="bermudan_swaption",
        inspected_modules=("trellis.models.rate_style_swaption",),
        product_ir=decompose_to_ir(
            "Bermudan swaption: analytical lower bound vs rate tree",
            instrument_type="bermudan_swaption",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_black76"
    assert plan.primitive_plan.engine_family == "analytical"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_bermudan_swaption_black76_lower_bound"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.blockers == ()


def test_unsupported_composite_primitive_plan_surfaces_blockers():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build=None,
        reasoning="test",
    )

    ir = decompose_to_ir("American Asian barrier option under Heston with early exercise")
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=None,
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=ir,
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.blockers
    assert "path_dependent_early_exercise_under_stochastic_vol" in plan.primitive_plan.blockers
    assert "primitive_plan_has_blockers" in plan.uncertainty_flags


def test_render_generation_plan_includes_primitive_plan_section():
    from trellis.agent.codegen_guardrails import build_generation_plan, render_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    text = render_generation_plan(plan)
    assert "Backend binding" in text
    assert "exercise_monte_carlo" in text
    assert "Route score" in text
    assert "longstaff_schwartz" in text
    assert "tsitsiklis_van_roy" in text
    assert "primal_dual_mc" in text


def test_build_generation_plan_uses_deterministic_cache():
    from trellis.agent.codegen_guardrails import (
        build_generation_plan,
        clear_generation_plan_cache,
        generation_plan_cache_stats,
    )
    from trellis.agent.knowledge.decompose import decompose_to_ir

    clear_generation_plan_cache()
    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir("American put option on equity", instrument_type="american_option")

    first = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=product_ir,
    )
    second = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=product_ir,
    )
    stats = generation_plan_cache_stats()

    assert first is second
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["size"] >= 1
