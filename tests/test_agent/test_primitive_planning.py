"""Tests for deterministic primitive planning from ProductIR."""

from __future__ import annotations

import pytest

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
    assert primitive_symbols == {"price_american_equity_option_lsm_monte_carlo"}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert primitive_modules == {"trellis.models.equity_option_monte_carlo"}
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
    assert {
        "price_single_barrier_option_pde_result",
        "SingleBarrierPDEConfig",
        "resolve_single_barrier_inputs",
    } <= primitive_symbols
    assert "trellis.models.single_barrier_option" in primitive_modules
    assert any("absorbing boundary" in note for note in plan.primitive_plan.notes)


def test_builds_mc_plan_for_barrier_option_uses_single_barrier_helper():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="barrier_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="barrier_option",
        inspected_modules=(
            "trellis.models.single_barrier_option",
            "trellis.models.monte_carlo.engine",
            "trellis.models.processes.gbm",
        ),
        product_ir=decompose_to_ir(
            "Barrier call: PDE absorbing BC vs MC discrete monitoring",
            instrument_type="barrier_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "monte_carlo_paths"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "price_single_barrier_option_monte_carlo_result",
        "SingleBarrierMonteCarloConfig",
        "resolve_single_barrier_inputs",
        "single_barrier_state_payoff",
    } <= primitive_symbols
    assert "price_double_barrier_option_monte_carlo_result" not in primitive_symbols


def test_double_barrier_text_emits_trait_for_route_conditions():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    ir = decompose_to_ir("Double barrier option via PDE", instrument_type="barrier_option")

    assert "double_barrier" in ir.payoff_traits


def test_builds_pde_plan_for_double_barrier_uses_absorbing_grid_primitives():
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
            "trellis.models.analytical.support.barriers",
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
        product_ir=decompose_to_ir(
            "Double barrier option via PDE",
            instrument_type="barrier_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "pde_theta_1d"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "price_double_barrier_option_pde_result",
        "DoubleBarrierPDEConfig",
        "resolve_double_barrier_inputs",
        "terminal_double_barrier_payoff",
        "Grid",
        "BlackScholesOperator",
        "theta_method_1d",
    } <= primitive_symbols


def test_builds_mc_plan_for_double_barrier_uses_process_engine_and_payoff_primitives():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="barrier_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="barrier_option",
        inspected_modules=(
            "trellis.models.analytical.support.barriers",
            "trellis.models.monte_carlo.engine",
            "trellis.models.processes.gbm",
        ),
        product_ir=decompose_to_ir(
            "Double barrier option via PDE",
            instrument_type="barrier_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "monte_carlo_paths"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "price_double_barrier_option_monte_carlo_result",
        "DoubleBarrierMonteCarloConfig",
        "resolve_double_barrier_inputs",
        "double_barrier_state_payoff",
        "GBM",
        "MonteCarloEngine",
    } <= primitive_symbols
    assert "price_event_aware_monte_carlo" not in primitive_symbols


def test_builds_mc_plan_for_autocallable_uses_event_contract_helper_without_qmc_requirement():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.autocallable"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="autocallable",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="autocallable",
        inspected_modules=("trellis.models.autocallable",),
        product_ir=decompose_to_ir(
            "Autocallable note with coupon, autocall barrier, terminal protection",
            instrument_type="autocallable",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "monte_carlo_paths"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "AutocallableMonteCarloConfig",
        "price_autocallable_monte_carlo_result",
        "resolve_autocallable_inputs",
    } <= primitive_symbols
    assert "sobol_normals" not in primitive_symbols


def test_plan_build_uses_static_autocallable_spec_under_offline_guard():
    from trellis.agent.offline_agents import offline_local_agent_llm_guard
    from trellis.agent.planner import plan_build

    with offline_local_agent_llm_guard():
        plan = plan_build(
            "Autocallable note with quarterly observations, coupon, and terminal protection",
            {"discount_curve", "black_vol_surface"},
            instrument_type="autocallable",
            preferred_method="monte_carlo",
        )

    assert plan.spec_schema is not None
    assert plan.spec_schema.spec_name == "AutocallableSpec"
    field_names = {field.name for field in plan.spec_schema.fields}
    assert {
        "observation_times",
        "autocall_barrier",
        "protection_barrier",
        "coupon_rate",
    } <= field_names


def test_builds_heston_adi_plan_for_pde_method():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.pde.heston_adi"],
        required_market_data={"discount_curve", "spot", "model_parameters"},
        model_to_build="heston_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="heston_option",
        inspected_modules=("trellis.models.pde.heston_adi", "trellis.models.transforms.heston"),
        product_ir=decompose_to_ir(
            "2D PDE: Heston (S, V) via ADI splitting",
            instrument_type="heston_option",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "heston_adi_2d"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "resolve_heston_adi_pde_inputs",
        "price_heston_option_adi_pde_result",
    } <= primitive_symbols


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
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.notes == ()


def test_builds_primitive_plan_for_cev_pde():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.schema import ProductIR

    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.equity_option_pde"],
        required_market_data={"discount_curve"},
        model_to_build="european_option",
        reasoning="test",
    )
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        payoff_traits=("cev_process",),
        exercise_style="european",
        model_family="cev_diffusion",
        candidate_engine_families=("pde",),
        route_families=("pde_solver", "equity_tree"),
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.equity_option_pde",),
        product_ir=product_ir,
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "cev_theta_pde"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_cev_option_pde"}
    assert primitive_modules == {"trellis.models.equity_option_pde"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.notes == ()


def test_builds_primitive_plan_for_cev_spot_lattice():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.schema import ProductIR

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.equity_option_tree"],
        required_market_data={"discount_curve"},
        model_to_build="european_option",
        reasoning="test",
    )
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        payoff_traits=("cev_process",),
        exercise_style="european",
        model_family="cev_diffusion",
        candidate_engine_families=("lattice",),
        route_families=("pde_solver", "equity_tree"),
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.equity_option_tree",),
        product_ir=product_ir,
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "cev_spot_lattice"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_cev_option_tree"}
    assert primitive_modules == {"trellis.models.equity_option_tree"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.notes == ()


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
    assert {"price_swaption_black76"} <= primitive_symbols
    assert "black76_call" not in primitive_symbols
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.notes == ()
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
    assert plan.primitive_plan.route == "credit_default_swap"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {"build_cds_schedule", "interval_default_probability", "price_cds_monte_carlo", "get_numpy"} <= primitive_symbols
    assert "MonteCarloEngine" not in primitive_symbols
    assert plan.primitive_plan.notes == ()


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
    assert primitive_symbols == {"price_fx_vanilla_analytical"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.blockers == ()


def test_builds_fx_barrier_plan_for_knock_in_fx_context():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.quant import PricingPlan

    description = "Price an FX knock-in call with domestic/foreign discounting."
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.fx_barrier_option"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
        model_to_build="barrier_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(description, instrument_type="barrier_option")

    assert product_ir.model_family == "fx"
    assert {"fx_rates", "forward_curve", "spot"} <= product_ir.required_market_data

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="barrier_option",
        inspected_modules=("trellis.models.fx_barrier_option",),
        product_ir=product_ir,
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_fx_barrier"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_fx_barrier_option_analytical"}
    assert plan.primitive_plan.adapters == ()
    assert plan.primitive_plan.blockers == ()


def test_builds_quanto_analytical_plan_with_shared_resolution_and_black76():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.quant import select_pricing_method_for_product_ir

    product_ir = decompose_to_ir(
        "Quanto option: quanto-adjusted BS vs MC cross-currency",
        instrument_type="quanto_option",
    )

    plan = build_generation_plan(
        pricing_plan=select_pricing_method_for_product_ir(
            product_ir,
            preferred_method="analytical",
        ),
        instrument_type="quanto_option",
        inspected_modules=("trellis.models.black", "trellis.models.resolution.quanto"),
        product_ir=product_ir,
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "equity_quanto"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_quanto_option_analytical_from_market_state"}
    assert plan.primitive_plan.adapters == ()


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
    # QUA-816: route-card `adapters` retired from `local_vol_monte_carlo`;
    # constructive guidance flows through `lane_obligations._construction_steps_for`
    # for `EventAwareMonteCarloIR` with `process_family=local_vol_1d` instead.
    # The typed `state_process` / `path_simulation` / `pricing_kernel`
    # primitives above are the source of truth.
    assert plan.primitive_plan.adapters == ()
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

    # QUA-915: ZCB-option family collapsed. The analytical branch of
    # ``short_rate_bond_option`` resolves to the Jamshidian helper and
    # the adapter block stays empty.
    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "short_rate_bond_option"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_zcb_option_jamshidian"}
    assert plan.primitive_plan.adapters == ()


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

    # QUA-915: the rate-tree branch of the collapsed route resolves to
    # the Hull-White lattice helper.
    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "short_rate_bond_option"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_zcb_option_tree"}
    assert plan.primitive_plan.adapters == ()


def test_builds_short_rate_bond_rate_tree_plan():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.short_rate_bond"],
        required_market_data={"discount_curve", "model_parameters"},
        model_to_build="short_rate_bond",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="short_rate_bond",
        inspected_modules=("trellis.models.short_rate_bond",),
        product_ir=decompose_to_ir(
            "Vasicek bond pricing: tree vs analytical",
            instrument_type="short_rate_bond",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "short_rate_zero_coupon_bond"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_short_rate_zero_coupon_bond_tree"}
    assert plan.primitive_plan.adapters == ()


def test_builds_short_rate_bond_analytical_plan():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.short_rate_bond"],
        required_market_data={"discount_curve", "model_parameters"},
        model_to_build="short_rate_bond",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="short_rate_bond",
        inspected_modules=("trellis.models.short_rate_bond",),
        product_ir=decompose_to_ir(
            "CIR bond pricing: tree vs analytical",
            instrument_type="short_rate_bond",
        ),
    )

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "short_rate_zero_coupon_bond"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {"price_short_rate_zero_coupon_bond_analytical"}
    assert plan.primitive_plan.adapters == ()


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
    assert "price_american_equity_option_lsm_monte_carlo" in text


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
