"""Tests for deterministic primitive-route scoring and selection."""

from __future__ import annotations

from trellis.agent.quant import PricingPlan


def test_callable_bond_ranks_exercise_lattice_above_generic_tree():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="callable_bond",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "Callable bond with semiannual coupon and call schedule",
        instrument_type="callable_bond",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert [plan.route for plan in ranked[:2]] == [
        "exercise_lattice",
        "rate_tree_backward_induction",
    ]
    assert ranked[0].route_family == "rate_lattice"
    assert ranked[0].score > ranked[1].score


def test_american_put_ranks_exercise_mc_above_plain_mc():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "American put option on equity",
        instrument_type="american_option",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert [plan.route for plan in ranked[:2]] == [
        "exercise_monte_carlo",
        "monte_carlo_paths",
    ]
    assert ranked[0].score > ranked[1].score


def test_american_put_ranks_equity_tree_above_rate_tree():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=[
            "trellis.models.trees.binomial",
            "trellis.models.trees.backward_induction",
        ],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "American put option on equity",
        instrument_type="american_option",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert [plan.route for plan in ranked[:2]] == [
        "exercise_lattice",
        "rate_tree_backward_induction",
    ]
    assert ranked[0].engine_family == "tree"
    assert ranked[0].route_family == "equity_tree"
    assert ranked[0].score > ranked[1].score


def test_unsupported_composite_routes_are_penalized_by_blockers():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build=None,
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "American Asian barrier option under Heston with early exercise",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert ranked
    assert ranked[0].blockers
    assert "path_dependent_early_exercise_under_stochastic_vol" in ranked[0].blockers
    assert ranked[0].score < 0.0


def test_cds_ranks_credit_default_swap_route_above_generic_analytical():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=[
            "trellis.core.date_utils",
            "trellis.curves.credit_curve",
        ],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="credit_default_swap",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "CDS pricing: hazard rate MC vs survival prob analytical",
        instrument_type="cds",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert ranked
    assert ranked[0].route == "credit_default_swap_analytical"
    assert ranked[0].route_family == "credit_default_swap"
    assert ranked[0].score > 0.0


def test_zcb_option_ranks_jamshidian_route_above_generic_analytical():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.zcb_option"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="zcb_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical",
        instrument_type="zcb_option",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert ranked
    assert ranked[0].route == "zcb_option_analytical"
    assert ranked[0].score > 0.0


def test_european_option_pde_ranks_helper_route_above_generic_pde():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.equity_option_pde"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "European call: theta-method convergence order measurement",
        instrument_type="european_option",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert ranked
    assert ranked[0].route == "vanilla_equity_theta_pde"
    assert ranked[0].score > ranked[1].score


def test_nth_to_default_uses_dedicated_credit_basket_route():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.copulas.gaussian"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="nth_to_default",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "First-to-default basket on five names with Gaussian copula",
        instrument_type="nth_to_default",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert ranked
    assert ranked[0].route == "nth_to_default_monte_carlo"
    assert ranked[0].route_family == "nth_to_default"
    assert ranked[0].score > 0.0


def test_generation_plan_selects_highest_scored_route():
    from trellis.agent.codegen_guardrails import build_generation_plan, rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="callable_bond",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "Callable bond with semiannual coupon and call schedule",
        instrument_type="callable_bond",
    )

    ranked = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="callable_bond",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=product_ir,
    )

    assert generation_plan.primitive_plan is not None
    assert generation_plan.primitive_plan.route == ranked[0].route
    assert generation_plan.primitive_plan.score == ranked[0].score
