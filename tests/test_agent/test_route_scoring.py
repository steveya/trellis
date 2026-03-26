"""Tests for deterministic primitive-route scoring and selection."""

from __future__ import annotations

from trellis.agent.quant import PricingPlan


def test_callable_bond_ranks_exercise_lattice_above_generic_tree():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount", "black_vol"},
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
    assert ranked[0].score > ranked[1].score


def test_american_put_ranks_exercise_mc_above_plain_mc():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount", "black_vol"},
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


def test_unsupported_composite_routes_are_penalized_by_blockers():
    from trellis.agent.codegen_guardrails import rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount", "black_vol"},
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


def test_generation_plan_selects_highest_scored_route():
    from trellis.agent.codegen_guardrails import build_generation_plan, rank_primitive_routes
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount", "black_vol"},
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
