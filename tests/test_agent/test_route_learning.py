"""Tests for offline route-learning data generation and ranking."""

from __future__ import annotations


def test_default_synthetic_product_cases_include_supported_and_blocked_examples():
    from trellis.agent.route_learning import default_synthetic_product_cases

    cases = default_synthetic_product_cases()

    descriptions = {case.description for case in cases}
    assert "American put option on equity" in descriptions
    assert "Callable bond with semiannual coupon and call schedule" in descriptions
    assert "American Asian barrier option under Heston with early exercise" in descriptions


def test_build_route_training_rows_emits_proceed_and_block_decisions():
    from trellis.agent.route_learning import (
        build_route_training_rows,
        default_synthetic_product_cases,
    )

    rows = build_route_training_rows(default_synthetic_product_cases())

    assert rows
    decisions = {row.decision for row in rows}
    assert "proceed" in decisions
    assert "block" in decisions
    assert any(
        row.route == "exercise_lattice" and row.decision == "proceed"
        for row in rows
    )
    assert any(
        row.instrument == "barrier_option"
        and row.decision == "block"
        for row in rows
    )


def test_fit_linear_route_ranker_prefers_exercise_routes_for_known_products():
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.quant import PricingPlan
    from trellis.agent.route_learning import (
        build_route_training_rows,
        default_synthetic_product_cases,
        fit_linear_route_ranker,
        learned_route_decision,
    )

    ranker = fit_linear_route_ranker(
        build_route_training_rows(default_synthetic_product_cases()),
    )

    callable_decision = learned_route_decision(
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount", "black_vol"},
            model_to_build="callable_bond",
            reasoning="test",
        ),
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
        ranker=ranker,
    )
    assert callable_decision.decision == "proceed"
    assert callable_decision.selected_route == "exercise_lattice"

    american_decision = learned_route_decision(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.models.monte_carlo.engine"],
            required_market_data={"discount", "black_vol"},
            model_to_build="american_option",
            reasoning="test",
        ),
        product_ir=decompose_to_ir(
            "American put option on equity",
            instrument_type="american_option",
        ),
        ranker=ranker,
    )
    assert american_decision.decision == "proceed"
    assert american_decision.selected_route == "exercise_monte_carlo"


def test_learned_route_decision_keeps_blocked_composite_blocked():
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.quant import PricingPlan
    from trellis.agent.route_learning import (
        build_route_training_rows,
        default_synthetic_product_cases,
        fit_linear_route_ranker,
        learned_route_decision,
    )

    ranker = fit_linear_route_ranker(
        build_route_training_rows(default_synthetic_product_cases()),
    )

    decision = learned_route_decision(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.models.monte_carlo.engine"],
            required_market_data={"discount", "black_vol"},
            model_to_build=None,
            reasoning="test",
        ),
        product_ir=decompose_to_ir(
            "American Asian barrier option under Heston with early exercise",
        ),
        ranker=ranker,
    )

    assert decision.decision == "block"
    assert decision.selected_route == "exercise_monte_carlo"
    assert "path_dependent_early_exercise_under_stochastic_vol" in decision.blockers
