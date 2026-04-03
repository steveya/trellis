"""Tests for structured blocker taxonomy and missing-primitive planning."""

from __future__ import annotations

from trellis.agent.quant import PricingPlan


def test_generation_plan_attaches_structured_blocker_report_for_blocked_composite():
    from trellis.agent.codegen_guardrails import build_generation_plan
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

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=None,
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=product_ir,
    )

    assert plan.blocker_report is not None
    assert plan.blocker_report.should_block
    blocker = plan.blocker_report.blockers[0]
    assert blocker.id == "path_dependent_early_exercise_under_stochastic_vol"
    assert blocker.category == "numerical_substrate_gap"
    assert blocker.primitive_kind == "exercise_control"
    assert blocker.target_package == "trellis.models.exercise"


def test_plan_blockers_interprets_missing_module_and_missing_symbol():
    from trellis.agent.blocker_planning import plan_blockers

    report = plan_blockers((
        "missing_module:trellis.models.exercise",
        "missing_symbol:trellis.models.pde.theta_method.theta_method_1d",
    ))

    assert len(report.blockers) == 2
    categories = {blocker.category for blocker in report.blockers}
    assert "implementation_gap" in categories
    assert "export_or_registry_gap" in categories


def test_render_generation_plan_includes_structured_blocker_actions():
    from trellis.agent.codegen_guardrails import build_generation_plan, render_generation_plan
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
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=None,
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=product_ir,
    )

    text = render_generation_plan(plan)
    assert "Structured blocker report" in text
    assert "numerical_substrate_gap" in text
    assert "trellis.models.exercise" in text
    assert "tests/test_models/test_generalized_methods.py" in text
