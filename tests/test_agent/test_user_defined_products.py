"""Tests for structured user-defined product compilation."""

from __future__ import annotations


CALLABLE_BOND_SPEC = """\
name: custom_callable_note
payoff_family: callable_fixed_income
payoff_traits:
  - callable
  - fixed_coupons
exercise_style: issuer_call
schedule_dependence: true
state_dependence: schedule_dependent
model_family: interest_rate
candidate_engine_families:
  - lattice
  - exercise
required_market_data:
  - discount
  - black_vol
preferred_method: rate_tree
"""


BLOCKED_COMPOSITE_SPEC = """\
name: heston_american_asian_barrier
payoff_family: composite_option
payoff_traits:
  - american
  - asian
  - barrier
  - stochastic_vol
exercise_style: american
schedule_dependence: false
state_dependence: path_dependent
model_family: stochastic_volatility
candidate_engine_families:
  - exercise
  - monte_carlo
required_market_data:
  - discount
  - black_vol
preferred_method: monte_carlo
"""


def test_parse_user_product_spec_from_yaml():
    from trellis.agent.user_defined_products import parse_user_product_spec

    spec = parse_user_product_spec(CALLABLE_BOND_SPEC)

    assert spec.name == "custom_callable_note"
    assert spec.exercise_style == "issuer_call"
    assert spec.schedule_dependence is True
    assert spec.preferred_method == "rate_tree"


def test_compile_supported_user_defined_callable_product_to_existing_route():
    from trellis.agent.user_defined_products import compile_user_defined_product

    compiled = compile_user_defined_product(CALLABLE_BOND_SPEC)

    assert compiled.product_ir.instrument == "custom_callable_note"
    assert compiled.pricing_plan.method == "rate_tree"
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "exercise_lattice"
    assert compiled.generation_plan.blocker_report is None
    assert compiled.generation_plan.new_primitive_workflow is None
    assert compiled.knowledge_text
    assert "custom_callable_note" in compiled.knowledge_text


def test_compile_blocked_user_defined_product_surfaces_blocker_and_workflow():
    from trellis.agent.user_defined_products import compile_user_defined_product

    compiled = compile_user_defined_product(BLOCKED_COMPOSITE_SPEC)

    assert compiled.product_ir.instrument == "heston_american_asian_barrier"
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "exercise_monte_carlo"
    assert compiled.generation_plan.blocker_report is not None
    assert compiled.generation_plan.blocker_report.should_block
    assert compiled.generation_plan.new_primitive_workflow is not None
    assert "## Structured blocker report" in compiled.rendered_plan
    assert "## New primitive workflow" in compiled.rendered_plan
