"""Tests for new-primitive workflow planning from blocker reports."""

from __future__ import annotations


def test_builds_new_foundational_primitive_workflow_for_stochastic_vol_exercise_gap():
    from trellis.agent.blocker_planning import plan_blockers
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.new_primitive_workflow import plan_new_primitive_workflow

    product_ir = decompose_to_ir(
        "American Asian barrier option under Heston with early exercise",
    )
    blocker_report = plan_blockers(
        product_ir.unresolved_primitives,
        product_ir=product_ir,
    )

    workflow = plan_new_primitive_workflow(
        blocker_report,
        product_ir=product_ir,
    )

    assert workflow.items
    item = workflow.items[0]
    assert item.action_kind == "new_foundational_primitive"
    assert item.target_package == "trellis.models.exercise"
    assert "tests/test_models/test_generalized_methods.py" in item.tests_to_add
    assert "trellis/agent/knowledge/canonical/cookbooks.yaml" in item.knowledge_files_to_update


def test_builds_library_repair_workflow_for_missing_symbol_gap():
    from trellis.agent.blocker_planning import plan_blockers
    from trellis.agent.new_primitive_workflow import plan_new_primitive_workflow

    blocker_report = plan_blockers((
        "missing_symbol:trellis.models.pde.theta_method.theta_method_1d",
    ))

    workflow = plan_new_primitive_workflow(blocker_report)

    assert workflow.items
    item = workflow.items[0]
    assert item.action_kind == "library_repair"
    assert item.target_package == "trellis.models.pde.theta_method"
    assert "tests/test_agent/test_import_registry.py" in item.tests_to_add


def test_render_new_primitive_workflow_mentions_math_tests_docs_and_knowledge():
    from trellis.agent.blocker_planning import plan_blockers
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.new_primitive_workflow import (
        plan_new_primitive_workflow,
        render_new_primitive_workflow,
    )

    product_ir = decompose_to_ir(
        "American Asian barrier option under Heston with early exercise",
    )
    workflow = plan_new_primitive_workflow(
        plan_blockers(product_ir.unresolved_primitives, product_ir=product_ir),
        product_ir=product_ir,
    )

    text = render_new_primitive_workflow(workflow)
    assert "Mathematical contract" in text
    assert "tests/test_models/test_generalized_methods.py" in text
    assert "docs/api/models.rst" in text
    assert "trellis/agent/knowledge/canonical/cookbooks.yaml" in text
