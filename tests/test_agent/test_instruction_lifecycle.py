"""Tests for structured instruction resolution and precedence."""

from __future__ import annotations

from types import SimpleNamespace

from trellis.agent.knowledge.instructions import resolve_instruction_records
from trellis.agent.knowledge.schema import InstructionRecord, ProductIR


def _analytical_pricing_plan() -> SimpleNamespace:
    # QUA-909: Black76 now requires ``black_vol_surface`` in
    # ``required_market_data``; declare it so the route lifecycle test
    # actually dispatches to ``analytical_black76`` as intended.
    return SimpleNamespace(
        method="analytical",
        method_modules=("trellis.models.black",),
        required_market_data=("discount_curve", "black_vol_surface"),
        model_to_build="swaption",
        reasoning="route lifecycle test",
        modeling_requirements=(),
    )


def _swaption_product_ir() -> ProductIR:
    # QUA-909: Black76's positive payoff-family filter uses the canonical
    # ``swaption`` label (what ``make_rate_style_swaption_contract`` emits
    # in ``trellis/agent/semantic_contracts.py``). Use that here so the
    # lifecycle fixture matches production dispatch shape instead of the
    # internal ``rate_style_swaption`` subfamily used before QUA-909.
    return ProductIR(
        instrument="swaption",
        payoff_family="swaption",
        exercise_style="european",
        model_family="interest_rate",
        schedule_dependence=True,
    )


def _basket_product_ir() -> ProductIR:
    return ProductIR(
        instrument="basket_path_payoff",
        payoff_family="basket_path_payoff",
        payoff_traits=("ranked_observation",),
        schedule_dependence=True,
    )


def test_resolver_preserves_non_conflicting_route_hints():
    records = (
        InstructionRecord(
            id="basket:helper",
            title="Use the route helper directly",
            instruction_type="hard_constraint",
            source_kind="route_card",
            scope_methods=("monte_carlo",),
            scope_instruments=("basket_path_payoff",),
            scope_routes=("correlated_basket_monte_carlo",),
            precedence_rank=100,
            statement="Use the route helper directly inside evaluate().",
        ),
        InstructionRecord(
            id="basket:schedule",
            title="Use the shared schedule builder",
            instruction_type="route_hint",
            source_kind="route_card",
            scope_methods=("monte_carlo",),
            scope_instruments=("basket_path_payoff",),
            scope_routes=("correlated_basket_monte_carlo",),
            precedence_rank=90,
            statement="Use trellis.core.date_utils.generate_schedule before pricing.",
        ),
        InstructionRecord(
            id="basket:note",
            title="Avoid inline grids",
            instruction_type="historical_note",
            source_kind="route_card",
            scope_methods=("monte_carlo",),
            scope_instruments=("basket_path_payoff",),
            scope_routes=("correlated_basket_monte_carlo",),
            precedence_rank=80,
            statement="Do not hard-code observation or payment grids inside the payoff body.",
        ),
    )

    resolved = resolve_instruction_records(
        records,
        method="monte_carlo",
        instrument_type="basket_path_payoff",
        route="correlated_basket_monte_carlo",
        product_ir=_basket_product_ir(),
    )

    assert [instruction.id for instruction in resolved.effective_instructions] == [
        "basket:helper",
        "basket:schedule",
        "basket:note",
    ]
    assert not resolved.conflicts


def test_resolver_reports_conflicts_and_superseded_records():
    records = (
        InstructionRecord(
            id="route:old",
            title="Old rule",
            instruction_type="hard_constraint",
            source_kind="route_card",
            scope_methods=("analytical",),
            scope_instruments=("swaption",),
            scope_routes=("analytical_black76",),
            precedence_rank=10,
            statement="Use the legacy route helper.",
        ),
        InstructionRecord(
            id="route:new",
            title="New rule",
            instruction_type="hard_constraint",
            source_kind="canonical",
            scope_methods=("analytical",),
            scope_instruments=("swaption",),
            scope_routes=("analytical_black76",),
            precedence_rank=20,
            supersedes=("route:old",),
            statement="Use the approved route helper.",
        ),
        InstructionRecord(
            id="route:hint",
            title="Schedule hint",
            instruction_type="route_hint",
            source_kind="route_card",
            scope_methods=("analytical",),
            scope_instruments=("swaption",),
            scope_routes=("analytical_black76",),
            precedence_rank=5,
            statement="Use trellis.core.date_utils.generate_schedule before pricing.",
        ),
    )

    resolved = resolve_instruction_records(
        records,
        method="analytical",
        instrument_type="swaption",
        route="analytical_black76",
        product_ir=ProductIR(
            instrument="swaption",
            payoff_family="rate_style_swaption",
            schedule_dependence=True,
        ),
    )

    assert [instruction.id for instruction in resolved.effective_instructions] == [
        "route:new",
        "route:hint",
    ]
    assert [instruction.id for instruction in resolved.dropped_instructions] == [
        "route:old",
    ]
    assert resolved.conflicts == ()


def test_resolver_reports_conflicting_hard_constraints():
    records = (
        InstructionRecord(
            id="route:one",
            title="Rule one",
            instruction_type="hard_constraint",
            source_kind="route_card",
            scope_methods=("analytical",),
            scope_instruments=("swaption",),
            scope_routes=("analytical_black76",),
            precedence_rank=10,
            statement="Use route helper A.",
        ),
        InstructionRecord(
            id="route:two",
            title="Rule two",
            instruction_type="hard_constraint",
            source_kind="canonical",
            scope_methods=("analytical",),
            scope_instruments=("swaption",),
            scope_routes=("analytical_black76",),
            precedence_rank=20,
            statement="Use route helper B.",
        ),
    )

    resolved = resolve_instruction_records(
        records,
        method="analytical",
        instrument_type="swaption",
        route="analytical_black76",
        product_ir=ProductIR(
            instrument="swaption",
            payoff_family="rate_style_swaption",
            schedule_dependence=True,
        ),
    )

    assert [instruction.id for instruction in resolved.effective_instructions] == [
        "route:two",
        "route:one",
    ]
    assert len(resolved.conflicts) == 1
    assert resolved.conflicts[0].winner_id == "route:two"
    assert resolved.conflicts[0].conflicting_ids == ("route:two", "route:one")


def test_generation_plan_materializes_resolved_instructions():
    from trellis.agent.codegen_guardrails import build_generation_plan

    plan = build_generation_plan(
        pricing_plan=_analytical_pricing_plan(),
        instrument_type="swaption",
        inspected_modules=("trellis.models.black",),
        product_ir=_swaption_product_ir(),
    )

    assert plan.resolved_instructions is not None
    assert plan.resolved_instructions.route == "analytical_black76"
    # QUA-909: with a canonical ``payoff_family="swaption"`` +
    # ``exercise_style="european"`` ProductIR the specific swaption when
    # clause activates, routing through
    # ``price_swaption_black76`` as a ``route_helper``. The route-helper
    # instruction lands first because the specific-helper clause preempts
    # the shared schedule-builder primitive that only appears on the
    # ``when: default`` fallback clause.
    instruction_ids = [
        instruction.id
        for instruction in plan.resolved_instructions.effective_instructions
    ]
    assert instruction_ids[0] == "analytical_black76:route-helper"
    assert plan.resolved_instructions.conflicts == ()
