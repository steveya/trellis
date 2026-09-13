"""Bounded swaption exercise values carry no contractual settlement claim."""

from dataclasses import replace

import pytest

from trellis.agent.semantic_contract_validation import validate_semantic_contract
from trellis.agent.semantic_contracts import (
    make_rate_style_swaption_contract,
    specialize_semantic_contract_for_method,
)


CONVENTION = "positive_payer_underlying_swap_npv"
EXERCISE_DATE = "2025-11-15"


def _contract(**kwargs):
    options = {
        "description": "European payer swap NPV exercise-value proof",
        "observation_schedule": (EXERCISE_DATE,),
        "exercise_style": "european",
        "term_fields": {
            "payer_receiver": "payer",
            "exercise_value_convention": CONVENTION,
        },
    }
    options.update(kwargs)
    return make_rate_style_swaption_contract(**options)


def test_default_timeline_distinguishes_omitted_and_empty_settlement_dates():
    from trellis.agent.semantic_contracts import _default_semantic_timeline

    schedule = ("2025-11-15", "2026-11-15")
    assert _default_semantic_timeline(schedule).settlement_dates == schedule[-1:]
    assert _default_semantic_timeline(schedule, settlement_dates=()).settlement_dates == ()
    assert _default_semantic_timeline(schedule, settlement_dates=[]).settlement_dates == ()


@pytest.mark.global_workflow
@pytest.mark.parametrize("method", ["analytical", "rate_tree", "monte_carlo"])
def test_loaded_t73_compiles_valuation_without_settlement_timeline_roles(method):
    from trellis.agent.dsl_algebra import contract_signature
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import task_to_semantic_contract
    from trellis.core.types import TimelineRole

    task = next(t for t in load_task_manifest("TASKS_PROOF_LEGACY.yaml") if t["id"] == "T73")
    contract = task_to_semantic_contract({**task, "title": "Uninformative label"})
    blueprint = compile_semantic_contract(contract, preferred_method=method)
    lowering = blueprint.dsl_lowering
    assert not lowering.errors
    assert blueprint.contract.product.timeline.settlement_dates == ()
    family_ir = lowering.family_ir
    if family_ir is not None:
        assert TimelineRole.SETTLEMENT not in family_ir.timeline_roles
    assert lowering.normalized_expr is not None
    assert TimelineRole.SETTLEMENT not in contract_signature(lowering.normalized_expr).timeline_roles
    program = getattr(family_ir, "event_program", None)
    if program is not None:
        assert "settlement" not in program.event_kinds
        valuations = [
            (bucket, event) for bucket in program.timeline for event in bucket.events
            if event.event_name == "exercise_value"
        ]
        assert len(valuations) == 1
        bucket, event = valuations[0]
        assert bucket.event_date == EXERCISE_DATE
        assert event.event_kind == "valuation"
        assert event.schedule_role == "decision_dates"
        assert event.phase == "decision"
        assert event.value_semantics == CONVENTION
        assert "settlement_dates" not in bucket.schedule_roles
    if method == "monte_carlo":
        assert program is not None
        assert "valuation" in family_ir.event_kinds
        assert "settlement" not in family_ir.event_kinds


@pytest.mark.global_workflow
@pytest.mark.parametrize("method", ["analytical", "rate_tree", "monte_carlo"])
@pytest.mark.parametrize("encoding", ["dict", "yaml"])
def test_loaded_t73_serialized_contract_preserves_exact_exercise_value(method, encoding):
    import yaml

    from trellis.agent.dsl_algebra import contract_signature
    from trellis.agent.platform_requests import _yaml_safe_value
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import parse_semantic_contract
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import task_to_semantic_contract
    from trellis.core.types import TimelineRole

    task = next(t for t in load_task_manifest("TASKS_PROOF_LEGACY.yaml") if t["id"] == "T73")
    original = task_to_semantic_contract(task)
    payload = _yaml_safe_value(original)
    assert payload["product"]["timeline"]["settlement_dates"] == []
    parsed = parse_semantic_contract(yaml.safe_dump(payload) if encoding == "yaml" else payload)

    assert parsed.product.event_machine == original.product.event_machine
    assert parsed.product.timeline == original.product.timeline
    assert parsed.product.obligations == original.product.obligations
    report = validate_semantic_contract(parsed)
    assert report.ok, report.errors
    blueprint = compile_semantic_contract(parsed, preferred_method=method)
    assert not blueprint.dsl_lowering.errors
    assert blueprint.contract.product.timeline.settlement_dates == ()
    assert TimelineRole.SETTLEMENT not in contract_signature(
        blueprint.dsl_lowering.normalized_expr
    ).timeline_roles


@pytest.mark.parametrize("encoding", ["typed", "dict", "yaml"])
def test_explicit_event_machine_roundtrip_preserves_nested_records(encoding):
    import yaml

    from trellis.agent.event_machine import EventGuard
    from trellis.agent.platform_requests import _yaml_safe_value
    from trellis.agent.semantic_contracts import parse_semantic_contract

    original = _contract()
    machine = original.product.event_machine
    first = machine.transitions[0]
    machine = replace(
        machine,
        states=(replace(machine.states[0], state_variables=("rate", "rate")), *machine.states[1:]),
        transitions=(replace(
            first, priority=7, event_kind="decision",
            guard=EventGuard("rate > strike", parameters=("rate", "strike", "strike")),
            action=replace(first.action, parameters=("strike", "strike")),
        ), *machine.transitions[1:]),
    )
    payload = _yaml_safe_value(replace(original, product=replace(original.product, event_machine=machine)))
    if encoding == "typed":
        payload["product"]["event_machine"] = machine
    parsed = parse_semantic_contract(yaml.safe_dump(payload) if encoding == "yaml" else payload)

    assert parsed.product.event_machine == machine
    if encoding == "typed":
        assert parsed.product.event_machine is machine


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(extra_event_field="must_not_be_dropped"),
        lambda m: m.update(states=["not_a_state_record"]),
        lambda m: m.update(transitions={"not": "a_sequence"}),
        lambda m: m["transitions"][0].update(guard=[]),
        lambda m: m["transitions"][0]["action"].update(parameters="not_a_sequence"),
        lambda m: m["transitions"][0].update(priority=True),
    ],
)
def test_serialized_event_machine_rejects_unknown_or_malformed_records(mutation):
    from trellis.agent.platform_requests import _yaml_safe_value
    from trellis.agent.semantic_contracts import parse_semantic_contract

    payload = _yaml_safe_value(_contract())
    mutation(payload["product"]["event_machine"])

    with pytest.raises((TypeError, ValueError)):
        parse_semantic_contract(payload)


@pytest.mark.parametrize("field,value", [("action_type", "settle"), ("description", "changed")])
def test_serialized_exercise_value_rejects_mutated_event_action(field, value):
    from trellis.agent.platform_requests import _yaml_safe_value
    from trellis.agent.semantic_contracts import parse_semantic_contract

    payload = _yaml_safe_value(_contract())
    payload["product"]["event_machine"]["transitions"][0]["action"][field] = value
    parsed = parse_semantic_contract(payload)

    assert getattr(parsed.product.event_machine.transitions[0].action, field) == value
    assert not validate_semantic_contract(parsed).ok


@pytest.mark.parametrize("method", ["analytical", "rate_tree", "monte_carlo"])
def test_exercise_value_mode_is_retained_when_rebuilt_for_each_method(method):
    original = _contract()
    contract = specialize_semantic_contract_for_method(
        original, preferred_method=method
    )

    report = validate_semantic_contract(contract)

    assert report.ok, report.errors
    product = contract.product
    assert product.term_fields["exercise_value_convention"] == CONVENTION
    assert product.settlement_rule == "exercise_value_only"
    assert product.maturity_settlement_rule == "exercise_value_only"
    assert len(product.obligations) == 1
    obligation = product.obligations[0]
    assert obligation.obligation_id == "exercise_value"
    assert obligation.amount_expression == CONVENTION
    assert obligation.settlement_kind == "valuation"
    assert obligation.settle_date_rule == "exercise_date"
    assert product.event_transitions == (
        "price_swaption_at_exercise", "value_at_exercise"
    )
    assert tuple(t.name for t in product.event_machine.transitions) == (
        "price_swaption_at_exercise", "value_at_exercise"
    )
    assert all(
        "cash" not in t.name and "settle" not in t.action.action_type
        for t in product.event_machine.transitions
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"exercise_value_convention": "cash_annuity"},
        {"exercise_value_convention": None},
        {"payer_receiver": "receiver"},
        {"payer_receiver": None},
        {"is_payer": False},
        {"settlement_type": "cash"},
        {"settlement_type": "physical"},
        {"settlement_timing": "exercise_date"},
        {"settlement_method": "par_yield"},
        {"cash_settlement_method": "collateralized_cash_price"},
    ],
)
def test_exercise_value_factory_rejects_unsupported_or_contradictory_terms(changes):
    terms = {"payer_receiver": "payer", "exercise_value_convention": CONVENTION}
    terms.update(changes)

    with pytest.raises(ValueError, match="exercise.value"):
        _contract(term_fields=terms)


@pytest.mark.parametrize(
    "options",
    [
        {"exercise_style": "bermudan"},
        {"observation_schedule": (EXERCISE_DATE, "2026-11-15")},
    ],
)
def test_exercise_value_factory_requires_one_european_exercise(options):
    with pytest.raises(ValueError, match="exercise.value"):
        _contract(**options)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: replace(p, settlement_rule="cash_settle_at_exercise"),
        lambda p: replace(p, maturity_settlement_rule="cash_settle_at_exercise"),
        lambda p: replace(p, timeline=replace(p.timeline, settlement_dates=(EXERCISE_DATE,))),
        lambda p: replace(p, timeline=replace(p.timeline, decision_dates=())),
        lambda p: replace(p, exercise_style="bermudan"),
        lambda p: replace(p, term_fields={**p.term_fields, "payer_receiver": "receiver"}),
        lambda p: replace(p, term_fields={**p.term_fields, "exercise_value_convention": "other"}),
        lambda p: replace(p, term_fields={**p.term_fields, "settlement_type": "cash"}),
        lambda p: replace(p, obligations=()),
        lambda p: replace(p, obligations=p.obligations * 2),
        lambda p: replace(p, obligations=(replace(p.obligations[0], obligation_id="cash_settlement"),)),
        lambda p: replace(p, obligations=(replace(p.obligations[0], amount_expression="cash_annuity"),)),
        lambda p: replace(p, obligations=(replace(p.obligations[0], settlement_kind="cash"),)),
        lambda p: replace(p, obligations=(replace(p.obligations[0], settle_date_rule="cash_settle_at_exercise"),)),
        lambda p: replace(p, event_transitions=("price_swaption_at_exercise", "settle_at_exercise")),
        lambda p: replace(p, event_machine=replace(
            p.event_machine, transitions=(replace(
                p.event_machine.transitions[0],
                action=replace(p.event_machine.transitions[0].action, action_type="settle"),
            ), *p.event_machine.transitions[1:]),
        )),
    ],
)
def test_exercise_value_validator_rejects_mutated_semantics(mutation):
    original = _contract()
    mutated = replace(original, product=mutation(original.product))

    report = validate_semantic_contract(mutated)

    assert not report.ok


@pytest.mark.legacy_compat
def test_ordinary_swaption_factory_retains_legacy_cash_semantics():
    contract = make_rate_style_swaption_contract(
        description="Existing European swaption request",
        observation_schedule=(EXERCISE_DATE,),
    )

    assert validate_semantic_contract(contract).ok
    assert contract.product.settlement_rule == "cash_settle_at_exercise"
    assert contract.product.maturity_settlement_rule == "cash_settle_at_exercise"
    assert contract.product.obligations[0].settlement_kind == "cash"
    assert contract.product.event_transitions[-1] == "settle_at_exercise"
