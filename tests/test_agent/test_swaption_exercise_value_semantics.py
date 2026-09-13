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
