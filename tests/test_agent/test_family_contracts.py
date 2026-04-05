"""Tests for typed family contracts and blueprint compilation."""

from __future__ import annotations

from dataclasses import asdict

import pytest


def test_quanto_contract_template_validates():
    from trellis.agent.family_contract_templates import get_family_contract_template
    from trellis.agent.family_contract_validation import validate_family_contract

    report = validate_family_contract(get_family_contract_template("quanto_option"))

    assert report.ok
    assert report.errors == ()
    assert report.normalized_contract is not None
    assert report.normalized_contract.product.family_id == "quanto_option"


def test_checked_in_family_templates_include_quanto_only():
    from trellis.agent.family_contract_templates import list_family_contract_templates

    assert list_family_contract_templates() == ("quanto_option",)


def test_quanto_contract_rejects_missing_correlation():
    from trellis.agent.family_contract_templates import get_family_contract_template
    from trellis.agent.family_contract_validation import validate_family_contract

    payload = asdict(get_family_contract_template("quanto_option"))
    payload["market_data"]["required_inputs"] = [
        item
        for item in payload["market_data"]["required_inputs"]
        if item["input_id"] != "underlier_fx_correlation"
    ]

    report = validate_family_contract(payload)

    assert not report.ok
    assert any("underlier_fx_correlation" in error for error in report.errors)


def test_quanto_family_template_bridges_to_semantic_contract():
    from trellis.agent.family_contract_templates import family_template_as_semantic_contract

    contract = family_template_as_semantic_contract("quanto_option")

    assert contract is not None
    assert contract.product.semantic_id == "quanto_option"
    assert contract.product.instrument_class == "quanto_option"
    assert contract.product.payoff_family == "vanilla_option"
    assert contract.methods.preferred_method == "analytical"
    assert contract.methods.candidate_methods == ("analytical", "monte_carlo")
    required_inputs = {item.input_id for item in contract.market_data.required_inputs}
    assert "underlier_fx_correlation" in required_inputs
    assert "fx_vol" in required_inputs
    assert "trellis.models.processes.correlated_gbm" in contract.blueprint.target_modules
    assert "quanto_adjustment_analytical" in contract.blueprint.primitive_families
    assert "correlated_gbm_monte_carlo" in contract.blueprint.primitive_families
    assert contract.validation.semantic_checks == (
        "check_quanto_required_inputs",
        "check_quanto_cross_currency_semantics",
    )
