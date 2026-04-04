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


@pytest.mark.legacy_compat
def test_quanto_contract_compiles_to_expected_blueprint():
    from trellis.agent.family_contract_compiler import compile_family_contract
    from trellis.agent.family_contract_templates import get_family_contract_template

    compiled = compile_family_contract(get_family_contract_template("quanto_option"))

    assert compiled.family_id == "quanto_option"
    assert compiled.preferred_method == "analytical"
    assert compiled.candidate_methods == ("analytical", "monte_carlo")
    assert compiled.product_ir.instrument == "quanto_option"
    assert compiled.product_ir.payoff_family == "vanilla_option"
    assert compiled.product_ir.schedule_dependence is False
    assert compiled.product_ir.state_dependence == "terminal_markov"
    assert set(compiled.product_ir.required_market_data) >= {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "fx_rates",
        "spot",
        "model_parameters",
    }
    assert "underlier_fx_correlation" in compiled.required_market_data
    assert "fx_vol" in compiled.required_market_data
    assert "trellis.models.processes.correlated_gbm" in compiled.target_modules
    assert "quanto_adjustment_analytical" in compiled.primitive_routes
    assert "correlated_gbm_monte_carlo" in compiled.primitive_routes
    assert compiled.spec_schema_hint == "QuantoOptionAnalyticalPayoff"
