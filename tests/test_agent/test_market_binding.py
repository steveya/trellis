"""Tests for compiled required-data and market-binding specs."""

from __future__ import annotations

from types import SimpleNamespace


def test_required_data_spec_compiles_from_semantic_contract():
    from trellis.agent.market_binding import build_required_data_spec
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
    )

    spec = build_required_data_spec(contract)

    assert "underlier_spot" in spec.required_input_ids
    assert "spot" in spec.required_capabilities
    assert "discount_curve" in spec.required_capabilities


def test_market_binding_spec_projects_legacy_connector_hints_from_compiled_bindings():
    from trellis.agent.market_binding import (
        build_market_binding_spec,
        build_required_data_spec,
    )
    from trellis.agent.semantic_contracts import make_quanto_option_contract
    from trellis.agent.valuation_context import build_valuation_context

    contract = make_quanto_option_contract(
        description="Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        underliers=("SAP",),
        observation_schedule=("2025-11-15",),
    )
    required_spec = build_required_data_spec(contract)
    binding_spec = build_market_binding_spec(
        contract,
        valuation_context=build_valuation_context(
            market_snapshot=SimpleNamespace(source="mock"),
            reporting_currency="USD",
            requested_outputs=["price", "vega"],
        ),
        required_data_spec=required_spec,
    )

    hints = binding_spec.to_connector_binding_hints()
    assert binding_spec.market_source == "mock"
    assert binding_spec.reporting_currency == "USD"
    assert binding_spec.requested_outputs == ("price", "vega")
    assert hints["model_parameters"]["capability"] == "model_parameters"
    assert "quanto_correlation" in hints["model_parameters"]["aliases"]
    assert hints["model_parameters"]["binding_source"] == "valuation_context.market_snapshot"
