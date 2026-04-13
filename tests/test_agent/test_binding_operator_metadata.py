"""Tests for binding-first operator metadata resolution."""

from __future__ import annotations


def test_resolve_binding_operator_metadata_returns_canonical_entry_for_known_binding():
    from trellis.agent.binding_operator_metadata import resolve_binding_operator_metadata

    metadata = resolve_binding_operator_metadata(
        binding_id="trellis.models.quanto_option.price_quanto_option_analytical_from_market_state",
        engine_family="analytical",
        route_family="analytical",
        route_id="quanto_adjustment_analytical",
    )

    assert metadata is not None
    assert metadata.display_name == "Quanto option analytical binding"
    assert metadata.diagnostic_label == "quanto_analytical_binding"
    assert "semantic quanto option pricing" in metadata.short_description


def test_resolve_binding_operator_metadata_derives_fallback_without_route_prose():
    from trellis.agent.binding_operator_metadata import resolve_binding_operator_metadata

    metadata = resolve_binding_operator_metadata(
        binding_id="trellis.models.synthetic.price_pathwise_exotic_helper",
        engine_family="monte_carlo",
        route_family="basket_credit",
        route_id="synthetic_exotic_binding",
    )

    assert metadata is not None
    assert metadata.display_name == "Pathwise Exotic (monte_carlo / basket_credit)"
    assert metadata.diagnostic_label == "synthetic_exotic_binding"
    assert "trellis.models.synthetic.price_pathwise_exotic_helper" in metadata.short_description
