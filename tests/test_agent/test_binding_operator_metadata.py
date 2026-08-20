"""Tests for binding-first operator metadata resolution."""

from __future__ import annotations


def test_resolve_binding_operator_metadata_returns_canonical_entry_for_known_binding():
    from trellis.agent.binding_operator_metadata import resolve_binding_operator_metadata

    metadata = resolve_binding_operator_metadata(
        binding_id="trellis.models.quanto_option.price_quanto_option_analytical_from_market_state",
        engine_family="analytical",
        route_family="analytical",
        route_id="equity_quanto",
    )

    assert metadata is not None
    assert metadata.display_name == "Quanto option analytical binding"
    assert metadata.diagnostic_label == "quanto_analytical_binding"
    assert "semantic quanto option pricing" in metadata.short_description


def test_resolve_binding_operator_metadata_names_cds_composition_bindings():
    from trellis.agent.binding_operator_metadata import resolve_binding_operator_metadata

    analytical = resolve_binding_operator_metadata(
        binding_id="trellis.models.contingent_cashflows.expected_first_event_weights",
        engine_family="analytical",
        route_family="event_triggered_two_legged_contract",
        route_id="credit_default_swap",
    )
    monte_carlo = resolve_binding_operator_metadata(
        binding_id="trellis.models.contingent_cashflows.sample_first_event_weights",
        engine_family="analytical",
        route_family="event_triggered_two_legged_contract",
        route_id="credit_default_swap",
    )

    assert analytical.display_name == "CDS expected first-event composition"
    assert analytical.diagnostic_label == "credit_default_swap_analytical_binding"
    assert monte_carlo.display_name == "CDS sampled first-event composition"
    assert monte_carlo.diagnostic_label == "credit_default_swap_monte_carlo_binding"


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


def test_resolve_binding_operator_metadata_uses_symbol_slug_when_route_id_is_empty():
    from trellis.agent.binding_operator_metadata import resolve_binding_operator_metadata

    metadata = resolve_binding_operator_metadata(
        binding_id="trellis.models.synthetic.price_bound_helper",
        engine_family="analytical",
        route_family="analytical",
        route_id="",
    )

    assert metadata is not None
    assert metadata.display_name == "Bound (analytical)"
    assert metadata.diagnostic_label == "price_bound_helper"


def test_resolve_binding_operator_metadata_returns_canonical_entry_for_black76_put():
    from trellis.agent.binding_operator_metadata import resolve_binding_operator_metadata

    metadata = resolve_binding_operator_metadata(
        binding_id="trellis.models.black.black76_put",
        engine_family="analytical",
        route_family="analytical",
        route_id="analytical_black76",
    )

    assert metadata is not None
    assert metadata.display_name == "Black-76 analytical put binding"
    assert metadata.diagnostic_label == "black76_put_analytical_binding"
