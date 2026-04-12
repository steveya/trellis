from __future__ import annotations

from trellis.agent.backend_bindings import (
    find_backend_binding_by_route_id,
    load_backend_binding_catalog,
    resolve_backend_binding_spec,
)
from trellis.agent.knowledge.schema import ProductIR


def test_binding_catalog_loads_core_route_backed_bindings():
    catalog = load_backend_binding_catalog()

    route_ids = {binding.route_id for binding in catalog.bindings}
    assert {
        "zcb_option_rate_tree",
        "analytical_black76",
        "credit_default_swap_analytical",
        "analytical_garman_kohlhagen",
        "waterfall_cashflows",
    } <= route_ids


def test_resolve_backend_binding_spec_captures_helper_schedule_and_cashflow_roles():
    catalog = load_backend_binding_catalog()

    cds = find_backend_binding_by_route_id("credit_default_swap_monte_carlo", catalog)
    waterfall = find_backend_binding_by_route_id("waterfall_cashflows", catalog)

    assert cds is not None
    assert waterfall is not None

    cds_resolved = resolve_backend_binding_spec(
        cds,
        product_ir=ProductIR(
            instrument="cds",
            payoff_family="credit_default_swap",
            schedule_dependence=True,
            state_dependence="pathwise_only",
        ),
    )
    waterfall_resolved = resolve_backend_binding_spec(
        waterfall,
        product_ir=ProductIR(
            instrument="waterfall",
            payoff_family="waterfall",
            schedule_dependence=True,
            state_dependence="schedule_state",
        ),
    )

    assert cds_resolved.binding_id == "trellis.models.credit_default_swap.price_cds_monte_carlo"
    assert cds_resolved.helper_refs == (
        "trellis.models.credit_default_swap.price_cds_monte_carlo",
    )
    assert cds_resolved.schedule_builder_refs == (
        "trellis.models.credit_default_swap.build_cds_schedule",
    )
    assert waterfall_resolved.cashflow_engine_refs == (
        "trellis.models.cashflow_engine.waterfall.Waterfall",
        "trellis.models.cashflow_engine.waterfall.Tranche",
    )
    assert waterfall_resolved.binding_id == "trellis.models.cashflow_engine.waterfall.Waterfall"


def test_resolve_backend_binding_spec_uses_route_conditionals_for_exact_targets():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert binding is not None

    swaption_resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            schedule_dependence=True,
            state_dependence="schedule_state",
        ),
    )

    assert (
        swaption_resolved.helper_refs
        == ("trellis.models.rate_style_swaption.price_swaption_black76",)
    )
    assert (
        swaption_resolved.binding_id
        == "trellis.models.rate_style_swaption.price_swaption_black76"
    )
