from __future__ import annotations

from types import SimpleNamespace

from trellis.agent.backend_bindings import (
    clear_backend_binding_catalog_cache,
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


def test_binding_catalog_canonical_load_is_not_derived_from_route_registry(monkeypatch):
    from trellis.agent import route_registry as route_registry_module

    clear_backend_binding_catalog_cache()
    monkeypatch.setattr(
        route_registry_module,
        "load_route_registry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("route registry should not load")),
    )

    catalog = load_backend_binding_catalog()

    assert find_backend_binding_by_route_id("analytical_garman_kohlhagen", catalog) is not None


def test_binding_catalog_skips_malformed_primitive_rows(monkeypatch):
    from trellis.agent import backend_bindings as backend_bindings_module

    clear_backend_binding_catalog_cache()
    monkeypatch.setattr(
        backend_bindings_module,
        "_load_canonical_bindings",
        lambda: (
            backend_bindings_module._binding_from_raw(
                {
                    "route_id": "synthetic_binding_route",
                    "engine_family": "analytical",
                    "route_family": "synthetic_family",
                    "primitives": [
                        {"module": "trellis.models.synthetic", "symbol": "good_helper", "role": "route_helper"},
                        {"module": "trellis.models.synthetic", "role": "route_helper"},
                    ],
                }
            ),
        ),
    )

    try:
        catalog = load_backend_binding_catalog()
        binding = find_backend_binding_by_route_id("synthetic_binding_route", catalog)

        assert binding is not None
        assert binding.primitives == (
            backend_bindings_module.PrimitiveRef(
                "trellis.models.synthetic",
                "good_helper",
                "route_helper",
            ),
        )
    finally:
        clear_backend_binding_catalog_cache()


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
    assert cds_resolved.primitives[0].symbol == "build_cds_schedule"
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


def test_nth_to_default_binding_has_no_schedule_builder_surface():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("nth_to_default_monte_carlo", catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="nth_to_default",
            payoff_family="nth_to_default",
            exercise_style="none",
            schedule_dependence=True,
            state_dependence="schedule_state",
        ),
    )

    assert resolved.binding_id == "trellis.instruments.nth_to_default.price_nth_to_default_basket"
    assert resolved.helper_refs == ("trellis.instruments.nth_to_default.price_nth_to_default_basket",)
    assert resolved.schedule_builder_refs == ()


def test_binding_catalog_cache_tracks_binding_catalog_freshness(monkeypatch):
    clear_backend_binding_catalog_cache()

    from trellis.agent import backend_bindings as backend_bindings_module

    catalog_one = SimpleNamespace(bindings=())
    catalog_two = SimpleNamespace(bindings=())
    calls = iter((catalog_one, catalog_two))

    monkeypatch.setattr(
        backend_bindings_module,
        "_cache_key",
        lambda *, include_discovered: (include_discovered, 1.0, "rev-a", ()),
    )
    monkeypatch.setattr(
        backend_bindings_module,
        "_load_canonical_bindings",
        lambda: tuple(next(calls).bindings),
    )

    first = load_backend_binding_catalog()
    second = load_backend_binding_catalog()
    assert first is second

    monkeypatch.setattr(
        backend_bindings_module,
        "_cache_key",
        lambda *, include_discovered: (include_discovered, 2.0, "rev-a", ()),
    )

    third = load_backend_binding_catalog()
    assert third is not first
