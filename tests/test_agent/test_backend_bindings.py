from __future__ import annotations

import pytest

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
        # QUA-915: ZCB-option family collapsed; the tree helper is now
        # reached through the ``short_rate_bond_option`` binding entry.
        "short_rate_bond_option",
        "analytical_black76",
        "credit_default_swap",
        "credit_basket_nth_to_default",
        "analytical_fx_barrier",
        "monte_carlo_fx_barrier",
        "analytical_garman_kohlhagen",
        "waterfall_cashflows",
    } <= route_ids


def test_binding_catalog_covers_retired_fallback_routes():
    """QUA-794: bindings for these lanes must remain in the canonical catalog.

    ``family_lowering_ir`` retired its
    ``route_id == X and binding_spec is None`` fallbacks for these routes on
    the basis that the catalog always resolves a binding_spec.  If a binding
    disappears from the catalog, this test fires before the missing-binding
    path exercises production builds.

    Slice 1 (PR #600): ``analytical_black76``, ``transform_fft``,
    ``monte_carlo_paths``, ``local_vol_monte_carlo``.

    Slice 2 (this PR): ``vanilla_equity_theta_pde``, ``pde_theta_1d``,
    ``exercise_lattice``, ``correlated_basket_monte_carlo``,
    ``credit_default_swap``,
    ``credit_basket_nth_to_default``.
    """
    catalog = load_backend_binding_catalog()
    route_ids = {binding.route_id for binding in catalog.bindings}
    assert {
        # slice 1
        "analytical_black76",
        "levy_reference_analytical",
        "transform_fft",
        "monte_carlo_paths",
        "local_vol_monte_carlo",
        # slice 2
        "vanilla_equity_theta_pde",
        "cev_theta_pde",
        "cev_spot_lattice",
        "pde_theta_1d",
        "exercise_lattice",
        "correlated_basket_monte_carlo",
        "credit_default_swap",
        "credit_basket_nth_to_default",
        "analytical_fx_barrier",
        "monte_carlo_fx_barrier",
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


def test_resolve_backend_binding_spec_uses_cev_exact_helpers():
    catalog = load_backend_binding_catalog()
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
        model_family="cev_diffusion",
        candidate_engine_families=("pde", "lattice"),
    )

    pde_binding = find_backend_binding_by_route_id("cev_theta_pde", catalog)
    tree_binding = find_backend_binding_by_route_id("cev_spot_lattice", catalog)

    assert pde_binding is not None
    assert tree_binding is not None
    pde_resolved = resolve_backend_binding_spec(pde_binding, product_ir=product_ir)
    tree_resolved = resolve_backend_binding_spec(tree_binding, product_ir=product_ir)

    assert pde_resolved.exact_target_refs == (
        "trellis.models.equity_option_pde.price_cev_option_pde",
    )
    assert pde_resolved.helper_refs == (
        "trellis.models.equity_option_pde.price_cev_option_pde",
    )
    assert tree_resolved.exact_target_refs == (
        "trellis.models.equity_option_tree.price_cev_option_tree",
    )
    assert tree_resolved.helper_refs == (
        "trellis.models.equity_option_tree.price_cev_option_tree",
    )


def test_resolve_backend_binding_spec_uses_vanilla_pde_primitive_composition():
    catalog = load_backend_binding_catalog()
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
        model_family="equity_diffusion",
        candidate_engine_families=("pde",),
    )
    binding = find_backend_binding_by_route_id("vanilla_equity_theta_pde", catalog)

    assert binding is not None
    resolved = resolve_backend_binding_spec(binding, product_ir=product_ir)

    assert resolved.exact_target_refs == (
        "trellis.models.pde.event_aware.solve_event_aware_pde",
    )
    assert resolved.helper_refs == ()
    assert {
        (primitive.symbol, primitive.role)
        for primitive in resolved.primitives
    } >= {
        ("resolve_single_state_diffusion_inputs", "market_binding"),
        ("terminal_intrinsic_from_resolved", "payoff_primitive"),
        ("build_event_aware_pde_problem", "problem_builder"),
        ("solve_event_aware_pde", "pricing_kernel"),
        ("interpolate_pde_values", "interpolation"),
    }


@pytest.mark.parametrize(
    ("route_id", "expected_symbols"),
    [
        (
            "analytical_black76",
            {
                "resolve_single_state_diffusion_inputs",
                "single_factor_lognormal_sum_contract",
                "weighted_lognormal_sum_moments",
                "match_lognormal_moments",
                "black76_call",
                "black76_put",
                "year_fraction",
            },
        ),
        (
            "monte_carlo_paths",
            {
                "resolve_single_state_diffusion_inputs",
                "WeightedObservationContract",
                "weighted_observation_payoff",
                "GBM",
                "MonteCarloEngine",
                "StateAwarePayoff",
                "year_fraction",
                "get_numpy",
            },
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_arithmetic_asian_primitive_composition(
    route_id,
    expected_symbols,
):
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id(route_id, catalog)
    product_ir = ProductIR(
        instrument="asian_option",
        payoff_family="asian_option",
        payoff_traits=("asian", "arithmetic_average"),
        exercise_style="european",
        state_dependence="path_dependent",
        schedule_dependence=True,
        model_family="equity_diffusion",
    )

    assert binding is not None
    resolved = resolve_backend_binding_spec(binding, product_ir=product_ir)

    assert resolved.helper_refs == ()
    assert expected_symbols == {primitive.symbol for primitive in resolved.primitives}
    assert not any("asian_option.price_" in ref for ref in resolved.primitive_refs)


@pytest.mark.parametrize(
    "payoff_traits",
    [
        ("asian", "geometric_average"),
        ("asian", "arithmetic_average", "floating_strike"),
        ("asian", "arithmetic_average", "multi_asset"),
    ],
)
def test_resolve_backend_binding_spec_does_not_narrow_broader_asian_contracts(
    payoff_traits,
):
    catalog = load_backend_binding_catalog()
    product_ir = ProductIR(
        instrument="asian_option",
        payoff_family="asian_option",
        payoff_traits=payoff_traits,
        exercise_style="european",
        state_dependence="path_dependent",
        schedule_dependence=True,
        model_family="equity_diffusion",
    )
    forbidden = {
        "WeightedObservationContract",
        "weighted_observation_payoff",
        "single_factor_lognormal_sum_contract",
        "weighted_lognormal_sum_moments",
        "match_lognormal_moments",
    }

    for route_id in ("analytical_black76", "monte_carlo_paths"):
        binding = find_backend_binding_by_route_id(route_id, catalog)
        assert binding is not None
        resolved = resolve_backend_binding_spec(binding, product_ir=product_ir)
        assert forbidden.isdisjoint(
            primitive.symbol for primitive in resolved.primitives
        )


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

    cds = find_backend_binding_by_route_id("credit_default_swap", catalog)
    waterfall = find_backend_binding_by_route_id("waterfall_cashflows", catalog)

    assert cds is not None
    assert waterfall is not None

    cds_resolved = resolve_backend_binding_spec(
        cds,
        product_ir=ProductIR(
            instrument="cds",
            payoff_family="event_triggered_two_legged_contract",
            schedule_dependence=True,
            state_dependence="pathwise_only",
        ),
        method="monte_carlo",
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

    assert swaption_resolved.helper_refs == ()
    assert swaption_resolved.market_binding_refs == (
        "trellis.models.rate_style_swaption.resolve_swaption_black76_inputs",
    )
    assert swaption_resolved.pricing_kernel_refs == (
        "trellis.models.rate_style_swaption.price_swaption_black76_raw",
    )
    assert swaption_resolved.primitive_refs == (
        "trellis.models.rate_style_swaption.resolve_swaption_black76_inputs",
        "trellis.models.rate_style_swaption.price_swaption_black76_raw",
    )
    assert (
        swaption_resolved.binding_id
        == "trellis.models.rate_style_swaption.price_swaption_black76_raw"
    )

    bermudan_resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="bermudan_swaption",
            payoff_family="swaption",
            exercise_style="bermudan",
            schedule_dependence=True,
            state_dependence="schedule_state",
        ),
    )

    assert bermudan_resolved.helper_refs == ()
    assert bermudan_resolved.schedule_builder_refs == (
        "trellis.core.date_utils.normalize_explicit_dates",
    )
    assert bermudan_resolved.market_binding_refs == (
        "trellis.models.rate_style_swaption.resolve_swaption_black76_inputs",
    )
    assert bermudan_resolved.pricing_kernel_refs == (
        "trellis.models.rate_style_swaption.price_swaption_black76_raw",
    )
    assert bermudan_resolved.exact_target_refs == (
        "trellis.models.rate_style_swaption.price_swaption_black76_raw",
    )
    assert bermudan_resolved.binding_id == (
        "trellis.models.rate_style_swaption.price_swaption_black76_raw"
    )


def test_resolve_backend_binding_spec_composes_european_swaption_monte_carlo():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("monte_carlo_paths", catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            schedule_dependence=True,
            state_dependence="schedule_state",
            model_family="interest_rate",
        ),
    )

    assert resolved.helper_refs == ()
    assert resolved.market_binding_refs == (
        "trellis.models.rate_style_swaption.resolve_swaption_black76_inputs",
    )
    assert resolved.schedule_builder_refs == (
        "trellis.core.date_utils.build_payment_timeline",
    )
    assert resolved.exact_target_refs == (
        "trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo",
    )
    assert resolved.binding_id == (
        "trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo"
    )
    assert {primitive.symbol: primitive.role for primitive in resolved.primitives} == {
        "resolve_swaption_black76_inputs": "market_binding",
        "build_payment_timeline": "schedule_builder",
        "resolve_hull_white_monte_carlo_process_inputs": "process_binding",
        "build_discounted_swap_pv_payload": "settlement_payload",
        "build_short_rate_discount_reducer": "path_reducer",
        "EventAwareMonteCarloEvent": "event_contract",
        "EventAwareMonteCarloProblemSpec": "problem_spec",
        "build_event_aware_monte_carlo_problem": "problem_builder",
        "price_event_aware_monte_carlo": "monte_carlo_estimator",
    }


def test_resolve_backend_binding_spec_composes_european_swaption_rate_lattice():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("rate_tree_backward_induction", catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            schedule_dependence=True,
            state_dependence="schedule_state",
            model_family="interest_rate",
        ),
    )

    assert resolved.helper_refs == ()
    assert resolved.market_binding_refs == (
        "trellis.models.bermudan_swaption_tree.resolve_bermudan_swaption_tree_inputs",
    )
    assert resolved.exact_target_refs == (
        "trellis.models.trees.algebra.price_on_lattice",
    )
    assert resolved.binding_id == "trellis.models.trees.algebra.price_on_lattice"
    assert {primitive.symbol: primitive.role for primitive in resolved.primitives} == {
        "BermudanSwaptionTreeSpec": "contract_spec",
        "resolve_swaption_curve_basis_spread": "curve_basis_binding",
        "resolve_bermudan_swaption_tree_inputs": "market_binding",
        "BINOMIAL_1F_TOPOLOGY": "topology",
        "UNIFORM_ADDITIVE_MESH": "mesh",
        "TERM_STRUCTURE_TARGET": "calibration_target",
        "build_lattice": "lattice_builder",
        "compile_bermudan_swaption_contract_spec": "contract_compiler",
        "price_on_lattice": "pricing_kernel",
    }


def test_resolve_backend_binding_spec_uses_basket_option_exact_helpers():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    transform = find_backend_binding_by_route_id("transform_fft", catalog)

    product_ir = ProductIR(
        instrument="basket_option",
        payoff_family="basket_option",
        payoff_traits=("two_asset_terminal_basket", "vanilla_option"),
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )

    assert analytical is not None
    assert monte_carlo is not None
    assert transform is not None

    analytical_resolved = resolve_backend_binding_spec(analytical, product_ir=product_ir)
    monte_carlo_resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)
    transform_resolved = resolve_backend_binding_spec(transform, product_ir=product_ir)

    assert analytical_resolved.helper_refs == (
        "trellis.models.basket_option.price_basket_option_analytical",
    )
    assert monte_carlo_resolved.helper_refs == (
        "trellis.models.basket_option.price_basket_option_monte_carlo",
    )
    assert transform_resolved.helper_refs == (
        "trellis.models.basket_option.price_basket_option_transform_proxy",
    )


def test_monte_carlo_binding_resolves_vanilla_terminal_claim_primitives():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert resolved.exact_target_refs == ()
    assert set(resolved.primitive_refs) == {
        "trellis.models.monte_carlo.single_state_diffusion.price_single_state_terminal_claim_monte_carlo_result",
        "trellis.models.resolution.single_state_diffusion.terminal_intrinsic_from_resolved",
    }


def test_resolve_backend_binding_spec_uses_heston_transform_helper():
    catalog = load_backend_binding_catalog()
    transform = find_backend_binding_by_route_id("transform_fft", catalog)

    assert transform is not None

    resolved = resolve_backend_binding_spec(
        transform,
        product_ir=ProductIR(
            instrument="heston_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="stochastic_volatility",
        ),
    )

    assert resolved.helper_refs == (
        "trellis.models.transforms.heston.price_heston_option_transform",
    )
    assert (
        resolved.binding_id
        == "trellis.models.transforms.heston.price_heston_option_transform"
    )


def test_resolve_backend_binding_spec_uses_digital_transform_helper():
    catalog = load_backend_binding_catalog()
    transform = find_backend_binding_by_route_id("transform_fft", catalog)

    assert transform is not None

    resolved = resolve_backend_binding_spec(
        transform,
        product_ir=ProductIR(
            instrument="digital_option",
            payoff_family="digital_option",
            payoff_traits=("digital_payoff",),
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == (
        "trellis.models.equity_option_transforms.price_equity_digital_option_transform",
    )
    assert (
        resolved.binding_id
        == "trellis.models.equity_option_transforms.price_equity_digital_option_transform"
    )


def test_resolve_backend_binding_spec_uses_heston_adi_result_identity():
    catalog = load_backend_binding_catalog()
    heston_adi = find_backend_binding_by_route_id("heston_adi_2d", catalog)

    assert heston_adi is not None

    resolved = resolve_backend_binding_spec(
        heston_adi,
        product_ir=ProductIR(
            instrument="heston_option",
            payoff_family="vanilla_option",
            payoff_traits=("stochastic_vol",),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="stochastic_volatility",
        ),
    )

    assert resolved.binding_id == (
        "trellis.models.pde.heston_adi.price_heston_option_adi_pde_result"
    )
    assert resolved.exact_target_refs == (
        "trellis.models.pde.heston_adi.price_heston_option_adi_pde_result",
    )
    assert resolved.market_binding_refs == (
        "trellis.models.pde.heston_adi.resolve_heston_adi_pde_inputs",
    )


@pytest.mark.parametrize(
    "route_id,expected_ref,expected_market_binding",
    [
        (
            "pde_theta_1d",
            "trellis.models.single_barrier_option.price_single_barrier_option_pde_result",
            "trellis.models.single_barrier_option.resolve_single_barrier_inputs",
        ),
        (
            "monte_carlo_paths",
            "trellis.models.single_barrier_option.price_single_barrier_option_monte_carlo_result",
            "trellis.models.single_barrier_option.resolve_single_barrier_inputs",
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_single_barrier_exact_helpers(
    route_id,
    expected_ref,
    expected_market_binding,
):
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id(route_id, catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("barrier", "single_barrier"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        ),
        method="pde_solver" if route_id == "pde_theta_1d" else "monte_carlo",
    )

    assert resolved.binding_id == expected_ref
    assert resolved.exact_target_refs == (expected_ref,)
    assert resolved.market_binding_refs == (expected_market_binding,)


@pytest.mark.parametrize(
    "route_id,method,expected_ref,expected_market_binding",
    [
        (
            "analytical_fx_barrier",
            "analytical",
            "trellis.models.analytical.barrier.barrier_option_price",
            "trellis.models.fx_barrier_option.resolve_fx_barrier_inputs",
        ),
        (
            "monte_carlo_fx_barrier",
            "monte_carlo",
            "trellis.models.monte_carlo.engine.MonteCarloEngine",
            "trellis.models.fx_barrier_option.resolve_fx_barrier_inputs",
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_fx_barrier_primitive_composition(
    route_id,
    method,
    expected_ref,
    expected_market_binding,
):
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id(route_id, catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("barrier", "single_barrier"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="fx",
        ),
        method=method,
    )

    assert resolved.binding_id == expected_ref
    assert resolved.exact_target_refs == (expected_ref,)
    assert resolved.market_binding_refs == (expected_market_binding,)
    assert resolved.helper_refs == ()


@pytest.mark.parametrize(
    "route_id,method,expected_ref",
    [
        (
            "analytical_garman_kohlhagen",
            "analytical",
            "trellis.models.analytical.fx.garman_kohlhagen_price_raw",
        ),
        (
            "monte_carlo_fx_vanilla",
            "monte_carlo",
            "trellis.models.monte_carlo.engine.MonteCarloEngine",
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_fx_vanilla_primitive_composition(
    route_id,
    method,
    expected_ref,
):
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id(route_id, catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="fx_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="fx",
        ),
        method=method,
    )

    assert resolved.binding_id == expected_ref
    assert resolved.exact_target_refs == (expected_ref,)
    assert resolved.market_binding_refs == (
        "trellis.models.fx_vanilla.resolve_fx_vanilla_inputs",
    )
    assert resolved.helper_refs == ()


@pytest.mark.parametrize(
    "method,expected_ref,expected_extra_ref",
    [
        (
            "analytical",
            "trellis.models.black.black76_call",
            "trellis.models.analytical.support.quanto_adjusted_forward",
        ),
        (
            "monte_carlo",
            "trellis.models.monte_carlo.engine.MonteCarloEngine",
            "trellis.models.processes.correlated_gbm.CorrelatedGBM",
        ),
        (
            "qmc",
            "trellis.models.monte_carlo.engine.MonteCarloEngine",
            "trellis.models.monte_carlo.variance_reduction.sobol_normals",
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_quanto_primitive_composition(
    method,
    expected_ref,
    expected_extra_ref,
):
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("equity_quanto", catalog)

    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="quanto_option",
            payoff_family="vanilla_option",
            payoff_traits=("fx_translation",),
            exercise_style="european",
            state_dependence="terminal_markov",
        ),
        method=method,
    )

    assert resolved.binding_id == expected_ref
    assert expected_extra_ref in resolved.primitive_refs
    assert resolved.market_binding_refs == (
        "trellis.models.resolution.quanto.resolve_quanto_inputs",
    )
    assert resolved.helper_refs == ()


@pytest.mark.parametrize(
    "product_ir,expected_route_family,expected_helper_refs,expected_kernel_refs",
    [
        pytest.param(
            ProductIR(
                instrument="barrier_option",
                payoff_family="barrier_option",
                exercise_style="european",
                state_dependence="terminal_markov",
                model_family="equity_diffusion",
            ),
            "analytical",
            ("trellis.models.analytical.barrier.barrier_option_price",),
            (),
            id="barrier",
        ),
        pytest.param(
            ProductIR(
                instrument="digital_option",
                payoff_family="digital_option",
                payoff_traits=(
                    "discounting",
                    "terminal_markov",
                    "vol_surface_dependence",
                ),
                exercise_style="european",
                state_dependence="terminal_markov",
                model_family="equity_diffusion",
            ),
            "analytical",
            (),
            (
                "trellis.models.black.black76_cash_or_nothing_call",
                "trellis.models.black.black76_cash_or_nothing_put",
                "trellis.models.black.black76_asset_or_nothing_call",
                "trellis.models.black.black76_asset_or_nothing_put",
            ),
            id="digital",
        ),
        pytest.param(
            ProductIR(
                instrument="lookback_option",
                payoff_family="lookback_option",
                payoff_traits=(
                    "discounting",
                    "path_dependent",
                    "vol_surface_dependence",
                ),
                exercise_style="european",
                state_dependence="path_dependent",
                model_family="equity_diffusion",
            ),
            "analytical",
            (),
            (),
            id="lookback",
        ),
        pytest.param(
            ProductIR(
                instrument="chooser_option",
                payoff_family="chooser_option",
                payoff_traits=(
                    "discounting",
                    "terminal_markov",
                    "vol_surface_dependence",
                ),
                exercise_style="european",
                state_dependence="terminal_markov",
                model_family="equity_diffusion",
            ),
            "analytical",
            (),
            (
                "trellis.models.black.black76_call",
                "trellis.models.black.black76_put",
            ),
            id="chooser",
        ),
        pytest.param(
            ProductIR(
                instrument="compound_option",
                payoff_family="compound_option",
                payoff_traits=(
                    "discounting",
                    "terminal_markov",
                    "vol_surface_dependence",
                ),
                exercise_style="european",
                state_dependence="terminal_markov",
                model_family="equity_diffusion",
            ),
            "analytical",
            (),
            (
                "trellis.models.black.black76_call",
                "trellis.models.black.black76_put",
            ),
            id="compound",
        ),
        pytest.param(
            ProductIR(
                instrument="variance_swap",
                payoff_family="variance_swap",
                payoff_traits=(
                    "discounting",
                    "path_dependent",
                    "vol_surface_dependence",
                ),
                exercise_style="none",
                state_dependence="path_dependent",
                schedule_dependence=False,
                model_family="generic",
            ),
            "analytical",
            (),
            (),
            id="variance-swap",
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_exact_helpers_for_absorbed_black76_equity_exotics(
    product_ir,
    expected_route_family,
    expected_helper_refs,
    expected_kernel_refs,
):
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert analytical is not None

    resolved = resolve_backend_binding_spec(analytical, product_ir=product_ir)

    assert resolved.route_family == expected_route_family
    assert resolved.helper_refs == expected_helper_refs
    assert resolved.pricing_kernel_refs == expected_kernel_refs


def test_resolve_backend_binding_spec_exposes_complete_variance_swap_composition():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert analytical is not None

    resolved = resolve_backend_binding_spec(
        analytical,
        product_ir=ProductIR(
            instrument="variance_swap",
            payoff_family="variance_swap",
            payoff_traits=(
                "discounting",
                "path_dependent",
                "vol_surface_dependence",
            ),
            exercise_style="none",
            state_dependence="path_dependent",
            model_family="generic",
        ),
    )

    assert {
        "trellis.core.date_utils.year_fraction",
        "trellis.curves.interpolation.linear_interp",
        "trellis.models.analytical.support.discount_factor_from_zero_rate",
    } <= set(resolved.primitive_refs)


def test_resolve_backend_binding_spec_exposes_complete_chooser_composition():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert analytical is not None

    resolved = resolve_backend_binding_spec(
        analytical,
        product_ir=ProductIR(
            instrument="chooser_option",
            payoff_family="chooser_option",
            payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.core.date_utils.year_fraction",
        "trellis.models.analytical.support.forward_from_dividend_yield",
        "trellis.models.analytical.support.discount_factor_from_zero_rate",
        "trellis.models.analytical.support.discounted_value",
        "trellis.models.black.black76_call",
        "trellis.models.black.black76_put",
        "trellis.models.analytical.support.probability.bivariate_standard_normal_cdf",
        "trellis.models.calibration.solve_request.ObjectiveBundle",
        "trellis.models.calibration.solve_request.SolveBounds",
        "trellis.models.calibration.solve_request.SolveRequest",
        "trellis.models.calibration.solve_request.execute_solve_request",
    } <= set(resolved.primitive_refs)


def test_resolve_backend_binding_spec_exposes_complete_lookback_composition():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert analytical is not None

    resolved = resolve_backend_binding_spec(
        analytical,
        product_ir=ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=(
                "continuous_monitoring",
                "discounting",
                "fixed_strike",
                "path_dependent",
                "vol_surface_dependence",
            ),
            exercise_style="european",
            state_dependence="path_dependent",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.core.date_utils.year_fraction",
        "trellis.models.analytical.support.normalized_option_type",
        "trellis.models.analytical.support.discount_factor_from_zero_rate",
        "trellis.models.analytical.support.probability.standard_normal_cdf",
    } <= set(resolved.primitive_refs)


def test_resolve_backend_binding_spec_exposes_complete_compound_composition():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert analytical is not None

    resolved = resolve_backend_binding_spec(
        analytical,
        product_ir=ProductIR(
            instrument="compound_option",
            payoff_family="compound_option",
            payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.core.date_utils.year_fraction",
        "trellis.models.analytical.support.forward_from_dividend_yield",
        "trellis.models.analytical.support.discount_factor_from_zero_rate",
        "trellis.models.analytical.support.discounted_value",
        "trellis.models.black.black76_call",
        "trellis.models.black.black76_put",
        "trellis.models.analytical.support.probability.standard_normal_cdf",
        "trellis.models.analytical.support.probability.bivariate_standard_normal_cdf",
        "trellis.models.calibration.solve_request.ObjectiveBundle",
        "trellis.models.calibration.solve_request.SolveBounds",
        "trellis.models.calibration.solve_request.SolveRequest",
        "trellis.models.calibration.solve_request.execute_solve_request",
    } <= set(resolved.primitive_refs)


def test_resolve_backend_binding_spec_uses_cliquet_primitive_composition():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)

    assert analytical is not None

    resolved = resolve_backend_binding_spec(
        analytical,
        product_ir=ProductIR(
            instrument="cliquet_option",
            payoff_family="cliquet_option",
            payoff_traits=("vanilla_option",),
            exercise_style="european",
            state_dependence="path_dependent",
            schedule_dependence=True,
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert resolved.pricing_kernel_refs == (
        "trellis.models.black.black76_call",
        "trellis.models.black.black76_put",
        "trellis.models.analytical.support.expectations.gauss_hermite_product_expectation",
    )
    assert "trellis.models.observation_returns.ObservationReturnContract" in resolved.primitive_refs
    assert "trellis.models.observation_returns.bounded_observation_return_sum" in resolved.primitive_refs


def test_resolve_backend_binding_spec_composes_variance_swap_monte_carlo():
    catalog = load_backend_binding_catalog()
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    product_ir = ProductIR(
        instrument="variance_swap",
        payoff_family="variance_swap",
        payoff_traits=(
            "discounting",
            "path_dependent",
            "vol_surface_dependence",
        ),
        exercise_style="none",
        state_dependence="path_dependent",
        schedule_dependence=False,
        model_family="generic",
    )

    assert monte_carlo is not None

    resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)

    assert resolved.helper_refs == ()
    assert resolved.pricing_kernel_refs == ()
    assert {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.models.monte_carlo.path_statistics.SquaredLogReturnContract",
        "trellis.models.monte_carlo.path_statistics.annualized_squared_log_return_sum",
        "trellis.models.monte_carlo.path_statistics.build_squared_log_return_reducer",
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.path_state.MonteCarloPathRequirement",
        "trellis.models.monte_carlo.path_state.StateAwarePayoff",
        "trellis.core.differentiable.get_numpy",
    } <= set(resolved.primitive_refs)


def test_resolve_backend_binding_spec_composes_fixed_lookback_monte_carlo():
    catalog = load_backend_binding_catalog()
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    product_ir = ProductIR(
        instrument="lookback_option",
        payoff_family="lookback_option",
        payoff_traits=(
            "discounting",
            "fixed_strike",
            "lookback",
            "path_dependent",
            "continuous_monitoring",
            "vol_surface_dependence",
        ),
        exercise_style="european",
        state_dependence="path_dependent",
        schedule_dependence=False,
        model_family="equity_diffusion",
    )

    assert monte_carlo is not None

    resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)

    assert resolved.helper_refs == ()
    assert resolved.pricing_kernel_refs == ()
    assert {
        "trellis.models.resolution.single_state_diffusion.resolve_scalar_diffusion_market_inputs",
        "trellis.models.analytical.support.normalized_option_type",
        "trellis.models.monte_carlo.transition_state.ConditionalBridgeExtremumContract",
        "trellis.models.monte_carlo.transition_state.build_conditional_bridge_extremum_reducer",
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.path_state.MonteCarloPathRequirement",
        "trellis.models.monte_carlo.path_state.StateAwarePayoff",
        "trellis.core.differentiable.get_numpy",
    } <= set(resolved.primitive_refs)

    unsupported_contracts = (
        ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=("lookback", "path_dependent"),
            exercise_style="european",
            state_dependence="path_dependent",
            model_family="equity_diffusion",
        ),
        ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=(
                "lookback",
                "path_dependent",
                "fixed_strike",
                "continuous_monitoring",
            ),
            exercise_style="european",
            state_dependence="path_dependent",
            model_family="stochastic_volatility",
        ),
        ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=(
                "lookback",
                "path_dependent",
                "floating_strike",
                "continuous_monitoring",
            ),
            exercise_style="european",
            state_dependence="path_dependent",
            model_family="equity_diffusion",
        ),
        ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=(
                "lookback",
                "path_dependent",
                "fixed_strike",
                "discrete_monitoring",
            ),
            exercise_style="european",
            state_dependence="path_dependent",
            model_family="equity_diffusion",
        ),
    )
    for product_ir in unsupported_contracts:
        unsupported = resolve_backend_binding_spec(
            monte_carlo,
            product_ir=product_ir,
        )
        assert not any(
            ref.startswith("trellis.models.monte_carlo.transition_state.")
            for ref in unsupported.primitive_refs
        )
        assert not any(
            ref.startswith("trellis.models.lookback_option.")
            for ref in unsupported.primitive_refs
        )


def test_resolve_backend_binding_spec_uses_merton_jump_diffusion_helpers():
    catalog = load_backend_binding_catalog()
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    transform = find_backend_binding_by_route_id("transform_fft", catalog)
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        payoff_traits=("jump_diffusion",),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="jump_diffusion",
    )

    assert monte_carlo is not None
    assert transform is not None

    mc_resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)
    transform_resolved = resolve_backend_binding_spec(transform, product_ir=product_ir)

    assert mc_resolved.helper_refs == (
        "trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_monte_carlo",
    )
    assert transform_resolved.helper_refs == (
        "trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_transform",
    )


def test_resolve_backend_binding_spec_uses_sabr_forward_option_helpers():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("sabr_hagan_analytical", catalog)
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        payoff_traits=("sabr",),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="sabr",
    )

    assert analytical is not None
    assert monte_carlo is not None

    analytical_resolved = resolve_backend_binding_spec(analytical, product_ir=product_ir)
    mc_resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)

    assert analytical_resolved.helper_refs == (
        "trellis.models.sabr_option.price_sabr_forward_option_hagan",
    )
    assert mc_resolved.helper_refs == (
        "trellis.models.sabr_option.price_sabr_forward_option_monte_carlo",
    )


@pytest.mark.parametrize(
    ("model_family", "transform_helper", "mc_helper", "reference_helper"),
    [
        (
            "variance_gamma",
            "trellis.models.levy_option.price_variance_gamma_option_transform",
            "trellis.models.levy_option.price_variance_gamma_option_monte_carlo",
            "trellis.models.levy_option.price_variance_gamma_option_reference",
        ),
        (
            "cgmy",
            "trellis.models.levy_option.price_cgmy_option_transform",
            "trellis.models.levy_option.price_cgmy_option_monte_carlo",
            "trellis.models.levy_option.price_cgmy_option_reference",
        ),
        (
            "kou",
            "trellis.models.levy_option.price_kou_option_transform",
            "trellis.models.levy_option.price_kou_option_monte_carlo",
            "trellis.models.levy_option.price_kou_option_reference",
        ),
        (
            "bates",
            "trellis.models.bates_option.price_bates_option_transform",
            "trellis.models.bates_option.price_bates_option_monte_carlo",
            None,
        ),
    ],
)
def test_resolve_backend_binding_spec_uses_levy_option_helpers(
    model_family,
    transform_helper,
    mc_helper,
    reference_helper,
):
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("levy_reference_analytical", catalog)
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    transform = find_backend_binding_by_route_id("transform_fft", catalog)
    product_ir = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        payoff_traits=(model_family,),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family=model_family,
    )

    assert analytical is not None
    assert monte_carlo is not None
    assert transform is not None

    analytical_resolved = resolve_backend_binding_spec(analytical, product_ir=product_ir)
    mc_resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)
    transform_resolved = resolve_backend_binding_spec(transform, product_ir=product_ir)

    assert analytical_resolved.helper_refs == (
        () if reference_helper is None else (reference_helper,)
    )
    assert mc_resolved.helper_refs == (mc_helper,)
    assert transform_resolved.helper_refs == (transform_helper,)


def test_resolve_backend_binding_spec_keeps_generic_multi_asset_baskets_off_two_asset_exact_helpers():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)
    transform = find_backend_binding_by_route_id("transform_fft", catalog)

    product_ir = ProductIR(
        instrument="basket_option",
        payoff_family="basket_option",
        payoff_traits=("vanilla_option",),
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )

    assert analytical is not None
    assert transform is not None

    analytical_resolved = resolve_backend_binding_spec(analytical, product_ir=product_ir)
    transform_resolved = resolve_backend_binding_spec(transform, product_ir=product_ir)

    assert analytical_resolved.helper_refs != (
        "trellis.models.basket_option.price_basket_option_analytical",
    )
    assert transform_resolved.helper_refs != (
        "trellis.models.basket_option.price_basket_option_transform_proxy",
    )


def test_resolve_backend_binding_spec_uses_rate_cap_floor_exact_helpers():
    catalog = load_backend_binding_catalog()
    analytical = find_backend_binding_by_route_id("analytical_black76", catalog)
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)

    product_ir = ProductIR(
        instrument="cap",
        payoff_family="period_rate_option_strip",
        exercise_style="none",
        schedule_dependence=True,
        state_dependence="schedule_dependent",
        model_family="interest_rate",
    )

    assert analytical is not None
    assert monte_carlo is not None

    analytical_resolved = resolve_backend_binding_spec(analytical, product_ir=product_ir)
    monte_carlo_resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)

    assert analytical_resolved.helper_refs == (
        "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",
    )
    assert analytical_resolved.binding_id == (
        "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical"
    )
    assert monte_carlo_resolved.helper_refs == (
        "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo",
    )
    assert monte_carlo_resolved.binding_id == (
        "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo"
    )


def test_nth_to_default_bindings_have_no_schedule_builder_surface():
    catalog = load_backend_binding_catalog()
    product_ir = ProductIR(
        instrument="nth_to_default",
        payoff_family="nth_to_default",
        exercise_style="none",
        schedule_dependence=True,
        state_dependence="schedule_state",
    )

    binding = find_backend_binding_by_route_id("credit_basket_nth_to_default", catalog)
    assert binding is not None
    for method in ("analytical", "monte_carlo"):
        resolved = resolve_backend_binding_spec(
            binding,
            product_ir=product_ir,
            method=method,
        )

        assert resolved.binding_id == "trellis.instruments.nth_to_default.price_nth_to_default_basket"
        assert resolved.helper_refs == ("trellis.instruments.nth_to_default.price_nth_to_default_basket",)
        assert resolved.schedule_builder_refs == ()


def test_resolve_backend_binding_spec_uses_credit_loss_distribution_exact_helpers():
    catalog = load_backend_binding_catalog()
    copula = find_backend_binding_by_route_id("copula_loss_distribution", catalog)
    monte_carlo = find_backend_binding_by_route_id("monte_carlo_paths", catalog)
    transform = find_backend_binding_by_route_id("transform_fft", catalog)

    product_ir = ProductIR(
        instrument="credit_loss_distribution",
        payoff_family="credit_loss_distribution",
        exercise_style="none",
        state_dependence="terminal_markov",
        model_family="credit_copula",
    )

    assert copula is not None
    assert monte_carlo is not None
    assert transform is not None

    copula_resolved = resolve_backend_binding_spec(copula, product_ir=product_ir)
    monte_carlo_resolved = resolve_backend_binding_spec(monte_carlo, product_ir=product_ir)
    transform_resolved = resolve_backend_binding_spec(transform, product_ir=product_ir)

    assert copula_resolved.helper_refs == (
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_recursive",
    )
    assert monte_carlo_resolved.helper_refs == (
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_monte_carlo",
    )
    assert transform_resolved.helper_refs == (
        "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_transform_proxy",
    )


def test_exercise_monte_carlo_binding_resolves_american_equity_lsm_primitives():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("exercise_monte_carlo", catalog)
    assert binding is not None

    product_ir = ProductIR(
        instrument="american_put",
        payoff_family="vanilla_option",
        exercise_style="american",
        model_family="equity_diffusion",
    )

    resolved = resolve_backend_binding_spec(binding, product_ir=product_ir)

    assert resolved.binding_id == "exercise:exercise:fallback"
    assert resolved.helper_refs == ()
    assert resolved.exact_target_refs == ()
    assert set(resolved.primitive_refs) == {
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.lsm.longstaff_schwartz",
        "trellis.models.monte_carlo.single_state_diffusion.resolve_single_state_monte_carlo_inputs",
        "trellis.models.resolution.single_state_diffusion.terminal_intrinsic_from_resolved",
    }


def test_exercise_lattice_binding_resolves_american_equity_algebra_primitives():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("exercise_lattice", catalog)
    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="american_put",
            payoff_family="vanilla_option",
            exercise_style="american",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert resolved.exact_target_refs == ()
    assert set(resolved.primitive_refs) == {
        "trellis.models.resolution.single_state_diffusion.resolve_single_state_diffusion_inputs",
        "trellis.models.resolution.single_state_diffusion.terminal_intrinsic_from_resolved",
        "trellis.models.trees.algebra.equity_tree",
        "trellis.models.trees.algebra.with_control",
        "trellis.models.trees.algebra.compile_lattice_recipe",
        "trellis.models.trees.algebra.build_lattice",
        "trellis.models.trees.algebra.price_on_lattice",
    }


def test_exercise_lattice_binding_adds_bermudan_schedule_mapping_primitives():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id("exercise_lattice", catalog)
    assert binding is not None

    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=ProductIR(
            instrument="american_put",
            payoff_family="vanilla_option",
            exercise_style="bermudan",
            model_family="equity_diffusion",
        ),
    )

    assert resolved.helper_refs == ()
    assert resolved.exact_target_refs == ()
    assert {
        "trellis.core.date_utils.year_fraction",
        "trellis.models.monte_carlo.event_state.event_step_indices",
    }.issubset(resolved.primitive_refs)


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
