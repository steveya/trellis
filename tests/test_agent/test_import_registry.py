"""Tests for the structured import registry."""

import trellis.agent.knowledge.import_registry as import_registry
from trellis.agent.knowledge.import_registry import (
    find_symbol_modules,
    get_import_registry,
    get_registry_snapshot,
    is_valid_import,
    list_module_exports,
    module_exists,
    resolve_import_candidates,
)


def test_registry_snapshot_contains_known_modules():
    snapshot = get_registry_snapshot()
    assert "trellis.models.black" in snapshot
    assert "trellis.core.market_state" in snapshot


def test_fpml_import_surface_is_visible_to_import_registry():
    module = "trellis.io.fpml"
    symbols = {
        "FPML_5_13_CONFIRMATION",
        "FpMLFieldProvenance",
        "FpMLImportReport",
        "FpMLInspectionLimits",
        "fpml_import_report_summary",
        "inspect_fpml_document",
        "normalize_fpml_document",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])


def test_list_module_exports_returns_known_symbols():
    exports = list_module_exports("trellis.models.pde.theta_method")
    assert "theta_method_1d" in exports


def test_registry_includes_local_exported_values_without_package_reexports():
    algebra_exports = set(list_module_exports("trellis.models.trees.algebra"))

    assert {
        "BINOMIAL_1F_TOPOLOGY",
        "TERM_STRUCTURE_TARGET",
        "UNIFORM_ADDITIVE_MESH",
    } <= algebra_exports
    assert "price_himalaya_option_monte_carlo" not in list_module_exports(
        "trellis.models.monte_carlo"
    )

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert {
        "BINOMIAL_1F_TOPOLOGY",
        "TERM_STRUCTURE_TARGET",
        "UNIFORM_ADDITIVE_MESH",
        "build_lattice",
        "price_on_lattice",
        "value_on_lattice",
    } <= set(static_registry["trellis.models.trees.algebra"])


def test_lattice_rollback_observation_primitives_are_visible_to_import_registry():
    module = "trellis.models.trees.lattice"
    symbols = {
        "LatticeRollbackObservation",
        "LatticeRollbackResult",
        "lattice_backward_induction_result",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])


def test_find_symbol_modules_returns_known_module():
    modules = find_symbol_modules("theta_method_1d")
    assert "trellis.models.pde.theta_method" in modules


def test_find_symbol_modules_returns_garman_kohlhagen_kernel_module():
    modules = find_symbol_modules("garman_kohlhagen_call")
    assert "trellis.models.black" in modules


def test_analytical_discount_factor_is_visible_to_import_registry():
    module = "trellis.models.analytical.support"
    symbol = "discount_factor_from_zero_rate"

    assert symbol in list_module_exports(module)
    assert module in find_symbol_modules(symbol)
    assert is_valid_import(module, symbol)

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbol in static_registry[module]


def test_observation_return_primitives_are_visible_to_import_registry():
    module = "trellis.models.observation_returns"
    symbols = {
        "ObservationReturnContract",
        "bounded_observation_return_sum",
        "build_observation_return_reducer",
        "observation_return_payoff",
        "simple_observation_returns",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    assert "ObservationReturnContract" in registry_text

    expectation_module = "trellis.models.analytical.support.expectations"
    expectation_symbol = "gauss_hermite_product_expectation"
    assert expectation_symbol in list_module_exports(expectation_module)
    assert expectation_module in find_symbol_modules(expectation_symbol)

    process_module = "trellis.models.processes.gbm"
    process_symbol = "PiecewiseConstantGBM"
    assert process_symbol in list_module_exports(process_module)
    assert process_module in find_symbol_modules(process_symbol)
    assert is_valid_import(process_module, process_symbol)


def test_weighted_observation_primitives_are_visible_to_import_registry():
    module = "trellis.models.observation_aggregation"
    symbols = {
        "WeightedObservationContract",
        "build_weighted_observation_reducer",
        "weighted_observation_payoff",
        "weighted_observation_sum",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_path_statistic_primitives_are_visible_to_import_registry():
    module = "trellis.models.monte_carlo.path_statistics"
    symbols = {
        "RunningExtremumContract",
        "SquaredLogReturnContract",
        "annualized_squared_log_return_sum",
        "build_running_extremum_reducer",
        "build_squared_log_return_reducer",
        "discrete_path_extremum",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])
    assert "PathReducer" in static_registry[
        "trellis.models.monte_carlo.path_state"
    ]
    resolution_module = "trellis.models.resolution.single_state_diffusion"
    assert "resolve_scalar_diffusion_market_inputs" in set(
        list_module_exports(resolution_module)
    )
    assert is_valid_import(
        resolution_module,
        "resolve_scalar_diffusion_market_inputs",
    )
    assert "resolve_scalar_diffusion_market_inputs" in static_registry[
        resolution_module
    ]


def test_transition_state_primitives_are_visible_to_import_registry():
    module = "trellis.models.monte_carlo.transition_state"
    symbols = {
        "ConditionalBridgeExtremumContract",
        "MonteCarloRandomInputs",
        "ScalarConditionalBridgeProcess",
        "ScalarTransitionObservation",
        "ScalarTransitionReducer",
        "build_conditional_bridge_extremum_reducer",
        "coerce_transition_uniforms",
        "conditional_log_bridge_extremum",
        "replay_scalar_transition_reducers",
        "resolve_scalar_bridge_parameters",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])
    assert "sobol_transition_inputs" in static_registry[
        "trellis.models.monte_carlo.variance_reduction"
    ]


def test_weighted_lognormal_moment_primitives_are_visible_to_import_registry():
    module = "trellis.models.analytical.support.lognormal_moments"
    symbols = {
        "LognormalMomentMatch",
        "WeightedLognormalSumContract",
        "WeightedLognormalSumMoments",
        "match_lognormal_moments",
        "single_factor_lognormal_sum_contract",
        "weighted_lognormal_sum_moments",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_gaussian_probability_primitives_are_visible_to_import_registry():
    module = "trellis.models.analytical.support.probability"
    symbols = {
        "bivariate_standard_normal_cdf",
        "standard_normal_cdf",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    for symbol in symbols:
        assert module in find_symbol_modules(symbol)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])

    scalar_root_module = "trellis.models.calibration.solve_request"
    scalar_root_symbols = {
        "ObjectiveBundle",
        "SolveBounds",
        "SolveRequest",
        "execute_solve_request",
    }
    assert scalar_root_symbols <= set(list_module_exports(scalar_root_module))
    for symbol in scalar_root_symbols:
        assert is_valid_import(scalar_root_module, symbol)
    assert scalar_root_symbols <= set(static_registry[scalar_root_module])


def test_quoted_observable_helpers_are_visible_to_import_registry():
    module = "trellis.models.quoted_observable"

    assert module_exists(module)
    assert "price_curve_quote_spread_analytical" in list_module_exports(module)
    assert "price_surface_quote_spread_analytical" in list_module_exports(module)
    assert find_symbol_modules("price_curve_quote_spread_analytical") == (module,)
    assert find_symbol_modules("price_surface_quote_spread_analytical") == (module,)
    assert is_valid_import(module, "QuotedObservableSpreadResult")

    registry_text = get_import_registry()
    assert "from trellis.models.quoted_observable import" in registry_text
    assert "price_curve_quote_spread_analytical" in registry_text


def test_retained_cliquet_monte_carlo_reference_is_visible_to_import_registry():
    module = "trellis.models.monte_carlo.event_aware"
    symbol = "price_equity_cliquet_option_monte_carlo"

    assert module_exists(module)
    assert symbol in list_module_exports(module)
    assert find_symbol_modules(symbol) == (module,)
    assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    assert symbol in registry_text


def test_american_lsm_helper_is_visible_to_import_registry():
    module = "trellis.models.equity_option_monte_carlo"
    symbol = "price_american_equity_option_lsm_monte_carlo"

    assert module_exists(module)
    assert symbol in list_module_exports(module)
    assert find_symbol_modules(symbol) == (module,)
    assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    assert symbol in registry_text


def test_counterparty_cva_helpers_are_visible_to_import_registry():
    module = "trellis.analytics.counterparty"
    symbols = {
        "price_interest_rate_swap_cva_analytical_approx",
        "price_interest_rate_swap_cva_monte_carlo",
        "price_interest_rate_swap_independent_cva",
        "price_interest_rate_swap_wrong_way_cva",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_credit_index_option_helpers_are_visible_to_import_registry():
    module = "trellis.models.credit_index_option"
    symbols = {
        "CreditIndexOptionSpec",
        "price_credit_index_option_black_on_spread",
        "price_credit_index_option_monte_carlo",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_local_vol_option_helpers_are_visible_to_import_registry():
    module = "trellis.models.local_vol_option"
    symbols = {
        "LocalVolPDEResult",
        "LocalVolVanillaOptionSpec",
        "price_local_vol_option_monte_carlo",
        "price_local_vol_option_pde",
        "price_local_vol_option_pde_result",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_variance_swap_monte_carlo_helper_is_visible_to_import_registry():
    module = "trellis.models.variance_swap"
    symbols = {
        "price_equity_variance_swap_monte_carlo",
        "price_equity_variance_swap_monte_carlo_result",
        "equity_variance_swap_outputs_monte_carlo",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_merton_jump_diffusion_helpers_are_visible_to_import_registry():
    module = "trellis.models.merton_jump_diffusion_option"
    symbols = {
        "price_merton_jump_diffusion_option_monte_carlo",
        "price_merton_jump_diffusion_option_monte_carlo_result",
        "price_merton_jump_diffusion_option_poisson_series",
        "price_merton_jump_diffusion_option_transform",
        "resolve_merton_jump_diffusion_option_inputs",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_sabr_forward_option_helpers_are_visible_to_import_registry():
    module = "trellis.models.sabr_option"
    symbols = {
        "price_sabr_forward_option_hagan",
        "price_sabr_forward_option_monte_carlo",
        "price_sabr_forward_option_monte_carlo_result",
        "resolve_sabr_forward_option_inputs",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_levy_option_helpers_are_visible_to_import_registry():
    module = "trellis.models.levy_option"
    symbols = {
        "price_cgmy_option_monte_carlo",
        "price_cgmy_option_monte_carlo_result",
        "price_cgmy_option_reference",
        "price_cgmy_option_transform",
        "price_kou_option_monte_carlo",
        "price_kou_option_monte_carlo_result",
        "price_kou_option_reference",
        "price_kou_option_transform",
        "price_variance_gamma_option_monte_carlo",
        "price_variance_gamma_option_monte_carlo_result",
        "price_variance_gamma_option_reference",
        "price_variance_gamma_option_transform",
        "resolve_levy_option_inputs",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_bates_option_helpers_are_visible_to_import_registry():
    module = "trellis.models.bates_option"
    symbols = {
        "bates_log_ratio_char_fn",
        "bates_log_spot_char_fn",
        "price_bates_option_monte_carlo",
        "price_bates_option_monte_carlo_result",
        "price_bates_option_transform",
        "price_bates_option_transform_result",
        "resolve_bates_option_inputs",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_short_rate_bond_helpers_are_visible_to_import_registry():
    module = "trellis.models.short_rate_bond"
    symbols = {
        "price_cir_zero_coupon_bond_analytical",
        "price_short_rate_zero_coupon_bond_analytical",
        "price_short_rate_zero_coupon_bond_tree",
        "price_vasicek_zero_coupon_bond_analytical",
        "resolve_short_rate_bond_inputs",
    }

    assert module_exists(module)
    exports = set(list_module_exports(module))
    assert symbols <= exports
    for symbol in symbols:
        assert find_symbol_modules(symbol) == (module,)
        assert is_valid_import(module, symbol)

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    for symbol in symbols:
        assert symbol in registry_text


def test_short_rate_lattice_binding_is_visible_to_import_registry():
    module = "trellis.models.short_rate_lattice"
    symbols = {
        "ResolvedShortRateLatticeInputs",
        "resolve_short_rate_lattice_inputs",
    }

    assert module_exists(module)
    assert symbols <= set(list_module_exports(module))
    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])


def test_cev_helpers_are_visible_to_import_registry():
    pde_module = "trellis.models.equity_option_pde"
    tree_module = "trellis.models.equity_option_tree"

    assert module_exists(pde_module)
    assert module_exists(tree_module)
    assert "price_cev_option_pde" in list_module_exports(pde_module)
    assert "price_cev_option_tree" in list_module_exports(tree_module)
    assert find_symbol_modules("price_cev_option_pde") == (pde_module,)
    assert find_symbol_modules("price_cev_option_tree") == (tree_module,)
    assert is_valid_import(pde_module, "price_cev_option_pde")
    assert is_valid_import(tree_module, "price_cev_option_tree")

    registry_text = get_import_registry()
    assert f"from {pde_module} import" in registry_text
    assert f"from {tree_module} import" in registry_text
    assert "price_cev_option_pde" in registry_text
    assert "price_cev_option_tree" in registry_text


def test_digital_pde_and_asian_helpers_are_visible_to_import_registry():
    pde_module = "trellis.models.equity_option_pde"
    transform_module = "trellis.models.equity_option_transforms"
    asian_module = "trellis.models.asian_option"

    assert module_exists(pde_module)
    assert module_exists(transform_module)
    assert module_exists(asian_module)
    assert "price_equity_digital_option_pde" in list_module_exports(pde_module)
    assert "price_equity_digital_option_transform" in list_module_exports(transform_module)
    assert "price_arithmetic_asian_option_analytical" in list_module_exports(asian_module)
    assert "price_arithmetic_asian_option_monte_carlo" in list_module_exports(asian_module)
    assert find_symbol_modules("price_equity_digital_option_pde") == (pde_module,)
    assert find_symbol_modules("price_equity_digital_option_transform") == (transform_module,)
    assert find_symbol_modules("price_arithmetic_asian_option_analytical") == (asian_module,)
    assert find_symbol_modules("price_arithmetic_asian_option_monte_carlo") == (asian_module,)
    assert is_valid_import(pde_module, "price_equity_digital_option_pde")
    assert is_valid_import(transform_module, "price_equity_digital_option_transform")
    assert is_valid_import(asian_module, "price_arithmetic_asian_option_analytical")
    assert is_valid_import(asian_module, "price_arithmetic_asian_option_monte_carlo")

    registry_text = get_import_registry()
    assert f"from {pde_module} import" in registry_text
    assert f"from {transform_module} import" in registry_text
    assert f"from {asian_module} import" in registry_text
    assert "price_equity_digital_option_pde" in registry_text
    assert "price_equity_digital_option_transform" in registry_text
    assert "price_arithmetic_asian_option_analytical" in registry_text
    assert "price_arithmetic_asian_option_monte_carlo" in registry_text


def test_single_barrier_helpers_are_visible_to_import_registry():
    module = "trellis.models.single_barrier_option"

    assert module_exists(module)
    assert "price_single_barrier_option_pde_result" in list_module_exports(module)
    assert "price_single_barrier_option_monte_carlo_result" in list_module_exports(module)
    assert find_symbol_modules("price_single_barrier_option_pde_result") == (module,)
    assert find_symbol_modules("price_single_barrier_option_monte_carlo_result") == (module,)
    assert is_valid_import(module, "SingleBarrierPDEConfig")
    assert is_valid_import(module, "SingleBarrierMonteCarloConfig")

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    assert "price_single_barrier_option_pde_result" in registry_text


def test_fx_barrier_helpers_are_visible_to_import_registry():
    module = "trellis.models.fx_barrier_option"

    assert module_exists(module)
    assert "price_fx_barrier_option_analytical" in list_module_exports(module)
    assert "price_fx_barrier_option_monte_carlo" in list_module_exports(module)
    assert find_symbol_modules("price_fx_barrier_option_analytical") == (module,)
    assert find_symbol_modules("price_fx_barrier_option_monte_carlo") == (module,)
    assert is_valid_import(module, "FXBarrierOptionSpec")

    registry_text = get_import_registry()
    assert f"from {module} import" in registry_text
    assert "price_fx_barrier_option_analytical" in registry_text


def test_fx_barrier_path_state_primitives_are_in_static_import_registry():
    module = "trellis.models.monte_carlo.path_state"
    symbols = {
        "BarrierMonitor",
        "MonteCarloPathRequirement",
        "StateAwarePayoff",
    }

    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )

    assert symbols <= set(static_registry[module])


def test_fx_vanilla_composition_primitives_are_in_static_import_registry():
    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )

    assert "resolve_fx_vanilla_inputs" in static_registry[
        "trellis.models.fx_vanilla"
    ]
    assert "garman_kohlhagen_price_raw" in static_registry[
        "trellis.models.analytical.fx"
    ]
    assert "terminal_value_payoff" in static_registry[
        "trellis.models.monte_carlo.path_state"
    ]


def test_quanto_composition_primitives_are_in_static_import_registry():
    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )

    assert "resolve_quanto_inputs" in static_registry[
        "trellis.models.resolution.quanto"
    ]
    assert {
        "discounted_value",
        "implied_zero_rate",
        "normalized_option_type",
        "quanto_adjusted_forward",
        "terminal_intrinsic",
    } <= set(static_registry["trellis.models.analytical.support"])
    assert "CorrelatedGBM" in static_registry[
        "trellis.models.processes.correlated_gbm"
    ]
    assert "MonteCarloEngine" in static_registry[
        "trellis.models.monte_carlo.engine"
    ]
    assert "sobol_normals" in static_registry[
        "trellis.models.monte_carlo.variance_reduction"
    ]


def test_digital_composition_support_reexports_are_discoverable():
    module = "trellis.models.analytical.support"
    symbols = {
        "asset_or_nothing_intrinsic",
        "cash_or_nothing_intrinsic",
        "discounted_value",
        "forward_from_dividend_yield",
    }

    assert symbols <= set(list_module_exports(module))
    static_registry = import_registry._parse_static_registry(
        import_registry._STATIC_REGISTRY
    )
    assert symbols <= set(static_registry[module])


def test_resolve_import_candidates_handles_known_and_unknown_symbols():
    candidates = resolve_import_candidates(["theta_method_1d", "definitely_not_real"])
    assert "trellis.models.pde.theta_method" in candidates["theta_method_1d"]
    assert candidates["definitely_not_real"] == ()


def test_is_valid_import_checks_symbol_membership():
    assert is_valid_import("trellis.models.black", "black76_call")
    assert not is_valid_import("trellis.models.black", "not_real")


def test_module_exists_rejects_unknown_modules():
    assert module_exists("trellis.models.black")
    assert not module_exists("trellis.models.not_a_real_module")


def test_formatted_registry_mentions_known_import():
    registry_text = get_import_registry()
    assert "from trellis.models.black import" in registry_text


def test_static_registry_fallback_covers_route_minimization_modules(monkeypatch):
    import_registry.reset_registry_cache()
    monkeypatch.setattr(
        import_registry,
        "_build_registry_data_from_introspection",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    snapshot = import_registry.get_registry_snapshot()

    assert "trellis.models.equity_option_transforms" in snapshot
    assert "price_vanilla_equity_option_transform" in snapshot["trellis.models.equity_option_transforms"]
    assert "trellis.models.quoted_observable" in snapshot
    assert "price_curve_quote_spread_analytical" in snapshot["trellis.models.quoted_observable"]
    assert "trellis.models.credit_basket_copula" in snapshot
    assert "price_credit_basket_tranche" in snapshot["trellis.models.credit_basket_copula"]
    assert "trellis.models.transforms.single_state_diffusion" in snapshot
    assert "resolve_single_state_diffusion_inputs" in snapshot["trellis.models.transforms.single_state_diffusion"]
    assert "trellis.models.transforms.heston" in snapshot
    assert "price_heston_option_transform" in snapshot["trellis.models.transforms.heston"]

    import_registry.reset_registry_cache()
