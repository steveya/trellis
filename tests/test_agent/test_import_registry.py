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


def test_list_module_exports_returns_known_symbols():
    exports = list_module_exports("trellis.models.pde.theta_method")
    assert "theta_method_1d" in exports


def test_find_symbol_modules_returns_known_module():
    modules = find_symbol_modules("theta_method_1d")
    assert "trellis.models.pde.theta_method" in modules


def test_find_symbol_modules_returns_garman_kohlhagen_kernel_module():
    modules = find_symbol_modules("garman_kohlhagen_call")
    assert "trellis.models.black" in modules


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


def test_cliquet_monte_carlo_helper_is_visible_to_import_registry():
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
