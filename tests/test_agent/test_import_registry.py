"""Tests for the structured import registry."""

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
