"""Tests for repo-aware agent tool handlers."""

from __future__ import annotations

import json

from trellis.agent.executor import _handle_tool_call


def test_find_symbol_tool_returns_matches():
    payload = json.loads(_handle_tool_call("find_symbol", {"symbol": "theta_method_1d"}))
    assert any(match["module"] == "trellis.models.pde.theta_method" for match in payload)


def test_list_exports_tool_returns_public_exports():
    payload = json.loads(_handle_tool_call(
        "list_exports",
        {"module_path": "trellis.models.pde.theta_method"},
    ))
    names = {item["name"] for item in payload}
    assert "theta_method_1d" in names


def test_resolve_import_candidates_tool_returns_mapping():
    payload = json.loads(_handle_tool_call(
        "resolve_import_candidates",
        {"symbols": ["theta_method_1d", "MonteCarloEngine"]},
    ))
    assert "trellis.models.pde.theta_method" in payload["theta_method_1d"]
    assert payload["MonteCarloEngine"]


def test_lookup_primitive_route_tool_returns_route_plan():
    payload = json.loads(_handle_tool_call(
        "lookup_primitive_route",
        {
            "description": "European equity call option",
            "instrument_type": "european_option",
            "preferred_method": "analytical",
        },
    ))

    assert payload["route"] == "analytical_black76"
    assert "trellis.models.black.black76_call" in payload["primitives"]


def test_select_invariant_pack_tool_returns_checks():
    payload = json.loads(_handle_tool_call(
        "select_invariant_pack",
        {
            "instrument_type": "callable_bond",
            "method": "rate_tree",
        },
    ))

    assert "check_non_negativity" in payload["checks"]
    assert "check_bounded_by_reference" in payload["checks"]


def test_build_comparison_harness_tool_returns_targets():
    payload = json.loads(_handle_tool_call(
        "build_comparison_harness",
        {
            "task": {
                "construct": ["lattice", "pde", "monte_carlo", "transforms"],
                "cross_validate": {
                    "internal": ["crr_tree", "bs_pde", "mc_exact", "fft", "cos"],
                    "analytical": "black_scholes",
                },
            },
        },
    ))

    assert payload["reference_target"] == "black_scholes"
    assert any(target["target_id"] == "fft" for target in payload["targets"])


def test_search_repo_tool_finds_source_match():
    payload = json.loads(_handle_tool_call(
        "search_repo",
        {"pattern": "theta_method_1d", "limit": 5},
    ))
    assert payload
    assert any("/trellis/models/pde/" in result["path"] for result in payload)


def test_search_tests_tool_finds_test_match():
    payload = json.loads(_handle_tool_call(
        "search_tests",
        {"pattern": "European payer swaption", "limit": 5},
    ))
    assert payload
    assert any("test_build_loop.py" in result["path"] for result in payload)


def test_search_lessons_tool_finds_lesson_match():
    payload = json.loads(_handle_tool_call(
        "search_lessons",
        {"pattern": "monte", "limit": 5},
    ))
    assert payload
    assert any(result["path"].endswith(".yaml") for result in payload)
