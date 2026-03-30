"""Tests for the compact API navigation map used by agent prompts and tools."""

from __future__ import annotations

import re
import importlib

from trellis.agent.knowledge.import_registry import module_exists
from trellis.agent.knowledge.api_map import format_api_map_for_prompt, get_api_map


_IMPORT_RE = re.compile(r"^from\s+([A-Za-z0-9_.]+)\s+import\s+(.+?)\s*$")


def test_api_map_contains_expected_core_entries():
    api_map = get_api_map()

    assert api_map["market_state"]["module"] == "trellis.core.market_state"
    assert api_map["payoff"]["module"] == "trellis.core.payoff"
    assert api_map["monte_carlo"]["module"] == "trellis.models.monte_carlo"
    assert api_map["equity_tree"]["module"] == "trellis.models.trees.binomial"
    assert api_map["rate_lattice"]["module"] == "trellis.models.trees.lattice"
    assert "utilities" in api_map

    for section_name in ("market_state", "payoff", "equity_tree", "rate_lattice"):
        assert module_exists(api_map[section_name]["module"])


def test_api_map_key_imports_are_registry_valid():
    api_map = get_api_map()

    for section_name in (
        "equity_tree",
        "rate_lattice",
        "monte_carlo",
        "qmc",
        "pde",
        "fft",
        "copulas",
        "analytical",
        "calibration",
    ):
        section = api_map[section_name]
        _assert_import_statements_valid(section["key_imports"])

    utilities = api_map["utilities"]
    for utility_name in (
        "black76",
        "garman_kohlhagen",
        "schedule",
        "day_count",
        "vol_surface",
        "cashflow_engine",
        "credit_curve",
    ):
        utility = utilities[utility_name]
        _assert_import_statements_valid(utility["imports"])


def test_api_map_formatter_includes_navigation_guidance():
    text = format_api_map_for_prompt(compact=True)

    assert "API Map" in text
    assert "MarketState" in text
    assert "equity_tree" in text
    assert "rate_lattice" in text
    assert "trellis.models.monte_carlo" in text
    assert "inspect_api_map" not in text


def _assert_import_statements_valid(import_statements: list[str]) -> None:
    for statement in import_statements:
        cleaned = statement.split("#", 1)[0].strip()
        match = _IMPORT_RE.match(cleaned)
        assert match is not None, f"Could not parse import statement: {statement!r}"

        module_path, symbols_text = match.groups()
        module = importlib.import_module(module_path)

        symbols = [symbol.strip() for symbol in symbols_text.split(",") if symbol.strip()]
        for symbol in symbols:
            assert hasattr(module, symbol), f"{symbol} is not exported by {module_path}"
