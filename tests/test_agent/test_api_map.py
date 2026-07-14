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
    assert (
        api_map["observation_return_composition"]["module"]
        == "trellis.models.observation_returns"
    )
    assert (
        api_map["scheduled_observation_composition"]["module"]
        == "trellis.models.observation_aggregation"
    )
    assert (
        api_map["weighted_lognormal_sum_composition"]["module"]
        == "trellis.models.analytical.support.lognormal_moments"
    )
    assert (
        api_map["quanto_option_composition"]["module"]
        == "trellis.models.resolution.quanto"
    )
    assert (
        api_map["digital_option_composition"]["module"]
        == "trellis.models.resolution.single_state_diffusion"
    )
    assert (
        api_map["rate_monte_carlo_composition"]["module"]
        == "trellis.models.monte_carlo.simulation_substrate"
    )
    assert api_map["equity_tree"]["module"] == "trellis.models.trees.algebra"
    assert api_map["rate_lattice"]["module"] == "trellis.models.trees.lattice"
    assert "utilities" in api_map

    for section_name in (
        "market_state",
        "payoff",
        "equity_tree",
        "rate_lattice",
        "rate_monte_carlo_composition",
    ):
        assert module_exists(api_map[section_name]["module"])


def test_api_map_key_imports_are_registry_valid():
    api_map = get_api_map()

    for section_name in (
        "equity_tree",
        "rate_lattice",
        "monte_carlo",
        "scheduled_observation_composition",
        "weighted_lognormal_sum_composition",
        "observation_return_composition",
        "digital_option_composition",
        "quanto_option_composition",
        "rate_monte_carlo_composition",
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
        "rate_style_swaption",
        "jamshidian_zcb_option",
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


def test_api_map_formatter_includes_all_canonical_utilities():
    text = format_api_map_for_prompt(compact=True)

    assert "rate_style_swaption" in text
    assert "jamshidian_zcb_option" in text
    assert "credit_curve" in text


def test_monte_carlo_api_map_prioritizes_american_lsm_primitives():
    section = get_api_map()["monte_carlo"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    assert "resolve_single_state_monte_carlo_inputs" in text
    assert "terminal_intrinsic_from_resolved" in text
    assert "longstaff_schwartz" in text
    assert "price_american_equity_option_lsm_monte_carlo" not in text


def test_monte_carlo_api_map_prioritizes_terminal_claim_composition():
    section = get_api_map()["monte_carlo"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    assert "price_single_state_terminal_claim_monte_carlo_result" in text
    assert "terminal_intrinsic_from_resolved" in text
    assert "price_vanilla_equity_option_monte_carlo" not in text


def test_api_map_exposes_product_neutral_observation_return_composition():
    section = get_api_map()["observation_return_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "ObservationReturnContract",
        "bounded_observation_return_sum",
        "observation_return_payoff",
        "gauss_hermite_product_expectation",
        "MonteCarloEngine",
        "PiecewiseConstantGBM",
    ):
        assert symbol in text
    assert "cliquet" not in text.lower()


def test_api_map_exposes_product_neutral_scheduled_observation_composition():
    section = get_api_map()["scheduled_observation_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "WeightedObservationContract",
        "weighted_observation_sum",
        "weighted_observation_payoff",
        "GBM",
        "MonteCarloEngine",
    ):
        assert symbol in text
    assert "asian" not in text.lower()


def test_api_map_exposes_product_neutral_weighted_lognormal_sum_composition():
    section = get_api_map()["weighted_lognormal_sum_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "WeightedLognormalSumContract",
        "single_factor_lognormal_sum_contract",
        "weighted_lognormal_sum_moments",
        "match_lognormal_moments",
        "black76_call",
        "black76_put",
    ):
        assert symbol in text
    assert "asian" not in text.lower()


def test_api_map_routes_arithmetic_asian_agents_to_primitive_composition():
    section = get_api_map()["arithmetic_asian_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "resolve_single_state_diffusion_inputs",
        "WeightedObservationContract",
        "resolve_uniform_grid_steps",
        "weighted_observation_payoff",
        "single_factor_lognormal_sum_contract",
        "weighted_lognormal_sum_moments",
        "match_lognormal_moments",
        "black76_call",
        "black76_put",
        "GBM",
        "MonteCarloEngine",
        "StateAwarePayoff",
    ):
        assert symbol in text
    assert "price_arithmetic_asian_option" not in text
    assert "arithmetic-average" in text
    assert "geometric" in text
    assert "floating-strike" in text


def test_api_map_prioritizes_fx_vanilla_primitive_composition():
    api_map = get_api_map()
    monte_carlo_text = "\n".join(
        (*api_map["monte_carlo"]["key_imports"], *api_map["monte_carlo"]["notes"])
    )
    analytical_text = "\n".join(
        (*api_map["analytical"]["key_imports"], *api_map["analytical"]["notes"])
    )

    for symbol in (
        "resolve_fx_vanilla_inputs",
        "GBM",
        "MonteCarloEngine",
        "terminal_value_payoff",
        "terminal_intrinsic",
    ):
        assert symbol in monte_carlo_text
    assert "price_fx_vanilla_monte_carlo" not in monte_carlo_text
    assert "resolve_fx_vanilla_inputs" in analytical_text
    assert "garman_kohlhagen_price_raw" in analytical_text
    assert "price_fx_vanilla_analytical" not in analytical_text


def test_api_map_prioritizes_quanto_primitive_composition():
    section = get_api_map()["quanto_option_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "resolve_quanto_inputs",
        "quanto_adjusted_forward",
        "black76_call",
        "black76_put",
        "CorrelatedGBM",
        "MonteCarloEngine",
        "terminal_value_payoff",
        "sobol_normals",
    ):
        assert symbol in text
    assert "price_quanto_option_analytical" not in text
    assert "price_quanto_option_monte_carlo" not in text


def test_api_map_prioritizes_digital_basis_composition():
    section = get_api_map()["digital_option_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "resolve_single_state_diffusion_inputs",
        "forward_from_dividend_yield",
        "discounted_value",
        "cash_or_nothing_intrinsic",
        "asset_or_nothing_intrinsic",
        "black76_cash_or_nothing_call",
        "black76_cash_or_nothing_put",
        "black76_asset_or_nothing_call",
        "black76_asset_or_nothing_put",
    ):
        assert symbol in text
    assert "price_equity_digital_option_analytical" not in text


def test_equity_tree_api_map_prioritizes_lattice_algebra_primitives():
    section = get_api_map()["equity_tree"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "resolve_single_state_diffusion_inputs",
        "equity_tree",
        "with_control",
        "compile_lattice_recipe",
        "build_lattice",
        "price_on_lattice",
    ):
        assert symbol in text
    assert "price_vanilla_equity_option_tree" not in text


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
