"""Tests for the compact API navigation map used by agent prompts and tools."""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

from trellis.agent.knowledge.import_registry import module_exists
from trellis.agent.knowledge import api_map as api_map_module
from trellis.agent.knowledge.api_map import (
    ApiMapQuery,
    format_api_map_for_prompt,
    get_api_map,
    select_api_map_sections,
)


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
        api_map["path_statistic_composition"]["module"]
        == "trellis.models.monte_carlo.path_statistics"
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
        "analytical_gaussian_composition",
    ):
        assert module_exists(api_map[section_name]["module"])


def test_api_map_key_imports_are_registry_valid():
    api_map = get_api_map()

    for section_name in (
        "equity_tree",
        "rate_lattice",
        "monte_carlo",
        "scheduled_observation_composition",
        "path_statistic_composition",
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
    text = format_api_map_for_prompt(
        compact=True,
        query=ApiMapQuery(
            instrument_type="digital_option",
            payoff_family="digital_option",
            method="analytical",
            model_family="equity_diffusion",
        ),
    )

    assert "API Map" in text
    assert "MarketState" in text
    assert "digital_option_composition" in text
    assert "resolve_single_state_diffusion_inputs" in text
    assert "black76_cash_or_nothing_call" in text
    assert "#### quanto_option_composition" not in text
    assert "Selected cards:" in text
    assert "Omitted cards:" in text
    assert "inspect_api_map" not in text


def test_api_map_no_query_formatter_is_bounded_complete_catalog():
    text = format_api_map_for_prompt(compact=True)
    selection = select_api_map_sections()
    api_map = get_api_map()
    expected_families = tuple(
        name
        for name, section in api_map.items()
        if name not in {"market_state", "payoff", "utilities"}
        and isinstance(section, dict)
        and section.get("module")
    )

    assert selection.catalog_only is True
    assert selection.available_families == expected_families
    assert not selection.selected_families
    for family_name in selection.available_families:
        assert family_name in text
    assert "rate_style_swaption" in text
    assert "jamshidian_zcb_option" in text
    assert "credit_curve" in text
    assert "from trellis" not in text
    assert len(text) <= 4000


def test_api_map_compact_default_enforces_four_thousand_character_budget():
    text = format_api_map_for_prompt(
        compact=True,
        query=ApiMapQuery(
            requested_families=(
                "monte_carlo",
                "pde",
                "rate_monte_carlo_composition",
                "quanto_option_composition",
            )
        ),
    )

    assert len(text) <= 4000
    assert "truncated" in text.lower()


def test_every_canonical_api_map_family_is_explicitly_reachable():
    available = select_api_map_sections().available_families

    for family_name in available:
        selection = select_api_map_sections(
            ApiMapQuery(requested_families=(family_name,))
        )
        assert family_name in selection.selected_families


def test_every_canonical_api_map_utility_is_explicitly_reachable():
    available = select_api_map_sections().available_utilities

    for utility_name in available:
        selection = select_api_map_sections(
            ApiMapQuery(requested_families=(utility_name,))
        )
        assert utility_name in selection.selected_utilities


def test_empty_api_map_treats_empty_query_as_catalog_only(monkeypatch):
    monkeypatch.setattr(api_map_module, "get_api_map", lambda: {})

    assert select_api_map_sections().catalog_only is True
    assert select_api_map_sections(ApiMapQuery()).catalog_only is True


def test_api_map_semantic_selection_reaches_composition_cards():
    cases = (
        (
            ApiMapQuery(
                payoff_family="digital_option",
                method="analytical",
            ),
            "digital_option_composition",
        ),
        (
            ApiMapQuery(
                instrument_type="quanto_option",
                payoff_family="quanto_option",
                method="monte_carlo",
                model_family="hybrid",
            ),
            "quanto_option_composition",
        ),
        (
            ApiMapQuery(
                instrument_type="arithmetic_asian_option",
                payoff_family="asian_option",
                method="analytical",
                features=("scheduled_observation", "arithmetic_average"),
            ),
            "arithmetic_asian_composition",
        ),
        (
            ApiMapQuery(
                features=("scheduled_observation",),
                method="monte_carlo",
            ),
            "scheduled_observation_composition",
        ),
        (
            ApiMapQuery(
                features=("weighted_lognormal_sum",),
                method="analytical",
            ),
            "weighted_lognormal_sum_composition",
        ),
        (
            ApiMapQuery(
                instrument_type="cliquet",
                payoff_family="cliquet",
                method="monte_carlo",
                features=("observation_return",),
            ),
            "observation_return_composition",
        ),
        (
            ApiMapQuery(
                instrument_type="fixed_lookback_option",
                payoff_family="lookback_option",
                method="monte_carlo",
                features=("running_extremum",),
            ),
            "path_statistic_composition",
        ),
        (
            ApiMapQuery(
                instrument_type="variance_swap",
                payoff_family="variance_swap",
                method="monte_carlo",
                features=("squared_log_return",),
            ),
            "path_statistic_composition",
        ),
        (
            ApiMapQuery(
                description="Price a cliquet with capped interval returns.",
            ),
            "observation_return_composition",
        ),
        (
            ApiMapQuery(
                instrument_type="bermudan_swaption",
                payoff_family="swaption",
                method="monte_carlo",
                model_family="interest_rate",
            ),
            "rate_monte_carlo_composition",
        ),
    )

    for query, expected_family in cases:
        selection = select_api_map_sections(query)
        assert expected_family in selection.selected_families


@pytest.mark.parametrize(
    "payoff_family",
    ["chooser_option", "compound_option", "lookback_option"],
)
def test_api_map_semantic_selection_reaches_gaussian_root_composition(
    payoff_family,
):
    selection = select_api_map_sections(
        ApiMapQuery(
            payoff_family=payoff_family,
            method="analytical",
            model_family="equity_diffusion",
        )
    )
    text = format_api_map_for_prompt(
        compact=True,
        query=ApiMapQuery(
            payoff_family=payoff_family,
            method="analytical",
            model_family="equity_diffusion",
        ),
    )

    assert "analytical_gaussian_composition" in selection.selected_families
    assert "standard_normal_cdf" in text
    assert "bivariate_standard_normal_cdf" in text
    assert "SolveRequest" in text
    assert "execute_solve_request" in text
    assert "from trellis.models.calibration.solve_request import" in text


def test_api_map_selection_and_rendering_are_stable_and_budgeted():
    query = ApiMapQuery(
        instrument_type="arithmetic_asian_option",
        payoff_family="asian_option",
        method="monte_carlo",
        features=("scheduled_observation", "arithmetic_average"),
    )

    first = select_api_map_sections(query)
    second = select_api_map_sections(query)
    text = format_api_map_for_prompt(
        compact=True,
        query=query,
        max_chars=1200,
    )

    assert first == second
    assert len(text) <= 1200
    assert "truncated" in text.lower()
    assert "arithmetic_asian_composition" in text


def test_api_map_reachability_does_not_depend_on_manual_family_tuple():
    source = inspect.getsource(api_map_module)

    assert "_FAMILY_ORDER" not in source


def test_api_map_rejects_unknown_explicit_cards_and_invalid_budgets():
    with pytest.raises(ValueError, match="Unknown API map cards"):
        select_api_map_sections(
            ApiMapQuery(requested_families=("not_a_canonical_card",))
        )

    with pytest.raises(ValueError, match="at least 240"):
        format_api_map_for_prompt(max_chars=0)


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


def test_api_map_prioritizes_product_neutral_path_statistic_composition():
    api_map = get_api_map()
    section = api_map["path_statistic_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))
    monte_carlo_text = "\n".join(
        (
            *api_map["monte_carlo"]["key_imports"],
            *api_map["monte_carlo"]["notes"],
        )
    )

    for symbol in (
        "RunningExtremumContract",
        "SquaredLogReturnContract",
        "discrete_path_extremum",
        "annualized_squared_log_return_sum",
        "build_running_extremum_reducer",
        "build_squared_log_return_reducer",
        "resolve_scalar_diffusion_market_inputs",
        "MonteCarloPathRequirement",
        "StateAwarePayoff",
        "MonteCarloEngine",
    ):
        assert symbol in text
    assert "settlement" in text
    assert "continuous" in text.lower()
    assert "annualization_convention" in text
    assert "trellis.models.variance_swap" not in text
    assert "price_equity_fixed_lookback_option_monte_carlo" not in monte_carlo_text
    assert "price_equity_variance_swap_monte_carlo" not in monte_carlo_text


def test_api_map_exposes_product_neutral_conditional_extremum_composition():
    api_map = get_api_map()
    section = api_map["conditional_extremum_composition"]
    text = "\n".join((*section["key_imports"], *section["notes"]))

    for symbol in (
        "ConditionalBridgeExtremumContract",
        "ScalarTransitionObservation",
        "ScalarTransitionReducer",
        "build_conditional_bridge_extremum_reducer",
        "MonteCarloRandomInputs",
        "sobol_transition_inputs",
        "MonteCarloPathRequirement",
        "MonteCarloEngine",
        "GBM",
    ):
        assert symbol in text
    assert "exact scalar" in text.lower()
    assert "one stochastic transition reducer" in text.lower()
    assert "settlement" in text.lower()
    assert "price_equity_fixed_lookback_option" not in text

    selection = select_api_map_sections(
        ApiMapQuery(
            instrument_type="lookback_option",
            method="monte_carlo",
            features=("continuous_monitoring",),
        )
    )
    assert "conditional_extremum_composition" in selection.selected_families


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
