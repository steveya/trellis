"""Tests for cookbook patterns."""

import pytest

from trellis.agent.cookbooks import COOKBOOKS, get_cookbook, get_all_cookbooks


class TestCookbooks:

    def test_all_method_families_have_cookbooks(self):
        expected = {
            "analytical",
            "rate_tree",
            "monte_carlo",
            "qmc",
            "copula",
            "waterfall",
            "pde_solver",
            "fft_pricing",
        }
        assert set(COOKBOOKS.keys()) == expected

    def test_get_cookbook_returns_content(self):
        for method in COOKBOOKS:
            cb = get_cookbook(method)
            assert len(cb) > 100, f"Cookbook for {method} is too short"

    def test_get_cookbook_unknown_returns_empty(self):
        assert get_cookbook("quantum_computing") == ""

    def test_cookbooks_contain_return_type(self):
        """Each cookbook should specify either Cashflows or PresentValue."""
        for method, cb in COOKBOOKS.items():
            assert "return" in cb, (
                f"Cookbook {method} missing return type guidance"
            )

    def test_analytical_returns_cashflows(self):
        cb = get_cookbook("analytical")
        assert "return pv" in cb
        assert "spec.spot" in cb
        assert "black76_call" in cb

    def test_analytical_cookbook_includes_fx_garman_kohlhagen_pattern(self):
        cb = get_cookbook("analytical")
        assert "garman_kohlhagen_call" in cb
        assert "garman_kohlhagen_put" in cb
        assert "market_state.fx_rates" in cb
        assert "df_domestic" in cb
        assert "df_foreign" in cb

    def test_rate_tree_returns_present_value(self):
        cb = get_cookbook("rate_tree")
        assert "return " in cb

    def test_monte_carlo_returns_present_value(self):
        cb = get_cookbook("monte_carlo")
        assert "return " in cb

    def test_copula_returns_present_value(self):
        cb = get_cookbook("copula")
        assert "return " in cb

    def test_waterfall_returns_cashflows(self):
        cb = get_cookbook("waterfall")
        assert "return pv" in cb

    def test_get_all_cookbooks(self):
        all_cb = get_all_cookbooks()
        for method in COOKBOOKS:
            assert get_cookbook(method) in all_cb

    def test_method_aliases_resolve(self):
        assert get_cookbook("pde") == get_cookbook("pde_solver")
        assert get_cookbook("fft") == get_cookbook("fft_pricing")
        assert get_cookbook("lattice") == get_cookbook("rate_tree")
        assert get_cookbook("quasi_monte_carlo") == get_cookbook("qmc")

    def test_cookbooks_contain_instrument_specific_markers(self):
        """Each cookbook should mark where the builder fills in instrument logic."""
        for method, cb in COOKBOOKS.items():
            assert "INSTRUMENT-SPECIFIC" in cb, (
                f"Cookbook {method} missing INSTRUMENT-SPECIFIC markers"
            )

    def test_mc_and_qmc_cookbooks_use_differentiable_numpy(self):
        for method in ("monte_carlo", "qmc"):
            cb = get_cookbook(method)
            assert "get_numpy" in cb
            assert "import numpy as np" not in cb

    def test_monte_carlo_cookbook_includes_accurate_early_exercise_guidance(self):
        cb = get_cookbook("monte_carlo")
        assert "longstaff_schwartz" in cb
        assert "approved early-exercise control primitive" in cb
        assert "LaguerreBasis" in cb
        assert 'method="lsm"' in cb

    def test_rate_tree_cookbook_includes_schedule_exercise_guidance(self):
        cb = get_cookbook("rate_tree")
        assert "build_rate_lattice" in cb
        assert "lattice_backward_induction" in cb
        assert "exercise_fn=min" in cb
        assert "exercise_fn=max" in cb
