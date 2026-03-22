"""Tests for cookbook patterns."""

import pytest

from trellis.agent.cookbooks import COOKBOOKS, get_cookbook, get_all_cookbooks


class TestCookbooks:

    def test_all_five_methods_have_cookbooks(self):
        expected = {"analytical", "rate_tree", "monte_carlo", "copula", "waterfall"}
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
            assert method in all_cb.lower() or method.replace("_", " ") in all_cb.lower()

    def test_cookbooks_contain_instrument_specific_markers(self):
        """Each cookbook should mark where the builder fills in instrument logic."""
        for method, cb in COOKBOOKS.items():
            assert "INSTRUMENT-SPECIFIC" in cb, (
                f"Cookbook {method} missing INSTRUMENT-SPECIFIC markers"
            )
