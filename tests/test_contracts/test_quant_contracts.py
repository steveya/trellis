"""Tier 2 contract tests: quant agent method selection for canary tasks (QUA-427).

These tests verify that the quant agent selects the correct pricing method
for instrument types used by canary tasks.  All use static plans (no LLM).

The quant agent's STATIC_PLANS are keyed by **instrument type** (e.g.
callable_bond, cds, swaption), not by engine family (lattice, credit, etc.).
"""

from __future__ import annotations

import pytest

from trellis.agent.quant import STATIC_PLANS, PricingPlan, select_pricing_method


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select(title: str, instrument_type: str | None = None) -> PricingPlan:
    """Call select_pricing_method with a canary task description."""
    return select_pricing_method(
        instrument_description=title,
        instrument_type=instrument_type,
    )


# ---------------------------------------------------------------------------
# T02 — Callable bond → rate_tree
# ---------------------------------------------------------------------------

class TestCallableBondQuantDecision:
    """Callable bond (T02/T17) must select rate_tree method."""

    @pytest.mark.tier2
    def test_selects_rate_tree(self):
        plan = _select(
            "Callable bond — BDT lognormal vs HW normal tree",
            instrument_type="callable_bond",
        )
        assert plan.method == "rate_tree", f"Expected rate_tree, got {plan.method}"

    @pytest.mark.tier2
    def test_method_modules_include_lattice(self):
        plan = _select(
            "Callable bond — BDT lognormal vs HW normal tree",
            instrument_type="callable_bond",
        )
        modules_str = " ".join(plan.method_modules)
        assert "tree" in modules_str or "lattice" in modules_str, (
            f"Expected tree/lattice in modules, got {plan.method_modules}"
        )

    @pytest.mark.tier2
    def test_has_reasoning(self):
        plan = _select(
            "Callable bond — BDT lognormal vs HW normal tree",
            instrument_type="callable_bond",
        )
        assert plan.reasoning, "Reasoning must be non-empty"


# ---------------------------------------------------------------------------
# T38 — CDS → analytical (from static plan)
# ---------------------------------------------------------------------------

class TestT38CdsQuantDecision:
    """T38: CDS uses analytical method from static plan."""

    @pytest.mark.tier2
    def test_selects_analytical(self):
        plan = _select(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="cds",
        )
        assert plan.method == "analytical", f"Expected analytical, got {plan.method}"

    @pytest.mark.tier2
    def test_has_required_market_data(self):
        plan = _select(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="cds",
        )
        assert isinstance(plan.required_market_data, set)


# ---------------------------------------------------------------------------
# T49 — CDO → copula
# ---------------------------------------------------------------------------

class TestT49CdoQuantDecision:
    """T49: CDO tranche must select copula method."""

    @pytest.mark.tier2
    def test_selects_copula(self):
        plan = _select(
            "CDO tranche: Gaussian vs Student-t copula",
            instrument_type="cdo",
        )
        assert plan.method == "copula", f"Expected copula, got {plan.method}"

    @pytest.mark.tier2
    def test_method_modules_include_copula(self):
        plan = _select(
            "CDO tranche: Gaussian vs Student-t copula",
            instrument_type="cdo",
        )
        modules_str = " ".join(plan.method_modules)
        assert "copula" in modules_str, (
            f"Expected copula in modules, got {plan.method_modules}"
        )


# ---------------------------------------------------------------------------
# T73 — Swaption → analytical
# ---------------------------------------------------------------------------

class TestT73SwaptionQuantDecision:
    """T73: Swaption selects analytical from static plan."""

    @pytest.mark.tier2
    def test_selects_analytical(self):
        plan = _select(
            "European swaption: Black76 vs HW tree vs HW MC",
            instrument_type="swaption",
        )
        assert plan.method == "analytical", f"Expected analytical, got {plan.method}"


# ---------------------------------------------------------------------------
# European option → analytical (T13 base instrument)
# ---------------------------------------------------------------------------

class TestEuropeanOptionQuantDecision:
    """European option selects analytical from static plan."""

    @pytest.mark.tier2
    def test_selects_analytical(self):
        plan = _select(
            "European call: theta-method convergence",
            instrument_type="european_option",
        )
        assert plan.method == "analytical", f"Expected analytical, got {plan.method}"


# ---------------------------------------------------------------------------
# Cross-cutting decision boundary contracts
# ---------------------------------------------------------------------------

class TestDecisionBoundaries:
    """Verify quant agent respects static plan boundaries."""

    @pytest.mark.tier2
    def test_instrument_types_map_to_expected_methods(self):
        """Key instrument types must map to specific methods."""
        expected = {
            "callable_bond": "rate_tree",
            "puttable_bond": "rate_tree",
            "bermudan_swaption": "rate_tree",
            "cdo": "copula",
            "american_option": "monte_carlo",
            "asian_option": "monte_carlo",
            "barrier_option": "monte_carlo",
            "bond": "analytical",
            "cap": "analytical",
            "floor": "analytical",
            "swap": "analytical",
            "swaption": "analytical",
            "cds": "analytical",
            "european_option": "analytical",
        }
        for instrument_type, expected_method in expected.items():
            if instrument_type not in STATIC_PLANS:
                continue
            plan = select_pricing_method(
                instrument_description=f"test {instrument_type}",
                instrument_type=instrument_type,
            )
            assert plan.method == expected_method, (
                f"{instrument_type}: expected {expected_method}, got {plan.method}"
            )

    @pytest.mark.tier2
    def test_all_static_plans_have_required_fields(self):
        """Every static PricingPlan must have non-empty method and reasoning."""
        for key, plan in STATIC_PLANS.items():
            assert plan.method, f"STATIC_PLANS[{key}]: method is empty"
            assert plan.reasoning, f"STATIC_PLANS[{key}]: reasoning is empty"

    @pytest.mark.tier2
    def test_static_plans_cover_canary_instrument_types(self):
        """Static plans must cover the key instrument types used by canary tasks."""
        required = {"callable_bond", "cds", "swaption", "cdo", "european_option"}
        covered = set(STATIC_PLANS.keys())
        missing = required - covered
        assert not missing, f"Missing static plans for canary instruments: {missing}"
