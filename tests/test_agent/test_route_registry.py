"""Registry tests: route_registry.py returns expected routes, primitives,
engine families, and route families for each (method, ProductIR, PricingPlan)
triple.
"""

from __future__ import annotations

import pytest

from dataclasses import replace

from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.family_lowering_ir import (
    EventAwareMonteCarloIR,
    MCControlSpec,
    MCMeasureSpec,
    MCPathRequirementSpec,
    MCPayoffReducerSpec,
    MCProcessSpec,
    MCStateSpec,
)
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.quant import PricingPlan
from trellis.agent.route_registry import (
    evaluate_route_admissibility,
    find_route_by_id,
    load_route_registry,
    match_candidate_routes,
    resolve_route_adapters,
    resolve_route_family,
    resolve_route_notes,
    resolve_route_primitives,
    validate_registry,
)


@pytest.fixture(scope="module")
def registry():
    return load_route_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(method: str, modules: list[str] | None = None, market_data: set[str] | None = None) -> PricingPlan:
    _default_modules = {
        "analytical": ["trellis.models.black"],
        "rate_tree": ["trellis.models.trees.lattice"],
        "monte_carlo": ["trellis.models.monte_carlo.engine"],
        "qmc": ["trellis.models.qmc"],
        "fft_pricing": ["trellis.models.transforms.fft_pricer"],
        "pde_solver": ["trellis.models.pde.theta_method"],
        "copula": ["trellis.models.copulas.factor"],
        "waterfall": ["trellis.models.cashflow_engine.waterfall"],
    }
    return PricingPlan(
        method=method,
        method_modules=modules or _default_modules.get(method, []),
        required_market_data=market_data or set(),
        model_to_build=None,
        reasoning="equivalence_test",
    )


def _new_routes(registry, method, product_ir, pricing_plan=None):
    """Call the new registry-based match."""
    return tuple(
        r.id for r in match_candidate_routes(registry, method, product_ir, pricing_plan=pricing_plan)
    )


def _prim_set(prims):
    """Convert primitives to comparable set of (module, symbol, role) tuples."""
    return {(p.module, p.symbol, p.role) for p in prims}


def _semantic_blueprint_for(contract, **compile_kwargs):
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract

    return compile_semantic_contract(contract, **compile_kwargs)


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

class TestRegistryValidation:
    def test_all_primitives_exist(self, registry):
        errors = validate_registry(registry)
        assert errors == (), f"Registry validation errors: {errors}"

    def test_core_promoted_routes_loaded(self, registry):
        route_ids = {route.id for route in registry.routes}
        assert {
            "analytical_black76",
            "analytical_credit_cds_par_spread_route",
            "exercise_lattice",
            "correlated_basket_monte_carlo",
            "correlated_gbm_monte_carlo",
            "credit_default_swap_analytical",
            "credit_default_swap_monte_carlo",
            "pde_theta_1d",
            "qmc_sobol_paths",
            "transform_fft",
            "zcb_option_analytical",
            "zcb_option_rate_tree",
        } <= route_ids

    def test_all_routes_promoted(self, registry):
        candidate_ids = {route.id for route in registry.routes if route.status == "candidate"}
        non_promoted = {route.id: route.status for route in registry.routes if route.status not in {"promoted", "candidate"}}
        assert non_promoted == {}
        assert candidate_ids == {"analytical_credit_cds_par_spread_route"}

    def test_typed_admissibility_hydrates_for_migrated_routes(self, registry):
        analytical = find_route_by_id("analytical_black76", registry)
        lattice = find_route_by_id("exercise_lattice", registry)
        basket = find_route_by_id("correlated_basket_monte_carlo", registry)

        assert analytical is not None
        assert analytical.admissibility.supported_control_styles == ("identity", "holder_max")
        assert analytical.admissibility.multicurrency_support == "single_currency_only"
        assert lattice is not None
        assert lattice.admissibility.supported_control_styles == ("identity", "holder_max", "issuer_min")
        assert basket is not None
        assert basket.admissibility.event_support == "automatic"

    def test_reuse_module_paths_hydrate_for_checked_in_deterministic_routes(self, registry):
        analytical_fx = find_route_by_id("analytical_garman_kohlhagen", registry)
        monte_carlo_fx = find_route_by_id("monte_carlo_paths", registry)
        analytical_quanto = find_route_by_id("quanto_adjustment_analytical", registry)
        monte_carlo_quanto = find_route_by_id("correlated_gbm_monte_carlo", registry)

        assert analytical_fx is not None
        assert analytical_fx.reuse_module_paths == ("instruments/_agent/fxvanillaanalytical.py",)
        assert monte_carlo_fx is not None
        assert monte_carlo_fx.reuse_module_paths == ("instruments/_agent/fxvanillamontecarlo.py",)
        assert analytical_quanto is not None
        assert analytical_quanto.reuse_module_paths == ("instruments/_agent/quantooptionanalytical.py",)
        assert monte_carlo_quanto is not None
        assert monte_carlo_quanto.reuse_module_paths == ("instruments/_agent/quantooptionmontecarlo.py",)


# ---------------------------------------------------------------------------
# Quanto routes
# ---------------------------------------------------------------------------

class TestQuantoRoutes:
    IR = ProductIR(instrument="quanto_option", payoff_family="vanilla_option", exercise_style="european")

    def test_analytical_candidate(self, registry):
        new = _new_routes(registry, "analytical", self.IR)
        assert new == ("quanto_adjustment_analytical",)

    def test_monte_carlo_candidate(self, registry):
        new = _new_routes(registry, "monte_carlo", self.IR)
        assert new == ("correlated_gbm_monte_carlo",)

    def test_qmc_candidate(self, registry):
        new = _new_routes(registry, "qmc", self.IR)
        assert new == ("correlated_gbm_monte_carlo",)

    def test_analytical_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "quanto_adjustment_analytical"][0]
        new_prims = resolve_route_primitives(spec, self.IR)
        new_adapters = resolve_route_adapters(spec, self.IR)
        expected_prims = {
            ("trellis.models.resolution.quanto", "resolve_quanto_inputs", "market_binding"),
            ("trellis.models.black", "black76_call", "pricing_kernel"),
            ("trellis.models.black", "black76_put", "pricing_kernel"),
            ("trellis.models.analytical.quanto", "price_quanto_option_analytical", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims
        assert "reuse_shared_quanto_market_binding" in new_adapters

    def test_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "quanto_adjustment_analytical"][0]
        assert spec.engine_family == "analytical"


# ---------------------------------------------------------------------------
# Credit routes
# ---------------------------------------------------------------------------

class TestCreditRoutes:
    CDS_IR = ProductIR(instrument="cds", payoff_family="credit_default_swap")
    NTD_IR = ProductIR(instrument="nth_to_default", payoff_family="nth_to_default")

    def test_cds_analytical(self, registry):
        new = _new_routes(registry, "analytical", self.CDS_IR)
        assert new == ("credit_default_swap_analytical",)

    def test_cds_monte_carlo(self, registry):
        new = _new_routes(registry, "monte_carlo", self.CDS_IR)
        assert new == ("credit_default_swap_monte_carlo",)

    def test_nth_to_default_analytical(self, registry):
        new = _new_routes(registry, "analytical", self.NTD_IR)
        assert new == ()

    def test_nth_to_default_monte_carlo(self, registry):
        new = _new_routes(registry, "monte_carlo", self.NTD_IR)
        assert new == ("nth_to_default_monte_carlo",)

    def test_route_family(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_analytical"][0]
        new = resolve_route_family(spec, self.CDS_IR)
        assert new == "credit_default_swap"

    def test_cds_analytical_admissibility_uses_credit_family_state_tags(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_credit_default_swap_contract

        contract = make_credit_default_swap_contract(
            description="Single-name CDS analytical",
            observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20"),
        )
        blueprint = compile_semantic_contract(contract)
        spec = find_route_by_id("credit_default_swap_analytical", registry)

        decision = evaluate_route_admissibility(spec, semantic_blueprint=blueprint)

        assert decision.ok
        assert tuple(decision.failures) == ()

    def test_cds_monte_carlo_admissibility_uses_credit_family_state_tags(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_credit_default_swap_contract

        contract = make_credit_default_swap_contract(
            description="Single-name CDS Monte Carlo",
            observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20"),
            preferred_method="monte_carlo",
        )
        blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")
        spec = find_route_by_id("credit_default_swap_monte_carlo", registry)

        decision = evaluate_route_admissibility(spec, semantic_blueprint=blueprint)

        assert decision.ok
        assert tuple(decision.failures) == ()


# ---------------------------------------------------------------------------
# Basket routes
# ---------------------------------------------------------------------------

class TestBasketRoutes:
    BASKET_IR = ProductIR(
        instrument="basket_option",
        payoff_family="basket_path_payoff",
        payoff_traits=("ranked_observation",),
    )

    def test_monte_carlo_candidate(self, registry):
        new = _new_routes(registry, "monte_carlo", self.BASKET_IR)
        assert new == ("correlated_basket_monte_carlo",)

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "correlated_basket_monte_carlo"][0]
        new_prims = resolve_route_primitives(spec, self.BASKET_IR)
        expected_prims = {
            ("trellis.models.resolution.basket_semantics", "resolve_basket_semantics", "market_binding"),
            ("trellis.models.monte_carlo.semantic_basket", "price_ranked_observation_basket_monte_carlo", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_admissibility_rejects_strategic_control_for_basket_event_route(self, registry):
        from dataclasses import replace
        from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

        spec = find_route_by_id("correlated_basket_monte_carlo", registry)
        assert spec is not None

        contract = make_ranked_observation_basket_contract(
            description="Himalaya on AAPL, MSFT, NVDA",
            constituents=("AAPL", "MSFT", "NVDA"),
            observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
        )
        bp = _semantic_blueprint_for(contract)
        bp = replace(
            bp,
            contract=replace(
                bp.contract,
                product=replace(
                    bp.contract.product,
                    controller_protocol=replace(
                        bp.contract.product.controller_protocol,
                        controller_style="holder_max",
                        controller_role="holder",
                        admissible_actions=("exercise", "continue"),
                    ),
                ),
            ),
        )

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert not decision.ok
        assert "unsupported_control_style:holder_max" in decision.failures

    def test_admissibility_uses_typed_event_machine_not_legacy_event_transition_mirror(self, registry):
        from dataclasses import replace
        from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

        spec = find_route_by_id("correlated_basket_monte_carlo", registry)
        assert spec is not None

        contract = make_ranked_observation_basket_contract(
            description="Himalaya on AAPL, MSFT, NVDA",
            constituents=("AAPL", "MSFT", "NVDA"),
            observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
        )
        contract = replace(
            contract,
            product=replace(
                contract.product,
                event_transitions=(),
            ),
        )
        bp = _semantic_blueprint_for(contract)

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert decision.ok


# ---------------------------------------------------------------------------
# Exercise Monte Carlo routes
# ---------------------------------------------------------------------------

class TestExerciseMonteCarloRoutes:
    AMERICAN_IR = ProductIR(
        instrument="american_option",
        payoff_family="vanilla_option",
        exercise_style="american",
    )

    def test_candidates(self, registry):
        new = _new_routes(registry, "monte_carlo", self.AMERICAN_IR)
        assert "exercise_monte_carlo" in new
        assert "monte_carlo_paths" in new

    def test_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_monte_carlo"][0]
        assert spec.engine_family == "exercise"


# ---------------------------------------------------------------------------
# Monte Carlo paths (vanilla European)
# ---------------------------------------------------------------------------

class TestMonteCarloPathsRoutes:
    EUROPEAN_IR = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
    )

    def test_candidate(self, registry):
        new = _new_routes(registry, "monte_carlo", self.EUROPEAN_IR)
        assert "monte_carlo_paths" in new

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        new_prims = resolve_route_primitives(spec, self.EUROPEAN_IR)
        expected_prims = {
            (
                "trellis.models.equity_option_monte_carlo",
                "price_vanilla_equity_option_monte_carlo",
                "route_helper",
            ),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_admissibility_hydrates_process_and_path_contracts(self, registry):
        generic = find_route_by_id("monte_carlo_paths", registry)
        local_vol = find_route_by_id("local_vol_monte_carlo", registry)

        assert generic is not None
        assert generic.admissibility.supported_process_families == ("gbm_1d", "hull_white_1f")
        assert generic.admissibility.supported_state_tags == (
            "pathwise_only",
            "terminal_markov",
            "recombining_safe",
            "schedule_state",
        )
        assert generic.admissibility.supported_path_requirement_kinds == (
            "terminal_only",
            "full_path",
            "event_snapshots",
            "event_replay",
            "reducer_state",
        )
        assert generic.admissibility.supports_calibration is True
        assert local_vol is not None
        assert local_vol.admissibility.supported_process_families == ("local_vol_1d",)
        assert local_vol.admissibility.supported_path_requirement_kinds == (
            "terminal_only",
            "full_path",
            "event_snapshots",
            "event_replay",
            "reducer_state",
        )

    def test_monte_carlo_admissibility_accepts_matching_event_aware_family_ir(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        spec = find_route_by_id("monte_carlo_paths", registry)
        assert spec is not None

        contract = make_vanilla_option_contract(
            description="EUR call on AAPL, K=150, T=1y",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
            preferred_method="monte_carlo",
        )
        bp = compile_semantic_contract(contract, preferred_method="monte_carlo")
        family_ir = EventAwareMonteCarloIR(
            route_id="monte_carlo_paths",
            route_family="monte_carlo",
            product_instrument="european_option",
            payoff_family="vanilla_option",
            state_spec=MCStateSpec(
                state_variable="spot",
                state_tags=("terminal_markov",),
            ),
            process_spec=MCProcessSpec(
                process_family="gbm_1d",
                simulation_scheme="exact_lognormal",
            ),
            path_requirement_spec=MCPathRequirementSpec(
                requirement_kind="terminal_only",
            ),
            payoff_reducer_spec=MCPayoffReducerSpec(
                reducer_kind="terminal_payoff",
                output_semantics="vanilla_option_payoff",
            ),
            control_spec=MCControlSpec(
                control_style="identity",
                controller_role="holder",
            ),
            measure_spec=MCMeasureSpec(
                measure_family="risk_neutral",
                numeraire_binding="discount_curve",
            ),
        )
        bp = replace(
            bp,
            dsl_lowering=replace(bp.dsl_lowering, family_ir=family_ir),
        )

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert decision.ok
        assert decision.failures == ()

    def test_monte_carlo_admissibility_accepts_hull_white_event_aware_family(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

        spec = find_route_by_id("monte_carlo_paths", registry)
        assert spec is not None

        contract = make_rate_style_swaption_contract(
            description="5Yx10Y USD payer swaption Hull-White Monte Carlo",
            observation_schedule=("2027-03-15",),
            preferred_method="monte_carlo",
        )
        bp = compile_semantic_contract(contract, preferred_method="monte_carlo")
        family_ir = EventAwareMonteCarloIR(
            route_id="monte_carlo_paths",
            route_family="monte_carlo",
            product_instrument="swaption",
            payoff_family="swaption",
            state_spec=MCStateSpec(
                state_variable="short_rate",
                state_tags=("terminal_markov", "recombining_safe", "schedule_state"),
            ),
            process_spec=MCProcessSpec(
                process_family="hull_white_1f",
                simulation_scheme="exact_ou",
            ),
            path_requirement_spec=MCPathRequirementSpec(
                requirement_kind="event_replay",
                reducer_kinds=("discounted_swap_pv",),
            ),
            payoff_reducer_spec=MCPayoffReducerSpec(
                reducer_kind="compiled_schedule_payoff",
                output_semantics="swaption_exercise_payoff",
            ),
            control_spec=MCControlSpec(
                control_style="identity",
                controller_role="holder",
            ),
            measure_spec=MCMeasureSpec(
                measure_family="risk_neutral",
                numeraire_binding="discount_curve",
            ),
        )
        bp = replace(
            bp,
            dsl_lowering=replace(bp.dsl_lowering, family_ir=family_ir),
        )

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert decision.ok
        assert decision.failures == ()


# ---------------------------------------------------------------------------
# Local vol Monte Carlo
# ---------------------------------------------------------------------------

class TestLocalVolRoutes:
    IR = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
    )
    PLAN = _make_plan("monte_carlo", market_data={"local_vol_surface", "spot", "discount_curve"})

    def test_candidate(self, registry):
        new = _new_routes(registry, "monte_carlo", self.IR, pricing_plan=self.PLAN)
        assert "local_vol_monte_carlo" in new

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "local_vol_monte_carlo"][0]
        new_prims = resolve_route_primitives(spec, self.IR)
        expected_prims = {
            ("trellis.models.processes.local_vol", "LocalVol", "state_process"),
            ("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
            ("trellis.models.monte_carlo.local_vol", "local_vol_european_vanilla_price", "pricing_kernel"),
        }
        assert _prim_set(new_prims) == expected_prims


# ---------------------------------------------------------------------------
# QMC Sobol paths
# ---------------------------------------------------------------------------

class TestQMCRoutes:
    IR = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
    )

    def test_candidate(self, registry):
        new = _new_routes(registry, "qmc", self.IR)
        assert "qmc_sobol_paths" in new


# ---------------------------------------------------------------------------
# Rate tree routes
# ---------------------------------------------------------------------------

class TestRateTreeRoutes:
    CALLABLE_IR = ProductIR(
        instrument="callable_bond",
        payoff_family="callable_fixed_income",
        exercise_style="issuer_call",
        model_family="interest_rate",
    )
    PUTTABLE_IR = ProductIR(
        instrument="puttable_bond",
        payoff_family="puttable_fixed_income",
        exercise_style="holder_put",
        model_family="interest_rate",
    )
    BERMUDAN_IR = ProductIR(
        instrument="bermudan_swaption",
        payoff_family="swaption",
        exercise_style="bermudan",
        model_family="interest_rate",
    )
    EUROPEAN_SWAPTION_IR = ProductIR(
        instrument="swaption",
        payoff_family="swaption",
        exercise_style="european",
        model_family="interest_rate",
    )
    EQUITY_AMERICAN_IR = ProductIR(
        instrument="american_option",
        payoff_family="vanilla_option",
        exercise_style="american",
        model_family="equity_diffusion",
    )

    def test_callable_bond_candidates(self, registry):
        new = _new_routes(registry, "rate_tree", self.CALLABLE_IR)
        assert "exercise_lattice" in new
        assert "rate_tree_backward_induction" in new

    def test_puttable_bond_candidates(self, registry):
        new = _new_routes(registry, "rate_tree", self.PUTTABLE_IR)
        assert "exercise_lattice" in new
        assert "rate_tree_backward_induction" in new

    def test_callable_bond_route_family(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new = resolve_route_family(spec, self.CALLABLE_IR)
        assert new == "rate_lattice"

    def test_puttable_bond_route_family(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new = resolve_route_family(spec, self.PUTTABLE_IR)
        assert new == "rate_lattice"

    def test_equity_american_route_family(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new = resolve_route_family(spec, self.EQUITY_AMERICAN_IR)
        assert new == "equity_tree"

    def test_equity_american_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new_prims = resolve_route_primitives(spec, self.EQUITY_AMERICAN_IR)
        expected_prims = {
            ("trellis.models.equity_option_tree", "price_vanilla_equity_option_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_callable_bond_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new_prims = resolve_route_primitives(spec, self.CALLABLE_IR)
        expected_prims = {
            ("trellis.models.callable_bond_tree", "price_callable_bond_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_puttable_bond_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new_prims = resolve_route_primitives(spec, self.PUTTABLE_IR)
        expected_prims = {
            ("trellis.models.callable_bond_tree", "price_callable_bond_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_bermudan_swaption_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        new_prims = resolve_route_primitives(spec, self.BERMUDAN_IR)
        expected_prims = {
            ("trellis.models.bermudan_swaption_tree", "price_bermudan_swaption_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_rate_tree_swaption_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        new_prims = resolve_route_primitives(spec, self.EUROPEAN_SWAPTION_IR)
        expected_prims = {
            ("trellis.models.rate_style_swaption_tree", "price_swaption_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_rate_tree_swaption_notes_pin_helper_backed_route(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        notes = resolve_route_notes(spec, self.EUROPEAN_SWAPTION_IR)
        assert any("price_swaption_tree" in note for note in notes)

    def test_backward_induction_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        assert spec.engine_family == "lattice"

    def test_backward_induction_route_family(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        new = resolve_route_family(spec, None)
        assert new == "rate_lattice"


# ---------------------------------------------------------------------------
# Analytical routes
# ---------------------------------------------------------------------------

class TestAnalyticalRoutes:
    SWAPTION_IR = ProductIR(instrument="swaption", payoff_family="swaption", exercise_style="european")
    BERMUDAN_SWAPTION_IR = ProductIR(
        instrument="bermudan_swaption",
        payoff_family="swaption",
        exercise_style="bermudan",
        model_family="interest_rate",
    )
    VANILLA_IR = ProductIR(instrument="european_option", payoff_family="vanilla_option", exercise_style="european")
    ZCB_IR = ProductIR(instrument="zcb_option", payoff_family="zcb_option", exercise_style="european")

    def test_swaption_candidate(self, registry):
        new = _new_routes(registry, "analytical", self.SWAPTION_IR)
        assert new == ("analytical_black76",)

    def test_zcb_candidate(self, registry):
        new = _new_routes(registry, "analytical", self.ZCB_IR)
        assert new == ("zcb_option_analytical",)

    def test_vanilla_primitives_with_product_ir(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.VANILLA_IR)
        prim_symbols = {p.symbol for p in new_prims}
        assert "black76_call" in prim_symbols
        assert "black76_put" in prim_symbols
        assert "year_fraction" in prim_symbols

    def test_default_primitives_without_vanilla(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.SWAPTION_IR)
        assert _prim_set(new_prims) == {
            (
                "trellis.models.rate_style_swaption",
                "price_swaption_black76",
                "route_helper",
            ),
        }

    def test_swaption_notes_pin_helper_backed_route(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        notes = resolve_route_notes(spec, self.SWAPTION_IR)
        assert any("price_swaption_black76" in note for note in notes)
        assert any("Hull-White-implied Black vol" in note for note in notes)

    def test_bermudan_swaption_primitives_use_lower_bound_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.BERMUDAN_SWAPTION_IR)
        assert _prim_set(new_prims) == {
            (
                "trellis.models.rate_style_swaption",
                "price_bermudan_swaption_black76_lower_bound",
                "route_helper",
            ),
        }

    def test_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        assert spec.engine_family == "analytical"

    def test_zcb_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_analytical"][0]
        new_prims = resolve_route_primitives(spec, self.ZCB_IR)
        expected_prims = {
            ("trellis.models.zcb_option", "price_zcb_option_jamshidian", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_zcb_notes_pin_resolved_jamshidian_lane(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_analytical"][0]
        notes = resolve_route_notes(spec, self.ZCB_IR)
        assert any("resolve_zcb_option_hw_inputs" in note for note in notes)
        assert any("zcb_option_hw_raw" in note for note in notes)

    def test_admissibility_rejects_pathwise_state_tags_on_black76(self, registry):
        from dataclasses import replace
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        spec = find_route_by_id("analytical_black76", registry)
        assert spec is not None

        contract = make_vanilla_option_contract(
            description="EUR call on AAPL, K=150, T=1y",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
        )
        bp = _semantic_blueprint_for(contract)
        pathwise_field = replace(
            bp.contract.product.state_fields[0],
            tags=("pathwise_only",),
        )
        bp = replace(
            bp,
            contract=replace(
                bp.contract,
                product=replace(
                    bp.contract.product,
                    state_fields=(pathwise_field,),
                ),
            ),
        )

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert not decision.ok
        assert "unsupported_state_tag:pathwise_only" in decision.failures


class TestZCBRateTreeRoutes:
    IR = ProductIR(instrument="zcb_option", payoff_family="zcb_option", exercise_style="european")

    def test_candidate(self, registry):
        new = _new_routes(registry, "rate_tree", self.IR)
        assert "zcb_option_rate_tree" in new
        assert "rate_tree_backward_induction" in new

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_rate_tree"][0]
        new_prims = resolve_route_primitives(spec, self.IR)
        expected_prims = {
            ("trellis.models.zcb_option_tree", "price_zcb_option_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims


# ---------------------------------------------------------------------------
# FX analytical route (Garman-Kohlhagen)
# ---------------------------------------------------------------------------

class TestFXAnalyticalRoutes:
    FX_IR = ProductIR(instrument="fx_option", payoff_family="vanilla_option", exercise_style="european")
    FX_PLAN = _make_plan("analytical", market_data={"fx_rates", "forward_curve", "discount_curve"})

    def test_fx_candidate(self, registry):
        new = _new_routes(registry, "analytical", self.FX_IR, pricing_plan=self.FX_PLAN)
        assert new == ("analytical_garman_kohlhagen",)

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        new_prims = resolve_route_primitives(spec, self.FX_IR)
        expected_prims = {
            ("trellis.models.analytical.fx", "garman_kohlhagen_price_raw", "route_helper"),
            ("trellis.core.date_utils", "year_fraction", "time_measure"),
        }
        assert _prim_set(new_prims) == expected_prims


# ---------------------------------------------------------------------------
# Transform / PDE / Copula / Waterfall routes
# ---------------------------------------------------------------------------

class TestFallbackRoutes:
    CALLABLE_BOND_IR = ProductIR(
        instrument="callable_bond",
        payoff_family="callable_fixed_income",
        exercise_style="issuer_call",
        model_family="interest_rate",
    )

    def test_fft_candidate(self, registry):
        new = _new_routes(registry, "fft_pricing", None)
        assert new == ("transform_fft",)

    def test_pde_candidate(self, registry):
        new = _new_routes(registry, "pde_solver", None)
        assert new == ("pde_theta_1d",)

    def test_callable_bond_pde_candidate(self, registry):
        new = _new_routes(registry, "pde_solver", self.CALLABLE_BOND_IR)
        assert new == ("pde_theta_1d",)

    def test_vanilla_equity_pde_candidate(self, registry):
        ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
        )
        new = _new_routes(registry, "pde_solver", ir)
        assert new == ("vanilla_equity_theta_pde", "pde_theta_1d")

    def test_copula_candidate(self, registry):
        new = _new_routes(registry, "copula", None)
        assert new == ("copula_loss_distribution",)

    def test_waterfall_candidate(self, registry):
        new = _new_routes(registry, "waterfall", None)
        assert new == ("waterfall_cashflows",)

    def test_fft_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "transform_fft"][0]
        new_prims = resolve_route_primitives(spec, None)
        expected_prims = {
            ("trellis.models.transforms.fft_pricer", "fft_price", "transform_pricer"),
            ("trellis.models.transforms.cos_method", "cos_price", "transform_pricer"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_vanilla_equity_transform_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "transform_fft"][0]
        ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
        )
        new_prims = resolve_route_primitives(spec, ir)
        expected_prims = {
            (
                "trellis.models.equity_option_transforms",
                "price_vanilla_equity_option_transform",
                "route_helper",
            ),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_pde_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        new_prims = resolve_route_primitives(spec, None)
        expected_prims = {
            ("trellis.models.pde.grid", "Grid", "grid"),
            ("trellis.models.pde.operator", "BlackScholesOperator", "spatial_operator"),
            ("trellis.models.pde.theta_method", "theta_method_1d", "time_stepping"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_vanilla_equity_pde_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
        )
        new_prims = resolve_route_primitives(spec, ir)
        expected_prims = {
            ("trellis.models.equity_option_pde", "price_vanilla_equity_option_pde", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_holder_max_equity_pde_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        ir = ProductIR(
            instrument="american_option",
            payoff_family="vanilla_option",
            exercise_style="bermudan",
            model_family="equity_diffusion",
        )
        new_prims = resolve_route_primitives(spec, ir)
        expected_prims = {
            ("trellis.models.equity_option_pde", "price_event_aware_equity_option_pde", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_callable_bond_pde_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_fixed_income",
            exercise_style="issuer_call",
            model_family="interest_rate",
        )
        new_prims = resolve_route_primitives(spec, ir)
        expected_prims = {
            ("trellis.models.callable_bond_pde", "price_callable_bond_pde", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_pde_admissibility_hydrates_operator_and_event_contracts(self, registry):
        vanilla = find_route_by_id("vanilla_equity_theta_pde", registry)
        generic = find_route_by_id("pde_theta_1d", registry)

        assert vanilla is not None
        assert vanilla.admissibility.supported_operator_families == ("black_scholes_1d",)
        assert vanilla.admissibility.supported_event_transform_kinds == ()
        assert generic is not None
        assert generic.admissibility.supported_control_styles == ("identity", "holder_max", "issuer_min")
        assert generic.admissibility.supported_operator_families == ("black_scholes_1d", "hull_white_1f")
        assert generic.admissibility.supported_event_transform_kinds == (
            "add_cashflow",
            "project_max",
            "project_min",
        )

    def test_vanilla_pde_admissibility_rejects_wrong_operator_family(self, registry):
        from dataclasses import replace

        from trellis.agent.family_lowering_ir import PDEOperatorSpec
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        spec = find_route_by_id("vanilla_equity_theta_pde", registry)
        assert spec is not None

        contract = make_vanilla_option_contract(
            description="EUR put on AAPL, K=150, T=1y",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
            preferred_method="pde_solver",
        )
        bp = compile_semantic_contract(contract, preferred_method="pde_solver")
        family_ir = replace(
            bp.dsl_lowering.family_ir,
            operator_spec=replace(
                bp.dsl_lowering.family_ir.operator_spec,
                operator_family="hull_white_1f",
            ),
        )
        bp = replace(
            bp,
            dsl_lowering=replace(bp.dsl_lowering, family_ir=family_ir),
        )

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert not decision.ok
        assert "unsupported_operator_family:hull_white_1f" in decision.failures

    def test_vanilla_pde_admissibility_rejects_event_transforms_not_declared_by_route(self, registry):
        from dataclasses import replace

        from trellis.agent.family_lowering_ir import PDEEventTransformSpec
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        spec = find_route_by_id("vanilla_equity_theta_pde", registry)
        assert spec is not None

        contract = make_vanilla_option_contract(
            description="EUR put on AAPL, K=150, T=1y",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
            preferred_method="pde_solver",
        )
        bp = compile_semantic_contract(contract, preferred_method="pde_solver")
        family_ir = replace(
            bp.dsl_lowering.family_ir,
            event_transforms=(
                PDEEventTransformSpec(
                    transform_kind="project_min",
                    schedule_role="decision_dates",
                    value_semantics="issuer_call_projection",
                ),
            ),
        )
        bp = replace(
            bp,
            dsl_lowering=replace(bp.dsl_lowering, family_ir=family_ir),
        )

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert not decision.ok
        assert "unsupported_event_transform_kind:project_min" in decision.failures

    def test_callable_bond_pde_admissibility_uses_lowered_pde_state_not_raw_schedule_tags(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_callable_bond_contract

        spec = find_route_by_id("pde_theta_1d", registry)
        assert spec is not None

        contract = make_callable_bond_contract(
            description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            observation_schedule=("2026-01-15", "2027-01-15"),
            preferred_method="pde_solver",
        )
        bp = compile_semantic_contract(contract, preferred_method="pde_solver")

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert decision.ok
        assert decision.failures == ()
        assert "unsupported_state_tag:schedule_state" not in decision.failures

    def test_holder_max_equity_pde_admissibility_accepts_project_max_contract(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_american_option_contract

        spec = find_route_by_id("pde_theta_1d", registry)
        assert spec is not None

        contract = make_american_option_contract(
            description="Bermudan put on AAPL with quarterly exercise dates",
            underliers=("AAPL",),
            observation_schedule=("2026-03-20", "2026-06-20", "2026-09-20", "2026-12-20"),
            preferred_method="pde_solver",
            exercise_style="bermudan",
        )
        bp = compile_semantic_contract(contract, preferred_method="pde_solver")

        decision = evaluate_route_admissibility(spec, semantic_blueprint=bp)

        assert decision.ok
        assert decision.failures == ()

    def test_copula_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "copula_loss_distribution"][0]
        new_prims = resolve_route_primitives(spec, None)
        expected_prims = {
            ("trellis.models.copulas.factor", "FactorCopula", "loss_distribution"),
            ("trellis.models.copulas.student_t", "StudentTCopula", "loss_distribution"),
            ("trellis.models.credit_basket_copula", "price_credit_basket_tranche", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_waterfall_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "waterfall_cashflows"][0]
        new_prims = resolve_route_primitives(spec, None)
        expected_prims = {
            ("trellis.models.cashflow_engine.waterfall", "Waterfall", "cashflow_engine"),
            ("trellis.models.cashflow_engine.waterfall", "Tranche", "cashflow_engine"),
        }
        assert _prim_set(new_prims) == expected_prims


# ---------------------------------------------------------------------------
# Engine family coverage
# ---------------------------------------------------------------------------

class TestEngineFamilyCoverage:
    EXPECTED = {
        "analytical_credit_cds_par_spread_route": "analytical",
        "quanto_adjustment_analytical": "analytical",
        "correlated_gbm_monte_carlo": "monte_carlo",
        "credit_default_swap_analytical": "analytical",
        "credit_default_swap_monte_carlo": "monte_carlo",
        "nth_to_default_monte_carlo": "monte_carlo",
        "correlated_basket_monte_carlo": "monte_carlo",
        "exercise_monte_carlo": "exercise",
        "monte_carlo_paths": "monte_carlo",
        "local_vol_monte_carlo": "monte_carlo",
        "qmc_sobol_paths": "qmc",
        "exercise_lattice": "lattice",
        "rate_tree_backward_induction": "lattice",
        "zcb_option_rate_tree": "lattice",
        "analytical_black76": "analytical",
        "zcb_option_analytical": "analytical",
        "analytical_garman_kohlhagen": "analytical",
        "transform_fft": "fft_pricing",
        "vanilla_equity_theta_pde": "pde_solver",
        "pde_theta_1d": "pde_solver",
        "copula_loss_distribution": "copula",
        "waterfall_cashflows": "waterfall",
    }

    def test_all_engine_families_match(self, registry):
        for route in registry.routes:
            assert route.engine_family == self.EXPECTED[route.id], (
                f"Route {route.id}: registry says {route.engine_family}, "
                f"expected {self.EXPECTED[route.id]}"
            )

    def test_coverage_complete(self, registry):
        route_ids = {r.id for r in registry.routes}
        expected_ids = set(self.EXPECTED.keys())
        assert route_ids == expected_ids, f"Missing: {expected_ids - route_ids}, Extra: {route_ids - expected_ids}"


# ---------------------------------------------------------------------------
# Market data access (matches lite_review _ROUTE_REQUIRED_ACCESSES)
# ---------------------------------------------------------------------------

class TestMarketDataAccess:
    def test_analytical_black76_access(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        assert "discount_curve" in spec.market_data_access.required
        assert "black_vol_surface" in spec.market_data_access.required

    def test_garman_kohlhagen_access(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        required = spec.market_data_access.required
        assert "discount_curve" in required
        assert "forward_curve" in required
        assert "black_vol_surface" in required
        assert "spot" in required

    def test_local_vol_access(self, registry):
        spec = [r for r in registry.routes if r.id == "local_vol_monte_carlo"][0]
        required = spec.market_data_access.required
        assert "discount_curve" in required
        assert "local_vol_surface" in required
        assert "spot" in required
