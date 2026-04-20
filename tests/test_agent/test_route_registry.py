"""Registry tests: route_registry.py returns expected routes, primitives,
engine families, and route families for each (method, ProductIR, PricingPlan)
triple.
"""

from __future__ import annotations

import pytest

from dataclasses import replace
from types import SimpleNamespace

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
    RouteAdmissibilitySpec,
    RouteRegistry,
    RouteSpec,
    evaluate_route_admissibility,
    evaluate_route_capability_match,
    find_route_by_id,
    load_route_registry,
    match_candidate_routes,
    resolve_route_adapters,
    resolve_route_family,
    resolve_route_notes,
    resolve_route_primitives,
    validate_registry,
    compile_route_binding_authority,
)


@pytest.fixture(scope="module")
def registry():
    return load_route_registry()


@pytest.fixture(scope="module")
def analysis_registry():
    return load_route_registry(include_discovered=True)


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
            "exercise_lattice",
            "correlated_basket_monte_carlo",
            "equity_quanto",
            "credit_default_swap",
            "pde_theta_1d",
            "qmc_sobol_paths",
            "transform_fft",
            "short_rate_bond_option",
        } <= route_ids

    def test_all_routes_promoted(self, registry):
        candidate_ids = {route.id for route in registry.routes if route.status == "candidate"}
        non_promoted = {route.id: route.status for route in registry.routes if route.status not in {"promoted", "candidate"}}
        assert non_promoted == {}
        assert candidate_ids == set()

    def test_analysis_registry_opt_in_loads_discovered_routes(self, registry, analysis_registry):
        live_ids = {route.id for route in registry.routes}
        analysis_ids = {route.id for route in analysis_registry.routes}

        assert "analytical_credit_cds_par_spread_route" not in live_ids
        assert "analytical_credit_cds_par_spread_route" in analysis_ids

        discovered = find_route_by_id("analytical_credit_cds_par_spread_route", analysis_registry)
        assert discovered is not None
        assert discovered.status == "candidate"
        assert discovered.discovered_from is not None

    def test_family_first_capability_match_uses_route_family_and_schedule_state(self):
        spec = RouteSpec(
            id="synthetic_credit_mc",
            engine_family="monte_carlo",
            route_family="event_triggered_two_legged_contract",
            status="promoted",
            confidence=1.0,
            match_methods=("monte_carlo",),
            match_instruments=None,
            exclude_instruments=(),
            match_payoff_family=None,
            match_payoff_traits=None,
            match_exercise=None,
            exclude_exercise=(),
            match_required_market_data=None,
            exclude_required_market_data=None,
            primitives=(),
            conditional_primitives=(),
            conditional_route_family=None,
            adapters=(),
            notes=(),
            admissibility=RouteAdmissibilitySpec(
                supported_state_tags=("schedule_state", "pathwise_only"),
                supported_path_requirement_kinds=("event_snapshots",),
            ),
        )
        ir = ProductIR(
            instrument="cds",
            payoff_family="event_triggered_two_legged_contract",
            schedule_dependence=True,
            state_dependence="pathwise_only",
            candidate_engine_families=("monte_carlo",),
            route_families=("event_triggered_two_legged_contract",),
        )

        decision = evaluate_route_capability_match(spec, ir)

        assert decision.ok is True
        assert "route_family" in decision.matched_predicates
        assert "schedule_dependence" in decision.matched_predicates
        assert "state:pathwise_only" in decision.matched_predicates

    def test_match_candidate_routes_prefers_family_capability_predicates(self):
        registry = RouteRegistry(
            routes=(
                RouteSpec(
                    id="generic_mc",
                    engine_family="monte_carlo",
                    route_family="monte_carlo",
                    status="promoted",
                    confidence=1.0,
                    match_methods=("monte_carlo",),
                    match_instruments=None,
                    exclude_instruments=(),
                    match_payoff_family=None,
                    match_payoff_traits=None,
                    match_exercise=None,
                    exclude_exercise=(),
                    match_required_market_data=None,
                    exclude_required_market_data=None,
                    primitives=(),
                    conditional_primitives=(),
                    conditional_route_family=None,
                    adapters=(),
                    notes=(),
                    admissibility=RouteAdmissibilitySpec(
                        supported_state_tags=("terminal_markov",),
                    ),
                ),
                RouteSpec(
                    id="family_mc",
                    engine_family="monte_carlo",
                    route_family="event_triggered_two_legged_contract",
                    status="promoted",
                    confidence=1.0,
                    match_methods=("monte_carlo",),
                    match_instruments=None,
                    exclude_instruments=(),
                    match_payoff_family=None,
                    match_payoff_traits=None,
                    match_exercise=None,
                    exclude_exercise=(),
                    match_required_market_data=None,
                    exclude_required_market_data=None,
                    primitives=(),
                    conditional_primitives=(),
                    conditional_route_family=None,
                    adapters=(),
                    notes=(),
                    admissibility=RouteAdmissibilitySpec(
                        supported_state_tags=("schedule_state", "pathwise_only"),
                        supported_path_requirement_kinds=("event_snapshots",),
                    ),
                ),
            ),
            _method_index={"monte_carlo": (0, 1)},
        )
        ir = ProductIR(
            instrument="cds",
            payoff_family="event_triggered_two_legged_contract",
            schedule_dependence=True,
            state_dependence="pathwise_only",
            candidate_engine_families=("monte_carlo",),
            route_families=("event_triggered_two_legged_contract",),
        )

        matches = match_candidate_routes(registry, "monte_carlo", ir)

        assert tuple(route.id for route in matches) == ("family_mc",)

    def test_route_resolution_prefers_binding_catalog_over_route_shell_fields(self, monkeypatch):
        from trellis.agent import backend_bindings as backend_bindings_module

        spec = RouteSpec(
            id="synthetic_binding_route",
            engine_family="analytical",
            route_family="legacy_family",
            status="promoted",
            confidence=1.0,
            match_methods=("analytical",),
            match_instruments=None,
            exclude_instruments=(),
            match_payoff_family=None,
            match_payoff_traits=None,
            match_exercise=None,
            exclude_exercise=(),
            match_required_market_data=None,
            exclude_required_market_data=None,
            primitives=(
                PrimitiveRef("trellis.models.legacy", "old_helper", "route_helper"),
            ),
            conditional_primitives=(),
            conditional_route_family=None,
            adapters=(),
            notes=(),
        )
        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda route_id, product_ir=None, method=None: SimpleNamespace(
                primitives=(
                    PrimitiveRef("trellis.models.synthetic", "fresh_helper", "route_helper"),
                ),
                route_family="binding_family",
            ),
        )

        synthetic_ir = ProductIR(instrument="synthetic", payoff_family="synthetic")
        resolved_primitives = resolve_route_primitives(spec, synthetic_ir)

        assert _prim_set(resolved_primitives) == {
            ("trellis.models.synthetic", "fresh_helper", "route_helper"),
        }
        assert resolve_route_family(spec, synthetic_ir) == "binding_family"

    def test_route_resolution_reuses_preloaded_binding_spec(self, monkeypatch):
        from trellis.agent import backend_bindings as backend_bindings_module

        spec = RouteSpec(
            id="synthetic_binding_route",
            engine_family="analytical",
            route_family="legacy_family",
            status="promoted",
            confidence=1.0,
            match_methods=("analytical",),
            match_instruments=None,
            exclude_instruments=(),
            match_payoff_family=None,
            match_payoff_traits=None,
            match_exercise=None,
            exclude_exercise=(),
            match_required_market_data=None,
            exclude_required_market_data=None,
            primitives=(
                PrimitiveRef("trellis.models.legacy", "old_helper", "route_helper"),
            ),
            conditional_primitives=(),
            conditional_route_family=None,
            adapters=(),
            notes=(),
        )
        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("binding lookup should not run when binding_spec is supplied")
            ),
        )

        binding_spec = SimpleNamespace(
            primitives=(
                PrimitiveRef("trellis.models.synthetic", "fresh_helper", "route_helper"),
            ),
            route_family="binding_family",
        )

        assert resolve_route_primitives(spec, None, binding_spec=binding_spec) == binding_spec.primitives
        assert resolve_route_family(spec, None, binding_spec=binding_spec) == "binding_family"

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
        monte_carlo_fx = find_route_by_id("monte_carlo_fx_vanilla", registry)
        quanto = find_route_by_id("equity_quanto", registry)

        assert analytical_fx is not None
        assert analytical_fx.reuse_module_paths == ("instruments/_agent/fxvanillaanalytical.py",)
        assert monte_carlo_fx is not None
        assert monte_carlo_fx.reuse_module_paths == ("instruments/_agent/fxvanillamontecarlo.py",)
        assert quanto is not None
        assert quanto.reuse_module_paths == (
            "instruments/_agent/quantooptionanalytical.py",
            "instruments/_agent/quantooptionmontecarlo.py",
        )

    def test_compile_route_binding_authority_accepts_plan_carried_binding_identity_without_route(self):
        authority = compile_route_binding_authority(
            generation_plan=SimpleNamespace(
                primitive_plan=None,
                backend_binding_id="trellis.models.synthetic.price_bound_helper",
                backend_exact_target_refs=("trellis.models.synthetic.price_bound_helper",),
                backend_helper_refs=("trellis.models.synthetic.price_bound_helper",),
                backend_pricing_kernel_refs=(),
                backend_schedule_builder_refs=(),
                backend_cashflow_engine_refs=(),
                backend_market_binding_refs=(),
                backend_engine_family="analytical",
                backend_route_family="analytical",
                backend_compatibility_alias_policy="internal_only",
                approved_modules=("trellis.models.synthetic",),
                lane_plan_kind="exact_target_binding",
                method="analytical",
                lane_family="analytical",
                repo_revision="test-rev",
            ),
        )

        assert authority is not None
        assert authority.route_id is None
        assert authority.route_family == "analytical"
        assert authority.authority_kind == "exact_backend_fit"
        assert authority.compatibility_alias_policy == "internal_only"
        assert authority.backend_binding.binding_id == "trellis.models.synthetic.price_bound_helper"
        assert authority.backend_binding.engine_family == "analytical"
        assert authority.backend_binding.exact_backend_fit is True
        assert authority.backend_binding.exact_target_refs == (
            "trellis.models.synthetic.price_bound_helper",
        )
        assert authority.operator_metadata is not None
        assert authority.operator_metadata.display_name == "Bound (analytical)"
        assert authority.operator_metadata.diagnostic_label == "price_bound_helper"


# ---------------------------------------------------------------------------
# Quanto routes
# ---------------------------------------------------------------------------

class TestQuantoRoutes:
    IR = ProductIR(
        instrument="quanto_option",
        payoff_family="vanilla_option",
        payoff_traits=("fx_translation",),
        exercise_style="european",
    )
    PLAN_MARKET_DATA = {"discount_curve", "black_vol_surface", "fx_rates"}

    def test_analytical_candidate(self, registry):
        new = _new_routes(
            registry,
            "analytical",
            self.IR,
            pricing_plan=_make_plan("analytical", market_data=self.PLAN_MARKET_DATA),
        )
        assert new == ("equity_quanto",)

    def test_monte_carlo_candidate(self, registry):
        new = _new_routes(
            registry,
            "monte_carlo",
            self.IR,
            pricing_plan=_make_plan("monte_carlo", market_data=self.PLAN_MARKET_DATA),
        )
        assert new == ("equity_quanto",)

    def test_qmc_candidate(self, registry):
        new = _new_routes(
            registry,
            "qmc",
            self.IR,
            pricing_plan=_make_plan("qmc", market_data=self.PLAN_MARKET_DATA),
        )
        assert new == ("equity_quanto",)

    def test_analytical_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        new_prims = resolve_route_primitives(spec, self.IR, method="analytical")
        expected_prims = {
            (
                "trellis.models.quanto_option",
                "price_quanto_option_analytical_from_market_state",
                "route_helper",
            ),
        }
        assert _prim_set(new_prims) == expected_prims
        assert resolve_route_adapters(spec, self.IR, method="analytical") == ()

    def test_monte_carlo_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        new_prims = resolve_route_primitives(spec, self.IR, method="monte_carlo")
        expected_prims = {
            (
                "trellis.models.quanto_option",
                "price_quanto_option_monte_carlo_from_market_state",
                "route_helper",
            ),
        }
        assert _prim_set(new_prims) == expected_prims
        assert resolve_route_family(spec, self.IR, method="monte_carlo") == "monte_carlo"

    def test_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        assert spec.engine_family == "analytical"
        assert spec.compatibility_alias_policy == "internal_only"


# ---------------------------------------------------------------------------
# Credit routes
# ---------------------------------------------------------------------------

class TestCreditRoutes:
    CDS_IR = ProductIR(
        instrument="cds",
        payoff_family="event_triggered_two_legged_contract",
        route_families=("event_triggered_two_legged_contract",),
    )
    NTD_IR = ProductIR(instrument="nth_to_default", payoff_family="nth_to_default")

    def test_cds_analytical(self, registry):
        new = _new_routes(
            registry,
            "analytical",
            self.CDS_IR,
            pricing_plan=_make_plan("analytical", market_data={"credit_curve"}),
        )
        assert new == ("credit_default_swap",)

    def test_cds_monte_carlo(self, registry):
        new = _new_routes(
            registry,
            "monte_carlo",
            self.CDS_IR,
            pricing_plan=_make_plan("monte_carlo", market_data={"credit_curve"}),
        )
        assert new == ("credit_default_swap",)

    def test_nth_to_default_analytical(self, registry):
        plan = _make_plan("analytical", market_data={"discount_curve", "credit_curve"})
        new = _new_routes(registry, "analytical", self.NTD_IR, pricing_plan=plan)
        assert new == ("credit_basket_nth_to_default",)

    def test_nth_to_default_monte_carlo(self, registry):
        plan = _make_plan("monte_carlo", market_data={"discount_curve", "credit_curve"})
        new = _new_routes(registry, "monte_carlo", self.NTD_IR, pricing_plan=plan)
        assert new == ("credit_basket_nth_to_default",)

    _PRE_COLLAPSE_ANALYTICAL_PRIMITIVES = frozenset({
        ("trellis.core.date_utils", "year_fraction", "time_measure"),
        ("trellis.instruments.nth_to_default", "price_nth_to_default_basket", "route_helper"),
    })

    _PRE_COLLAPSE_MC_PRIMITIVES = frozenset({
        ("trellis.core.date_utils", "year_fraction", "time_measure"),
        ("trellis.core.differentiable", "get_numpy", "array_backend"),
        ("trellis.models.copulas.gaussian", "GaussianCopula", "default_time_sampler"),
        ("trellis.instruments.nth_to_default", "price_nth_to_default_basket", "route_helper"),
        ("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
    })

    def test_nth_to_default_collapsed_route_preserves_analytical_primitive_surface(self, registry):
        spec = find_route_by_id("credit_basket_nth_to_default", registry)
        assert spec is not None
        primitives = resolve_route_primitives(spec, self.NTD_IR, method="analytical")
        assert _prim_set(primitives) == self._PRE_COLLAPSE_ANALYTICAL_PRIMITIVES

    def test_nth_to_default_collapsed_route_preserves_mc_primitive_surface(self, registry):
        spec = find_route_by_id("credit_basket_nth_to_default", registry)
        assert spec is not None
        primitives = resolve_route_primitives(spec, self.NTD_IR, method="monte_carlo")
        assert _prim_set(primitives) == self._PRE_COLLAPSE_MC_PRIMITIVES

    def test_nth_to_default_collapsed_route_qmc_copula_methods_share_mc_surface(self, registry):
        spec = find_route_by_id("credit_basket_nth_to_default", registry)
        assert spec is not None
        mc = _prim_set(resolve_route_primitives(spec, self.NTD_IR, method="monte_carlo"))
        qmc = _prim_set(resolve_route_primitives(spec, self.NTD_IR, method="qmc"))
        copula = _prim_set(resolve_route_primitives(spec, self.NTD_IR, method="copula"))
        assert mc == qmc == copula == self._PRE_COLLAPSE_MC_PRIMITIVES

    def test_nth_to_default_collapsed_route_admissibility_envelope_is_union(self, registry):
        spec = find_route_by_id("credit_basket_nth_to_default", registry)
        assert spec is not None
        tags = set(spec.admissibility.supported_state_tags)
        assert {"pathwise_only", "remaining_pool", "schedule_state"}.issubset(tags)
        assert spec.admissibility.supported_control_styles == ("identity",)
        assert spec.admissibility.event_support == "automatic"
        assert spec.admissibility.supports_sensitivity_outputs is True

    def test_route_family(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        new = resolve_route_family(spec, self.CDS_IR)
        assert new == "event_triggered_two_legged_contract"

    def test_cds_analytical_does_not_require_instrument_key(self, registry):
        structural_only_ir = ProductIR(
            instrument="synthetic_event_wrapper",
            payoff_family="event_triggered_two_legged_contract",
            route_families=("event_triggered_two_legged_contract",),
            schedule_dependence=True,
            state_dependence="schedule_state",
        )

        new = _new_routes(
            registry,
            "analytical",
            structural_only_ir,
            pricing_plan=_make_plan("analytical", market_data={"credit_curve"}),
        )

        assert new == ("credit_default_swap",)

    def test_cds_analytical_admissibility_uses_credit_family_state_tags(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_credit_default_swap_contract

        contract = make_credit_default_swap_contract(
            description="Single-name CDS analytical",
            observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20"),
        )
        blueprint = compile_semantic_contract(contract)
        spec = find_route_by_id("credit_default_swap", registry)

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
        spec = find_route_by_id("credit_default_swap", registry)

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
        assert new[0] == "correlated_basket_monte_carlo"
        assert "monte_carlo_paths" in new

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "correlated_basket_monte_carlo"][0]
        new_prims = resolve_route_primitives(spec, self.BASKET_IR)
        expected_prims = {
            ("trellis.models.resolution.basket_semantics", "resolve_basket_semantics", "market_binding"),
            ("trellis.models.monte_carlo.semantic_basket", "price_ranked_observation_basket_monte_carlo", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims
        assert resolve_route_adapters(spec, self.BASKET_IR) == ()
        assert resolve_route_notes(spec, self.BASKET_IR) == ()

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

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_monte_carlo"][0]
        new_prims = resolve_route_primitives(spec, self.AMERICAN_IR)
        expected_prims = {
            ("trellis.models.processes.gbm", "GBM", "state_process"),
            ("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
            ("trellis.models.monte_carlo.lsm", "longstaff_schwartz", "exercise_control"),
            ("trellis.models.monte_carlo.tv_regression", "tsitsiklis_van_roy", "exercise_control"),
            ("trellis.models.monte_carlo.primal_dual", "primal_dual_mc", "exercise_control"),
            ("trellis.models.monte_carlo.stochastic_mesh", "stochastic_mesh", "exercise_control"),
        }
        assert _prim_set(new_prims) == expected_prims


# ---------------------------------------------------------------------------
# Monte Carlo paths (vanilla European)
# ---------------------------------------------------------------------------

class TestMonteCarloPathsRoutes:
    EUROPEAN_IR = ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
    )
    GENERIC_IR = ProductIR(
        instrument="path_dependent_note",
        payoff_family="path_dependent_generic",
        exercise_style="none",
        model_family="generic",
    )
    GENERIC_BASKET_IR = ProductIR(
        instrument="basket_option",
        payoff_family="basket_option",
        payoff_traits=("vanilla_option",),
        exercise_style="european",
        model_family="generic",
    )

    def test_candidate(self, registry):
        new = _new_routes(registry, "monte_carlo", self.EUROPEAN_IR)
        assert "monte_carlo_paths" in new

    def test_generic_basket_candidate_stays_on_generic_monte_carlo_surface(self, registry):
        new = _new_routes(registry, "monte_carlo", self.GENERIC_BASKET_IR)
        assert "monte_carlo_paths" in new
        assert "correlated_basket_monte_carlo" not in new

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

    def test_default_branch_preserves_base_adapters_and_notes_for_generic_requests(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]

        assert resolve_route_primitives(spec, self.GENERIC_IR) == spec.primitives
        assert resolve_route_adapters(spec, self.GENERIC_IR) == spec.adapters
        assert resolve_route_notes(spec, self.GENERIC_IR) == spec.notes

    def test_helper_backed_branches_do_not_reintroduce_route_notes_or_adapters(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]

        assert resolve_route_adapters(spec, self.EUROPEAN_IR) == ()
        assert resolve_route_notes(spec, self.EUROPEAN_IR) == ()
        assert resolve_route_adapters(spec, self.GENERIC_IR) == ()
        assert resolve_route_notes(spec, self.GENERIC_IR) == ()

    def test_fallback_lanes_have_no_route_card_adapters_or_notes(self, registry):
        """QUA-816 / QUA-880: route-card constructive adapters / notes retired
        for these fallback lanes.

        The thinned lanes rely on the typed binding primitives (``state_process``,
        ``path_simulation``, ``pricing_kernel``, ``low_discrepancy_sampler``,
        ``cashflow_engine``, ``exercise_control``) plus ``lane_obligations``
        construction steps and control obligations for constructive guidance.
        Reintroducing prose on these cards resurrects the dual-authority
        problem.

        QUA-880 slice 2 also retires the ``dynamic_notes`` on
        ``exercise_monte_carlo`` -- the early-exercise policy summaries
        now flow through ``_control_obligations_for``.

        ``pde_theta_1d`` is intentionally NOT in the thinned set (slice
        3 follow-on): migrating its kernel-contract notes regressed T13
        cassette-replay generation, so the PDE route-card prose stays
        on the card until the generation path is updated to consume
        them from the typed surface.
        """
        thinned_lane_ids = {
            # QUA-816 slice 1
            "local_vol_monte_carlo",
            "qmc_sobol_paths",
            "waterfall_cashflows",
            # QUA-880 slice 2
            "exercise_monte_carlo",
        }
        for spec in registry.routes:
            if spec.id not in thinned_lane_ids:
                continue
            assert spec.adapters == (), (
                f"route {spec.id} unexpectedly carries adapters: {spec.adapters}"
            )
            assert spec.notes == (), (
                f"route {spec.id} unexpectedly carries notes: {spec.notes}"
            )
            assert spec.dynamic_notes == (), (
                f"route {spec.id} unexpectedly carries dynamic_notes: {spec.dynamic_notes}"
            )
            assert spec.primitives, (
                f"route {spec.id} must keep typed primitives after QUA-816/880 cleanup"
            )

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

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "qmc_sobol_paths"][0]
        new_prims = resolve_route_primitives(spec, self.IR)
        expected_prims = {
            ("trellis.models.qmc", "sobol_normals", "low_discrepancy_sampler"),
            ("trellis.models.qmc", "brownian_bridge", "path_construction"),
            ("trellis.models.processes.gbm", "GBM", "state_process"),
        }
        assert _prim_set(new_prims) == expected_prims


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

    def test_callable_bond_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        notes = resolve_route_notes(spec, self.CALLABLE_IR)
        assert notes == ()

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

    def test_bermudan_swaption_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        notes = resolve_route_notes(spec, self.BERMUDAN_IR)
        assert notes == ()

    def test_rate_tree_swaption_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        new_prims = resolve_route_primitives(spec, self.EUROPEAN_SWAPTION_IR)
        expected_prims = {
            ("trellis.models.rate_style_swaption_tree", "price_swaption_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_rate_tree_swaption_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        notes = resolve_route_notes(spec, self.EUROPEAN_SWAPTION_IR)
        assert notes == ()

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
    BARRIER_IR = ProductIR(
        instrument="barrier_option",
        payoff_family="barrier_option",
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )
    DIGITAL_IR = ProductIR(
        instrument="digital_option",
        payoff_family="composite_option",
        payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )
    LOOKBACK_IR = ProductIR(
        instrument="lookback_option",
        payoff_family="composite_option",
        payoff_traits=("discounting", "path_dependent", "vol_surface_dependence"),
        exercise_style="european",
        state_dependence="path_dependent",
        model_family="equity_diffusion",
    )
    CHOOSER_IR = ProductIR(
        instrument="chooser_option",
        payoff_family="composite_option",
        payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )
    COMPOUND_IR = ProductIR(
        instrument="compound_option",
        payoff_family="composite_option",
        payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )
    CLIQUET_IR = ProductIR(
        instrument="cliquet_option",
        payoff_family="composite_option",
        payoff_traits=("vanilla_option",),
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )
    VARIANCE_SWAP_IR = ProductIR(
        instrument="variance_swap",
        payoff_family="variance_swap",
        payoff_traits=("discounting", "path_dependent", "vol_surface_dependence"),
        exercise_style="none",
        state_dependence="path_dependent",
        schedule_dependence=False,
        model_family="generic",
    )
    ZCB_IR = ProductIR(instrument="zcb_option", payoff_family="zcb_option", exercise_style="european")
    # QUA-909: ``analytical_black76`` now requires ``black_vol_surface`` in the
    # pricing-plan required market data (positive filter replacing the old
    # ``exclude_required_market_data: [fx_rates]`` clause). Tests that exercise
    # the Black76 match clause must pass a plan carrying that data; the bare
    # no-plan form stays exercised below to defend the ``pricing_plan is None``
    # branch in ``match_candidate_routes``.
    BLACK76_PLAN = _make_plan(
        "analytical",
        market_data={"discount_curve", "black_vol_surface"},
    )

    def test_swaption_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.SWAPTION_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_barrier_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.BARRIER_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_digital_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.DIGITAL_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_lookback_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.LOOKBACK_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_cliquet_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.CLIQUET_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_variance_swap_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.VARIANCE_SWAP_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_chooser_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.CHOOSER_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_compound_candidate(self, registry):
        new = _new_routes(
            registry, "analytical", self.COMPOUND_IR, pricing_plan=self.BLACK76_PLAN,
        )
        assert new == ("analytical_black76",)

    def test_zcb_candidate(self, registry):
        # QUA-915: ZCB option family collapsed into short_rate_bond_option.
        # The new route carries required_market_data: [discount_curve,
        # black_vol_surface]; provide a matching plan so the route is
        # visible in the ranked list alongside analytical_black76.
        plan = _make_plan(
            "analytical",
            market_data={"discount_curve", "black_vol_surface"},
        )
        new = _new_routes(registry, "analytical", self.ZCB_IR, pricing_plan=plan)
        assert "short_rate_bond_option" in new

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

    def test_swaption_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        assert resolve_route_notes(spec, self.SWAPTION_IR) == ()
        assert resolve_route_adapters(spec, self.SWAPTION_IR) == ()

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

    def test_bermudan_swaption_analytical_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        assert resolve_route_notes(spec, self.BERMUDAN_SWAPTION_IR) == ()
        assert resolve_route_adapters(spec, self.BERMUDAN_SWAPTION_IR) == ()

    def test_barrier_primitives_use_exact_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.BARRIER_IR)
        assert resolve_route_family(spec, self.BARRIER_IR) == "analytical"
        assert _prim_set(new_prims) == {
            ("trellis.models.analytical.barrier", "barrier_option_price", "route_helper"),
        }

    def test_digital_primitives_use_exact_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.DIGITAL_IR)
        assert resolve_route_family(spec, self.DIGITAL_IR) == "analytical"
        assert _prim_set(new_prims) == {
            (
                "trellis.models.analytical.equity_exotics",
                "price_equity_digital_option_analytical",
                "route_helper",
            ),
        }

    def test_lookback_primitives_use_exact_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.LOOKBACK_IR)
        assert resolve_route_family(spec, self.LOOKBACK_IR) == "analytical"
        assert _prim_set(new_prims) == {
            (
                "trellis.models.analytical.equity_exotics",
                "price_equity_fixed_lookback_option_analytical",
                "route_helper",
            ),
        }

    def test_chooser_primitives_use_exact_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.CHOOSER_IR)
        assert resolve_route_family(spec, self.CHOOSER_IR) == "analytical"
        assert _prim_set(new_prims) == {
            (
                "trellis.models.analytical.equity_exotics",
                "price_equity_chooser_option_analytical",
                "route_helper",
            ),
        }

    def test_compound_primitives_use_exact_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.COMPOUND_IR)
        assert resolve_route_family(spec, self.COMPOUND_IR) == "analytical"
        assert _prim_set(new_prims) == {
            (
                "trellis.models.analytical.equity_exotics",
                "price_equity_compound_option_analytical",
                "route_helper",
            ),
        }

    def test_cliquet_primitives_use_exact_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.CLIQUET_IR)
        assert resolve_route_family(spec, self.CLIQUET_IR) == "analytical"
        assert _prim_set(new_prims) == {
            (
                "trellis.models.analytical.equity_exotics",
                "price_equity_cliquet_option_analytical",
                "route_helper",
            ),
        }

    def test_variance_swap_primitives_use_exact_helper_and_output_kernel(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        new_prims = resolve_route_primitives(spec, self.VARIANCE_SWAP_IR)
        assert resolve_route_family(spec, self.VARIANCE_SWAP_IR) == "analytical"
        assert _prim_set(new_prims) == {
            (
                "trellis.models.analytical.equity_exotics",
                "price_equity_variance_swap_analytical",
                "route_helper",
            ),
            (
                "trellis.models.analytical.equity_exotics",
                "equity_variance_swap_outputs_analytical",
                "pricing_kernel",
            ),
        }

    def test_rate_cap_floor_strip_analytical_admissibility_accepts_structural_schedule_contract(self, registry):
        from dataclasses import replace

        from trellis.agent.semantic_contracts import make_rate_cap_floor_strip_contract

        black76_spec = find_route_by_id("analytical_black76", registry)
        assert black76_spec is not None

        contract = make_rate_cap_floor_strip_contract(
            description="5Y cap on SOFR with quarterly caplets under Black-76",
            instrument_class="cap",
            observation_schedule=("2026-03-20", "2026-06-20", "2026-09-20", "2026-12-20"),
            preferred_method="analytical",
        )
        bp = _semantic_blueprint_for(contract, preferred_method="analytical")

        decision = evaluate_route_admissibility(black76_spec, semantic_blueprint=bp)

        assert decision.ok
        assert "unsupported_event_support:automatic_triggers" not in decision.failures

        raw_bp = replace(
            bp,
            dsl_lowering=replace(bp.dsl_lowering, family_ir=None),
        )
        raw_decision = evaluate_route_admissibility(black76_spec, semantic_blueprint=raw_bp)
        assert not raw_decision.ok
        assert "unsupported_event_support:automatic_triggers" in raw_decision.failures

        # QUA-915: the ZCB-option family has collapsed into the
        # pattern-keyed ``short_rate_bond_option`` route. The
        # admissibility contract for a rate-cap-floor-strip contract is
        # still rejected (wrong product), but the failure is not an
        # automatic-event gate — matching the pre-collapse assertion.
        zcb_spec = find_route_by_id("short_rate_bond_option", registry)
        assert zcb_spec is not None
        zcb_decision = evaluate_route_admissibility(zcb_spec, semantic_blueprint=bp)
        assert not zcb_decision.ok
        assert all(
            failure != "unsupported_event_support:automatic_triggers"
            for failure in zcb_decision.failures
        )

    def test_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        assert spec.engine_family == "analytical"

    def test_zcb_primitives(self, registry):
        # QUA-915: the analytical-method branch of short_rate_bond_option
        # resolves to the Jamshidian helper (same kernel the pre-collapse
        # zcb_option_analytical route exposed).
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        new_prims = resolve_route_primitives(spec, self.ZCB_IR, method="analytical")
        expected_prims = {
            ("trellis.models.zcb_option", "price_zcb_option_jamshidian", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_zcb_analytical_route_is_thin(self, registry):
        # QUA-915: the collapsed route stays thin — the analytical branch
        # adds no prose notes on top of the base (empty) notes.
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        notes = resolve_route_notes(spec, self.ZCB_IR, method="analytical")
        assert notes == ()

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


class TestShortRateBondOptionRoutes:
    """QUA-915: pattern-keyed ZCB-option family replaces the old
    ``zcb_option_analytical`` + ``zcb_option_rate_tree`` pair. The
    rate-tree branch of ``short_rate_bond_option`` dispatches to the
    Hull-White helper; the analytical branch keeps the Jamshidian helper.
    """

    IR = ProductIR(instrument="zcb_option", payoff_family="zcb_option", exercise_style="european")

    def test_candidate(self, registry):
        # Provide a plan carrying the required_market_data declared on
        # ``short_rate_bond_option`` so match_candidate_routes does not
        # filter it out. ``rate_tree_backward_induction`` remains the
        # method-generic fallback route for rate_tree; both should
        # surface for the ZCB option under a rate-tree request.
        plan = _make_plan(
            "rate_tree",
            market_data={"discount_curve", "black_vol_surface"},
        )
        new = _new_routes(registry, "rate_tree", self.IR, pricing_plan=plan)
        assert "short_rate_bond_option" in new
        assert "rate_tree_backward_induction" in new

    def test_primitives_rate_tree(self, registry):
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        new_prims = resolve_route_primitives(spec, self.IR, method="rate_tree")
        expected_prims = {
            ("trellis.models.zcb_option_tree", "price_zcb_option_tree", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_primitives_analytical(self, registry):
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        new_prims = resolve_route_primitives(spec, self.IR, method="analytical")
        expected_prims = {
            ("trellis.models.zcb_option", "price_zcb_option_jamshidian", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_route_is_thin_per_method(self, registry):
        # The collapsed route carries no adapter or note prose for either
        # branch — both helper surfaces already own the assembly.
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        assert resolve_route_notes(spec, self.IR, method="rate_tree") == ()
        assert resolve_route_adapters(spec, self.IR, method="rate_tree") == ()
        assert resolve_route_notes(spec, self.IR, method="analytical") == ()
        assert resolve_route_adapters(spec, self.IR, method="analytical") == ()

    def test_route_family_resolves_per_method(self, registry):
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        assert resolve_route_family(spec, self.IR, method="rate_tree") == "rate_lattice"
        assert resolve_route_family(spec, self.IR, method="analytical") == "analytical"

    def test_no_double_match_with_analytical_black76(self, registry):
        # The discriminator between ``short_rate_bond_option`` and
        # ``analytical_black76`` is ``payoff_family``: black76 does not
        # list ``zcb_option`` in its positive filter, so a ZCB option
        # should only resolve to the collapsed route under analytical.
        plan = _make_plan(
            "analytical",
            market_data={"discount_curve", "black_vol_surface"},
        )
        new = _new_routes(registry, "analytical", self.IR, pricing_plan=plan)
        assert "short_rate_bond_option" in new
        assert "analytical_black76" not in new

    def test_skip_market_data_filters_surfaces_collapsed_route_without_plan(self, registry):
        # QUA-915 regression guard: ``short_rate_bond_option`` declares
        # ``required_market_data: [discount_curve, black_vol_surface]``
        # in its match clause.  Gap-audit callers (``gap_check``,
        # ``reflect``'s discovery booster, and
        # ``KnowledgeStore._promoted_routes_for``) call
        # ``match_candidate_routes`` with no pricing plan — they only
        # have a ``ProductDecomposition`` / minimal ``ProductIR``.
        # Without ``skip_market_data_filters=True`` the AND-filter over
        # an empty required-market-data set rejects every route that
        # declares market-data requirements, so similar-product
        # retrieval silently returns no promoted routes.  This test
        # defends the gap-audit mode end-to-end.
        no_plan_matches = match_candidate_routes(
            registry, "rate_tree", self.IR, promoted_only=True
        )
        assert "short_rate_bond_option" not in {r.id for r in no_plan_matches}

        gap_audit_matches = match_candidate_routes(
            registry,
            "rate_tree",
            self.IR,
            promoted_only=True,
            skip_market_data_filters=True,
        )
        assert "short_rate_bond_option" in {r.id for r in gap_audit_matches}


def test_rate_tree_routes_keep_backend_binding_metadata_only(registry):
    """QUA-915: the ZCB-option family collapsed into
    ``short_rate_bond_option``. The rate-tree-facing bindings still stay
    thin: no adapter prose, no notes, on either method branch.
    """
    exercise = find_route_by_id("exercise_lattice", registry)
    backward = find_route_by_id("rate_tree_backward_induction", registry)
    zcb_route = find_route_by_id("short_rate_bond_option", registry)

    assert exercise is not None
    assert backward is not None
    assert zcb_route is not None

    zcb_ir = ProductIR(
        instrument="zcb_option",
        payoff_family="zcb_option",
        exercise_style="european",
    )
    # Both method branches must stay thin — the collapsed route does not
    # smuggle adapter or note prose through either conditional block.
    assert resolve_route_adapters(zcb_route, zcb_ir, method="rate_tree") == ()
    assert resolve_route_notes(zcb_route, zcb_ir, method="rate_tree") == ()
    assert resolve_route_adapters(zcb_route, zcb_ir, method="analytical") == ()
    assert resolve_route_notes(zcb_route, zcb_ir, method="analytical") == ()


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
            ("trellis.models.fx_vanilla", "price_fx_vanilla_analytical", "route_helper"),
        }
        assert _prim_set(new_prims) == expected_prims


class TestFXMonteCarloRoutes:
    FX_IR = ProductIR(instrument="fx_option", payoff_family="vanilla_option", exercise_style="european")
    FX_PLAN = _make_plan(
        "monte_carlo",
        market_data={"fx_rates", "forward_curve", "discount_curve", "black_vol_surface", "spot"},
    )

    def test_fx_candidate(self, registry):
        new = _new_routes(registry, "monte_carlo", self.FX_IR, pricing_plan=self.FX_PLAN)
        assert "monte_carlo_fx_vanilla" in new

    def test_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_fx_vanilla"][0]
        new_prims = resolve_route_primitives(spec, self.FX_IR)
        expected_prims = {
            ("trellis.models.fx_vanilla", "price_fx_vanilla_monte_carlo", "route_helper"),
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
            model_family="equity_diffusion",
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

    def test_vanilla_equity_transform_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "transform_fft"][0]
        ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        )
        assert resolve_route_notes(spec, ir) == ()
        assert resolve_route_adapters(spec, ir) == ()

    def test_stochastic_vol_transform_primitives_fall_back_to_raw_kernels(self, registry):
        spec = [r for r in registry.routes if r.id == "transform_fft"][0]
        ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="stochastic_volatility",
        )
        new_prims = resolve_route_primitives(spec, ir)
        expected_prims = {
            ("trellis.models.transforms.fft_pricer", "fft_price", "transform_pricer"),
            ("trellis.models.transforms.cos_method", "cos_price", "transform_pricer"),
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

    def test_vanilla_equity_pde_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
        )
        assert resolve_route_notes(spec, ir) == ()
        assert resolve_route_adapters(spec, ir) == ()

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

    def test_holder_max_equity_pde_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        ir = ProductIR(
            instrument="american_option",
            payoff_family="vanilla_option",
            exercise_style="bermudan",
            model_family="equity_diffusion",
        )
        assert resolve_route_notes(spec, ir) == ()
        assert resolve_route_adapters(spec, ir) == ()

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

    def test_callable_bond_pde_helper_route_is_thin(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_fixed_income",
            exercise_style="issuer_call",
            model_family="interest_rate",
        )
        assert resolve_route_notes(spec, ir) == ()
        assert resolve_route_adapters(spec, ir) == ()

    def test_pde_admissibility_hydrates_operator_and_event_contracts(self, registry):
        vanilla = find_route_by_id("vanilla_equity_theta_pde", registry)
        generic = find_route_by_id("pde_theta_1d", registry)
        transform = find_route_by_id("transform_fft", registry)

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
        assert transform is not None
        assert transform.admissibility.supported_control_styles == ("identity",)
        assert transform.admissibility.supported_state_tags == ("terminal_markov",)
        assert transform.admissibility.supported_characteristic_families == (
            "gbm_log_spot",
            "heston_log_spot",
        )

    def test_transform_route_admissibility_accepts_european_holder_control_surface(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        spec = find_route_by_id("transform_fft", registry)
        assert spec is not None

        contract = make_vanilla_option_contract(
            description="European call on SPX priced with transforms",
            underliers=("SPX",),
            observation_schedule=("2026-06-20",),
            preferred_method="fft_pricing",
        )
        blueprint = compile_semantic_contract(contract, preferred_method="fft_pricing")

        decision = evaluate_route_admissibility(spec, semantic_blueprint=blueprint)

        assert decision.ok

    def test_transform_route_admissibility_requires_lowered_transform_family_contract(self, registry):
        from dataclasses import replace

        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        spec = find_route_by_id("transform_fft", registry)
        assert spec is not None

        contract = make_vanilla_option_contract(
            description="European call on SPX priced with transforms",
            underliers=("SPX",),
            observation_schedule=("2026-06-20",),
            preferred_method="fft_pricing",
        )
        blueprint = compile_semantic_contract(contract, preferred_method="fft_pricing")
        lowered_decision = evaluate_route_admissibility(spec, semantic_blueprint=blueprint)
        assert lowered_decision.ok

        raw_blueprint = replace(
            blueprint,
            dsl_lowering=replace(blueprint.dsl_lowering, family_ir=None),
        )
        raw_decision = evaluate_route_admissibility(spec, semantic_blueprint=raw_blueprint)

        assert not raw_decision.ok
        assert "unsupported_control_style:holder_max" in raw_decision.failures

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

    def test_callable_bond_pde_route_is_filtered_when_product_ir_locks_rate_lattice(self, registry):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_callable_bond_contract
        from trellis.agent.quant import select_pricing_method_for_product_ir

        spec = find_route_by_id("pde_theta_1d", registry)
        assert spec is not None

        contract = make_callable_bond_contract(
            description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            observation_schedule=("2026-01-15", "2027-01-15"),
            preferred_method="pde_solver",
        )
        bp = compile_semantic_contract(contract, preferred_method="pde_solver")
        pricing_plan = select_pricing_method_for_product_ir(
            bp.product_ir,
            preferred_method="pde_solver",
        )

        decision = evaluate_route_capability_match(spec, bp.product_ir)
        candidates = match_candidate_routes(
            registry,
            "pde_solver",
            bp.product_ir,
            pricing_plan=pricing_plan,
        )

        assert bp.dsl_lowering.route_family == "pde_solver"
        assert "pde_solver" in bp.product_ir.route_families
        assert not decision.ok
        assert "schedule_dependence_unsupported" in decision.failures
        assert "state_dependence_unsupported:schedule_dependent" in decision.failures
        assert candidates == ()

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

    def test_credit_and_copula_routes_are_thin_backend_bindings(self, registry):
        for route_id in (
            "credit_default_swap",
            "credit_basket_nth_to_default",
            "copula_loss_distribution",
        ):
            spec = find_route_by_id(route_id, registry)
            assert spec is not None
            assert spec.adapters == ()
            assert spec.notes == ()

    def test_waterfall_primitives(self, registry):
        spec = [r for r in registry.routes if r.id == "waterfall_cashflows"][0]
        new_prims = resolve_route_primitives(spec, None)
        expected_prims = {
            ("trellis.models.cashflow_engine.waterfall", "Waterfall", "cashflow_engine"),
            ("trellis.models.cashflow_engine.waterfall", "Tranche", "cashflow_engine"),
        }
        assert _prim_set(new_prims) == expected_prims

    def test_waterfall_has_no_notes(self, registry):
        spec = [r for r in registry.routes if r.id == "waterfall_cashflows"][0]
        assert spec.notes == ()


# ---------------------------------------------------------------------------
# Engine family coverage
# ---------------------------------------------------------------------------

class TestEngineFamilyCoverage:
    EXPECTED = {
        "equity_quanto": "analytical",
        "credit_default_swap": "analytical",
        "credit_basket_nth_to_default": "analytical",
        "correlated_basket_monte_carlo": "monte_carlo",
        "exercise_monte_carlo": "exercise",
        "monte_carlo_paths": "monte_carlo",
        "monte_carlo_fx_vanilla": "monte_carlo",
        "local_vol_monte_carlo": "monte_carlo",
        "qmc_sobol_paths": "qmc",
        "exercise_lattice": "lattice",
        "rate_tree_backward_induction": "lattice",
        "analytical_black76": "analytical",
        # QUA-915: collapsed ZCB-option family takes analytical as its
        # umbrella engine_family; the lattice helper is reached through
        # the rate_tree when-clause and resolve_route_family's
        # conditional_route_family override.
        "short_rate_bond_option": "analytical",
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


# ---------------------------------------------------------------------------
# Route-match parity harness (QUA-907) — shared gate for Phase 1 rewrites
# ---------------------------------------------------------------------------

class TestRouteMatchParityHarness:
    """Smoke tests for :func:`route_parity.assert_route_match_parity`.

    Every Phase 1 route rewrite under ``QUA-887`` must call the helper with
    its own fixtures. These tests exercise the harness against the canonical
    registry to guarantee the trivial parity path behaves correctly and that
    the structural asserts engage on real divergence.
    """

    @staticmethod
    def _basket_fixture() -> tuple[ProductIR, PricingPlan]:
        product_ir = ProductIR(
            instrument="basket_option",
            payoff_family="basket_path_payoff",
            payoff_traits=("ranked_observation",),
        )
        pricing_plan = _make_plan("monte_carlo")
        return product_ir, pricing_plan

    def test_trivial_noop_passes_parity(self):
        from tests.test_agent.route_parity import assert_route_match_parity

        fixtures = [self._basket_fixture()]
        # Empty dict == "change nothing". Must be a trivial pass.
        assert_route_match_parity("correlated_basket_monte_carlo", {}, fixtures)

    def test_equivalent_explicit_match_clause_passes_parity(self, registry):
        from tests.test_agent.route_parity import assert_route_match_parity

        spec = find_route_by_id("correlated_basket_monte_carlo", registry)
        assert spec is not None
        # Re-assert the current match clause with the same values. This is
        # still trivial parity but exercises every supported key in the
        # helper's parser, not just the no-op short-circuit.
        new_match_clause = {
            "methods": list(spec.match_methods),
            "payoff_family": list(spec.match_payoff_family or ()),
            "payoff_traits": list(spec.match_payoff_traits or ()),
        }
        fixtures = [self._basket_fixture()]
        assert_route_match_parity(
            "correlated_basket_monte_carlo",
            new_match_clause,
            fixtures,
        )

    def test_multiple_fixtures_all_checked(self):
        from tests.test_agent.route_parity import assert_route_match_parity

        # Same shape twice plus a second method the route also matches (qmc).
        # Under a no-op rewrite both must still pass.
        basket_mc = self._basket_fixture()
        basket_qmc = (
            ProductIR(
                instrument="basket_option",
                payoff_family="basket_path_payoff",
                payoff_traits=("ranked_observation",),
            ),
            _make_plan("qmc"),
        )
        assert_route_match_parity(
            "correlated_basket_monte_carlo",
            {},
            [basket_mc, basket_qmc],
        )

    def test_unknown_match_key_raises_value_error(self):
        from tests.test_agent.route_parity import assert_route_match_parity

        with pytest.raises(ValueError, match="unsupported match-clause keys"):
            assert_route_match_parity(
                "correlated_basket_monte_carlo",
                {"state_tags": ["schedule_state"]},
                [self._basket_fixture()],
            )

    def test_unknown_route_id_raises_lookup_error(self):
        from tests.test_agent.route_parity import assert_route_match_parity

        with pytest.raises(LookupError, match="not found"):
            assert_route_match_parity(
                "nonexistent_route_id_for_parity_test",
                {},
                [self._basket_fixture()],
            )

    def test_detects_match_clause_drift_as_assertion_error(self):
        from tests.test_agent.route_parity import assert_route_match_parity

        # Replace the real match clause with one that cannot match the
        # basket fixture (methods narrowed to pde_solver only). The harness
        # must flag this as a parity divergence, not silently pass. The
        # length-mismatch branch fires because the variant registry drops
        # ``correlated_basket_monte_carlo`` from the basket ranking entirely.
        with pytest.raises(
            AssertionError,
            match=r"ranked-list length mismatch",
        ):
            assert_route_match_parity(
                "correlated_basket_monte_carlo",
                {"methods": ["pde_solver"]},
                [self._basket_fixture()],
            )

    def test_detects_primitive_drift_as_assertion_error(self):
        from tests.test_agent.route_parity import assert_route_match_parity

        # Broaden the match clause with an extra payoff family that forces
        # a basket fixture onto a different rank ordering (by accepting the
        # vanilla family while still matching basket). The variant ranking
        # contains the same routes but changes the top plan's primitives
        # for a fixture that now matches on two clauses. This exercises the
        # full-fingerprint divergence branch instead of the length branch.
        fixtures = [
            (
                ProductIR(
                    instrument="vanilla_option",
                    payoff_family="vanilla_option",
                    exercise_style="european",
                ),
                _make_plan("monte_carlo"),
            )
        ]
        with pytest.raises(
            AssertionError,
            match=r"Route parity divergence|ranked-list length mismatch",
        ):
            assert_route_match_parity(
                "correlated_basket_monte_carlo",
                {
                    "methods": ["monte_carlo", "qmc"],
                    "payoff_family": ["basket_path_payoff", "vanilla_option"],
                    "payoff_traits": ["ranked_observation"],
                },
                fixtures,
            )
