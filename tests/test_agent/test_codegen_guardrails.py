"""Tests for generation plans and import validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trellis.agent.codegen_guardrails import (
    GenerationPlan,
    PrimitiveRef,
    build_generation_plan,
    rank_primitive_routes,
    render_generation_plan,
    render_review_contract_card,
    render_generation_route_card,
    render_semantic_repair_card,
    sanitize_generated_source,
    validate_generated_imports,
)
from trellis.agent.platform_requests import compile_build_request
from trellis.agent.quant import PricingPlan
from trellis.agent.knowledge.schema import ProductIR


VALID_SOURCE = """\
from trellis.core.date_utils import generate_schedule
from trellis.core.market_state import MarketState
from trellis.core.types import Frequency
from trellis.models.black import black76_call
"""

UNAPPROVED_SOURCE = """\
from trellis.models.processes.heston import Heston
"""

INVALID_SYMBOL_SOURCE = """\
from trellis.models.black import not_a_real_symbol
"""

AGENT_IMPORT_SOURCE = """\
from trellis.instruments._agent.europeanoptionanalytical import price
"""

AGENT_MODULE_IMPORT_SOURCE = """\
import trellis.instruments._agent.europeanoptionanalytical
"""

CONTROL_TIMELINE_SOURCE = """\
from trellis.models.trees.control import (
    build_exercise_timeline_from_dates,
    build_payment_timeline,
)
"""


def _analytical_plan():
    from trellis.agent.knowledge.schema import ProductIR

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface"},
        model_to_build="swaption",
        reasoning="test",
    )
    product_ir = ProductIR(
        instrument="swaption",
        payoff_family="swaption",
        exercise_style="european",
        model_family="interest_rate",
        schedule_dependence=True,
    )
    return build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="swaption",
        inspected_modules=("trellis.instruments.cap", "trellis.models.black"),
        product_ir=product_ir,
    )


def test_generation_plan_includes_common_modules_and_targets():
    plan = _analytical_plan()
    assert "trellis.core.market_state" in plan.approved_modules
    assert "trellis.models.black" in plan.approved_modules
    assert "tests/test_agent/test_build_loop.py" in plan.proposed_tests
    assert plan.repo_revision
    assert plan.symbol_map is not None
    assert plan.package_map is not None
    assert plan.test_map is not None


def test_generation_plan_carries_quanto_primitive_binding_identity_without_helpers():
    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        model="claude-sonnet-4-6",
    )

    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.backend_binding_id == (
        "trellis.models.black.black76_call"
    )
    assert compiled.generation_plan.backend_engine_family == "analytical"
    assert compiled.generation_plan.backend_route_family == "analytical"
    assert compiled.generation_plan.backend_exact_target_refs == (
        "trellis.models.black.black76_call",
        "trellis.models.black.black76_put",
    )
    assert compiled.generation_plan.backend_helper_refs == ()
    assert compiled.generation_plan.backend_compatibility_alias_policy == "internal_only"


def test_rank_primitive_routes_prefers_binding_spec_primitives_when_route_card_is_stale(monkeypatch):
    from trellis.agent import backend_bindings as backend_bindings_module
    from trellis.agent import route_registry as route_registry_module
    from trellis.agent import route_scorer as route_scorer_module

    stale_helper = PrimitiveRef("trellis.models.stale", "stale_helper", "route_helper")
    fresh_helper = PrimitiveRef("trellis.models.synthetic", "fresh_helper", "route_helper")
    spec = SimpleNamespace(
        id="synthetic_route",
        engine_family="analytical",
        route_family="analytical",
        score_hints={},
        aliases=(),
    )

    monkeypatch.setattr(
        route_registry_module,
        "load_route_registry",
        lambda: SimpleNamespace(routes=(spec,)),
    )
    monkeypatch.setattr(
        route_registry_module,
        "match_candidate_routes",
        lambda registry, method, product_ir, pricing_plan=None: (spec,),
    )
    monkeypatch.setattr(
        route_registry_module,
        "resolve_route_primitives",
        lambda spec, product_ir, binding_spec=None, method=None: (stale_helper,),
    )
    monkeypatch.setattr(route_registry_module, "resolve_route_adapters", lambda spec, product_ir, method=None: ())
    monkeypatch.setattr(route_registry_module, "resolve_route_notes", lambda spec, product_ir, method=None: ())
    monkeypatch.setattr(
        route_registry_module,
        "resolve_route_family",
        lambda spec, product_ir, binding_spec=None, method=None: "analytical",
    )
    monkeypatch.setattr(
        backend_bindings_module,
        "load_backend_binding_catalog",
        lambda registry=None: object(),
    )
    monkeypatch.setattr(
        backend_bindings_module,
        "resolve_backend_binding_by_route_id",
        lambda route_id, product_ir=None, primitive_plan=None, catalog=None, method=None: SimpleNamespace(
            primitives=(fresh_helper,),
            binding_id="trellis.models.synthetic.fresh_helper",
            aliases=(),
            exact_target_refs=("trellis.models.synthetic.fresh_helper",),
            helper_refs=("trellis.models.synthetic.fresh_helper",),
            pricing_kernel_refs=(),
            schedule_builder_refs=(),
            cashflow_engine_refs=(),
            market_binding_refs=(),
            compatibility_alias_policy="operator_visible",
            engine_family="analytical",
            route_family="analytical",
        ),
    )

    class _FakeScorer:
        def __init__(self, registry):
            self.registry = registry

        def score_route(self, ctx):
            return SimpleNamespace(final_score=1.0)

    monkeypatch.setattr(route_scorer_module, "RouteScorer", _FakeScorer)

    ranked = rank_primitive_routes(
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.synthetic"],
            required_market_data=set(),
            model_to_build="synthetic",
            reasoning="test",
        ),
        product_ir=None,
    )

    assert ranked[0].primitives == (fresh_helper,)
    assert ranked[0].backend_binding_id == "trellis.models.synthetic.fresh_helper"


def test_validate_generated_imports_accepts_valid_code():
    report = validate_generated_imports(VALID_SOURCE, _analytical_plan())
    assert report.ok


def test_validate_generated_imports_rejects_unapproved_module():
    report = validate_generated_imports(UNAPPROVED_SOURCE, _analytical_plan())
    assert not report.ok
    assert any("unapproved Trellis module" in error for error in report.errors)


def test_validate_generated_imports_rejects_invalid_symbol():
    report = validate_generated_imports(INVALID_SYMBOL_SOURCE, _analytical_plan())
    assert not report.ok
    assert any("not exported" in error for error in report.errors)


def test_validate_generated_imports_guides_invalid_sobol_class_alias():
    source = "from trellis.models.monte_carlo.schemes import SobolNormals\n"
    plan = GenerationPlan(
        method="monte_carlo",
        instrument_type="autocallable",
        inspected_modules=("trellis.models.monte_carlo.schemes",),
        approved_modules=("trellis.models.monte_carlo.schemes",),
        symbols_to_reuse=(),
        proposed_tests=(),
    )

    report = validate_generated_imports(source, plan)

    assert not report.ok
    assert any("sobol_normals" in error for error in report.errors)


def test_validate_generated_imports_rejects_admitted_agent_from_import():
    report = validate_generated_imports(AGENT_IMPORT_SOURCE, _analytical_plan())
    assert not report.ok
    assert any(
        "trellis.instruments._agent" in error and "admitted" in error.lower()
        for error in report.errors
    )


def test_validate_generated_imports_rejects_admitted_agent_module_import():
    report = validate_generated_imports(AGENT_MODULE_IMPORT_SOURCE, _analytical_plan())
    assert not report.ok
    assert any("trellis.instruments._agent" in error for error in report.errors)


def test_validate_generated_imports_accepts_control_timeline_facade_exports():
    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.control"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="callable_bond",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="callable_bond",
        inspected_modules=("trellis.models.trees.control",),
    )

    report = validate_generated_imports(CONTROL_TIMELINE_SOURCE, plan)

    assert report.ok


def test_analytical_generation_plan_approves_gaussian_probability_and_scalar_root():
    probability_module = "trellis.models.analytical.support.probability"
    scalar_root_module = "trellis.models.calibration.solve_request"
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=[probability_module],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="critical_state_option",
            reasoning="test",
        ),
        instrument_type="chooser_option",
        inspected_modules=(probability_module,),
        product_ir=ProductIR(
            instrument="chooser_option",
            payoff_family="chooser_option",
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )

    assert scalar_root_module in plan.approved_modules
    report = validate_generated_imports(
        "from trellis.models.analytical.support.probability import "
        "standard_normal_cdf, bivariate_standard_normal_cdf\n"
        "from trellis.models.calibration.solve_request import "
        "ObjectiveBundle, SolveBounds, SolveRequest, execute_solve_request\n",
        plan,
    )
    assert report.ok


def test_barrier_family_support_approves_shared_barrier_primitives():
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="barrier_option",
            reasoning="test",
        ),
        instrument_type="barrier_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("double_barrier",),
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )

    assert "trellis.models.analytical.support.barriers" in plan.approved_modules
    assert "trellis.models.single_barrier_option" in plan.approved_modules
    assert "trellis.models.double_barrier_option" in plan.approved_modules
    report = validate_generated_imports(
        "from trellis.models.single_barrier_option import "
        "SingleBarrierPDEConfig, SingleBarrierMonteCarloConfig, "
        "price_single_barrier_option_pde_result, "
        "price_single_barrier_option_monte_carlo_result\n"
        "from trellis.models.analytical.support.barriers import "
        "terminal_double_barrier_payoff, double_barrier_state_payoff\n"
        "from trellis.models.double_barrier_option import "
        "DoubleBarrierPDEConfig, DoubleBarrierMonteCarloConfig, "
        "price_double_barrier_option_pde_result, "
        "price_double_barrier_option_monte_carlo_result\n",
        plan,
    )
    assert report.ok


def test_heston_family_support_approves_adi_diagnostics():
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=[],
            required_market_data={"discount_curve", "model_parameters"},
            model_to_build="heston_option",
            reasoning="test",
        ),
        instrument_type="heston_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="heston_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="stochastic_volatility",
        ),
    )

    assert "trellis.models.pde.heston_adi" in plan.approved_modules
    report = validate_generated_imports(
        "from trellis.models.pde.heston_adi import "
        "HestonAdiPDEConfig, resolve_heston_adi_pde_inputs, "
        "price_heston_option_adi_pde_result\n",
        plan,
    )
    assert report.ok


def test_american_put_family_support_approves_equity_helper_surfaces():
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="american_put",
            reasoning="test",
        ),
        instrument_type="american_put",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="american_put",
            payoff_family="vanilla_option",
            exercise_style="american",
            model_family="equity_diffusion",
        ),
    )

    assert "trellis.models.equity_option_pde" in plan.approved_modules
    assert "trellis.models.equity_option_tree" in plan.approved_modules
    assert "trellis.models.equity_option_monte_carlo" in plan.approved_modules
    report = validate_generated_imports(
        "from trellis.models.equity_option_pde import price_event_aware_equity_option_pde\n"
        "from trellis.models.equity_option_tree import price_vanilla_equity_option_tree\n"
        "from trellis.models.equity_option_monte_carlo import "
        "price_american_equity_option_lsm_monte_carlo\n",
        plan,
    )
    assert report.ok


def test_european_option_family_support_approves_cev_helper_surfaces():
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=[],
            required_market_data={"discount_curve"},
            model_to_build="european_option",
            reasoning="test",
        ),
        instrument_type="european_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )

    assert "trellis.models.equity_option_pde" in plan.approved_modules
    assert "trellis.models.equity_option_tree" in plan.approved_modules
    report = validate_generated_imports(
        "from trellis.models.equity_option_pde import price_cev_option_pde\n"
        "from trellis.models.equity_option_tree import price_cev_option_tree\n",
        plan,
    )
    assert report.ok


def test_path_dependent_family_support_approves_asian_and_lookback_primitives():
    asian_plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="asian_option",
            reasoning="test",
        ),
        instrument_type="asian_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="asian_option",
            payoff_family="asian_option",
            payoff_traits=("asian", "arithmetic_average", "path_dependent"),
            exercise_style="european",
            model_family="equity_diffusion",
            schedule_dependence=True,
        ),
    )
    lookback_plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="lookback_option",
            reasoning="test",
        ),
        instrument_type="lookback_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=(
                "lookback",
                "path_dependent",
                "fixed_strike",
                "continuous_monitoring",
            ),
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )
    analytical_lookback_plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="lookback_option",
            reasoning="test",
        ),
        instrument_type="lookback_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=(
                "lookback",
                "path_dependent",
                "fixed_strike",
                "continuous_monitoring",
            ),
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )

    assert "trellis.models.asian_option" not in asian_plan.approved_modules
    assert {
        "trellis.models.observation_aggregation",
        "trellis.models.analytical.support.lognormal_moments",
        "trellis.models.resolution.single_state_diffusion",
        "trellis.models.monte_carlo.engine",
        "trellis.models.monte_carlo.path_state",
        "trellis.models.processes.gbm",
    } <= set(asian_plan.approved_modules)
    assert "trellis.models.lookback_option" not in lookback_plan.approved_modules
    assert lookback_plan.blocker_report is None
    assert {
        "trellis.models.analytical.support",
        "trellis.models.resolution.single_state_diffusion",
        "trellis.models.monte_carlo.engine",
        "trellis.models.monte_carlo.path_state",
        "trellis.models.monte_carlo.transition_state",
        "trellis.models.processes.gbm",
        "trellis.core.differentiable",
    } <= set(lookback_plan.approved_modules)
    assert "trellis.models.analytical.equity_exotics" not in lookback_plan.approved_modules
    assert (
        "trellis.models.analytical.equity_exotics"
        in analytical_lookback_plan.approved_modules
    )
    asian_report = validate_generated_imports(
        "from trellis.models.observation_aggregation import "
        "WeightedObservationContract, weighted_observation_payoff\n"
        "from trellis.models.monte_carlo.engine import MonteCarloEngine\n"
        "from trellis.models.monte_carlo.path_state import StateAwarePayoff\n"
        "from trellis.models.processes.gbm import GBM\n",
        asian_plan,
    )
    wrapper_report = validate_generated_imports(
        "from trellis.models.asian_option import "
        "price_arithmetic_asian_option_monte_carlo\n",
        asian_plan,
    )
    lookback_report = validate_generated_imports(
        "from trellis.models.monte_carlo.transition_state import "
        "ConditionalBridgeExtremumContract, build_conditional_bridge_extremum_reducer\n"
        "from trellis.models.monte_carlo.engine import MonteCarloEngine\n"
        "from trellis.models.monte_carlo.path_state import "
        "MonteCarloPathRequirement, StateAwarePayoff\n"
        "from trellis.models.processes.gbm import GBM\n",
        lookback_plan,
    )
    lookback_wrapper_report = validate_generated_imports(
        "from trellis.models.lookback_option import "
        "price_equity_fixed_lookback_option_monte_carlo\n",
        lookback_plan,
    )
    lookback_analytical_report = validate_generated_imports(
        "from trellis.models.analytical.equity_exotics import "
        "price_equity_fixed_lookback_option_analytical\n",
        lookback_plan,
    )

    assert asian_report.ok
    assert not wrapper_report.ok
    assert lookback_report.ok
    assert not lookback_wrapper_report.ok
    assert not lookback_analytical_report.ok

    sparse_lookback_plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="lookback_option",
            reasoning="ambiguous sparse lookback contract",
        ),
        instrument_type="lookback_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=("lookback", "path_dependent"),
            exercise_style="european",
            model_family="equity_diffusion",
        ),
    )
    assert not any(
        "resolve_scalar_diffusion_market_inputs" in ref
        or "ConditionalBridgeExtremumContract" in ref
        or "build_conditional_bridge_extremum_reducer" in ref
        for ref in sparse_lookback_plan.lane_exact_binding_refs
    )
    assert sparse_lookback_plan.blocker_report is not None
    assert sparse_lookback_plan.blocker_report.should_block
    assert {
        "lookback_strike_semantics",
        "lookback_monitoring_semantics",
    } <= {blocker.id for blocker in sparse_lookback_plan.blocker_report.blockers}


@pytest.mark.parametrize(
    ("payoff_traits", "model_family", "expected_blocker"),
    [
        (
            ("lookback", "path_dependent", "floating_strike", "continuous_monitoring"),
            "equity_diffusion",
            "lookback_floating_strike",
        ),
        (
            ("lookback", "path_dependent", "fixed_strike", "discrete_monitoring"),
            "equity_diffusion",
            "lookback_discrete_monitoring",
        ),
        (
            ("lookback", "path_dependent", "fixed_strike", "continuous_monitoring"),
            "stochastic_volatility",
            "lookback_transition_dynamics",
        ),
    ],
)
def test_unsupported_lookback_contracts_block_before_generation(
    payoff_traits,
    model_family,
    expected_blocker,
):
    from trellis.agent.build_gate import evaluate_pre_generation_gate

    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=[],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="lookback_option",
            reasoning="unsupported lookback contract",
        ),
        instrument_type="lookback_option",
        inspected_modules=(),
        product_ir=ProductIR(
            instrument="lookback_option",
            payoff_family="lookback_option",
            payoff_traits=payoff_traits,
            exercise_style="european",
            state_dependence="path_dependent",
            model_family=model_family,
        ),
    )

    assert plan.blocker_report is not None
    assert expected_blocker in {
        blocker.id for blocker in plan.blocker_report.blockers
    }
    decision = evaluate_pre_generation_gate(None, plan)
    assert decision.decision == "block"
    assert expected_blocker in decision.reason


def test_autocallable_generation_plan_approves_event_helper_surface():
    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.autocallable"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="autocallable",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="autocallable",
        inspected_modules=("trellis.models.autocallable",),
    )

    assert "trellis.models.autocallable" in plan.approved_modules
    report = validate_generated_imports(
        "from trellis.models.autocallable import "
        "AutocallableMonteCarloConfig, price_autocallable_monte_carlo_result\n",
        plan,
    )
    assert report.ok


def test_qmc_generation_plan_approves_qmc_family_modules():
    pricing_plan = PricingPlan(
        method="qmc",
        method_modules=["trellis.models.qmc"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="autocallable",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="autocallable",
        inspected_modules=("trellis.models.qmc",),
    )

    assert "trellis.models.qmc" in plan.approved_modules
    assert "sobol_normals" in plan.symbols_to_reuse
    assert "brownian_bridge" in plan.symbols_to_reuse


def test_basket_route_card_stays_backend_binding_only():
    description = (
        "Price a Himalaya-style basket on AAPL, MSFT, NVDA, and AMZN. "
        "Observe monthly on 2026-04-01, 2026-05-01, and 2026-06-01. "
        "At each observation, choose the best performer among the remaining names, remove it, "
        "lock that simple return, and settle the average locked returns once at maturity. "
        "Use discount curve, spot, vol surface, and correlation."
    )
    compiled = compile_build_request(
        description,
        instrument_type="basket_option",
        model="claude-sonnet-4-6",
    )

    card = render_generation_route_card(compiled.generation_plan)

    assert "correlated_basket_monte_carlo" in card
    assert "resolve_basket_semantics" in card
    assert "price_ranked_observation_basket_monte_carlo" in card
    assert "Required adapters:" not in card
    assert "Backend notes:" not in card
    assert "Parse `spec.underlyings` into a Python list of ticker strings" not in card
    assert "delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter" not in card
    assert "HimalayaBasketSpec" not in card
    assert "CorrelatedGBM" not in card
    assert "trellis.models.processes.correlated_gbm" not in card
    assert "trellis.models.basket" not in card
    assert "trellis.models.ranked_observation" not in card
    assert "trellis.models.payoff" not in card


def test_generation_route_card_for_generic_monte_carlo_stays_backend_binding_only():
    compiled = compile_build_request(
        "Path-dependent note with monthly averaging under Monte Carlo",
        instrument_type="structured_note",
        preferred_method="monte_carlo",
    )

    card = render_generation_route_card(compiled.generation_plan)

    assert "monte_carlo_paths" in card
    assert "price_event_aware_monte_carlo" in card
    assert "Required adapters:" not in card
    assert "Backend notes:" not in card
    assert "build_payoff_vector_from_paths" not in card
    assert "Prefer existing MC simulation helpers over bespoke path loops." not in card


def test_generation_plan_renders_compiled_semantic_and_validation_boundary():
    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        model="claude-sonnet-4-6",
    )

    text = render_generation_plan(compiled.generation_plan)

    assert "- Semantic contract: `quanto_option`" in text
    assert "instrument=`quanto_option`" in text
    assert "payoff=`vanilla_option`" in text
    assert "- Valuation context:" in text
    assert "market_source=`unbound_market_snapshot`" in text
    assert "- Lane boundary:" in text
    assert "family=`analytical`" in text
    assert "- Lane obligations:" in text
    assert "Plan kind: `exact_target_binding`" in text
    assert "- Lowering boundary:" in text
    assert "route_alias=`equity_quanto`" not in text
    assert "expr=`ContractAtom`" in text
    assert "targets=`trellis.models.resolution.quanto.resolve_quanto_inputs`" in text
    assert "trellis.models.analytical.support.quanto_adjusted_forward" in text
    assert "trellis.models.black.black76_call" in text
    assert "price_quanto_option_analytical_from_market_state" not in text
    assert "- Validation contract:" in text
    assert "bundle=`analytical:quanto_option`" in text
    assert "check_non_negativity" in text
    assert "check_quanto_required_inputs" in text
    assert "check_quanto_cross_currency_semantics" in text
    assert "- Route authority:" in text
    assert "authority=`exact_backend_fit`" in text


def test_generation_route_card_keeps_lane_obligations_ahead_of_route_authority():
    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        model="claude-sonnet-4-6",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "- Lane obligations:" in text
    assert "- Route authority:" in text
    assert "- Backend binding:" in text
    assert text.index("- Lane obligations:") < text.index("- Route authority:")
    assert text.index("- Route authority:") < text.index("- Backend binding:")
    assert "Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan." in text


def test_generation_route_card_for_fx_analytical_composes_resolver_and_kernel():
    compiled = compile_build_request(
        "FX vanilla option: Garman-Kohlhagen vs MC",
        instrument_type="european_option",
        preferred_method="analytical",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "resolve_fx_vanilla_inputs" in text
    assert "garman_kohlhagen_price_raw" in text
    assert "price_fx_vanilla_analytical" not in text
    assert "map_fx_spot_and_curves_to_garman_kohlhagen_inputs" not in text


def test_requested_method_augmentation_adds_route_family_hints_for_stale_decompositions():
    from trellis.agent.codegen_guardrails import _augment_product_ir_for_requested_method
    from trellis.agent.knowledge.decompose import decompose_to_ir

    product_ir = decompose_to_ir(
        "European equity call option",
        instrument_type="european_option",
    )
    assert "pde_solver" in product_ir.route_families
    assert "analytical" not in product_ir.route_families

    augmented = _augment_product_ir_for_requested_method(
        product_ir,
        preferred_method="analytical",
    )

    assert augmented is not None
    assert "analytical" in augmented.route_families
    assert "analytical" in augmented.candidate_engine_families
    assert "pde_solver" in augmented.route_families


def test_requested_method_augmentation_does_not_invent_route_family_when_ir_has_none():
    from trellis.agent.codegen_guardrails import _augment_product_ir_for_requested_method
    from trellis.agent.knowledge.schema import ProductIR

    product_ir = ProductIR(
        instrument="barrier_option",
        payoff_family="barrier_option",
        exercise_style="european",
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
    )

    augmented = _augment_product_ir_for_requested_method(
        product_ir,
        preferred_method="analytical",
    )

    assert augmented is not None
    assert augmented.route_families == ()
    assert "analytical" in augmented.candidate_engine_families


def test_requested_method_augmentation_adds_exercise_family_for_non_european_mc_routes():
    from trellis.agent.codegen_guardrails import _augment_product_ir_for_requested_method
    from trellis.agent.knowledge.schema import ProductIR

    product_ir = ProductIR(
        instrument="barrier_option",
        payoff_family="composite_option",
        payoff_traits=("american", "asian", "barrier", "early_exercise"),
        exercise_style="american",
        state_dependence="path_dependent",
        model_family="stochastic_volatility",
        candidate_engine_families=("monte_carlo",),
        route_families=("barrier_option",),
        supported=False,
    )

    augmented = _augment_product_ir_for_requested_method(
        product_ir,
        preferred_method="monte_carlo",
    )

    assert augmented is not None
    assert "exercise" in augmented.candidate_engine_families
    assert "exercise" in augmented.route_families
    assert "barrier_option" in augmented.route_families


def test_generation_route_card_for_swaption_analytical_stays_helper_only():
    compiled = compile_build_request(
        "European swaption on fixed-for-float swap",
        instrument_type="swaption",
        preferred_method="analytical",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "price_swaption_black76" in text
    assert "reuse_checked_in_rate_style_swaption_helper" not in text
    assert "Hull-White-implied Black vol" not in text
    assert "Backend notes:" not in text


def test_generation_route_card_for_quanto_exposes_primitive_composition_only():
    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        preferred_method="analytical",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "resolve_quanto_inputs" in text
    assert "quanto_adjusted_forward" in text
    assert "black76_call" in text
    assert "black76_put" in text
    assert "Helper authority:" not in text
    assert "price_quanto_option_analytical_from_market_state" not in text
    assert "trellis.models.analytical.quanto" not in text


def test_generation_route_card_for_zcb_option_analytical_stays_helper_only():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.zcb_option"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="zcb_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="zcb_option",
        inspected_modules=("trellis.models.zcb_option",),
        product_ir=decompose_to_ir(
            "ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical",
            instrument_type="zcb_option",
        ),
    )

    text = render_generation_route_card(plan)

    assert "price_zcb_option_jamshidian" in text
    assert "resolve_zcb_option_hw_inputs" not in text
    assert "zcb_option_hw_raw" not in text


def test_generation_route_card_for_zcb_option_tree_stays_helper_only():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.zcb_option_tree"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="zcb_option",
        reasoning="test",
    )

    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="zcb_option",
        inspected_modules=("trellis.models.zcb_option_tree",),
        product_ir=decompose_to_ir(
            "ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical",
            instrument_type="zcb_option",
        ),
    )

    text = render_generation_route_card(plan)

    assert "price_zcb_option_tree" in text
    assert "build_generic_lattice" not in text
    assert "MODEL_REGISTRY" not in text


def test_generation_route_card_for_vanilla_equity_pde_requires_primitive_composition():
    compiled = compile_build_request(
        "European equity call on AAPL",
        instrument_type="european_option",
        preferred_method="pde_solver",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "price_vanilla_equity_option_pde" not in text
    assert "resolve_single_state_diffusion_inputs" in text
    assert "build_event_aware_pde_problem" in text
    assert "solve_event_aware_pde" in text
    assert "interpolate_pde_values" in text


def test_generation_route_card_for_vanilla_equity_fft_stays_helper_only():
    compiled = compile_build_request(
        "European equity call on AAPL via FFT",
        instrument_type="european_option",
        preferred_method="fft_pricing",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "price_vanilla_equity_option_transform" in text
    assert "build_vector_safe_characteristic_function" not in text
    assert "GBM characteristic functions" not in text
    assert "Backend notes:" not in text


def test_semantic_repair_card_for_vanilla_pde_requires_primitive_composition():
    compiled = compile_build_request(
        "European equity call on AAPL",
        instrument_type="european_option",
        preferred_method="pde_solver",
    )

    text = render_semantic_repair_card(compiled.generation_plan)

    assert "price_vanilla_equity_option_pde" not in text
    assert "resolve_single_state_diffusion_inputs" in text
    assert "build_event_aware_pde_problem" in text
    assert "solve_event_aware_pde" in text
    assert "Required adapters:" not in text


def test_review_contract_card_renders_wrapper_route_and_validation_scope():
    compiled = compile_build_request(
        "European equity call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
        model="claude-sonnet-4-6",
    )

    text = render_review_contract_card(compiled.generation_plan)

    assert "## Compiled Route Contract" in text
    assert "bridge=`thin_compatibility_wrapper`" in text
    assert "wrapper=`european_option`" in text
    assert "route_alias=`analytical_black76`" not in text
    assert "bundle=`analytical:european_option`" in text
    assert "- Route authority:" in text
    assert "authority=`exact_backend_fit`" in text
    assert "trellis.models.black" in text


def test_schedule_dependent_route_card_mentions_shared_schedule_builder():
    plan = _analytical_plan()
    card = render_generation_route_card(plan)

    # Under the current positive-filter Black76 match clause a canonical
    # European swaption dispatches to the swaption-specific helper, which
    # owns its schedule construction internally rather than surfacing the
    # generic ``build_payment_timeline`` fallback primitive.
    assert "price_swaption_black76" in card
    assert "Instruction precedence: follow the lane obligations in this card first." in card


def test_pde_route_card_mentions_terminal_array_contract():
    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.pde.theta_method"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.pde.theta_method",),
    )

    card = render_generation_route_card(plan)

    assert "Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)" in card
    assert "BlackScholesOperator(sigma_fn, r_fn)" in card
    assert "Do not pass a callable terminal payoff into `theta_method_1d`" in card
    assert "rannacher_timesteps" in card


def test_american_option_route_card_describes_lattice_algebra_composition():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.trees",),
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "exercise_lattice"
    assert plan.primitive_plan.engine_family == "tree"
    assert plan.primitive_plan.route_family == "equity_tree"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    primitive_modules = {primitive.module for primitive in plan.primitive_plan.primitives}
    assert primitive_symbols == {
        "resolve_single_state_diffusion_inputs",
        "terminal_intrinsic_from_resolved",
        "equity_tree",
        "with_control",
        "compile_lattice_recipe",
        "build_lattice",
        "price_on_lattice",
    }
    assert primitive_modules == {
        "trellis.models.resolution.single_state_diffusion",
        "trellis.models.trees.algebra",
    }
    assert "Lane obligations:" in card
    assert "Resolve spot, strike, expiry, option type, and exercise style" in card
    assert "Compile `equity_tree(...)`" in card
    assert "Build the lattice with `build_lattice(...)`" in card
    assert "price the compiled contract with `price_on_lattice(...)`" in card
    assert "build_rate_lattice" not in primitive_symbols
    assert "price_vanilla_equity_option_tree" not in card
    assert "lsm_mc" not in card


def test_fx_monte_carlo_route_card_mentions_fx_rate_scalar_extraction():
    from trellis.agent.codegen_guardrails import clear_generation_plan_cache
    from trellis.agent.knowledge.decompose import decompose_to_ir

    clear_generation_plan_cache()

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine", "trellis.models.processes.gbm"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
        model_to_build="european_option",
        reasoning="FX MC route",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.monte_carlo.engine", "trellis.models.processes.gbm"),
        product_ir=decompose_to_ir("FX option (EURUSD): GK analytical vs MC", instrument_type="european_option"),
    )

    card = render_generation_route_card(plan)

    assert "Lane obligations:" in card
    assert "Lane family: `monte_carlo`" in card
    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "monte_carlo_fx_vanilla"
    primitive_refs = {
        f"{primitive.module}.{primitive.symbol}"
        for primitive in plan.primitive_plan.primitives
    }
    assert {
        "trellis.models.fx_vanilla.resolve_fx_vanilla_inputs",
        "trellis.models.processes.gbm.GBM",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "trellis.models.monte_carlo.path_state.terminal_value_payoff",
        "trellis.models.analytical.terminal_intrinsic",
    } <= primitive_refs
    assert "Resolve scalar FX spot from `market_state.fx_rates[spec.fx_pair].spot`" in card
    assert "price_fx_vanilla_monte_carlo" not in card
    assert "terminal_value_payoff" in card
    assert "FXRate` wrapper" in card


def test_barrier_option_route_card_mentions_grid_operator_and_rannacher():
    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.pde.theta_method"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="barrier_option",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="barrier_option",
        inspected_modules=(
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "pde_theta_1d"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {"Grid", "BlackScholesOperator", "theta_method_1d"} <= primitive_symbols
    assert "rannacher_timesteps" in card


def test_barrier_option_analytical_route_uses_absorbed_black76_helper_binding():
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.analytical.barrier"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="barrier_option",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="barrier_option",
        inspected_modules=("trellis.models.analytical.barrier",),
        product_ir=ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_black76"
    assert plan.primitive_plan.route_family == "analytical"
    primitive_refs = {
        f"{primitive.module}.{primitive.symbol}" for primitive in plan.primitive_plan.primitives
    }
    assert "trellis.models.analytical.barrier.barrier_option_price" in primitive_refs
    assert "instantiating `ResolvedBarrierInputs` in generated adapters" in card
    assert "barrier_option_price" in card


def test_cds_monte_carlo_route_uses_single_name_credit_default_swap_assembly():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="credit_default_swap",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="credit_default_swap",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=decompose_to_ir(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="credit_default_swap",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "credit_default_swap"
    assert plan.primitive_plan.route_family == "event_triggered_two_legged_contract"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert {
        "build_cds_schedule",
        "interval_default_probability",
        "price_cds_monte_carlo",
        "get_numpy",
    } <= primitive_symbols
    assert "MonteCarloEngine" not in primitive_symbols
    assert "GaussianCopula" not in primitive_symbols
    assert "build_cds_schedule" in card
    assert "price_cds_monte_carlo" in card
    assert "spread_quote" in card
    assert "n_paths" in card
    assert "Route family: `credit_default_swap`" in card
    assert "Route family: `event_triggered_two_legged_contract`" not in card
    assert "survival_probability" not in card
    assert "Required adapters:" not in card
    assert "use_credit_curve_hazard_rate_or_survival_probability" not in card
    assert "Do not hard-code n_paths=50000" not in card


def test_cds_analytical_route_card_surfaces_helper_signature_keywords():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="credit_default_swap",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="credit_default_swap",
        inspected_modules=("trellis.models.black",),
        product_ir=decompose_to_ir(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="credit_default_swap",
        ),
    )

    card = render_generation_route_card(plan)

    assert "price_cds_analytical" in card
    assert "build_cds_schedule" in card
    assert "spread_quote" in card
    assert "discount_curve" in card
    assert "survival_probability" not in card
    assert "Required adapters:" not in card
    assert "Do not reinterpret a single-name CDS" not in card


@pytest.mark.parametrize(
    "instrument_type,expected_helper_ref",
    [
        (
            "chooser_option",
            "trellis.models.analytical.equity_exotics.price_equity_chooser_option_analytical",
        ),
        (
            "compound_option",
            "trellis.models.analytical.equity_exotics.price_equity_compound_option_analytical",
        ),
    ],
)
def test_absorbed_black76_structural_exotic_route_uses_exact_helper_binding(
    instrument_type,
    expected_helper_ref,
):
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.analytical.equity_exotics"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build=instrument_type,
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=("trellis.models.analytical.equity_exotics",),
        product_ir=ProductIR(
            instrument=instrument_type,
            payoff_family=instrument_type,
            payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "analytical_black76"
    primitive_refs = {
        f"{primitive.module}.{primitive.symbol}" for primitive in plan.primitive_plan.primitives
    }
    assert expected_helper_ref in primitive_refs
    assert expected_helper_ref.rsplit(".", 1)[-1] in card


def test_nth_to_default_monte_carlo_route_uses_copula_assembly():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.copulas.gaussian"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="nth_to_default",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="nth_to_default",
        inspected_modules=("trellis.models.copulas.gaussian",),
        product_ir=decompose_to_ir(
            "First-to-default basket on five names with Gaussian copula",
            instrument_type="nth_to_default",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "credit_basket_nth_to_default"
    assert plan.primitive_plan.route_family == "nth_to_default"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert "GaussianCopula" in primitive_symbols
    assert "price_nth_to_default_basket" in card
    assert "single-name CDS" not in card
    assert "Required adapters:" not in card


def test_copula_loss_distribution_route_stays_helper_backed():
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="copula",
        method_modules=["trellis.models.copulas.factor"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="cdo",
        reasoning="test",
    )
    plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="cdo",
        inspected_modules=("trellis.models.copulas.factor",),
        product_ir=decompose_to_ir(
            "CDO tranche: Gaussian vs Student-t copula",
            instrument_type="cdo",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == "copula_loss_distribution"
    assert "price_credit_basket_tranche" in card
    assert "Required adapters:" not in card
    assert "Prefer the semantic-facing basket-credit helper" not in card


def test_sanitize_generated_source_strips_single_outer_fence():
    source = """\
```python
from trellis.core.market_state import MarketState

class Demo:
    pass
```
"""

    report = sanitize_generated_source(source)

    assert report.ok
    assert report.source_status == "sanitized"
    assert report.fence_removed
    assert report.fence_language == "python"
    assert report.fence_count == 2
    assert report.raw_source == source
    assert report.sanitized_source == "from trellis.core.market_state import MarketState\n\nclass Demo:\n    pass"


def test_sanitize_generated_source_accepts_raw_python_without_fences():
    source = """\
        from trellis.core.market_state import MarketState

        class Demo:
            pass
    """

    report = sanitize_generated_source(source)

    assert report.ok
    assert report.source_status == "accepted"
    assert not report.fence_removed
    assert report.fence_count == 0
    assert report.sanitized_source == "from trellis.core.market_state import MarketState\n\nclass Demo:\n    pass"


def test_sanitize_generated_source_rejects_ambiguous_fences():
    source = """\
Here is the code:
```python
class Demo:
    pass
```
"""

    report = sanitize_generated_source(source)

    assert not report.ok
    assert report.source_status == "rejected"
    assert "markdown fences" in report.errors[0]
