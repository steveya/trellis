"""Tests for generation plans and import validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trellis.agent.codegen_guardrails import (
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


def test_generation_plan_carries_backend_binding_identity_for_exact_helper_routes():
    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        model="claude-sonnet-4-6",
    )

    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.backend_binding_id == (
        "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    )
    assert compiled.generation_plan.backend_engine_family == "analytical"
    assert compiled.generation_plan.backend_route_family == "analytical"
    assert compiled.generation_plan.backend_exact_target_refs == (
        "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state",
    )
    assert compiled.generation_plan.backend_helper_refs == (
        "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state",
    )
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
    assert "price_quanto_option_analytical_from_market_state" in text
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


def test_generation_route_card_for_fx_exact_helper_stays_helper_only():
    compiled = compile_build_request(
        "FX vanilla option: Garman-Kohlhagen vs MC",
        instrument_type="european_option",
        preferred_method="analytical",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "price_fx_vanilla_analytical" in text
    assert "garman_kohlhagen_price_raw" not in text
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


def test_generation_route_card_for_quanto_exact_helper_stays_helper_only():
    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        preferred_method="analytical",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "price_quanto_option_analytical_from_market_state" in text
    assert "resolve_quanto_inputs" not in text
    assert "trellis.models.analytical.quanto.price_quanto_option_analytical" not in text
    assert "apply_quanto_adjustment_terms" not in text


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


def test_generation_route_card_for_vanilla_equity_pde_stays_helper_only():
    compiled = compile_build_request(
        "European equity call on AAPL",
        instrument_type="european_option",
        preferred_method="pde_solver",
    )

    text = render_generation_route_card(compiled.generation_plan)

    assert "price_vanilla_equity_option_pde" in text
    assert "reuse_checked_in_vanilla_equity_pde_helper" not in text
    assert "Grid + BlackScholesOperator + theta_method_1d" not in text
    assert "Backend notes:" not in text


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


def test_semantic_repair_card_for_helper_backed_pde_route_omits_route_notes_and_adapters():
    compiled = compile_build_request(
        "European equity call on AAPL",
        instrument_type="european_option",
        preferred_method="pde_solver",
    )

    text = render_semantic_repair_card(compiled.generation_plan)

    assert "price_vanilla_equity_option_pde" in text
    assert "Required adapters:" not in text
    assert "Route notes:" not in text


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


def test_american_option_route_card_mentions_equity_tree_helper():
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
    assert primitive_symbols == {"price_vanilla_equity_option_tree"}
    assert primitive_modules == {"trellis.models.equity_option_tree"}
    assert "Lane obligations:" in card
    assert "Resolve a spec-like contract with `spot`, `strike`, `expiry_date`" in card
    assert "price_vanilla_equity_option_tree(market_state, spec_like, model=\"crr\"|\"jarrow_rudd\", n_steps=...)" in card
    assert "build_rate_lattice" not in primitive_symbols
    assert "price_vanilla_equity_option_tree" in card
    assert "build_spot_lattice" in card
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
    assert "trellis.models.fx_vanilla.price_fx_vanilla_monte_carlo" in primitive_refs
    assert "Resolve scalar FX spot from `market_state.fx_rates[spec.fx_pair].spot`" in card
    assert "price_fx_vanilla_monte_carlo" in card
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
    "instrument_type,expected_route,expected_helper_ref",
    [
        (
            "chooser_option",
            "equity_chooser_analytical",
            "trellis.models.analytical.equity_exotics.price_equity_chooser_option_analytical",
        ),
        (
            "compound_option",
            "equity_compound_analytical",
            "trellis.models.analytical.equity_exotics.price_equity_compound_option_analytical",
        ),
    ],
)
def test_nested_composite_route_uses_exact_helper_binding(
    instrument_type,
    expected_route,
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
            payoff_family="composite_option",
            payoff_traits=("discounting", "terminal_markov", "vol_surface_dependence"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        ),
    )

    card = render_generation_route_card(plan)

    assert plan.primitive_plan is not None
    assert plan.primitive_plan.route == expected_route
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
