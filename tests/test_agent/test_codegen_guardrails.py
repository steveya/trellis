"""Tests for generation plans and import validation."""

from __future__ import annotations

from trellis.agent.codegen_guardrails import (
    build_generation_plan,
    render_generation_plan,
    render_review_contract_card,
    render_generation_route_card,
    sanitize_generated_source,
    validate_generated_imports,
)
from trellis.agent.platform_requests import compile_build_request
from trellis.agent.quant import PricingPlan


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

CONTROL_TIMELINE_SOURCE = """\
from trellis.models.trees.control import (
    build_exercise_timeline_from_dates,
    build_payment_timeline,
)
"""


def _analytical_plan():
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface"},
        model_to_build="swaption",
        reasoning="test",
    )
    return build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="swaption",
        inspected_modules=("trellis.instruments.cap", "trellis.models.black"),
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


def test_basket_route_card_calls_helper_directly():
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
    assert "Parse `spec.underlyings` into a Python list of ticker strings" in card
    assert "resolve_basket_semantics" in card
    assert "price_ranked_observation_basket_monte_carlo" in card
    assert "RankedObservationBasketSpec" in card
    assert "Bind the market state with `resolve_basket_semantics(...)`" in card
    assert "delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter" in card
    assert "import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket`" in card
    assert "do not import process primitives directly" in card
    assert "Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline." in card
    assert "CorrelatedGBM" not in card
    assert "trellis.models.processes.correlated_gbm" not in card
    assert "trellis.models.basket" not in card
    assert "trellis.models.ranked_observation" not in card
    assert "trellis.models.payoff" not in card


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
    assert "route=`quanto_adjustment_analytical`" in text
    assert "expr=`ThenExpr`" in text
    assert "price_quanto_option_analytical" in text
    assert "- Validation contract:" in text
    assert "bundle=`analytical:quanto_option`" in text
    assert "check_non_negativity" in text
    assert "quanto_adjustment_applied" in text
    assert "- Route authority:" in text
    assert "authority=`exact_backend_fit`" in text
    assert "canaries=`T105`" in text


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
    assert "route=`analytical_black76`" in text
    assert "bundle=`analytical:european_option`" in text
    assert "- Route authority:" in text
    assert "authority=`exact_backend_fit`" in text
    assert "trellis.models.black" in text


def test_schedule_dependent_route_card_mentions_shared_schedule_builder():
    plan = _analytical_plan()
    card = render_generation_route_card(plan)

    assert "build_payment_timeline" in card
    assert "Schedule construction:" in card
    assert "Do not hard-code observation or payment grids inside the payoff body." in card
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
    from trellis.agent.knowledge.decompose import decompose_to_ir

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
    assert "Resolve scalar FX spot from `market_state.fx_rates[spec.fx_pair].spot`" in card
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
    assert plan.primitive_plan.route == "credit_default_swap_monte_carlo"
    assert plan.primitive_plan.route_family == "credit_default_swap"
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
    assert "survival_probability" in card


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
    assert "spread_quote" in card
    assert "discount_curve" in card


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
    assert plan.primitive_plan.route == "nth_to_default_monte_carlo"
    assert plan.primitive_plan.route_family == "nth_to_default"
    primitive_symbols = {primitive.symbol for primitive in plan.primitive_plan.primitives}
    assert "GaussianCopula" in primitive_symbols
    assert "single-name CDS" in card


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
