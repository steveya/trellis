"""Tests for the deterministic lite-reviewer."""

from __future__ import annotations

from types import SimpleNamespace


def test_lite_review_rejects_hardcoded_market_inputs_for_required_capabilities():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.black import black76_call

def price(market_state):
    T = 1.0
    r = 0.05
    sigma = 0.2
    return black76_call(100.0, 100.0, sigma, T) * r
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(),
            adapters=(),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.hardcoded_discount_curve" in issue_codes
    assert "lite.hardcoded_black_vol_surface" in issue_codes


def test_lite_review_accepts_market_state_sourced_inputs():
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
def price(market_state):
    T = 1.0
    r = float(market_state.discount.zero_rate(T))
    sigma = float(market_state.vol_surface.black_vol(T, 100.0))
    return r + sigma
"""

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(source, pricing_plan=pricing_plan)

    assert report.ok


def test_lite_review_rejects_wall_clock_valuation_date():
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from datetime import date

def price(spec, market_state):
    T = (spec.expiry_date - date.today()).days / 365.0
    return float(T)
"""

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(source, pricing_plan=pricing_plan)

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.wall_clock_valuation_date" in issue_codes


def test_lite_review_rejects_analytical_black76_without_discount_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.black import black76_call

def price(spec, market_state):
    T = 1.0
    sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
    return black76_call(spec.spot, spec.strike, sigma, T)
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(),
            adapters=("map_spot_discount_and_vol_to_forward_black76",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.analytical_black76_discount_curve_access_missing" in issue_codes


def test_lite_review_rejects_analytical_black76_without_vol_surface_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.black import black76_call

def price(spec, market_state):
    T = 1.0
    df = float(market_state.discount.discount(T))
    sigma = spec.vol
    forward = spec.spot / df
    return df * black76_call(forward, spec.strike, sigma, T)
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(),
            adapters=("map_spot_discount_and_vol_to_forward_black76",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.analytical_black76_black_vol_surface_access_missing" in issue_codes


def test_lite_review_allows_swaption_black76_helper_with_explicit_hull_white_comparison_params():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.rate_style_swaption import price_swaption_black76

def price(spec, market_state):
    return float(
        price_swaption_black76(
            market_state,
            spec,
            mean_reversion=0.05,
            sigma=0.01,
        )
    )
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="swaption",
        inspected_modules=("trellis.models.rate_style_swaption",),
        approved_modules=("trellis.models.rate_style_swaption",),
        symbols_to_reuse=("price_swaption_black76",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.rate_style_swaption", "price_swaption_black76", "route_helper"),
            ),
            adapters=("reuse_checked_in_rate_style_swaption_helper",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.rate_style_swaption"],
        required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
        model_to_build="swaption",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.hardcoded_black_vol_surface" not in issue_codes


def test_lite_review_allows_swaption_tree_helper_with_explicit_hull_white_comparison_params():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.rate_style_swaption_tree import price_swaption_tree

def price(spec, market_state):
    return float(
        price_swaption_tree(
            market_state,
            spec,
            mean_reversion=0.05,
            sigma=0.01,
        )
    )
"""

    plan = GenerationPlan(
        method="rate_tree",
        instrument_type="swaption",
        inspected_modules=("trellis.models.rate_style_swaption_tree",),
        approved_modules=("trellis.models.rate_style_swaption_tree",),
        symbols_to_reuse=("price_swaption_tree",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="rate_tree_backward_induction",
            engine_family="lattice",
            primitives=(
                PrimitiveRef("trellis.models.rate_style_swaption_tree", "price_swaption_tree", "route_helper"),
            ),
            adapters=("reuse_checked_in_rate_style_swaption_helper",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.rate_style_swaption_tree"],
        required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
        model_to_build="swaption",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.hardcoded_black_vol_surface" not in issue_codes


def test_lite_review_allows_swaption_monte_carlo_helper_with_explicit_hull_white_comparison_params():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.rate_style_swaption import price_swaption_monte_carlo

def price(spec, market_state):
    return float(
        price_swaption_monte_carlo(
            market_state,
            spec,
            mean_reversion=0.05,
            sigma=0.01,
        )
    )
"""

    plan = GenerationPlan(
        method="monte_carlo",
        instrument_type="swaption",
        inspected_modules=("trellis.models.rate_style_swaption",),
        approved_modules=("trellis.models.rate_style_swaption",),
        symbols_to_reuse=("price_swaption_monte_carlo",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="monte_carlo_paths",
            engine_family="monte_carlo",
            primitives=(
                PrimitiveRef("trellis.models.rate_style_swaption", "price_swaption_monte_carlo", "route_helper"),
            ),
            adapters=("reuse_checked_in_rate_style_swaption_helper",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.rate_style_swaption"],
        required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
        model_to_build="swaption",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.hardcoded_black_vol_surface" not in issue_codes


def test_lite_review_rejects_garman_kohlhagen_without_foreign_curve_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.black import garman_kohlhagen_call

def price(spec, market_state):
    T = 1.0
    df_domestic = float(market_state.discount.discount(T))
    sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
    spot = float(market_state.fx_rates[spec.fx_pair].spot)
    return garman_kohlhagen_call(spot, spec.strike, sigma, T, df_domestic, 0.97)
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("garman_kohlhagen_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_garman_kohlhagen",
            engine_family="analytical",
            primitives=(),
            adapters=("map_fx_spot_and_curves_to_garman_kohlhagen_inputs",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.analytical_garman_kohlhagen_forward_curve_access_missing" in issue_codes


def test_lite_review_flags_missing_required_exact_helper_for_fx_route():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.analytical.fx import garman_kohlhagen_price_raw

def evaluate(self, market_state):
    return garman_kohlhagen_price_raw("call", resolved)
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.fx_vanilla",),
        approved_modules=("trellis.models.fx_vanilla", "trellis.models.analytical.fx"),
        symbols_to_reuse=("price_fx_vanilla_analytical", "garman_kohlhagen_price_raw"),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_garman_kohlhagen",
            engine_family="analytical",
            primitives=(
                PrimitiveRef(
                    "trellis.models.fx_vanilla",
                    "price_fx_vanilla_analytical",
                    "route_helper",
                ),
            ),
            adapters=(),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.fx_vanilla"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "spot"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.analytical_garman_kohlhagen_route_helper_missing" in issue_codes


def test_lite_review_does_not_treat_instance_method_as_exact_helper_call():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
def evaluate(self, market_state):
    return self.price_fx_vanilla_analytical(market_state, self._spec)
"""

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.fx_vanilla",),
        approved_modules=("trellis.models.fx_vanilla",),
        symbols_to_reuse=("price_fx_vanilla_analytical",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_garman_kohlhagen",
            engine_family="analytical",
            primitives=(
                PrimitiveRef(
                    "trellis.models.fx_vanilla",
                    "price_fx_vanilla_analytical",
                    "route_helper",
                ),
            ),
            adapters=(),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.fx_vanilla"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "spot"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.analytical_garman_kohlhagen_route_helper_missing" in issue_codes


def test_lite_review_rejects_monte_carlo_route_without_market_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM

def price(spec, market_state):
    process = GBM(mu=0.05, sigma=0.2)
    engine = MonteCarloEngine(process, n_paths=2000, n_steps=32, method="exact")
    return float(engine.price(spec.spot, 1.0, lambda paths: paths[:, -1], discount_rate=0.05)["price"])
"""

    plan = GenerationPlan(
        method="monte_carlo",
        instrument_type="barrier_option",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        approved_modules=("trellis.models.monte_carlo.engine", "trellis.models.processes.gbm"),
        symbols_to_reuse=("MonteCarloEngine", "GBM"),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="monte_carlo_paths",
            engine_family="monte_carlo",
            primitives=(),
            adapters=("build_payoff_vector_from_paths",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="barrier_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.monte_carlo_paths_discount_curve_access_missing" in issue_codes
    assert "lite.monte_carlo_paths_black_vol_surface_access_missing" in issue_codes


def test_lite_review_rejects_rate_tree_route_without_market_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction

def price(spec, market_state):
    lattice = build_rate_lattice(r0=0.05, sigma=0.2, dt=0.5, n_steps=10)
    return lattice_backward_induction(lattice, terminal_values=[100.0] * 11)
"""

    plan = GenerationPlan(
        method="rate_tree",
        instrument_type="callable_bond",
        inspected_modules=("trellis.models.trees.lattice",),
        approved_modules=("trellis.models.trees.lattice",),
        symbols_to_reuse=("build_rate_lattice", "lattice_backward_induction"),
        proposed_tests=("tests/test_agent/test_callable_bond.py",),
        primitive_plan=PrimitivePlan(
            route="exercise_lattice",
            engine_family="lattice",
            primitives=(),
            adapters=("map_cashflows_and_exercise_dates_to_tree_steps",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.trees.lattice"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="callable_bond",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.exercise_lattice_discount_curve_access_missing" in issue_codes
    assert "lite.exercise_lattice_black_vol_surface_access_missing" in issue_codes


def test_lite_review_rejects_local_vol_monte_carlo_without_surface_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.monte_carlo.local_vol import local_vol_european_vanilla_price

def price(spec, market_state):
    discount_curve = market_state.discount
    spot = market_state.spot
    return local_vol_european_vanilla_price(
        spot=spot,
        strike=spec.strike,
        maturity=1.0,
        discount_curve=discount_curve,
        local_vol_surface=lambda s, t: 0.2,
        option_type="call",
    )
"""

    plan = GenerationPlan(
        method="monte_carlo",
        instrument_type="european_option",
        inspected_modules=("trellis.models.monte_carlo.local_vol",),
        approved_modules=(
            "trellis.models.monte_carlo.local_vol",
            "trellis.models.processes.local_vol",
            "trellis.models.monte_carlo.engine",
        ),
        symbols_to_reuse=("local_vol_european_vanilla_price", "LocalVol", "MonteCarloEngine"),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="local_vol_monte_carlo",
            engine_family="monte_carlo",
            primitives=(),
            adapters=("map_market_state_local_vol_surface_spot_and_discount_into_local_vol_mc_inputs",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.local_vol", "trellis.models.processes.local_vol"],
        required_market_data={"discount_curve", "spot", "local_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.local_vol_monte_carlo_local_vol_surface_access_missing" in issue_codes


def test_lite_review_accepts_helper_backed_pde_route_without_direct_market_access():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.lite_review import review_generated_code
    from trellis.agent.quant import PricingPlan

    source = """\
from trellis.models.equity_option_pde import price_vanilla_equity_option_pde

def price(spec, market_state):
    return float(price_vanilla_equity_option_pde(market_state, spec, theta=1.0))
"""

    plan = GenerationPlan(
        method="pde_solver",
        instrument_type="european_option",
        inspected_modules=("trellis.models.equity_option_pde",),
        approved_modules=("trellis.models.equity_option_pde",),
        symbols_to_reuse=("price_vanilla_equity_option_pde",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="vanilla_equity_theta_pde",
            engine_family="pde_solver",
            primitives=(
                PrimitiveRef(
                    "trellis.models.equity_option_pde",
                    "price_vanilla_equity_option_pde",
                    "route_helper",
                ),
            ),
            adapters=("reuse_checked_in_vanilla_equity_pde_helper",),
            blockers=(),
        ),
    )
    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.equity_option_pde"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )

    report = review_generated_code(
        source,
        pricing_plan=pricing_plan,
        generation_plan=plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lite.vanilla_equity_theta_pde_discount_curve_access_missing" not in issue_codes
    assert "lite.vanilla_equity_theta_pde_black_vol_surface_access_missing" not in issue_codes


def test_builder_prompt_surface_uses_semantic_repair_for_lite_review():
    from trellis.agent.executor import _builder_prompt_surface_for_attempt

    surface = _builder_prompt_surface_for_attempt(
        attempt_number=2,
        retry_reason="lite_review",
    )

    assert surface == "semantic_repair"


def test_builder_prompt_surface_keeps_code_generation_retries_compact():
    from trellis.agent.executor import _builder_prompt_surface_for_attempt

    surface = _builder_prompt_surface_for_attempt(
        attempt_number=2,
        retry_reason="code_generation",
    )

    assert surface == "compact"


def test_builder_prompt_surface_keeps_validation_retries_compact():
    from trellis.agent.executor import _builder_prompt_surface_for_attempt

    surface = _builder_prompt_surface_for_attempt(
        attempt_number=2,
        retry_reason="validation",
    )

    assert surface == "compact"


def test_validate_build_skips_llm_reviewer_stages_after_deterministic_failures(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.quant import PricingPlan

    class DummyPayoff:
        pass

    spec_schema = SimpleNamespace(class_name="DummyPayoff", requirements=["analytical"], fields=[])

    monkeypatch.setattr(
        "trellis.agent.executor._make_test_payoff",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: ["deterministic gate failure"],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_bounded_by_reference",
        lambda *args, **kwargs: [],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM-backed reviewer stage should be skipped")

    monkeypatch.setattr("trellis.agent.critic.critique", fail_if_called)
    monkeypatch.setattr("trellis.agent.model_validator.validate_model", fail_if_called)

    failures = _validate_build(
        DummyPayoff,
        code="def evaluate(self, market_state):\n    return -1.0\n",
        description="European call option on equity",
        spec_schema=spec_schema,
        validation="thorough",
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount_curve"},
            model_to_build="european_option",
            reasoning="test",
        ),
        product_ir=SimpleNamespace(
            instrument="callable_bond",
            payoff_traits=("callable",),
            exercise_style="issuer_call",
            state_dependence="schedule_dependent",
            schedule_dependence=True,
            model_family="interest_rate",
            unresolved_primitives=(),
            supported=True,
        ),
        attempt_number=1,
    )

    assert failures == ["deterministic gate failure"]


def test_validate_build_uses_quanto_family_validation_bundle(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.platform_requests import compile_build_request

    class DummyPayoff:
        pass

    compiled = compile_build_request(
        "Quanto option on SAP settled in USD",
        instrument_type="quanto_option",
    )

    spec_schema = SimpleNamespace(
        class_name="QuantoOptionAnalyticalPayoff",
        spec_name="QuantoOptionSpec",
        requirements=["discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"],
        fields=[],
    )
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        "trellis.agent.executor._make_test_payoff",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "trellis.agent.executor._record_platform_event",
        lambda compiled_request, event, **kwargs: events.append((event, kwargs.get("details", {}))),
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_required_inputs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_cross_currency_semantics",
        lambda *args, **kwargs: [],
    )

    failures = _validate_build(
        DummyPayoff,
        code="def evaluate(self, market_state):\n    return 0.0\n",
        description="Quanto option on SAP settled in USD",
        spec_schema=spec_schema,
        validation="fast",
        compiled_request=compiled,
        pricing_plan=compiled.pricing_plan,
        product_ir=compiled.product_ir,
        attempt_number=1,
    )

    assert failures == []
    selected = next(details for event, details in events if event == "validation_bundle_selected")
    assert selected["bundle_id"] == "analytical:quanto_option"
    assert "check_quanto_required_inputs" in selected["checks"]
    assert "check_quanto_cross_currency_semantics" in selected["checks"]


def test_validate_build_supports_quanto_monte_carlo_market_inputs():
    from trellis.agent.executor import _validate_build
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.instruments._agent.quantooptionmontecarlo import QuantoOptionMonteCarloPayoff

    compiled = compile_build_request(
        "Quanto option: quanto-adjusted BS vs MC cross-currency",
        instrument_type="quanto_option",
        preferred_method="monte_carlo",
    )

    failures = _validate_build(
        QuantoOptionMonteCarloPayoff,
        code="def evaluate(self, market_state):\n    return 0.0\n",
        description="Quanto option: quanto-adjusted BS vs MC cross-currency",
        spec_schema=SPECIALIZED_SPECS["quanto_option_monte_carlo"],
        validation="fast",
        compiled_request=compiled,
        pricing_plan=compiled.pricing_plan,
        product_ir=compiled.product_ir,
        attempt_number=1,
    )

    assert failures == []


def test_format_validation_failure_feedback_includes_structured_diagnostics():
    from trellis.agent.executor import _format_validation_failure_feedback
    from trellis.agent.invariants import InvariantFailure

    feedback = _format_validation_failure_feedback(
        failures=["Price is negative: -2.000000"],
        failure_details=[
            InvariantFailure(
                check="check_non_negativity",
                message="Price is negative: -2.000000",
                actual=-2.0,
                exception_type="ValueError",
                exception_message="missing FX quote",
                context={
                    "spot": 100.0,
                    "fx_pairs": ("EURUSD",),
                    "model_parameter_keys": ("quanto_correlation",),
                },
            )
        ],
    )

    assert "check_non_negativity" in feedback
    assert "Actual: -2.0" in feedback
    assert "ValueError: missing FX quote" in feedback
    assert "spot=100.0" in feedback
