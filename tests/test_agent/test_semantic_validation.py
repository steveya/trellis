"""Tests for semantic validation of generated agent modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from trellis.agent.codegen_guardrails import build_generation_plan
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.quant import PricingPlan


ROOT = Path(__file__).resolve().parents[2]
AGENT_ARTIFACTS = ROOT / "trellis" / "instruments" / "_agent"


BAD_AMERICAN_SOURCE = """\
from __future__ import annotations

import numpy as np

from trellis.core.market_state import MarketState
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.lsm import LaguerreBasis
from trellis.models.processes.gbm import GBM


class BadAmericanOptionPayoff:
    def __init__(self, spec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = 1.0
        r = 0.05
        sigma = 0.2
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=4096, n_steps=64, method="lsm")

        def payoff_fn(paths):
            return np.maximum(spec.strike - paths, 0.0)

        basis = LaguerreBasis()
        _ = basis
        return float(engine.price(spec.spot, T, payoff_fn, discount_rate=r)["price"])
"""


BAD_TRANSFORM_SOURCE = """\
from __future__ import annotations

import cmath

from trellis.core.market_state import MarketState
from trellis.models.transforms.fft_pricer import fft_price


class BadTransformPayoff:
    def __init__(self, spec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = 1.0
        r = 0.05
        sigma = 0.2
        s0 = spec.s0

        def char_fn(u):
            return cmath.exp(1j * u * cmath.log(s0) - 0.5 * sigma * sigma * T)

        return float(fft_price(char_fn=char_fn, S0=s0, K=spec.strike, T=T, r=r))
"""


BAD_MC_SHAPE_SOURCE = """\
from __future__ import annotations

import numpy as np

from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM


def build_price(spot: float, strike: float) -> float:
    process = GBM(mu=0.05, sigma=0.2)
    engine = MonteCarloEngine(process, n_paths=1000, n_steps=32, method="exact")

    def payoff_fn(paths):
        return np.maximum(strike - paths, 0.0)

    return float(engine.price(spot, 1.0, payoff_fn, discount_rate=0.05)["price"])
"""


ALT_CONTROL_PRIMITIVE_SOURCE = """\
from __future__ import annotations

from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM
from trellis.models.monte_carlo.primal_dual import primal_dual_mc


def build_price(spot: float, strike: float) -> float:
    process = GBM(mu=0.05, sigma=0.2)
    _ = MonteCarloEngine(process, n_paths=1000, n_steps=32, method="exact")
    return float(primal_dual_mc(spot=spot, strike=strike, maturity=1.0, n_paths=1000))
"""


TVR_CONTROL_PRIMITIVE_SOURCE = """\
from __future__ import annotations

import numpy as np

from trellis.models.monte_carlo.tv_regression import tsitsiklis_van_roy


def price_from_paths(paths):
    exercise_dates = list(range(1, paths.shape[1]))

    def payoff_fn(S):
        return np.maximum(1.0 - S, 0.0)

    return float(tsitsiklis_van_roy(paths, exercise_dates, payoff_fn, 0.01, 1.0 / len(exercise_dates)))
"""


EQUITY_TREE_SOURCE = """\
from __future__ import annotations

from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction
"""


HELPER_ONLY_EQUITY_TREE_SOURCE = """\
from __future__ import annotations

from trellis.models.equity_option_tree import price_vanilla_equity_option_tree


def build_price(self, market_state):
    return float(price_vanilla_equity_option_tree(market_state, self._spec, model="crr"))
"""


HELPER_ONLY_EQUITY_PDE_SOURCE = """\
from __future__ import annotations

from trellis.models.equity_option_pde import price_vanilla_equity_option_pde


def build_price(self, market_state):
    return float(price_vanilla_equity_option_pde(market_state, self._spec, theta=0.5))
"""


HELPER_BACKED_CDS_MONTE_CARLO_SOURCE = """\
from __future__ import annotations

from trellis.models.credit_default_swap import build_cds_schedule, price_cds_monte_carlo


def build_price(self, market_state):
    spec = self._spec
    schedule = build_cds_schedule(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        spec.day_count,
    )
    spread = float(spec.spread)
    if spread > 1.0:
        spread *= 1e-4
    return float(
        price_cds_monte_carlo(
            notional=spec.notional,
            spread_quote=spread,
            recovery=spec.recovery,
            schedule=schedule,
            credit_curve=market_state.credit_curve,
            discount_curve=market_state.discount,
            n_paths=int(getattr(spec, "n_paths", 250000)),
            seed=42,
        )
    )
"""


HELPER_BACKED_CDO_TRANCHE_SOURCE = """\
from __future__ import annotations

from trellis.models.credit_basket_copula import price_credit_basket_tranche


def build_price(self, market_state):
    spec = self._spec
    return float(price_credit_basket_tranche(market_state, spec, copula_family="gaussian"))
"""


HULL_WHITE_EVENT_AWARE_MC_SOURCE = """\
from __future__ import annotations

from trellis.models.rate_style_swaption import price_swaption_monte_carlo


def build_price(self, market_state):
    return float(
        price_swaption_monte_carlo(
            market_state,
            self._spec,
            n_paths=32,
            seed=7,
            mean_reversion=0.05,
            sigma=0.01,
        )
    )
"""


RATE_LATTICE_SOURCE = """\
from __future__ import annotations

from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction
"""


RATE_LATTICE_POLICY_SOURCE = """\
from __future__ import annotations

from trellis.models.trees.control import resolve_lattice_exercise_policy
from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction


def price_callable(lattice, terminal_payoff, exercise_value):
    policy = resolve_lattice_exercise_policy(
        "issuer_call",
        exercise_steps=[3, 5, 7],
    )
    return lattice_backward_induction(
        lattice,
        terminal_payoff,
        exercise_value,
        exercise_policy=policy,
    )
"""


THIN_CALLABLE_ADAPTER_SOURCE = """\
from __future__ import annotations

from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec


def build_adapter(spec):
    payoff = CallableBondPayoff(
        CallableBondSpec(
            notional=spec.notional,
            coupon=spec.coupon,
            start_date=spec.start_date,
            end_date=spec.end_date,
            call_dates=spec.call_dates,
        )
    )
    return payoff
"""


HELPER_ONLY_CALLABLE_ROUTE_SOURCE = """\
from __future__ import annotations

from trellis.models.callable_bond_tree import price_callable_bond_tree


def build_price(self, market_state):
    spec = self._spec
    return float(price_callable_bond_tree(market_state, spec, model="hull_white"))
"""


BERMUDAN_LATTICE_POLICY_SOURCE = """\
from __future__ import annotations

from trellis.models.trees.control import resolve_lattice_exercise_policy
from trellis.models.trees.lattice import lattice_backward_induction


def price_swaption(lattice, swap_values, valid_exercise_steps):
    exercise_policy = resolve_lattice_exercise_policy(
        "bermudan",
        exercise_steps=valid_exercise_steps,
    )

    def exercise_value(step, node, lat, continuation):
        del lat, continuation
        return max(swap_values[step][node], 0.0)

    return lattice_backward_induction(
        lattice,
        0.0,
        exercise_value=exercise_value,
        exercise_policy=exercise_policy,
    )
"""


HELPER_ONLY_BERMUDAN_ROUTE_SOURCE = """\
from __future__ import annotations

from trellis.models.bermudan_swaption_tree import price_bermudan_swaption_tree


def build_price(self, market_state):
    spec = self._spec
    return float(price_bermudan_swaption_tree(market_state, spec, model="hull_white"))
"""

RAW_STRING_BERMUDAN_SPEC_SOURCE = """\
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    notional: float
    strike: float
    exercise_dates: str
    swap_end: date
"""


TYPED_BERMUDAN_SPEC_SOURCE = """\
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    notional: float
    strike: float
    exercise_dates: tuple[date, ...]
    swap_end: date
"""


INVALID_LATTICE_POLICY_KWARG_SOURCE = """\
from __future__ import annotations

from trellis.models.trees.control import resolve_lattice_exercise_policy


def build_policy(exercise_steps):
    return resolve_lattice_exercise_policy(
        "holder_put",
        exercise_steps=exercise_steps,
        exercise_fn=max,
    )
"""


def _artifact(name: str) -> str:
    return (AGENT_ARTIFACTS / name).read_text()


def test_extracts_mc_method_and_missing_control_primitive_from_bad_american_artifact():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(BAD_AMERICAN_SOURCE)

    assert "lsm" in signals.monte_carlo_methods
    assert signals.exercise_control_primitives == ()
    assert not signals.uses_longstaff_schwartz
    assert "trellis.models.monte_carlo.lsm" in signals.laguerre_import_modules
    assert "payoff_fn" in signals.path_matrix_callbacks


def test_extracts_approved_alternative_early_exercise_control_primitive():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(ALT_CONTROL_PRIMITIVE_SOURCE)

    assert signals.exercise_control_primitives == ("primal_dual_mc",)


def test_extracts_tsitsiklis_van_roy_early_exercise_control_primitive():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(TVR_CONTROL_PRIMITIVE_SOURCE)

    assert signals.exercise_control_primitives == ("tsitsiklis_van_roy",)


def test_extracts_transform_usage_and_scalar_math():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(BAD_TRANSFORM_SOURCE)

    assert "fft_price" in signals.transform_pricers
    assert "char_fn" in signals.scalar_math_functions


def test_extracts_lattice_exercise_contract_from_rate_lattice_policy_source():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(BERMUDAN_LATTICE_POLICY_SOURCE)

    assert "rate_lattice" in signals.engine_families
    assert "bermudan" in signals.lattice_exercise_types
    assert signals.lattice_has_exercise_steps
    assert "max" in signals.lattice_exercise_functions


def test_extracts_equity_tree_engine_family_from_binomial_tree_imports():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(EQUITY_TREE_SOURCE)

    assert signals.engine_families == ("equity_tree",)


def test_extracts_equity_tree_engine_family_from_helper_import():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(HELPER_ONLY_EQUITY_TREE_SOURCE)

    assert signals.engine_families == ("equity_tree",)


def test_extracts_pde_engine_family_from_helper_import():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(HELPER_ONLY_EQUITY_PDE_SOURCE)

    assert signals.engine_families == ("pde_solver",)


def test_extracts_rate_lattice_engine_family_from_rate_lattice_imports():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(RATE_LATTICE_SOURCE)

    assert signals.engine_families == ("rate_lattice",)


def test_extracts_lattice_policy_contract_from_policy_helper():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(RATE_LATTICE_POLICY_SOURCE)

    assert "issuer_call" in signals.lattice_exercise_styles
    assert "bermudan" in signals.lattice_exercise_types
    assert signals.lattice_has_exercise_steps
    assert "min" in signals.lattice_exercise_functions


def test_rejects_current_american_agent_artifact():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        BAD_AMERICAN_SOURCE,
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "mc.invalid_method_mode" in issue_codes
    assert "exercise.invalid_basis_import" in issue_codes
    assert "exercise.missing_control_primitive" in issue_codes
    assert "mc.invalid_payoff_shape" in issue_codes


def test_does_not_require_longstaff_when_other_approved_control_is_used():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        ALT_CONTROL_PRIMITIVE_SOURCE,
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "exercise.missing_control_primitive" not in issue_codes


def test_accepts_swaption_artifact():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        _artifact("swaption.py"),
        product_ir=decompose_to_ir("European payer swaption", instrument_type="swaption"),
    )

    assert report.ok


def test_rejects_scalar_transform_char_fn():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        BAD_TRANSFORM_SOURCE,
        product_ir=decompose_to_ir(
            "Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "transform.scalar_char_fn" in issue_codes


@pytest.mark.parametrize(
    "description,expected_missing_field",
    [
        ("Finite element method (FEM) vs finite difference for European", "semantic_product_shape"),
        (
            "Crank-Nicolson Rannacher smoothing for discontinuous payoffs\n\nImplementation target: black_scholes_digital",
            "semantic_product_shape",
        ),
    ],
)
def test_classify_semantic_gap_treats_vanilla_option_words_as_shape_cues(
    description: str,
    expected_missing_field: str,
):
    from trellis.agent.semantic_contract_validation import classify_semantic_gap

    report = classify_semantic_gap(description)

    assert expected_missing_field not in report.missing_contract_fields
    assert report.requires_clarification is False


def test_classify_semantic_gap_treats_cds_as_credit_request():
    from trellis.agent.semantic_contract_validation import classify_semantic_gap

    report = classify_semantic_gap(
        "CDS pricing: hazard rate MC vs survival prob analytical",
        instrument_type="cds",
    )

    assert report.requires_clarification is False
    assert "semantic_product_shape" not in report.missing_contract_fields
    assert "discount_curve" in report.missing_market_inputs
    assert "credit_curve" in report.missing_market_inputs

def test_accepts_fixed_t39_transform_artifact():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        _artifact("buildapayoff.py"),
        product_ir=decompose_to_ir(
            "Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities",
        ),
    )

    assert report.ok


def test_rejects_raw_string_schedule_fields():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        RAW_STRING_BERMUDAN_SPEC_SOURCE,
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "schedule.raw_string_field" in issue_codes


def test_accepts_typed_tuple_schedule_fields():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        TYPED_BERMUDAN_SPEC_SOURCE,
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "schedule.raw_string_field" not in issue_codes


def test_rejects_invalid_lattice_policy_keywords():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(INVALID_LATTICE_POLICY_KWARG_SOURCE)

    issue_codes = {issue.code for issue in report.issues}
    assert "lattice.invalid_policy_kwarg" in issue_codes


def test_callable_adapter_does_not_claim_lattice_objective():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        THIN_CALLABLE_ADAPTER_SOURCE,
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
    )

    assert report.ok


def test_rejects_bermudan_lattice_with_issuer_objective():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        BERMUDAN_LATTICE_POLICY_SOURCE.replace('"bermudan"', '"issuer_call"', 1),
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lattice.exercise_objective_mismatch" in issue_codes


def test_rejects_schedule_dependent_lattice_without_exercise_steps():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        BERMUDAN_LATTICE_POLICY_SOURCE.replace(
            "        exercise_steps=valid_exercise_steps,\n",
            "",
        ),
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lattice.exercise_schedule_missing" in issue_codes


def test_accepts_helper_only_bermudan_swaption_route_without_low_level_tree_contract():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.bermudan_swaption_tree"],
        required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
        model_to_build="bermudan_swaption",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "Bermudan swaption: tree vs LSM MC",
        instrument_type="bermudan_swaption",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="bermudan_swaption",
        inspected_modules=("trellis.models.bermudan_swaption_tree",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HELPER_ONLY_BERMUDAN_ROUTE_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "engine.family_incompatible_with_ir" not in issue_codes
    assert "lattice.exercise_schedule_missing" not in issue_codes
    assert report.ok


def test_accepts_callable_lattice_with_policy_helper():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        RATE_LATTICE_POLICY_SOURCE,
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
    )

    assert report.ok


def test_accepts_helper_only_callable_route_without_low_level_lattice_contract():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.callable_bond_tree"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="callable_bond",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "Callable bond with semiannual coupon and call schedule",
        instrument_type="callable_bond",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="callable_bond",
        inspected_modules=("trellis.models.callable_bond_tree",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HELPER_ONLY_CALLABLE_ROUTE_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lattice.exercise_type_mismatch" not in issue_codes
    assert "lattice.exercise_schedule_missing" not in issue_codes
    assert "lattice.exercise_objective_mismatch" not in issue_codes
    assert "engine.family_incompatible_with_ir" not in issue_codes
    assert report.ok


def test_accepts_helper_only_equity_tree_route_without_low_level_tree_contract():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="rate_tree",
        method_modules=["trellis.models.equity_option_tree"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "American put option on equity",
        instrument_type="american_option",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.equity_option_tree",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HELPER_ONLY_EQUITY_TREE_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "engine.family_incompatible_with_ir" not in issue_codes
    assert "assembly.required_primitive_missing" not in issue_codes
    assert report.ok


def test_accepts_helper_only_equity_pde_route_without_low_level_pde_contract():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="pde_solver",
        method_modules=["trellis.models.equity_option_pde"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "European call: theta-method convergence order measurement",
        instrument_type="european_option",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.equity_option_pde",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HELPER_ONLY_EQUITY_PDE_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "engine.family_incompatible_with_ir" not in issue_codes
    assert "assembly.required_primitive_missing" not in issue_codes
    assert report.ok


def test_accepts_helper_backed_cds_route_without_internal_event_probability_call():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.credit_default_swap"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="credit_default_swap",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "CDS pricing: hazard rate MC vs survival prob analytical",
        instrument_type="credit_default_swap",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="credit_default_swap",
        inspected_modules=("trellis.models.credit_default_swap",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HELPER_BACKED_CDS_MONTE_CARLO_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "assembly.required_primitive_missing" not in issue_codes
    assert report.ok


def test_accepts_helper_backed_cdo_tranche_route_without_internal_copula_calls():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="copula",
        method_modules=["trellis.models.credit_basket_copula"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="cdo",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "CDO tranche on a 100-name IG portfolio with attachment 3% and detachment 7%",
        instrument_type="cdo",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="cdo",
        inspected_modules=("trellis.models.credit_basket_copula",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HELPER_BACKED_CDO_TRANCHE_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "assembly.required_primitive_missing" not in issue_codes
    assert report.ok


def test_accepts_one_required_state_process_when_route_declares_role_alternatives():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "forward_curve", "black_vol_surface"},
        model_to_build="swaption",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "European payer swaption under Hull-White Monte Carlo",
        instrument_type="swaption",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="swaption",
        inspected_modules=("trellis.models.monte_carlo.engine", "trellis.models.monte_carlo.event_aware"),
        product_ir=product_ir,
    )

    report = validate_semantics(
        HULL_WHITE_EVENT_AWARE_MC_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "assembly.required_primitive_missing" not in issue_codes


def test_rejects_matrix_payoff_passed_to_mc_engine():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        BAD_MC_SHAPE_SOURCE,
        product_ir=decompose_to_ir("American put option on equity", instrument_type="american_option"),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "mc.invalid_payoff_shape" in issue_codes


def test_requires_selected_primitives_from_generation_plan():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="american_option",
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "American put option on equity",
        instrument_type="american_option",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="american_option",
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        BAD_AMERICAN_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "exercise.missing_control_primitive" in issue_codes
    assert "assembly.required_primitive_missing" not in issue_codes


def test_rejects_generation_plan_with_blockers():
    from trellis.agent.semantic_validation import validate_semantics

    pricing_plan = PricingPlan(
        method="monte_carlo",
        method_modules=["trellis.models.monte_carlo.engine"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build=None,
        reasoning="test",
    )
    product_ir = decompose_to_ir(
        "American Asian barrier option under Heston with early exercise",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=None,
        inspected_modules=("trellis.models.monte_carlo.engine",),
        product_ir=product_ir,
    )

    report = validate_semantics(
        BAD_AMERICAN_SOURCE,
        product_ir=product_ir,
        generation_plan=generation_plan,
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "assembly.route_has_blockers" in issue_codes
