"""Tests for semantic validation of generated agent modules."""

from __future__ import annotations

from pathlib import Path

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


def test_extracts_lattice_exercise_contract_from_callable_artifact():
    from trellis.agent.semantic_validation import extract_semantic_signals

    signals = extract_semantic_signals(_artifact("callablebond.py"))

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


def test_accepts_fixed_t39_transform_artifact():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        _artifact("buildapayoff.py"),
        product_ir=decompose_to_ir(
            "Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities",
        ),
    )

    assert report.ok


def test_rejects_callable_lattice_with_holder_objective():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        _artifact("callablebond.py").replace("exercise_fn=min", "exercise_fn=max"),
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lattice.exercise_objective_mismatch" in issue_codes


def test_rejects_bermudan_lattice_with_issuer_objective():
    from trellis.agent.semantic_validation import validate_semantics

    report = validate_semantics(
        _artifact("bermudanswaption.py").replace("exercise_fn=max", "exercise_fn=min"),
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
        _artifact("bermudanswaption.py").replace(
            'exercise_steps=valid_exercise_steps,\n',
            "",
        ),
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )

    issue_codes = {issue.code for issue in report.issues}
    assert "lattice.exercise_schedule_missing" in issue_codes


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
        required_market_data={"discount", "black_vol"},
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
        required_market_data={"discount", "black_vol"},
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
