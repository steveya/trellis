"""Proving-ground tests for generated rate-tree exercise products."""

from __future__ import annotations

from datetime import date
from importlib import import_module
from pathlib import Path
import sys

from trellis.agent.codegen_guardrails import build_generation_plan
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.quant import PricingPlan
from trellis.agent.semantic_validation import validate_semantics
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.vol_surface import FlatVol


ROOT = Path(__file__).resolve().parents[2]
AGENT_ARTIFACTS = ROOT / "trellis" / "instruments" / "_agent"


def _artifact_source(name: str) -> str:
    return (AGENT_ARTIFACTS / name).read_text()


def _callable_plan():
    return build_generation_plan(
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="callable_bond",
            reasoning="test",
        ),
        instrument_type="callable_bond",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
    )


def _puttable_plan():
    return build_generation_plan(
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="puttable_bond",
            reasoning="test",
        ),
        instrument_type="puttable_bond",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=decompose_to_ir(
            "Puttable bond with semiannual coupon and put schedule",
            instrument_type="puttable_bond",
        ),
    )


def _bermudan_plan():
    return build_generation_plan(
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
            model_to_build="bermudan_swaption",
            reasoning="test",
        ),
        instrument_type="bermudan_swaption",
        inspected_modules=("trellis.models.trees.lattice",),
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
    )


def _bermudan_rate_tree_source() -> str:
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    plan = _bermudan_plan()
    skeleton = _generate_skeleton(
        STATIC_SPECS["bermudan_swaption"],
        "Bermudan payer swaption",
        generation_plan=plan,
    )
    generated = _materialize_deterministic_exact_binding_module(skeleton, plan)
    assert generated is not None
    return generated.code


def test_callable_artifact_is_rate_tree_route_compliant():
    report = validate_semantics(
        _artifact_source("callablebond.py"),
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
        generation_plan=_callable_plan(),
    )
    assert report.ok


def test_puttable_artifact_is_rate_tree_route_compliant():
    report = validate_semantics(
        _artifact_source("puttablebond.py"),
        product_ir=decompose_to_ir(
            "Puttable bond with semiannual coupon and put schedule",
            instrument_type="puttable_bond",
        ),
        generation_plan=_puttable_plan(),
    )
    assert report.ok


def test_callable_and_puttable_artifacts_own_primitive_composition():
    required_symbols = {
        "resolve_short_rate_lattice_inputs",
        "build_embedded_fixed_income_event_timeline",
        "compile_embedded_fixed_income_lattice_contract_spec",
        "build_lattice",
        "price_on_lattice",
        "present_value_fixed_coupon_bond",
    }
    for artifact_name in ("callablebond.py", "puttablebond.py"):
        source = _artifact_source(artifact_name)
        assert "price_callable_bond_tree" not in source
        for symbol in required_symbols:
            assert symbol in source

    assert 'expected_control_style="issuer_min"' in _artifact_source(
        "callablebond.py"
    )
    assert 'expected_control_style="holder_max"' in _artifact_source(
        "puttablebond.py"
    )


def test_callable_artifact_wrong_declared_control_fails_semantic_validation():
    source = _artifact_source("callablebond.py").replace(
        'expected_control_style="issuer_min"',
        'expected_control_style="holder_max"',
    )

    report = validate_semantics(
        source,
        product_ir=decompose_to_ir(
            "Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
        ),
        generation_plan=_callable_plan(),
    )

    assert any(
        issue.code == "lattice.exercise_objective_mismatch"
        for issue in report.issues
    )


def test_bermudan_artifact_is_rate_tree_route_compliant():
    report = validate_semantics(
        _bermudan_rate_tree_source(),
        product_ir=decompose_to_ir(
            "Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
        ),
        generation_plan=_bermudan_plan(),
    )
    assert report.ok


def test_callable_artifact_prices_plausibly_against_reference_tree():
    sys.path.insert(0, str(ROOT))
    from tests.test_tasks.test_t02_bdt_callable import (
        FLAT_RATE,
        HW_A,
        HW_SIGMA,
        N_STEPS,
        T,
        _price_callable_bond,
    )
    from trellis.models.trees.lattice import build_generic_lattice
    from trellis.models.trees.models import MODEL_REGISTRY

    mod = import_module("trellis.instruments._agent.callablebond")
    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(FLAT_RATE, max_tenor=max(T + 1, 31.0)),
        vol_surface=FlatVol(0.20),
    )
    spec = mod.CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=settle,
        end_date=date(2034, 11, 15),
        call_dates=(
            date(2027, 11, 15),
            date(2029, 11, 15),
            date(2031, 11, 15),
        ),
    )
    artifact_price = price_payoff(mod.CallableBondPayoff(spec), market_state)

    lattice = build_generic_lattice(
        MODEL_REGISTRY["hull_white"],
        r0=FLAT_RATE,
        sigma=HW_SIGMA,
        a=HW_A,
        T=T,
        n_steps=N_STEPS,
        discount_curve=market_state.discount,
    )
    reference_price = _price_callable_bond(lattice)

    assert abs(artifact_price - reference_price) / reference_price < 0.05


def test_callable_artifact_works_with_calibrated_model_parameters_only():
    sys.path.insert(0, str(ROOT))
    mod = import_module("trellis.instruments._agent.callablebond")
    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
            "sigma": 0.004,
        },
    )
    spec = mod.CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=settle,
        end_date=date(2034, 11, 15),
        call_dates=(
            date(2027, 11, 15),
            date(2029, 11, 15),
            date(2031, 11, 15),
        ),
    )

    assert price_payoff(mod.CallableBondPayoff(spec), market_state) > 0.0


def test_bermudan_artifact_prices_plausibly_against_reference_tree():
    sys.path.insert(0, str(ROOT))
    from tests.test_tasks.test_t04_bermudan_swaption import (
        EXERCISE_YEARS,
        FIXED_RATE,
        HW_A,
        HW_SIGMA,
        NOTIONAL,
        T_SWAP_TENOR,
        _price_bermudan_swaption_on_tree,
    )
    from trellis.models.trees.lattice import build_generic_lattice
    from trellis.models.trees.models import MODEL_REGISTRY

    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        vol_surface=FlatVol(0.20),
    )
    class _Spec:
        notional = NOTIONAL
        strike = FIXED_RATE
        exercise_dates = (
            date(2025, 11, 15),
            date(2026, 11, 15),
            date(2027, 11, 15),
            date(2028, 11, 15),
            date(2029, 11, 15),
        )
        swap_end = date(2030, 11, 15)
        from trellis.core.types import DayCountConvention, Frequency
        swap_frequency = Frequency.SEMI_ANNUAL
        day_count = DayCountConvention.ACT_360
        rate_index = None
        is_payer = True

    namespace: dict[str, object] = {}
    source = _bermudan_rate_tree_source()
    exec(compile(source, "<bermudan_rate_tree>", "exec"), namespace)  # noqa: S102
    spec = namespace["BermudanSwaptionSpec"](
        notional=_Spec.notional,
        strike=_Spec.strike,
        exercise_dates=_Spec.exercise_dates,
        swap_end=_Spec.swap_end,
        swap_frequency=_Spec.swap_frequency,
        day_count=_Spec.day_count,
        rate_index=_Spec.rate_index,
        is_payer=_Spec.is_payer,
    )
    payoff = namespace["BermudanSwaptionPayoff"](spec)
    artifact_price = price_payoff(payoff, market_state)

    lattice = build_generic_lattice(
        MODEL_REGISTRY["hull_white"],
        r0=0.05,
        sigma=HW_SIGMA,
        a=HW_A,
        T=10.0,
        n_steps=200,
        discount_curve=market_state.discount,
    )
    reference_price = _price_bermudan_swaption_on_tree(
        lattice,
        EXERCISE_YEARS,
        T_SWAP_TENOR,
        FIXED_RATE,
        NOTIONAL,
    )

    assert abs(artifact_price - reference_price) / reference_price < 0.20


def test_zcb_artifact_normalizes_quoted_option_type():
    sys.path.insert(0, str(ROOT))
    mod = import_module("trellis.instruments._agent.zcboption")
    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05, max_tenor=12.0),
        vol_surface=FlatVol(0.20),
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.1,
            "sigma": 0.01,
        },
    )
    spec = mod.ZCBOptionSpec(
        notional=100.0,
        strike=63.0,
        expiry_date=date(2027, 11, 15),
        bond_maturity_date=date(2033, 11, 15),
        option_type="'call'",
    )

    assert price_payoff(mod.ZCBOptionPayoff(spec), market_state) > 0.0


def test_zcb_artifact_accepts_case_variants_and_bond_market_aliases():
    sys.path.insert(0, str(ROOT))
    mod = import_module("trellis.instruments._agent.zcboption")
    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05, max_tenor=12.0),
        vol_surface=FlatVol(0.20),
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.1,
            "sigma": 0.01,
        },
    )

    for option_type in ("Call", "PUT", "payer", "receiver"):
        spec = mod.ZCBOptionSpec(
            notional=100.0,
            strike=63.0,
            expiry_date=date(2027, 11, 15),
            bond_maturity_date=date(2033, 11, 15),
            option_type=option_type,
        )
        assert price_payoff(mod.ZCBOptionPayoff(spec), market_state) > 0.0


def test_zcb_artifact_does_not_probe_vol_surface_when_sigma_is_calibrated():
    sys.path.insert(0, str(ROOT))
    mod = import_module("trellis.instruments._agent.zcboption")
    settle = date(2024, 11, 15)

    class ExplodingVolSurface:
        def black_vol(self, t: float, strike: float) -> float:
            raise AssertionError("zcb artifact should not probe vol_surface when sigma is present")

    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05, max_tenor=12.0),
        vol_surface=ExplodingVolSurface(),
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.1,
            "sigma": 0.01,
        },
    )
    spec = mod.ZCBOptionSpec(
        notional=100.0,
        strike=63.0,
        expiry_date=date(2027, 11, 15),
        bond_maturity_date=date(2033, 11, 15),
        option_type="call",
    )

    assert price_payoff(mod.ZCBOptionPayoff(spec), market_state) > 0.0
