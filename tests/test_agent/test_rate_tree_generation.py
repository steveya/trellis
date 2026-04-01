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
BERMUDAN_RATE_TREE_SOURCE = """
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.bermudan_swaption_tree import price_bermudan_swaption_tree


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    notional: float
    strike: float
    exercise_dates: str
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class BermudanSwaptionPayoff:
    def __init__(self, spec: BermudanSwaptionSpec):
        self._spec = spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "black_vol", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        if market_state.discount is None:
            raise ValueError("discount curve required")
        if market_state.vol_surface is None:
            raise ValueError("black vol surface required")
        return float(price_bermudan_swaption_tree(market_state, self._spec, model="hull_white"))
"""


def _artifact_source(name: str) -> str:
    return (AGENT_ARTIFACTS / name).read_text()


def _callable_plan():
    return build_generation_plan(
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount", "black_vol"},
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


def _bermudan_plan():
    return build_generation_plan(
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=["trellis.models.trees.lattice"],
            required_market_data={"discount", "black_vol", "forward_rate"},
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


def test_bermudan_artifact_is_rate_tree_route_compliant():
    report = validate_semantics(
        BERMUDAN_RATE_TREE_SOURCE,
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
        call_dates="2027-11-15,2029-11-15,2031-11-15",
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
    from trellis.models.bermudan_swaption_tree import price_bermudan_swaption_tree
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
        exercise_dates = "2025-11-15,2026-11-15,2027-11-15,2028-11-15,2029-11-15"
        swap_end = date(2030, 11, 15)
        from trellis.core.types import DayCountConvention, Frequency
        swap_frequency = Frequency.SEMI_ANNUAL
        day_count = DayCountConvention.ACT_360
        rate_index = None
        is_payer = True

    spec = _Spec()
    artifact_price = price_bermudan_swaption_tree(market_state, spec, model="hull_white")

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
