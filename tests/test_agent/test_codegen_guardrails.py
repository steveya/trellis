"""Tests for generation plans and import validation."""

from __future__ import annotations

from trellis.agent.codegen_guardrails import (
    build_generation_plan,
    validate_generated_imports,
)
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


def _analytical_plan():
    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount", "forward_rate", "black_vol"},
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


def test_qmc_generation_plan_approves_qmc_family_modules():
    pricing_plan = PricingPlan(
        method="qmc",
        method_modules=["trellis.models.qmc"],
        required_market_data={"discount", "black_vol"},
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
