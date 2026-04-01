"""Tests for semantic validators."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.route_registry import load_route_registry, resolve_route_primitives, RouteSpec
from trellis.agent.semantic_validators import validate_generated_semantics
from trellis.agent.semantic_validators.algorithm_contract import AlgorithmContractValidator
from trellis.agent.semantic_validators.base import SemanticFinding, SemanticValidationReport
from trellis.agent.semantic_validators.market_data import MarketDataValidator
from trellis.agent.semantic_validators.parameter_binding import ParameterBindingValidator


@pytest.fixture(scope="module")
def registry():
    return load_route_registry()


def _make_plan(route: str, engine_family: str = "analytical") -> GenerationPlan:
    return GenerationPlan(
        method=engine_family,
        instrument_type=None,
        inspected_modules=(),
        approved_modules=(),
        symbols_to_reuse=(),
        proposed_tests=(),
        primitive_plan=PrimitivePlan(
            route=route,
            engine_family=engine_family,
            primitives=(),
            adapters=(),
            blockers=(),
        ),
    )


# ---------------------------------------------------------------------------
# MarketDataValidator
# ---------------------------------------------------------------------------

class TestMarketDataValidator:
    def test_passes_when_required_access_present(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        source = '''
def evaluate(self, market_state):
    df = market_state.discount(T)
    vol = market_state.vol_surface(T, K)
    return black76_call(F, K, T, vol, df)
'''
        validator = MarketDataValidator()
        findings = validator.validate(source, _make_plan("analytical_black76"), spec)
        errors = [f for f in findings if f.severity == "error"]
        assert len(errors) == 0

    def test_flags_missing_discount_access(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        source = '''
def evaluate(self, market_state):
    vol = market_state.vol_surface(T, K)
    return black76_call(F, K, T, vol, 1.0)
'''
        validator = MarketDataValidator()
        findings = validator.validate(source, _make_plan("analytical_black76"), spec)
        errors = [f for f in findings if f.severity == "error"]
        assert any("discount_curve" in f.category for f in errors)

    def test_flags_hardcoded_rate(self):
        source = '''
def evaluate(self, market_state):
    r = 0.05
    return price(r)
'''
        validator = MarketDataValidator()
        findings = validator.validate(source, _make_plan("test"), None)
        warnings = [f for f in findings if f.category == "hardcoded_market_data"]
        assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# ParameterBindingValidator
# ---------------------------------------------------------------------------

class TestParameterBindingValidator:
    def test_passes_when_params_from_spec(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        source = '''
def evaluate(self, market_state):
    T = spec.maturity
    K = spec.strike
    return black76_call(F, K, T, vol, df)
'''
        validator = ParameterBindingValidator()
        findings = validator.validate(source, _make_plan("analytical_black76"), spec)
        param_findings = [f for f in findings if f.category.startswith("missing_")]
        assert len(param_findings) == 0

    def test_flags_suspicious_literal(self):
        source = '''
def evaluate(self, market_state):
    strike = 100.0
    maturity = 1.0
    return price(strike, maturity)
'''
        validator = ParameterBindingValidator()
        findings = validator.validate(source, _make_plan("test"), None)
        assert any("hardcoded_parameter" in f.category for f in findings)


# ---------------------------------------------------------------------------
# AlgorithmContractValidator
# ---------------------------------------------------------------------------

class TestAlgorithmContractValidator:
    def test_passes_when_engine_matches(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        source = '''
from trellis.models.monte_carlo.engine import MonteCarloEngine
engine = MonteCarloEngine(process, n_paths=10000)
paths = engine.simulate(T, n_steps)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("monte_carlo_paths", "monte_carlo"), spec)
        assert not any(f.category == "engine_family_mismatch" for f in findings)

    def test_flags_missing_engine(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        source = '''
def evaluate(self, market_state):
    return 42.0
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("monte_carlo_paths", "monte_carlo"), spec)
        assert any(f.category == "engine_family_mismatch" for f in findings)

    def test_flags_missing_route_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "quanto_adjustment_analytical"][0]
        source = '''
def evaluate(self, market_state):
    return black76_call(F, K, T, vol, df)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("quanto_adjustment_analytical"), spec)
        assert any(f.category == "route_helper_not_called" for f in findings)

    def test_flags_missing_callable_bond_route_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        callable_ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_fixed_income",
            exercise_style="bermudan",
            model_family="short_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, callable_ir))
        source = '''
def evaluate(self, market_state):
    return lattice_backward_induction(lattice, terminal_payoff)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("exercise_lattice", "lattice"), spec)
        assert any(f.category == "route_helper_not_called" for f in findings)

    def test_flags_missing_discount(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        source = '''
def evaluate(self, market_state):
    vol = market_state.vol_surface(T, K)
    return black76_call(F, K, T, vol, 1.0)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("analytical_black76"), spec)
        assert any(f.category == "missing_discount_application" for f in findings)

    def test_helper_backed_pde_route_does_not_require_low_level_engine_signatures(self, registry):
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        source = '''
from trellis.models.equity_option_pde import price_vanilla_equity_option_pde

def evaluate(self, market_state):
    return float(price_vanilla_equity_option_pde(market_state, self._spec, theta=0.5))
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("vanilla_equity_theta_pde", "pde_solver"), spec)
        assert not any(f.category == "engine_family_mismatch" for f in findings)

    def test_treats_lattice_policy_helper_as_exercise_logic(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        source = '''
from trellis.models.trees.control import resolve_lattice_exercise_policy

policy = resolve_lattice_exercise_policy("issuer_call", exercise_steps=[10, 20])
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("exercise_lattice", "lattice"), spec)
        assert not any(f.category == "missing_exercise_logic" for f in findings)


# ---------------------------------------------------------------------------
# Integrated validation
# ---------------------------------------------------------------------------

class TestIntegratedValidation:
    def test_warning_mode_always_passes(self, registry):
        source = "def evaluate(self, market_state): return 42.0"
        plan = _make_plan("analytical_black76")
        report = validate_generated_semantics(source, plan, mode="warning")
        assert report.ok  # warnings never block

    def test_blocking_mode_can_fail(self, registry):
        spec = [r for r in registry.routes if r.id == "quanto_adjustment_analytical"][0]
        source = "def evaluate(self, market_state): return 42.0"
        plan = _make_plan("quanto_adjustment_analytical")
        report = validate_generated_semantics(source, plan, route_spec=spec, mode="blocking")
        # Should have errors (missing route helper, missing market data, etc.)
        assert len(report.findings) > 0

    def test_returns_report_with_findings(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        source = '''
def evaluate(self, market_state):
    r = 0.05
    return r * 100
'''
        plan = _make_plan("analytical_black76")
        report = validate_generated_semantics(source, plan, route_spec=spec)
        assert isinstance(report, SemanticValidationReport)
        assert len(report.findings) > 0
