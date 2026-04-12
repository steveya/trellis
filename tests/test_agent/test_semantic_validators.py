"""Tests for semantic validators."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
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


def _make_plan(
    route: str,
    engine_family: str = "analytical",
    *,
    primitives: tuple[PrimitiveRef, ...] = (),
    adapters: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    route_family: str = "",
) -> GenerationPlan:
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
            primitives=primitives,
            adapters=adapters,
            blockers=(),
            notes=notes,
            route_family=route_family,
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

    def test_flags_raw_fx_rate_used_in_arithmetic(self):
        source = '''
def evaluate(self, market_state):
    spot = market_state.fx_rates[spec.fx_pair]
    return spot * 1.01
'''
        validator = MarketDataValidator()
        findings = validator.validate(source, _make_plan("test"), None)
        assert any(f.category == "fx_rate_scalar_extraction_missing" for f in findings)


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

    def test_importing_route_helper_without_calling_it_still_fails(self, registry):
        spec = [r for r in registry.routes if r.id == "quanto_adjustment_analytical"][0]
        source = '''
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state

def evaluate(self, market_state):
    return black76_call(F, K, T, vol, df)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("quanto_adjustment_analytical"), spec)
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

    def test_flags_exact_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        callable_ir = ProductIR(
            instrument="american_option",
            payoff_family="vanilla_option",
            exercise_style="american",
            model_family="equity_diffusion",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, callable_ir))
        source = '''
from trellis.models.equity_option_tree import price_vanilla_equity_option_tree

def evaluate(self, market_state):
    return price_vanilla_equity_option_tree(
        market_state=market_state,
        underlying=self._spec.underlying,
        expiry_date=self._spec.expiry_date,
        strike=self._spec.strike,
        exercise="american",
        steps=200,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("exercise_lattice", "lattice"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_fx_exact_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        source = '''
from trellis.models.fx_vanilla import price_fx_vanilla_analytical

def evaluate(self, market_state):
    return price_fx_vanilla_analytical(self._spec.option_type, resolved)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("analytical_garman_kohlhagen"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_fx_exact_helper_with_mixed_positional_and_keyword_args(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        source = '''
from trellis.models.fx_vanilla import price_fx_vanilla_analytical

def evaluate(self, market_state):
    return price_fx_vanilla_analytical(market_state, spec=self._spec)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("analytical_garman_kohlhagen"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_quanto_exact_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "quanto_adjustment_analytical"][0]
        source = '''
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state

def evaluate(self, market_state):
    return price_quanto_option_analytical_from_market_state(
        spec=self._spec,
        resolved_inputs=resolved,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("quanto_adjustment_analytical"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_callable_bond_tree_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        callable_ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_fixed_income",
            exercise_style="issuer_call",
            model_family="short_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, callable_ir))
        source = '''
from trellis.models.callable_bond_tree import price_callable_bond_tree

def evaluate(self, market_state):
    return price_callable_bond_tree(spec=self._spec, market_state=market_state, maturity=5.0)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("exercise_lattice", "lattice"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_callable_bond_tree_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        callable_ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_fixed_income",
            exercise_style="issuer_call",
            model_family="short_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, callable_ir))
        source = '''
from trellis.models.callable_bond_tree import price_callable_bond_tree

def evaluate(self, market_state):
    return price_callable_bond_tree(market_state, self._spec, model="hull_white", sigma=0.01)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("exercise_lattice", "lattice"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_rate_tree_swaption_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        swaption_ir = ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            model_family="interest_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, swaption_ir))
        source = '''
from trellis.models.rate_style_swaption_tree import price_swaption_tree

def evaluate(self, market_state):
    return price_swaption_tree(spec=self._spec, market_state=market_state, exercise_steps=12)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("rate_tree_backward_induction", "lattice"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_rate_tree_swaption_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        swaption_ir = ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            model_family="interest_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, swaption_ir))
        source = '''
from trellis.models.rate_style_swaption_tree import price_swaption_tree

def evaluate(self, market_state):
    return price_swaption_tree(market_state, self._spec, model="hull_white", mean_reversion=0.05)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("rate_tree_backward_induction", "lattice"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_zcb_option_tree_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_rate_tree"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, zcb_ir))
        source = '''
from trellis.models.zcb_option_tree import price_zcb_option_tree

def evaluate(self, market_state):
    return price_zcb_option_tree(self._spec, market_state, steps=100)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("zcb_option_rate_tree", "lattice"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_zcb_option_tree_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_rate_tree"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, zcb_ir))
        source = '''
from trellis.models.zcb_option_tree import price_zcb_option_tree

def evaluate(self, market_state):
    return price_zcb_option_tree(market_state, self._spec, model="ho_lee", n_steps=200)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("zcb_option_rate_tree", "lattice"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_zcb_option_jamshidian_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_analytical"][0]
        source = '''
from trellis.models.zcb_option import price_zcb_option_jamshidian

def evaluate(self, market_state):
    return price_zcb_option_jamshidian(self._spec, market_state, strike=0.63)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("zcb_option_analytical"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_zcb_option_jamshidian_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "zcb_option_analytical"][0]
        source = '''
from trellis.models.zcb_option import price_zcb_option_jamshidian

def evaluate(self, market_state):
    return price_zcb_option_jamshidian(market_state, self._spec, mean_reversion=0.1)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("zcb_option_analytical"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_credit_default_swap_analytical_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_analytical"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_analytical

def evaluate(self, market_state):
    return price_cds_analytical(
        notional=self._spec.notional,
        spread=self._spec.spread,
        recovery=self._spec.recovery_rate,
        schedule=schedule,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount_curve,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap_analytical"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_credit_default_swap_analytical_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_analytical"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_analytical

def evaluate(self, market_state):
    return price_cds_analytical(
        notional=self._spec.notional,
        spread_quote=self._spec.spread_quote,
        recovery=self._spec.recovery,
        schedule=schedule,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount_curve,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap_analytical"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_credit_default_swap_analytical_positional_calls(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_analytical"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_analytical

def evaluate(self, market_state):
    return price_cds_analytical(
        self._spec.notional,
        self._spec.spread_quote,
        self._spec.recovery,
        schedule,
        market_state.credit_curve,
        market_state.discount_curve,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap_analytical"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_credit_default_swap_monte_carlo_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_monte_carlo"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_monte_carlo

def evaluate(self, market_state):
    return price_cds_monte_carlo(
        notional=self._spec.notional,
        spread=self._spec.spread,
        recovery=self._spec.recovery_rate,
        schedule=schedule,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount_curve,
        paths=self._spec.n_paths,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap_monte_carlo", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_credit_default_swap_monte_carlo_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_monte_carlo"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_monte_carlo

def evaluate(self, market_state):
    return price_cds_monte_carlo(
        notional=self._spec.notional,
        spread_quote=self._spec.spread_quote,
        recovery=self._spec.recovery,
        schedule=schedule,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount_curve,
        n_paths=self._spec.n_paths,
        seed=self._spec.seed,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap_monte_carlo", "monte_carlo"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_credit_default_swap_monte_carlo_positional_calls(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap_monte_carlo"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_monte_carlo

def evaluate(self, market_state):
    return price_cds_monte_carlo(
        self._spec.notional,
        self._spec.spread_quote,
        self._spec.recovery,
        schedule,
        market_state.credit_curve,
        market_state.discount_curve,
        self._spec.n_paths,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap_monte_carlo", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_nth_to_default_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "nth_to_default_monte_carlo"][0]
        source = '''
from trellis.instruments.nth_to_default import price_nth_to_default_basket

def evaluate(self, market_state):
    return price_nth_to_default_basket(
        notional=self._spec.notional,
        n_names=len(self._spec.reference_entities),
        n_th=self._spec.nth_default,
        maturity=self._spec.maturity,
        correlation=self._spec.correlation,
        recovery=self._spec.recovery,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount_curve,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("nth_to_default_monte_carlo", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_nth_to_default_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "nth_to_default_monte_carlo"][0]
        source = '''
from trellis.instruments.nth_to_default import price_nth_to_default_basket

def evaluate(self, market_state):
    return price_nth_to_default_basket(
        notional=self._spec.notional,
        n_names=len(self._spec.reference_entities),
        n_th=self._spec.nth_default,
        horizon=self._spec.horizon,
        correlation=self._spec.correlation,
        recovery=self._spec.recovery,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount_curve,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("nth_to_default_monte_carlo", "monte_carlo"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_nth_to_default_positional_calls(self, registry):
        spec = [r for r in registry.routes if r.id == "nth_to_default_monte_carlo"][0]
        source = '''
from trellis.instruments.nth_to_default import price_nth_to_default_basket

def evaluate(self, market_state):
    return price_nth_to_default_basket(
        self._spec.notional,
        len(self._spec.reference_entities),
        self._spec.nth_default,
        self._spec.horizon,
        self._spec.correlation,
        self._spec.recovery,
        market_state.credit_curve,
        market_state.discount_curve,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("nth_to_default_monte_carlo", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_credit_basket_tranche_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "copula_loss_distribution"][0]
        source = '''
from trellis.models.credit_basket_copula import price_credit_basket_tranche

def evaluate(self, market_state):
    return price_credit_basket_tranche(
        spec=self._spec,
        copula_family="gaussian",
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("copula_loss_distribution", "copula"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_credit_basket_tranche_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "copula_loss_distribution"][0]
        source = '''
from trellis.models.credit_basket_copula import price_credit_basket_tranche

def evaluate(self, market_state):
    return price_credit_basket_tranche(
        market_state,
        self._spec,
        copula_family="gaussian",
        degrees_of_freedom=5.0,
        n_paths=40000,
        seed=42,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("copula_loss_distribution", "copula"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

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

    def test_uses_resolved_primitive_plan_when_route_spec_is_not_passed(self):
        source = '''
def evaluate(self, market_state):
    return lattice_backward_induction(lattice, terminal_payoff)
'''
        plan = _make_plan(
            "exercise_lattice",
            "lattice",
            primitives=(
                PrimitiveRef(
                    module="trellis.models.callable_bond_tree",
                    symbol="price_callable_bond_tree",
                    role="route_helper",
                ),
            ),
            route_family="callable_bond",
        )

        report = validate_generated_semantics(source, plan)

        assert not report.ok
        assert any(f.category == "route_helper_not_called" for f in report.findings)
        assert report.mode == "blocking"

    def test_fx_rate_scalar_extraction_is_blocking_by_default(self):
        source = '''
def evaluate(self, market_state):
    spot = market_state.fx_rates["EURUSD"]
    return spot * 1.01
'''
        report = validate_generated_semantics(source, _make_plan("test"))

        assert not report.ok
        assert any(f.category == "fx_rate_scalar_extraction_missing" for f in report.findings)
        assert report.mode == "blocking"
