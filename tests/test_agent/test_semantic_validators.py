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
    instrument_type: str | None = None,
    primitives: tuple[PrimitiveRef, ...] = (),
    adapters: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    route_family: str = "",
) -> GenerationPlan:
    return GenerationPlan(
        method=engine_family,
        instrument_type=instrument_type,
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

    def test_heston_model_parameter_route_rejects_black_vol_surface_access(self, registry):
        source = '''
def evaluate(self, market_state):
    sigma = market_state.vol_surface.black_vol(1.0, spec.strike)
    return sigma * spec.strike
'''
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        plan = _make_plan(
            "vanilla_equity_theta_pde",
            "pde_solver",
            instrument_type="heston_option",
        )

        validator = MarketDataValidator()
        findings = validator.validate(source, plan, spec)

        assert any(f.category == "heston_black_vol_surface_mismatch" for f in findings)

    def test_heston_black_vol_surface_mismatch_is_blocking_in_aggregate(self, registry):
        source = '''
def evaluate(self, market_state):
    sigma = market_state.vol_surface.black_vol(1.0, spec.strike)
    return sigma * spec.strike
'''
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        plan = _make_plan(
            "vanilla_equity_theta_pde",
            "pde_solver",
            instrument_type="heston_option",
        )

        report = validate_generated_semantics(source, plan, spec)

        assert not report.ok
        assert any(f.category == "heston_black_vol_surface_mismatch" for f in report.errors)


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

    def test_flags_incomplete_quanto_primitive_composition(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        source = '''
def evaluate(self, market_state):
    return black76_call(F, K, T, vol, df)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("equity_quanto"), spec)
        assert any(f.category == "required_primitive_not_called" for f in findings)

    def test_flags_missing_equity_barrier_pricing_kernel(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        barrier_ir = ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("barrier", "single_barrier", "terminal_markov"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        )
        primitives = resolve_route_primitives(spec, barrier_ir)
        source = '''
def evaluate(self, market_state):
    return 0.0
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan(
                "analytical_black76",
                instrument_type="barrier_option",
                primitives=primitives,
            ),
            spec,
        )

        assert any(
            finding.category == "required_primitive_not_called"
            and "barrier_option_price" in finding.message
            for finding in findings
        )

    def test_barrier_pricing_kernel_satisfies_analytical_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        barrier_ir = ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("barrier", "single_barrier", "terminal_markov"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="equity_diffusion",
        )
        primitives = resolve_route_primitives(spec, barrier_ir)
        source = '''
def evaluate(self, market_state):
    return barrier_option_price(spot, strike, barrier, rate, vol, time, q=carry)
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan(
                "analytical_black76",
                instrument_type="barrier_option",
                primitives=primitives,
            ),
            spec,
        )

        assert not any(
            finding.category == "engine_family_mismatch"
            for finding in findings
        )

    def test_fx_barrier_pricing_kernel_satisfies_analytical_engine_family(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_fx_barrier"][0]
        barrier_ir = ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("barrier", "single_barrier", "terminal_markov"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="fx",
        )
        primitives = resolve_route_primitives(spec, barrier_ir)
        source = '''
def evaluate(self, market_state):
    return barrier_option_price(spot, strike, barrier, rate, vol, time, q=carry)
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan(
                "analytical_fx_barrier",
                instrument_type="barrier_option",
                primitives=primitives,
            ),
            spec,
        )

        assert not any(
            finding.category == "engine_family_mismatch"
            for finding in findings
        )

    def test_fx_barrier_rejects_substituted_analytical_kernel(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_fx_barrier"][0]
        barrier_ir = ProductIR(
            instrument="barrier_option",
            payoff_family="barrier_option",
            payoff_traits=("barrier", "single_barrier", "terminal_markov"),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="fx",
        )
        primitives = resolve_route_primitives(spec, barrier_ir)
        source = '''
def evaluate(self, market_state):
    return garman_kohlhagen_price_raw(resolved)
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan(
                "analytical_fx_barrier",
                instrument_type="barrier_option",
                primitives=primitives,
            ),
            spec,
        )

        assert any(
            finding.category == "required_primitive_not_called"
            and "barrier_option_price" in finding.message
            for finding in findings
        )

    def test_non_owning_pricing_kernel_does_not_hide_missing_engine(self, registry):
        spec = [r for r in registry.routes if r.id == "local_vol_monte_carlo"][0]
        local_vol_ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            payoff_traits=("terminal_markov",),
            exercise_style="european",
            state_dependence="terminal_markov",
            model_family="local_volatility",
        )
        primitives = resolve_route_primitives(spec, local_vol_ir)
        source = '''
def evaluate(self, market_state):
    return local_vol_european_vanilla_price(spot, strike, rate, vol, time)
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan(
                "local_vol_monte_carlo",
                engine_family="monte_carlo",
                instrument_type="european_option",
                primitives=primitives,
            ),
            spec,
        )

        assert any(
            finding.category == "engine_family_mismatch"
            for finding in findings
        )

    def test_analytical_black76_helper_owned_rate_strip_does_not_require_internal_kernels(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        rate_strip_ir = ProductIR(
            instrument="cap",
            payoff_family="period_rate_option_strip",
            exercise_style="none",
            schedule_dependence=True,
            state_dependence="schedule_dependent",
            model_family="interest_rate",
        )
        primitives = resolve_route_primitives(spec, rate_strip_ir)
        source = '''
def evaluate(self, market_state):
    return price_rate_cap_floor_strip_analytical(market_state, self._spec)
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan(
                "analytical_black76",
                instrument_type="cap",
                primitives=primitives,
            ),
            spec,
        )

        assert not any(
            finding.category == "required_primitive_not_called"
            for finding in findings
        )

    def test_rejects_autocallable_shortcut_for_monte_carlo_paths(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        source = '''
from trellis.models.monte_carlo.variance_reduction import sobol_normals

def evaluate(self, market_state):
    return 0.0
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan(
                "monte_carlo_paths",
                "monte_carlo",
                primitives=(
                    PrimitiveRef(
                        "trellis.models.monte_carlo.event_aware",
                        "price_event_aware_monte_carlo",
                        "route_helper",
                    ),
                ),
            ),
            spec,
        )
        assert any(f.category == "route_helper_not_called" for f in findings)

    def test_rejects_retired_cliquet_helper_for_monte_carlo_paths(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        source = '''
from trellis.models.monte_carlo.event_aware import price_equity_cliquet_option_monte_carlo

def evaluate(self, market_state):
    return price_equity_cliquet_option_monte_carlo(market_state, self._spec)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan(
                "monte_carlo_paths",
                "monte_carlo",
                instrument_type="cliquet_option",
                primitives=(
                    PrimitiveRef(
                        "trellis.models.monte_carlo.event_aware",
                        "price_event_aware_monte_carlo",
                        "route_helper",
                    ),
                ),
            ),
            spec,
        )
        assert any(f.category == "route_helper_not_called" for f in findings)

    def test_rejects_double_barrier_terminal_payoff_without_pde_engine(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        source = '''
from trellis.models.analytical.support.barriers import terminal_double_barrier_payoff

def evaluate(self, market_state):
    return terminal_double_barrier_payoff([self._spec.spot], self._spec)[0]
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan("pde_theta_1d", "pde_solver"),
            spec,
        )
        assert any(f.category == "engine_family_mismatch" for f in findings)

    @pytest.mark.parametrize(
        ("method", "source"),
        [
            (
                "pde_solver",
                '''
from trellis.models.double_barrier_option import price_double_barrier_option_pde_result

def evaluate(self, market_state):
    return price_double_barrier_option_pde_result(market_state, self._spec).price
''',
            ),
            (
                "monte_carlo",
                '''
from trellis.models.double_barrier_option import price_double_barrier_option_monte_carlo_result

def evaluate(self, market_state):
    return price_double_barrier_option_monte_carlo_result(market_state, self._spec).price
''',
            ),
        ],
    )
    def test_accepts_helper_backed_double_barrier_route_without_low_level_findings(
        self,
        registry,
        method,
        source,
    ):
        from trellis.agent.platform_requests import compile_build_request

        compiled = compile_build_request(
            "Double barrier option via checked helper",
            instrument_type="barrier_option",
            preferred_method=method,
        )
        route_id = compiled.generation_plan.primitive_plan.route
        spec = [r for r in registry.routes if r.id == route_id][0]

        validator = AlgorithmContractValidator()
        findings = validator.validate(source, compiled.generation_plan, spec)

        assert findings == ()

    def test_flags_double_barrier_helper_signature_mismatch(self, registry):
        from trellis.agent.platform_requests import compile_build_request

        compiled = compile_build_request(
            "Double barrier option via checked helper",
            instrument_type="barrier_option",
            preferred_method="pde_solver",
        )
        route_id = compiled.generation_plan.primitive_plan.route
        spec = [r for r in registry.routes if r.id == route_id][0]
        source = '''
from trellis.models.double_barrier_option import price_double_barrier_option_pde_result

def evaluate(self, market_state):
    return price_double_barrier_option_pde_result(
        market_state=market_state,
        spec=self._spec,
        spot=self._spec.spot,
    ).price
'''

        validator = AlgorithmContractValidator()
        findings = validator.validate(source, compiled.generation_plan, spec)

        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_double_barrier_helper_duplicate_positional_keyword(self, registry):
        from trellis.agent.platform_requests import compile_build_request

        compiled = compile_build_request(
            "Double barrier option via checked helper",
            instrument_type="barrier_option",
            preferred_method="pde_solver",
        )
        route_id = compiled.generation_plan.primitive_plan.route
        spec = [r for r in registry.routes if r.id == route_id][0]
        source = '''
from trellis.models.double_barrier_option import price_double_barrier_option_pde_result

def evaluate(self, market_state):
    return price_double_barrier_option_pde_result(
        market_state,
        self._spec,
        spec=self._spec,
    ).price
'''

        validator = AlgorithmContractValidator()
        findings = validator.validate(source, compiled.generation_plan, spec)

        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_incidental_checked_helper_call_does_not_own_unrelated_route(self, registry):
        spec = [r for r in registry.routes if r.id == "pde_theta_1d"][0]
        source = '''
from trellis.models.heston import price_heston_option_monte_carlo

def evaluate(self, market_state):
    price_heston_option_monte_carlo(market_state, self._spec)
    return self._spec.spot
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan("pde_theta_1d", "pde_solver"),
            spec,
        )

        assert any(f.category == "engine_family_mismatch" for f in findings)

    def test_heston_adi_result_surface_satisfies_engine_signature(self, registry):
        spec = [r for r in registry.routes if r.id == "heston_adi_2d"][0]
        source = '''
from trellis.models.pde.heston_adi import price_heston_option_adi_pde_result

def evaluate(self, market_state):
    return price_heston_option_adi_pde_result(market_state, self._spec).price
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan("heston_adi_2d", "pde_solver"),
            spec,
        )
        assert not any(f.category == "engine_family_mismatch" for f in findings)

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

    def test_importing_retired_quanto_wrapper_does_not_satisfy_composition(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        source = '''
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state

def evaluate(self, market_state):
    return black76_call(F, K, T, vol, df)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("equity_quanto"), spec)
        assert any(f.category == "required_primitive_not_called" for f in findings)

    def test_prefers_plan_primitives_over_route_card_for_route_helper_checks(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        spec = replace(spec, primitives=())
        plan = _make_plan(
            "analytical_garman_kohlhagen",
            primitives=(
                PrimitiveRef(
                    "trellis.models.fx_vanilla",
                    "price_fx_vanilla_analytical",
                    "route_helper",
                ),
            ),
        )
        source = '''
def evaluate(self, market_state):
    return 42.0
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, plan, spec)
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

    def test_equity_tree_compatibility_helper_is_not_exact_route_authority(self, registry):
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
        assert all(primitive.role != "route_helper" for primitive in spec.primitives)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_fx_product_helper_is_not_exact_route_authority(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        source = '''
from trellis.models.fx_vanilla import price_fx_vanilla_analytical

def evaluate(self, market_state):
    return price_fx_vanilla_analytical(self._spec.option_type, resolved)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("analytical_garman_kohlhagen"), spec)
        assert all(primitive.role != "route_helper" for primitive in spec.primitives)
        assert any(f.category == "engine_family_mismatch" for f in findings)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_fx_raw_garman_kohlhagen_kernel(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_garman_kohlhagen"][0]
        source = '''
from trellis.models.analytical.fx import garman_kohlhagen_price_raw
from trellis.models.fx_vanilla import resolve_fx_vanilla_inputs

def evaluate(self, market_state):
    resolved = resolve_fx_vanilla_inputs(market_state, self._spec)
    return resolved.notional * garman_kohlhagen_price_raw(
        resolved.option_type,
        resolved.garman_kohlhagen,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("analytical_garman_kohlhagen"), spec)
        assert not any(f.category == "engine_family_mismatch" for f in findings)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_vanilla_equity_transform_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "transform_fft"][0]
        vanilla_ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, vanilla_ir))
        source = '''
from trellis.models.equity_option_transforms import price_vanilla_equity_option_transform

def evaluate(self, market_state):
    return price_vanilla_equity_option_transform(
        market_state=market_state,
        spec=self._spec,
        spot=self._spec.spot,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("transform_fft", "fft_pricing"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_vanilla_equity_transform_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "transform_fft"][0]
        vanilla_ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, vanilla_ir))
        source = '''
from trellis.models.equity_option_transforms import price_vanilla_equity_option_transform

def evaluate(self, market_state):
    return price_vanilla_equity_option_transform(market_state, self._spec, method="fft")
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("transform_fft", "fft_pricing"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_vanilla_equity_monte_carlo_helper_is_not_exact_route_authority(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        vanilla_ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, vanilla_ir))
        source = '''
from trellis.models.equity_option_monte_carlo import price_vanilla_equity_option_monte_carlo

def evaluate(self, market_state):
    return price_vanilla_equity_option_monte_carlo(
        market_state=market_state,
        spec=self._spec,
        spot=self._spec.spot,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("monte_carlo_paths", "monte_carlo"), spec)
        assert all(primitive.role != "route_helper" for primitive in spec.primitives)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_vanilla_equity_monte_carlo_compatibility_helper_has_no_route_signature_contract(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        vanilla_ir = ProductIR(
            instrument="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="equity_diffusion",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, vanilla_ir))
        source = '''
from trellis.models.equity_option_monte_carlo import price_vanilla_equity_option_monte_carlo

def evaluate(self, market_state):
    return price_vanilla_equity_option_monte_carlo(market_state, self._spec, n_paths=50000)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("monte_carlo_paths", "monte_carlo"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_legacy_vanilla_pde_helper_is_not_route_helper_authority(self, registry):
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        source = '''
from trellis.models.equity_option_pde import price_vanilla_equity_option_pde

def evaluate(self, market_state):
    return price_vanilla_equity_option_pde(
        market_state=market_state,
        spec=self._spec,
        strike=self._spec.strike,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("vanilla_equity_theta_pde", "pde_solver"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)
        assert any(f.category == "engine_family_mismatch" for f in findings)

    def test_rejects_legacy_vanilla_equity_pde_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        source = '''
from trellis.models.equity_option_pde import price_vanilla_equity_option_pde

def evaluate(self, market_state):
    return price_vanilla_equity_option_pde(market_state, self._spec, theta=0.5)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("vanilla_equity_theta_pde", "pde_solver"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)
        assert any(f.category == "engine_family_mismatch" for f in findings)

    def test_rejects_quanto_product_wrapper_as_route_implementation(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        source = '''
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state

def evaluate(self, market_state):
    return price_quanto_option_analytical_from_market_state(
        spec=self._spec,
        resolved_inputs=resolved,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("equity_quanto"), spec)
        assert any(f.category == "required_primitive_not_called" for f in findings)

    @pytest.mark.parametrize(
        "non_call_source",
        [
            "# black76_call(forward, strike, vol, T)",
            '"""black76_call(forward, strike, vol, T)"""',
            "def black76_call(forward, strike, vol, T):\n        return 0.0",
        ],
    )
    def test_rejects_non_call_text_for_required_quanto_primitive(
        self,
        registry,
        non_call_source,
    ):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        source = f'''
def evaluate(self, market_state):
    resolved = resolve_quanto_inputs(market_state, self._spec)
    option_type = normalized_option_type(self._spec.option_type)
    if resolved.T <= 0.0:
        return terminal_intrinsic(option_type, spot=resolved.spot, strike=self._spec.strike)
    forward = quanto_adjusted_forward(
        spot=resolved.spot,
        domestic_df=resolved.domestic_df,
        foreign_df=resolved.foreign_df,
        corr=resolved.corr,
        sigma_underlier=resolved.sigma_underlier,
        sigma_fx=resolved.sigma_fx,
        T=resolved.T,
    )
    {non_call_source}
    call = 0.0
    put = black76_put(forward, self._spec.strike, resolved.sigma_underlier, resolved.T)
    return discounted_value(call if option_type == "call" else put, resolved.domestic_df)
'''
        validator = AlgorithmContractValidator()

        findings = validator.validate(source, _make_plan("equity_quanto"), spec)

        assert any(
            finding.category == "required_primitive_not_called"
            and "'black76_call'" in finding.message
            for finding in findings
        )

    def test_accepts_complete_quanto_analytical_primitive_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        source = '''
def evaluate(self, market_state):
    resolved = resolve_quanto_inputs(market_state, self._spec)
    option_type = normalized_option_type(self._spec.option_type)
    if resolved.T <= 0.0:
        return terminal_intrinsic(option_type, spot=resolved.spot, strike=self._spec.strike)
    forward = quanto_adjusted_forward(
        spot=resolved.spot,
        domestic_df=resolved.domestic_df,
        foreign_df=resolved.foreign_df,
        corr=resolved.corr,
        sigma_underlier=resolved.sigma_underlier,
        sigma_fx=resolved.sigma_fx,
        T=resolved.T,
    )
    call = black76_call(forward, self._spec.strike, resolved.sigma_underlier, resolved.T)
    put = black76_put(forward, self._spec.strike, resolved.sigma_underlier, resolved.T)
    return discounted_value(call if option_type == "call" else put, resolved.domestic_df)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("equity_quanto"), spec)
        assert not any(f.category == "required_primitive_not_called" for f in findings)

    @pytest.mark.parametrize(
        ("method", "source"),
        [
            (
                "monte_carlo",
                '''
def evaluate(self, market_state):
    resolved = resolve_quanto_inputs(market_state, self._spec)
    option_type = normalized_option_type(self._spec.option_type)
    forward = quanto_adjusted_forward(resolved.spot, resolved.foreign_df, resolved.domestic_df, resolved.corr, resolved.sigma_underlier, resolved.sigma_fx, resolved.T)
    call = black76_call(forward, resolved.strike, resolved.sigma_underlier, resolved.T)
    put = black76_put(forward, resolved.strike, resolved.sigma_underlier, resolved.T)
    return discounted_value(call if option_type == "call" else put, resolved.domestic_df)
''',
            ),
            (
                "analytical",
                '''
def evaluate(self, market_state):
    resolved = resolve_quanto_inputs(market_state, self._spec)
    rate = implied_zero_rate(resolved.domestic_df, resolved.T)
    process = CorrelatedGBM(mu=[rate, rate], sigma=[0.2, 0.1], corr=[[1.0, 0.0], [0.0, 1.0]])
    engine = MonteCarloEngine(process)
    payoff = terminal_value_payoff(lambda terminal: terminal_intrinsic(terminal[..., 0], resolved.strike, "call"))
    return engine.price(get_numpy().array([resolved.spot, resolved.fx_spot]), resolved.T, 4, payoff)
''',
            ),
        ],
    )
    def test_rejects_quanto_method_substitution(self, registry, method, source):
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan("equity_quanto", method),
            spec,
        )
        assert any(f.category == "required_primitive_not_called" for f in findings)

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

    def test_rejects_callable_bond_tree_positional_optional_argument(self, registry):
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
    return price_callable_bond_tree(market_state, self._spec, "hull_white")
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("exercise_lattice", "lattice"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_rate_tree_swaption_missing_required_composition_primitive(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        swaption_ir = ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            model_family="interest_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, swaption_ir))
        source = '''
from trellis.models.bermudan_swaption_tree import BermudanSwaptionTreeSpec

def evaluate(self, market_state):
    return BermudanSwaptionTreeSpec(
        notional=self._spec.notional,
        strike=self._spec.strike,
        exercise_dates=(self._spec.expiry_date,),
        swap_end=self._spec.swap_end,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("rate_tree_backward_induction", "lattice"), spec)
        assert any(
            f.category == "required_primitive_not_called"
            and "resolve_swaption_curve_basis_spread" in f.message
            for f in findings
        )

    def test_accepts_rate_tree_swaption_generic_lattice_composition(self, registry):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        swaption_ir = ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            model_family="interest_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, swaption_ir))
        source = '''
from trellis.models.bermudan_swaption_tree import (
    BermudanSwaptionTreeSpec,
    compile_bermudan_swaption_contract_spec,
    resolve_bermudan_swaption_tree_inputs,
)
from trellis.models.rate_style_swaption import resolve_swaption_curve_basis_spread
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    TERM_STRUCTURE_TARGET,
    UNIFORM_ADDITIVE_MESH,
    build_lattice,
    price_on_lattice,
)

def evaluate(self, market_state):
    spread = resolve_swaption_curve_basis_spread(market_state, self._spec)
    spec = BermudanSwaptionTreeSpec(
        notional=self._spec.notional,
        strike=self._spec.strike - spread,
        exercise_dates=(self._spec.expiry_date,),
        swap_end=self._spec.swap_end,
    )
    resolved = resolve_bermudan_swaption_tree_inputs(market_state, spec)
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        "hull_white",
        TERM_STRUCTURE_TARGET(market_state.discount),
        r0=resolved.r0,
        sigma=resolved.sigma,
        a=resolved.mean_reversion,
        T=resolved.tree_horizon,
        n_steps=resolved.n_steps,
    )
    contract = compile_bermudan_swaption_contract_spec(
        lattice,
        spec=spec,
        settlement=resolved.settlement,
    )
    return price_on_lattice(lattice, contract)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("rate_tree_backward_induction", "lattice"), spec)
        assert not any(f.severity == "error" for f in findings)

    def test_rate_tree_swaption_function_reference_does_not_satisfy_composition(
        self, registry
    ):
        spec = [r for r in registry.routes if r.id == "rate_tree_backward_induction"][0]
        swaption_ir = ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            model_family="interest_rate",
        )
        spec = replace(spec, primitives=resolve_route_primitives(spec, swaption_ir))
        source = '''
from trellis.models.trees.algebra import price_on_lattice

def evaluate(self, market_state):
    unused_kernel = price_on_lattice
    return 0.0
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan("rate_tree_backward_induction", "lattice"),
            spec,
        )

        assert any(
            finding.category == "required_primitive_not_called"
            and "price_on_lattice" in finding.message
            for finding in findings
        )

    def test_flags_zcb_option_tree_helper_signature_mismatch(self, registry):
        # QUA-915: ZCB-option family collapsed into ``short_rate_bond_option``.
        # The rate-tree branch still routes to ``price_zcb_option_tree``.
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(
            spec,
            primitives=resolve_route_primitives(spec, zcb_ir, method="rate_tree"),
        )
        source = '''
from trellis.models.zcb_option_tree import price_zcb_option_tree

def evaluate(self, market_state):
    return price_zcb_option_tree(self._spec, market_state, steps=100)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option", "lattice"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_zcb_option_tree_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(
            spec,
            primitives=resolve_route_primitives(spec, zcb_ir, method="rate_tree"),
        )
        source = '''
from trellis.models.zcb_option_tree import price_zcb_option_tree

def evaluate(self, market_state):
    return price_zcb_option_tree(market_state, self._spec, model="ho_lee", n_steps=200)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option", "lattice"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_zcb_option_jamshidian_helper_signature_mismatch(self, registry):
        # QUA-915: the analytical branch of the collapsed route still
        # routes to ``price_zcb_option_jamshidian`` and must keep the
        # same helper-signature enforcement it had pre-collapse.
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(
            spec,
            primitives=resolve_route_primitives(spec, zcb_ir, method="analytical"),
        )
        source = '''
from trellis.models.zcb_option import price_zcb_option_jamshidian

def evaluate(self, market_state):
    return price_zcb_option_jamshidian(self._spec, market_state, strike=0.63)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_zcb_option_jamshidian_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(
            spec,
            primitives=resolve_route_primitives(spec, zcb_ir, method="analytical"),
        )
        source = '''
from trellis.models.zcb_option import price_zcb_option_jamshidian

def evaluate(self, market_state):
    return price_zcb_option_jamshidian(market_state, self._spec, mean_reversion=0.1)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_zcb_option_jamshidian_positional_optional_argument(self, registry):
        spec = [r for r in registry.routes if r.id == "short_rate_bond_option"][0]
        zcb_ir = ProductIR(
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
        )
        spec = replace(
            spec,
            primitives=resolve_route_primitives(spec, zcb_ir, method="analytical"),
        )
        source = '''
from trellis.models.zcb_option import price_zcb_option_jamshidian

def evaluate(self, market_state):
    return price_zcb_option_jamshidian(market_state, self._spec, 0.1)
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_credit_default_swap_analytical_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
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
        findings = validator.validate(source, _make_plan("credit_default_swap"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_credit_default_swap_analytical_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
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
        findings = validator.validate(source, _make_plan("credit_default_swap"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_credit_default_swap_analytical_positional_calls(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
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
        findings = validator.validate(source, _make_plan("credit_default_swap"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_credit_default_swap_monte_carlo_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
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
        findings = validator.validate(source, _make_plan("credit_default_swap", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_credit_default_swap_monte_carlo_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
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
        findings = validator.validate(source, _make_plan("credit_default_swap", "monte_carlo"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_credit_default_swap_monte_carlo_positional_calls(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
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
        findings = validator.validate(source, _make_plan("credit_default_swap", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_flags_nth_to_default_helper_signature_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_basket_nth_to_default"][0]
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
        findings = validator.validate(source, _make_plan("credit_basket_nth_to_default", "monte_carlo"), spec)
        assert any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_accepts_nth_to_default_helper_surface(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_basket_nth_to_default"][0]
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
        findings = validator.validate(source, _make_plan("credit_basket_nth_to_default", "monte_carlo"), spec)
        assert not any(f.category == "route_helper_signature_mismatch" for f in findings)

    def test_rejects_nth_to_default_positional_calls(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_basket_nth_to_default"][0]
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
        findings = validator.validate(source, _make_plan("credit_basket_nth_to_default", "monte_carlo"), spec)
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

    def test_helper_backed_pde_route_does_not_satisfy_engine_contract(self, registry):
        spec = [r for r in registry.routes if r.id == "vanilla_equity_theta_pde"][0]
        source = '''
from trellis.models.equity_option_pde import price_vanilla_equity_option_pde

def evaluate(self, market_state):
    return float(price_vanilla_equity_option_pde(market_state, self._spec, theta=0.5))
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("vanilla_equity_theta_pde", "pde_solver"), spec)
        assert any(f.category == "engine_family_mismatch" for f in findings)

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
        spec = [r for r in registry.routes if r.id == "equity_quanto"][0]
        source = "def evaluate(self, market_state): return 42.0"
        plan = _make_plan("equity_quanto")
        report = validate_generated_semantics(source, plan, route_spec=spec, mode="blocking")
        # Should have errors (missing required primitives, market data, etc.)
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
