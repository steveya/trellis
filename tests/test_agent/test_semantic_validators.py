"""Tests for semantic validators."""

from __future__ import annotations

import textwrap
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
    payoff_spec_name: str = "",
    payoff_spec_fields: tuple[tuple[str, str, str | None], ...] = (),
) -> GenerationPlan:
    return GenerationPlan(
        method=engine_family,
        instrument_type=instrument_type,
        inspected_modules=(),
        approved_modules=(),
        symbols_to_reuse=(),
        proposed_tests=(),
        payoff_spec_name=payoff_spec_name,
        payoff_spec_fields=payoff_spec_fields,
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


def _cds_composition_source(
    *,
    return_expression: str = (
        "protection_leg - premium_leg - accrued_on_event + accrued_to_valuation"
    ),
    survival_index: str = "interval_stop - 1",
    event_index: str = "interval_index",
    premium_discount: str = (
        "market_state.discount.discount(grid.period_payment_times[period_index])"
    ),
    event_discount: str = (
        "market_state.discount.discount(interval.settlement_time)"
    ),
    initial_survival: str | None = None,
    premium_notional: str = "self._spec.notional",
    protection_notional: str = "self._spec.notional",
    event_notional: str = "self._spec.notional",
    premium_rate: str = "spread",
    event_rate: str = "spread",
    recovery: str = "self._spec.recovery",
    valuation_adjustment: str = (
        "self._spec.notional * spread * period.accrual_fraction "
        "* grid.elapsed_period_fractions[period_index]"
    ),
    time_origin: str = "self._spec.valuation_date or self._spec.start_date",
    weight_grid: str = "grid",
    extra_setup: str = "",
    post_weights_setup: str = "",
    schedule_start: str = "self._spec.start_date",
    schedule_end: str = "self._spec.end_date",
    schedule_frequency: str = "self._spec.frequency",
    schedule_day_count: str = "self._spec.day_count",
    schedule_calendar: str = "WEEKEND_ONLY",
    schedule_bda: str = "BusinessDayAdjustment.FOLLOWING",
    schedule_roll: str = "RollConvention.NONE",
    schedule_stub: str = "StubType.SHORT_LAST",
    schedule_payment_lag: str = "0",
    conditional_credit_curve: str = "market_state.credit_curve",
    weight_symbol: str = "expected_first_event_weights",
    weight_controls: str = "",
    additional_weight_call: str = "",
    weight_owner: str = "weights",
    scheduled_accrual: str = "period.accrual_fraction",
    event_accrual: str = (
        "period.accrual_fraction * interval.period_fraction_elapsed"
    ),
) -> str:
    """Build a compact, structurally complete CDS composition for validator tests."""
    if initial_survival is None:
        initial_survival = (
            "market_state.credit_curve.survival_probability("
            f"{weight_grid}.intervals[0].start_time)"
        )
    return f'''
def evaluate(self, market_state):
    from trellis.conventions.calendar import BusinessDayAdjustment, WEEKEND_ONLY
    from trellis.conventions.schedule import RollConvention, StubType
    from trellis.core.date_utils import build_period_schedule
    from trellis.models.contingent_cashflows import (
        CouponAccrual,
        ProtectionPayment,
        build_default_event_grid,
        conditional_event_probabilities_from_curve,
        coupon_cashflow_pv,
        {weight_symbol},
        protection_payment_pv,
    )

    schedule = build_period_schedule(
        {schedule_start},
        {schedule_end},
        {schedule_frequency},
        day_count={schedule_day_count},
        time_origin={time_origin},
        calendar={schedule_calendar},
        bda={schedule_bda},
        roll_convention={schedule_roll},
        stub={schedule_stub},
        payment_lag_days={schedule_payment_lag},
    )
    grid = build_default_event_grid(schedule)
    {extra_setup}
    conditional = conditional_event_probabilities_from_curve(
        {conditional_credit_curve},
        {weight_grid}.intervals,
    )
    initial_survival_weight = {initial_survival}
    weights = {weight_symbol}(
        conditional,
        initial_survival_weight=initial_survival_weight,
        {weight_controls}
    )
    {additional_weight_call}
    {post_weights_setup}
    spread = float(self._spec.spread)
    if spread > 1.0:
        spread *= 1e-4
    premium_leg = 0.0
    protection_leg = 0.0
    accrued_on_event = 0.0
    accrued_to_valuation = 0.0
    interval_start = 0
    for period_index, period in enumerate(grid.periods):
        interval_stop = grid.period_interval_stops[period_index]
        if interval_stop <= interval_start:
            interval_start = interval_stop
            continue
        survival_weight = {weight_owner}.survival_weights[{survival_index}]
        premium_leg += coupon_cashflow_pv(CouponAccrual(
            notional={premium_notional},
            rate={premium_rate},
            accrual={scheduled_accrual},
            discount_factor={premium_discount},
            weight=survival_weight,
        ))
        accrued_to_valuation += {valuation_adjustment}
        for interval_index in range(interval_start, interval_stop):
            interval = grid.intervals[interval_index]
            event_weight = {weight_owner}.event_weights[{event_index}]
            event_discount = {event_discount}
            protection_leg += protection_payment_pv(ProtectionPayment(
                notional={protection_notional},
                recovery={recovery},
                default_probability=event_weight,
                discount_factor=event_discount,
            ))
            accrued_on_event += coupon_cashflow_pv(CouponAccrual(
                notional={event_notional},
                rate={event_rate},
                accrual={event_accrual},
                discount_factor=event_discount,
                weight=event_weight,
            ))
        interval_start = interval_stop
    return float({return_expression})
'''


def _guard_cds_accumulator(
    source: str,
    *,
    accumulator: str,
    condition: str,
) -> str:
    """Move one loop accumulator block beneath a synthetic branch guard."""
    lines = source.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith(f"{accumulator} +=")
    )
    indentation = len(lines[start]) - len(lines[start].lstrip())
    stop = start + 1
    while stop < len(lines):
        line = lines[stop]
        if line.strip() and len(line) - len(line.lstrip()) <= indentation:
            break
        stop += 1
    guarded = [" " * indentation + f"if {condition}:"] + [
        "    " + line for line in lines[start:stop]
    ]
    return "\n".join(lines[:start] + guarded + lines[stop:]) + "\n"


def _guard_cds_interval_loop(source: str, *, condition: str) -> str:
    """Move the interval loop beneath a synthetic branch guard."""
    lines = source.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("for interval_index in range(")
    )
    indentation = len(lines[start]) - len(lines[start].lstrip())
    stop = start + 1
    while stop < len(lines):
        line = lines[stop]
        if line.strip() and len(line) - len(line.lstrip()) <= indentation:
            break
        stop += 1
    guarded = [" " * indentation + f"if {condition}:"] + [
        "    " + line for line in lines[start:stop]
    ]
    return "\n".join(lines[:start] + guarded + lines[stop:]) + "\n"


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

    def test_flags_missing_callable_bond_primitive_composition(self, registry):
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
        assert any(f.category == "required_primitive_not_called" for f in findings)

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

    def test_callable_bond_tree_wrapper_does_not_satisfy_primitive_composition(self, registry):
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
        assert any(f.category == "required_primitive_not_called" for f in findings)

    def test_callable_bond_tree_wrapper_is_not_route_authority(self, registry):
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
        assert any(f.category == "required_primitive_not_called" for f in findings)

    def test_callable_bond_tree_positional_call_does_not_satisfy_composition(self, registry):
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
        assert any(f.category == "required_primitive_not_called" for f in findings)

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

    def test_rejects_zcb_option_tree_compatibility_helper(self, registry):
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
        assert any(f.category == "zcb_option_forbidden_helper" for f in findings)

    def test_accepts_zcb_option_generic_lattice_composition(self, registry):
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
from trellis.models.resolution.short_rate_claims import resolve_discount_bond_claim_inputs
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    UNIFORM_ADDITIVE_MESH,
    TERM_STRUCTURE_TARGET,
    build_lattice,
)
from trellis.models.trees.control import lattice_step_from_time
from trellis.models.trees.lattice import (
    lattice_backward_induction,
    lattice_backward_induction_result,
)
from trellis.models.trees.models import MODEL_REGISTRY

def evaluate(self, market_state):
    claim = resolve_discount_bond_claim_inputs(market_state, self._spec, model="ho_lee")
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        MODEL_REGISTRY["ho_lee"],
        TERM_STRUCTURE_TARGET(market_state.discount),
        r0=claim.regime.initial_rate,
        sigma=claim.regime.sigma,
        a=claim.regime.mean_reversion,
        maturity=claim.bond_maturity_time,
        n_steps=200,
    )
    expiry_step = lattice_step_from_time(
        claim.expiry_time, dt=lattice.dt, n_steps=lattice.n_steps
    )
    bond_step = lattice_step_from_time(
        claim.bond_maturity_time, dt=lattice.dt, n_steps=lattice.n_steps
    )
    bond = lattice_backward_induction_result(
        lattice,
        terminal_value=1.0,
        terminal_step=bond_step,
        observation_steps=(expiry_step,),
    )
    bond_values = bond.observation_at(expiry_step).post_control_values
    return lattice_backward_induction(
        lattice,
        terminal_payoff=lambda step, node, lattice_: max(
            bond_values[node] - claim.strike_unit, 0.0
        ) * claim.notional,
        terminal_step=expiry_step,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option", "lattice"), spec)
        assert not any(f.severity == "error" for f in findings)

    def test_rejects_zcb_option_jamshidian_compatibility_helper(self, registry):
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
        assert any(f.category == "zcb_option_forbidden_helper" for f in findings)

    def test_accepts_zcb_option_raw_jamshidian_composition(self, registry):
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
from trellis.models.analytical.jamshidian import ResolvedJamshidianInputs, zcb_option_hw_raw
from trellis.models.resolution.short_rate_claims import resolve_discount_bond_claim_inputs

def evaluate(self, market_state):
    claim = resolve_discount_bond_claim_inputs(
        market_state,
        self._spec,
        model="hull_white",
    )
    inputs = ResolvedJamshidianInputs(
        discount_factor_expiry=claim.discount_factor_expiry,
        discount_factor_bond=claim.discount_factor_bond,
        strike=claim.strike_unit,
        T_exp=claim.expiry_time,
        T_bond=claim.bond_maturity_time,
        sigma=claim.regime.sigma,
        a=claim.regime.mean_reversion,
    )
    return claim.notional * zcb_option_hw_raw(inputs)[claim.option_type]
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("short_rate_bond_option"), spec)
        assert not any(f.severity == "error" for f in findings)

    def test_rejects_credit_default_swap_analytical_product_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_analytical

def evaluate(self, market_state):
    return price_cds_analytical(
        notional=self._spec.notional,
        spread_quote=self._spec.spread,
        recovery=self._spec.recovery,
        schedule=schedule,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap"), spec)
        assert any(
            f.category == "credit_default_swap_forbidden_helper"
            for f in findings
        )

    def test_rejects_credit_default_swap_monte_carlo_product_helper(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = '''
from trellis.models.credit_default_swap import price_cds_monte_carlo

def evaluate(self, market_state):
    return price_cds_monte_carlo(
        notional=self._spec.notional,
        spread_quote=self._spec.spread,
        recovery=self._spec.recovery,
        schedule=schedule,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount,
        n_paths=self._spec.n_paths,
        seed=42,
    )
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            source,
            _make_plan("credit_default_swap", "monte_carlo"),
            spec,
        )
        assert any(
            f.category == "credit_default_swap_forbidden_helper"
            for f in findings
        )

    @pytest.mark.parametrize(
        "weight_call",
        (
            "expected_first_event_weights(conditional)",
            "expected_first_event_weights(conditional, initial_survival_weight=1.0)",
            (
                "expected_first_event_weights(conditional, "
                "initial_survival_weight=market_state.credit_curve."
                "survival_probability(grid.intervals[-1].start_time))"
            ),
            "sample_first_event_weights(conditional, n_paths=10000, seed=42)",
        ),
    )
    def test_rejects_credit_default_swap_weights_without_initial_survival(
        self,
        registry,
        weight_call,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = f'''
def evaluate(self, market_state):
    weights = {weight_call}
    return weights
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_initial_survival_missing"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("extra_setup", "weight_controls"),
        (
            ("opaque_args = ()", "*opaque_args,"),
            ("opaque_kwargs = {}", "**opaque_kwargs,"),
        ),
    )
    def test_rejects_credit_default_swap_opaque_weight_arguments(
        self,
        registry,
        extra_setup,
        weight_controls,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(
            extra_setup=extra_setup,
            weight_controls=weight_controls,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_first_event_call_shape"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "initial_survival",
        (
            (
                "0.5 * market_state.credit_curve.survival_probability("
                "grid.intervals[0].start_time)"
            ),
            (
                "market_state.credit_curve.survival_probability("
                "grid.intervals[0].start_time) + 0.1"
            ),
        ),
    )
    def test_rejects_credit_default_swap_scaled_initial_survival(
        self,
        registry,
        initial_survival,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(initial_survival=initial_survival),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_initial_survival_missing"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "time_origin",
        (
            "self._spec.start_date",
            "self._spec.end_date",
        ),
    )
    def test_rejects_credit_default_swap_non_valuation_event_grid(
        self,
        registry,
        time_origin,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(time_origin=time_origin),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_valuation_origin"
            for finding in findings
        )

    def test_rejects_credit_default_swap_weight_grid_mismatch(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        extra_setup = '''
    other_schedule = build_period_schedule(
        self._spec.start_date,
        self._spec.end_date,
        self._spec.frequency,
        day_count=self._spec.day_count,
        time_origin=self._spec.valuation_date or self._spec.start_date,
    )
    other_grid = build_default_event_grid(other_schedule)
'''

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                weight_grid="other_grid",
                extra_setup=extra_setup,
            ),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_valuation_origin"
            for finding in findings
        )

    def test_rejects_credit_default_swap_single_period_composition(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = '''
from trellis.core.date_utils import build_period_schedule
from trellis.models.contingent_cashflows import (
    CouponAccrual,
    ProtectionPayment,
    build_default_event_grid,
    conditional_event_probabilities_from_curve,
    coupon_cashflow_pv,
    expected_first_event_weights,
    protection_payment_pv,
)

def evaluate(self, market_state):
    schedule = build_period_schedule(
        self._spec.start_date,
        self._spec.end_date,
        self._spec.frequency,
        day_count=self._spec.day_count,
        time_origin=self._spec.start_date,
    )
    grid = build_default_event_grid(schedule)
    conditional = conditional_event_probabilities_from_curve(
        market_state.credit_curve,
        grid.intervals,
    )
    credit_curve = market_state.credit_curve
    initial_survival_weight = credit_curve.survival_probability(
        grid.intervals[0].start_time,
    )
    weights = expected_first_event_weights(
        conditional,
        initial_survival_weight=initial_survival_weight,
    )
    premium = coupon_cashflow_pv(CouponAccrual(
        notional=self._spec.notional,
        rate=self._spec.spread,
        accrual=grid.periods[0].accrual_fraction,
        discount_factor=market_state.discount.discount(grid.period_payment_times[0]),
        weight=weights.survival_weights[0],
    ))
    protection = protection_payment_pv(ProtectionPayment(
        notional=self._spec.notional,
        recovery=self._spec.recovery,
        default_probability=weights.event_weights[0],
        discount_factor=market_state.discount.discount(grid.intervals[0].settlement_time),
    ))
    return protection - premium
'''
        validator = AlgorithmContractValidator()
        findings = validator.validate(source, _make_plan("credit_default_swap"), spec)
        assert any(
            f.category == "credit_default_swap_incomplete_event_grid"
            for f in findings
        )
        assert not any(
            f.category == "credit_default_swap_initial_survival_missing"
            for f in findings
        )

    @pytest.mark.parametrize(
        ("period_iterable", "interval_iterable"),
        (
            ("grid.periods[:1]", "range(interval_start, interval_stop)"),
            ("grid.periods", "range(0, 1)"),
        ),
    )
    def test_rejects_credit_default_swap_partial_grid_loops(
        self,
        registry,
        period_iterable,
        interval_iterable,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = f'''
def evaluate(self, market_state):
    initial_survival_weight = market_state.credit_curve.survival_probability(
        grid.intervals[0].start_time,
    )
    weights = expected_first_event_weights(
        conditional,
        initial_survival_weight=initial_survival_weight,
    )
    premium_leg = 0.0
    protection_leg = 0.0
    interval_start = 0
    for period_index, period in enumerate({period_iterable}):
        interval_stop = grid.period_interval_stops[period_index]
        premium_leg += coupon_cashflow_pv(CouponAccrual(
            notional=1.0,
            rate=0.01,
            accrual=period.accrual_fraction,
            discount_factor=1.0,
            weight=weights.survival_weights[interval_stop - 1],
        ))
        for interval_index in {interval_iterable}:
            interval = grid.intervals[interval_index]
            protection_leg += protection_payment_pv(ProtectionPayment(
                notional=1.0,
                recovery=0.4,
                default_probability=weights.event_weights[interval_index],
                discount_factor=1.0,
            ))
        interval_start = interval_stop
    return protection_leg - premium_leg
'''

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "return_expression",
        (
            "premium_leg - protection_leg - accrued_on_event + accrued_to_valuation",
            "protection_leg - premium_leg + accrued_on_event + accrued_to_valuation",
            "protection_leg - premium_leg",
        ),
    )
    def test_rejects_credit_default_swap_wrong_leg_signs(
        self,
        registry,
        return_expression,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(return_expression=return_expression),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_sign_convention"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("accumulator", "condition"),
        (
            ("premium_leg", "False"),
            ("protection_leg", "False"),
            ("protection_leg", "event_weight <= 0.0"),
        ),
    )
    def test_rejects_credit_default_swap_legs_hidden_in_invalid_branch(
        self,
        registry,
        accumulator,
        condition,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _guard_cds_accumulator(
            _cds_composition_source(),
            accumulator=accumulator,
            condition=condition,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_leg_after_unconditional_continue(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "            protection_leg +=",
            "            continue\n            protection_leg +=",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("anchor", "guard"),
        (
            (
                "        survival_weight =",
                "        if True:\n            continue\n",
            ),
            (
                "        survival_weight =",
                "        if True:\n            break\n",
            ),
            (
                "            interval =",
                "            if True:\n                continue\n",
            ),
            (
                "            interval =",
                "            if True:\n                break\n",
            ),
        ),
    )
    def test_rejects_credit_default_swap_conditional_loop_exit_before_assembly(
        self,
        registry,
        anchor,
        guard,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            anchor,
            guard + anchor,
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "replacement",
        (
            "            break\n"
            "        interval_start = interval_stop\n"
            "    return",
            "            if True:\n"
            "                break\n"
            "        interval_start = interval_stop\n"
            "    return",
            "        interval_start = interval_stop\n"
            "        break\n"
            "    return",
            "        interval_start = interval_stop\n"
            "        if True:\n"
            "            break\n"
            "    return",
        ),
    )
    def test_rejects_credit_default_swap_loop_exit_after_assembly(
        self,
        registry,
        replacement,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "        interval_start = interval_stop\n    return",
            replacement,
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_assert_before_assembly(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            "    assert self._spec.notional\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_missing_empty_period_guard(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "        if interval_stop <= interval_start:\n"
            "            interval_start = interval_stop\n"
            "            continue\n",
            "",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_late_empty_period_guard(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        guard = (
            "        if interval_stop <= interval_start:\n"
            "            interval_start = interval_stop\n"
            "            continue\n"
        )
        source = (
            _cds_composition_source()
            .replace(guard, "", 1)
            .replace(
                "        premium_leg += coupon_cashflow_pv(",
                guard + "        premium_leg += coupon_cashflow_pv(",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_accepts_credit_default_swap_recognized_early_continue_guards(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "            event_discount =",
            "            if event_weight <= 0.0:\n"
            "                continue\n"
            "            event_discount =",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    @pytest.mark.parametrize(
        ("anchor", "mutation"),
        (
            (
                "        if interval_stop <= interval_start:",
                "        interval_stop = -1\n",
            ),
            (
                "        if interval_stop <= interval_start:",
                "        interval_start = 1000000\n",
            ),
            (
                "            if event_weight <= 0.0:",
                "            event_weight = 0.0\n",
            ),
        ),
    )
    def test_rejects_credit_default_swap_mutated_early_continue_guard_control(
        self,
        registry,
        anchor,
        mutation,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            _cds_composition_source()
            .replace(
                "            event_discount =",
                "            if event_weight <= 0.0:\n"
                "                continue\n"
                "            event_discount =",
                1,
            )
            .replace(anchor, mutation + anchor, 1)
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("original", "replacement"),
        (
            (
                "        premium_leg += coupon_cashflow_pv(",
                "        premium_leg += -coupon_cashflow_pv(",
            ),
            (
                "            protection_leg += protection_payment_pv(",
                "            protection_leg += 0.0 * protection_payment_pv(",
            ),
            (
                "            accrued_on_event += coupon_cashflow_pv(",
                "            accrued_on_event += -coupon_cashflow_pv(",
            ),
        ),
    )
    def test_rejects_credit_default_swap_wrapped_leg_cashflow_call(
        self,
        registry,
        original,
        replacement,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            original,
            replacement,
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(finding.severity == "error" for finding in findings)

    @pytest.mark.parametrize(
        "symbol",
        ("coupon_cashflow_pv", "protection_payment_pv"),
    )
    def test_rejects_credit_default_swap_impersonated_cashflow_primitive(
        self,
        registry,
        symbol,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            f"{symbol}(",
            f"self.{symbol}(",
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(finding.severity == "error" for finding in findings)

    @pytest.mark.parametrize(
        "symbol",
        ("coupon_cashflow_pv", "protection_payment_pv"),
    )
    def test_rejects_credit_default_swap_shadowed_cashflow_import(
        self,
        registry,
        symbol,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            f"    {symbol} = lambda cashflow: 0.0\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(finding.severity == "error" for finding in findings)

    @pytest.mark.parametrize(
        "symbol",
        (
            "build_period_schedule",
            "BusinessDayAdjustment",
            "WEEKEND_ONLY",
            "RollConvention",
            "StubType",
            "CouponAccrual",
            "ProtectionPayment",
            "build_default_event_grid",
            "conditional_event_probabilities_from_curve",
            "expected_first_event_weights",
        ),
    )
    def test_rejects_credit_default_swap_shadowed_route_primitive(
        self,
        registry,
        symbol,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            f"    {symbol} = lambda *args, **kwargs: None\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_cashflow_primitive_binding"
            for finding in findings
        )

    def test_accepts_credit_default_swap_top_level_cashflow_imports(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        local_imports = (
            "    from trellis.conventions.calendar import "
            "BusinessDayAdjustment, WEEKEND_ONLY\n"
            "    from trellis.conventions.schedule import RollConvention, StubType\n"
            "    from trellis.core.date_utils import build_period_schedule\n"
            "    from trellis.models.contingent_cashflows import (\n"
            "        CouponAccrual,\n"
            "        ProtectionPayment,\n"
            "        build_default_event_grid,\n"
            "        conditional_event_probabilities_from_curve,\n"
            "        coupon_cashflow_pv,\n"
            "        expected_first_event_weights,\n"
            "        protection_payment_pv,\n"
            "    )\n\n"
        )
        top_level_imports = (
            "from trellis.conventions.calendar import "
            "BusinessDayAdjustment, WEEKEND_ONLY\n"
            "from trellis.conventions.schedule import RollConvention, StubType\n"
            "from trellis.core.date_utils import build_period_schedule\n"
            "from trellis.models.contingent_cashflows import (\n"
            "    CouponAccrual,\n"
            "    ProtectionPayment,\n"
            "    build_default_event_grid,\n"
            "    conditional_event_probabilities_from_curve,\n"
            "    coupon_cashflow_pv,\n"
            "    expected_first_event_weights,\n"
            "    protection_payment_pv,\n"
            ")\n"
        )
        source = top_level_imports + _cds_composition_source().replace(
            local_imports,
            "",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    def test_accepts_credit_default_swap_pricing_value_scaffold_import(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from trellis.core.payoff import PricingValue\n"
            + _cds_composition_source().replace(
                "def evaluate(self, market_state):",
                "def evaluate(self, market_state) -> PricingValue:",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    def test_accepts_credit_default_swap_postponed_annotation_name(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from __future__ import annotations\n"
            + _cds_composition_source().replace(
                "def evaluate(self, market_state):",
                "def evaluate(self, market_state) -> MissingType:",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    def test_rejects_credit_default_swap_wildcard_import(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source() + "\nfrom user_helpers import *\n"

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize("scope", ("module", "evaluate"))
    def test_rejects_credit_default_swap_unapproved_import(
        self,
        registry,
        scope,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source()
        if scope == "module":
            source = "import user_helpers\n\n" + source
        else:
            source = source.replace(
                "    schedule = build_period_schedule(",
                "    import user_helpers\n"
                "    schedule = build_period_schedule(",
                1,
            )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "symbol",
        ("coupon_cashflow_pv", "protection_payment_pv"),
    )
    def test_rejects_credit_default_swap_aliased_cashflow_import(
        self,
        registry,
        symbol,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            f"        {symbol},",
            f"        {symbol} as {symbol},",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_cashflow_primitive_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_late_local_cashflow_imports(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        local_imports = (
            "    from trellis.conventions.calendar import "
            "BusinessDayAdjustment, WEEKEND_ONLY\n"
            "    from trellis.conventions.schedule import RollConvention, StubType\n"
            "    from trellis.core.date_utils import build_period_schedule\n"
            "    from trellis.models.contingent_cashflows import (\n"
            "        CouponAccrual,\n"
            "        ProtectionPayment,\n"
            "        build_default_event_grid,\n"
            "        conditional_event_probabilities_from_curve,\n"
            "        coupon_cashflow_pv,\n"
            "        expected_first_event_weights,\n"
            "        protection_payment_pv,\n"
            "    )\n\n"
        )
        source = (
            _cds_composition_source()
            .replace(local_imports, "", 1)
            .replace(
                "    return float(",
                local_imports + "    return float(",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_cashflow_primitive_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "symbol",
        ("coupon_cashflow_pv", "protection_payment_pv"),
    )
    @pytest.mark.parametrize(
        "mutation",
        (
            'globals()["{symbol}"] = lambda cashflow: 0.0',
            'globals().update({{"{symbol}": lambda cashflow: 0.0}})',
            "import sys\nsys.modules[__name__].{symbol} = lambda cashflow: 0.0",
            'exec("{symbol} = lambda cashflow: 0.0", globals())',
            (
                "from builtins import globals as namespace\n"
                'namespace().update({{"{symbol}": lambda cashflow: 0.0}})'
            ),
        ),
    )
    def test_rejects_credit_default_swap_indirect_global_primitive_rebinding(
        self,
        registry,
        symbol,
        mutation,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        local_imports = (
            "    from trellis.conventions.calendar import "
            "BusinessDayAdjustment, WEEKEND_ONLY\n"
            "    from trellis.conventions.schedule import RollConvention, StubType\n"
            "    from trellis.core.date_utils import build_period_schedule\n"
            "    from trellis.models.contingent_cashflows import (\n"
            "        CouponAccrual,\n"
            "        ProtectionPayment,\n"
            "        build_default_event_grid,\n"
            "        conditional_event_probabilities_from_curve,\n"
            "        coupon_cashflow_pv,\n"
            "        expected_first_event_weights,\n"
            "        protection_payment_pv,\n"
            "    )\n\n"
        )
        top_level_imports = (
            "from trellis.conventions.calendar import "
            "BusinessDayAdjustment, WEEKEND_ONLY\n"
            "from trellis.conventions.schedule import RollConvention, StubType\n"
            "from trellis.core.date_utils import build_period_schedule\n"
            "from trellis.models.contingent_cashflows import (\n"
            "    CouponAccrual,\n"
            "    ProtectionPayment,\n"
            "    build_default_event_grid,\n"
            "    conditional_event_probabilities_from_curve,\n"
            "    coupon_cashflow_pv,\n"
            "    expected_first_event_weights,\n"
            "    protection_payment_pv,\n"
            ")\n"
        )
        source = (
            top_level_imports
            + mutation.format(symbol=symbol)
            + "\n"
            + _cds_composition_source().replace(local_imports, "", 1)
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_cashflow_primitive_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_decorated_evaluate(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def replace_evaluate(function):\n"
            "    return lambda self, market_state: 0.0\n\n"
            + _cds_composition_source().replace(
                "def evaluate(self, market_state):",
                "@replace_evaluate\ndef evaluate(self, market_state):",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_nested_evaluate_definition(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        composition = _cds_composition_source().strip()
        source = "def build_payoff():\n" + "\n".join(
            f"    {line}" if line else ""
            for line in composition.splitlines()
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_post_class_evaluate_rebinding(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        composition = _cds_composition_source().strip()
        source = "class Payoff:\n" + "\n".join(
            f"    {line}" if line else ""
            for line in composition.splitlines()
        )
        source += "\nPayoff.evaluate = lambda self, market_state: 0.0\n"

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_spread_normalization_after_assembly(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        normalization = "    if spread > 1.0:\n        spread *= 1e-4\n"
        source = (
            _cds_composition_source()
            .replace(normalization, "", 1)
            .replace(
                "    return float(",
                normalization + "    return float(",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_missing_spread_normalization(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    if spread > 1.0:\n        spread *= 1e-4\n",
            "",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_branch_hidden_spread_normalization(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    if spread > 1.0:\n        spread *= 1e-4\n",
            "    if False:\n"
            "        if spread > 1.0:\n"
            "            spread *= 1e-4\n",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    def test_accepts_credit_default_swap_normalized_spread_alias_after_guard(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        normalization = "    if spread > 1.0:\n        spread *= 1e-4\n"
        source = (
            _cds_composition_source()
            .replace(
                normalization,
                normalization + "    coupon_rate = spread\n",
                1,
            )
            .replace("rate=spread", "rate=coupon_rate", 2)
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    def test_rejects_credit_default_swap_spread_alias_before_guard(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            _cds_composition_source()
            .replace(
                "    if spread > 1.0:",
                "    coupon_rate = spread\n    if spread > 1.0:",
                1,
            )
            .replace("rate=spread", "rate=coupon_rate", 2)
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_composition_in_unused_helper(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = """
def evaluate(self, market_state):
    return 123.0
""" + _cds_composition_source().replace(
            "def evaluate(self, market_state):",
            "def unused_cds_composition(self, market_state):",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_composition_in_nested_unused_helper(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        composition_lines = _cds_composition_source().strip().splitlines()
        source = "\n".join(
            [
                composition_lines[0],
                "    def unused_cds_composition(self, market_state):",
                *("    " + line for line in composition_lines[1:]),
                "    return 123.0",
            ]
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_composition_after_early_return(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            "    return 123.0\n    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("anchor", "conditional_exit"),
        (
            (
                "    schedule = build_period_schedule(",
                "    if True:\n        return 123.0",
            ),
            (
                "    schedule = build_period_schedule(",
                "    if market_state is None:\n        return 123.0",
            ),
            (
                "    return float(",
                "    if True:\n        return 123.0",
            ),
            (
                "    return float(",
                "    for _ in range(1):\n        return 123.0",
            ),
            (
                "    return float(",
                "    if True:\n        raise RuntimeError('invalid CDS')",
            ),
        ),
    )
    def test_rejects_credit_default_swap_conditional_exit_before_signed_return(
        self,
        registry,
        anchor,
        conditional_exit,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            anchor,
            f"{conditional_exit}\n{anchor}",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_accepts_credit_default_swap_required_market_guards(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            "    if market_state.credit_curve is None:\n"
            "        raise ValueError('credit curve is required')\n"
            "    if market_state.discount is None:\n"
            "        raise ValueError('discount curve is required')\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    @pytest.mark.parametrize(
        "mutation",
        (
            "market_state = MarketProxy(market_state)",
            "del market_state",
        ),
    )
    def test_rejects_credit_default_swap_mutated_market_state_parameter(
        self,
        registry,
        mutation,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "class MarketProxy:\n"
            "    def __init__(self, market_state):\n"
            "        self.credit_curve = market_state.credit_curve\n"
            "        self.discount = market_state.discount\n\n"
            + _cds_composition_source().replace(
                "    schedule = build_period_schedule(",
                f"    {mutation}\n    schedule = build_period_schedule(",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_market_state_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("module_setup", "exception_expression", "raise_suffix"),
        (
            (
                "def ValueError(message):\n"
                "    while True:\n"
                "        pass\n\n",
                "'credit curve is required'",
                "",
            ),
            ("", "1 / self._spec.notional", ""),
            (
                "def spin():\n"
                "    while True:\n"
                "        pass\n\n",
                "'credit curve is required'",
                " from spin()",
            ),
        ),
    )
    def test_rejects_credit_default_swap_unsafe_market_guards(
        self,
        registry,
        module_setup,
        exception_expression,
        raise_suffix,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = module_setup + _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            "    if market_state.credit_curve is None:\n"
            f"        raise ValueError({exception_expression}){raise_suffix}\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_period_loop_hidden_in_branch(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        lines = _cds_composition_source().splitlines()
        source = "\n".join(
            lines[:2] + ["    if False:"] + ["    " + line for line in lines[2:]]
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("anchor", "replacement"),
        (
            (
                "enumerate(grid.periods)",
                "enumerate(grid.periods, start=1)",
            ),
            (
                "range(interval_start, interval_stop)",
                "range(interval_start, interval_stop, step=1)",
            ),
        ),
    )
    def test_rejects_credit_default_swap_modified_grid_iteration_call(
        self,
        registry,
        anchor,
        replacement,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(anchor, replacement, 1)

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize("symbol", ("enumerate", "range"))
    def test_rejects_credit_default_swap_shadowed_grid_iteration_builtin(
        self,
        registry,
        symbol,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            f"    {symbol} = lambda *args: ()\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_shadowed_float_builtin(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            "    float = lambda value: 0.0\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_reassigned_credit_curve_alias(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "market_state.credit_curve",
            "credit_curve",
        )
        source = source.replace(
            "    schedule = build_period_schedule(",
            "    credit_curve = market_state.credit_curve\n"
            "    credit_curve = other_curve\n"
            "    schedule = build_period_schedule(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_credit_curve_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("anchor", "mutation"),
        (
            (
                "    for period_index, period in enumerate(grid.periods):\n",
                "        period_index = 0\n",
            ),
            (
                "    for period_index, period in enumerate(grid.periods):\n",
                "        period = grid.periods[0]\n",
            ),
            (
                "        for interval_index in range(interval_start, interval_stop):\n",
                "            interval_index = 0\n",
            ),
        ),
    )
    def test_rejects_credit_default_swap_rebound_grid_loop_target(
        self,
        registry,
        anchor,
        mutation,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            anchor,
            anchor + mutation,
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "cursor_setup",
        (
            "    if False:\n        interval_start = 0\n",
            "    interval_start = 0\n    interval_start = 1\n",
        ),
    )
    def test_rejects_credit_default_swap_invalid_cursor_initialization(
        self,
        registry,
        cursor_setup,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    interval_start = 0\n",
            cursor_setup,
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_interval_loop_hidden_in_branch(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _guard_cds_interval_loop(
            _cds_composition_source(),
            condition="False",
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("accumulator", "category"),
        (
            ("premium_leg", "credit_default_swap_incomplete_event_grid"),
            ("protection_leg", "credit_default_swap_incomplete_event_grid"),
            ("accrued_on_event", "credit_default_swap_accrual_mapping"),
            ("accrued_to_valuation", "credit_default_swap_economic_binding"),
        ),
    )
    def test_rejects_credit_default_swap_non_additive_leg_update(
        self,
        registry,
        accumulator,
        category,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            f"{accumulator} +=",
            f"{accumulator} -=",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(finding.category == category for finding in findings)

    @pytest.mark.parametrize(
        "accumulator",
        (
            "premium_leg",
            "protection_leg",
            "accrued_on_event",
            "accrued_to_valuation",
        ),
    )
    def test_rejects_credit_default_swap_leg_reassigned_after_assembly(
        self,
        registry,
        accumulator,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    return float(",
            f"    {accumulator} = 0.0\n    return float(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_sign_convention"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "accumulator",
        (
            "premium_leg",
            "protection_leg",
            "accrued_on_event",
            "accrued_to_valuation",
        ),
    )
    @pytest.mark.parametrize(
        "mutation_template",
        (
            "{accumulator} += 1000000.0",
            "{accumulator} *= 0.0",
            "{accumulator}, ignored = 0.0, None",
            "({accumulator} := 0.0)",
        ),
    )
    def test_rejects_credit_default_swap_unrecognized_accumulator_mutation(
        self,
        registry,
        accumulator,
        mutation_template,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        mutation = mutation_template.format(accumulator=accumulator)
        source = _cds_composition_source().replace(
            "    return float(",
            f"    {mutation}\n    return float(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_sign_convention"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "accumulator",
        (
            "premium_leg",
            "protection_leg",
            "accrued_on_event",
            "accrued_to_valuation",
        ),
    )
    def test_rejects_credit_default_swap_nonzero_leg_initialization(
        self,
        registry,
        accumulator,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            f"    {accumulator} = 0.0",
            f"    {accumulator} = 1.0",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_sign_convention"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("survival_index", "event_index"),
        (
            ("0", "interval_index"),
            ("interval_stop - 1", "0"),
        ),
    )
    def test_rejects_credit_default_swap_inactive_event_weights(
        self,
        registry,
        survival_index,
        event_index,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                survival_index=survival_index,
                event_index=event_index,
            ),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_weight_mapping"
            for finding in findings
        )

    def test_rejects_credit_default_swap_nested_decoy_weight_keyword(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "            weight=survival_weight,",
            "            weight=(dict(weight=survival_weight) and 0.0),",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_weight_mapping"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("constructor_line", "extra_keyword_line"),
        (
            (
                "        premium_leg += coupon_cashflow_pv(CouponAccrual(",
                "            probe=spin(),",
            ),
            (
                "            protection_leg += protection_payment_pv(ProtectionPayment(",
                "                probe=spin(),",
            ),
        ),
    )
    def test_rejects_credit_default_swap_extra_cashflow_constructor_keyword(
        self,
        registry,
        constructor_line,
        extra_keyword_line,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def spin():\n"
            "    while True:\n"
            "        pass\n\n"
            + _cds_composition_source().replace(
                constructor_line,
                f"{constructor_line}\n{extra_keyword_line}",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_weight_mapping"
            for finding in findings
        )

    def test_rejects_credit_default_swap_unbound_weight_owner(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                post_weights_setup=(
                    "other_weights = FirstEventWeights("
                    "tuple(0.0 for _ in conditional), "
                    "tuple(0.0 for _ in conditional))"
                ),
                weight_owner="other_weights",
            ),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_weight_mapping"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("method", "weight_symbol", "weight_controls", "additional_weight_call"),
        (
            (
                "analytical",
                "sample_first_event_weights",
                "n_paths=self._spec.n_paths, seed=42,",
                (
                    "expected_first_event_weights(conditional, "
                    "initial_survival_weight=initial_survival_weight)"
                ),
            ),
            (
                "monte_carlo",
                "expected_first_event_weights",
                "",
                (
                    "sample_first_event_weights(conditional, "
                    "initial_survival_weight=initial_survival_weight, "
                    "n_paths=self._spec.n_paths, seed=42)"
                ),
            ),
        ),
    )
    def test_rejects_credit_default_swap_weights_from_wrong_method(
        self,
        registry,
        method,
        weight_symbol,
        weight_controls,
        additional_weight_call,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                weight_symbol=weight_symbol,
                weight_controls=weight_controls,
                additional_weight_call=additional_weight_call,
            ),
            _make_plan("credit_default_swap", method),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_weight_mapping"
            for finding in findings
        )

    def test_rejects_credit_default_swap_unselected_weight_call(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(
            extra_setup=(
                "from trellis.models.contingent_cashflows import "
                "sample_first_event_weights"
            ),
            additional_weight_call=(
                "sample_first_event_weights(conditional, "
                "initial_survival_weight=initial_survival_weight, "
                "n_paths=0 if self._spec.notional == 0 else 1, seed=None)"
            ),
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap", "analytical"),
            spec,
        )

        assert any(
            finding.category
            == "credit_default_swap_unselected_first_event_primitive"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "loop_setup",
        (
            "while True:\n        pass",
            "for _ in iter(int, 1):\n        pass",
        ),
    )
    def test_rejects_credit_default_swap_nonterminating_preassembly_loop(
        self,
        registry,
        loop_setup,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(
            extra_setup=loop_setup,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap", "analytical"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "extra_setup",
        (
            "if self._spec.notional == 0:\n        1 / 0",
            (
                "try:\n"
                "        1 / self._spec.notional\n"
                "    except ZeroDivisionError:\n"
                "        raise"
            ),
            (
                "from contextlib import nullcontext\n"
                "    with nullcontext():\n"
                "        1 / self._spec.notional"
            ),
            (
                "match self._spec.notional:\n"
                "        case 0:\n"
                "            1 / 0"
            ),
            "1 / self._spec.notional",
            "probe = 1 / self._spec.notional",
            "_ = 1 / 0 if self._spec.notional == 0 else 0",
            "_ = tuple(1 for _ in iter(int, 1))",
        ),
    )
    def test_rejects_credit_default_swap_implicit_preassembly_exception(
        self,
        registry,
        extra_setup,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(
            extra_setup=extra_setup,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap", "analytical"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_accepts_credit_default_swap_exact_initial_survival_fallback(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(
            initial_survival=(
                "float(market_state.credit_curve.survival_probability("
                "grid.intervals[0].start_time)) if grid.intervals else 1.0"
            ),
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap", "analytical"),
            spec,
        )

        assert not any(
            finding.category
            in {
                "credit_default_swap_incomplete_event_grid",
                "credit_default_swap_initial_survival_missing",
            }
            for finding in findings
        )

    @pytest.mark.parametrize("n_paths", ("10", "self._spec.n_paths"))
    def test_rejects_credit_default_swap_unbound_path_count(
        self,
        registry,
        n_paths,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                weight_symbol="sample_first_event_weights",
                weight_controls=f"n_paths={n_paths}, seed=42,",
            ),
            _make_plan("credit_default_swap", "monte_carlo"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_path_count_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "n_paths",
        (
            "self._spec.n_paths or 250000",
            'getattr(self._spec, "n_paths", 250000) or 250000',
        ),
    )
    def test_accepts_credit_default_swap_active_monte_carlo_path_count(
        self,
        registry,
        n_paths,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                weight_symbol="sample_first_event_weights",
                weight_controls=f"n_paths={n_paths}, seed=42,",
            ),
            _make_plan("credit_default_swap", "monte_carlo"),
            spec,
        )

        assert not any(f.severity == "error" for f in findings)

    @pytest.mark.parametrize(
        ("seed", "extra_setup"),
        (
            ("None", ""),
            ("41", ""),
            ("seed_value", "seed_value = None"),
            (
                "seed_value",
                "seed_value = 42\n    seed_value = None",
            ),
            (
                "seed_value",
                "seed_value = 42\n    seed_value += 1",
            ),
            ("1" + "0" * 400, ""),
        ),
    )
    def test_rejects_credit_default_swap_unreproducible_sampling_seed(
        self,
        registry,
        seed,
        extra_setup,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                weight_symbol="sample_first_event_weights",
                weight_controls=(
                    "n_paths=self._spec.n_paths or 250000, "
                    f"seed={seed},"
                ),
                extra_setup=extra_setup,
            ),
            _make_plan("credit_default_swap", "monte_carlo"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_seed_binding"
            for finding in findings
        )

    def test_accepts_credit_default_swap_reproducible_sampling_seed_alias(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                weight_symbol="sample_first_event_weights",
                weight_controls=(
                    "n_paths=self._spec.n_paths or 250000, seed=seed_value,"
                ),
                extra_setup="seed_value = 42",
            ),
            _make_plan("credit_default_swap", "monte_carlo"),
            spec,
        )

        assert not any(f.severity == "error" for f in findings)

    @pytest.mark.parametrize(
        ("premium_discount", "event_discount"),
        (
            (
                "1.0",
                "market_state.discount.discount(interval.settlement_time)",
            ),
            (
                "market_state.discount.discount(grid.period_payment_times[period_index])",
                "1.0",
            ),
            (
                "market_state.discount.discount(grid.period_payment_times[0])",
                "market_state.discount.discount(interval.settlement_time)",
            ),
            (
                "market_state.discount.discount(grid.period_payment_times[period_index])",
                "market_state.discount.discount(grid.intervals[0].settlement_time)",
            ),
        ),
    )
    def test_rejects_credit_default_swap_wrong_discount_coordinates(
        self,
        registry,
        premium_discount,
        event_discount,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                premium_discount=premium_discount,
                event_discount=event_discount,
            ),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_discount_mapping"
            for finding in findings
        )

    def test_rejects_credit_default_swap_nested_decoy_discount_keyword(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "            discount_factor=market_state.discount.discount("
            "grid.period_payment_times[period_index]),",
            "            discount_factor=(dict(discount_factor="
            "market_state.discount.discount("
            "grid.period_payment_times[period_index])) and 1.0),",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_discount_mapping"
            for finding in findings
        )

    def test_rejects_credit_default_swap_reassigned_discount_alias(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "            protection_leg += protection_payment_pv(",
            "            event_discount = 0.0\n"
            "            protection_leg += protection_payment_pv(",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_discount_mapping"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("premium_discount", "event_discount"),
        (
            (
                "other_discount.discount(grid.period_payment_times[period_index])",
                "market_state.discount.discount(interval.settlement_time)",
            ),
            (
                "market_state.discount.discount(grid.period_payment_times[period_index])",
                "other_discount.discount(interval.settlement_time)",
            ),
        ),
    )
    def test_rejects_credit_default_swap_unbound_discount_curve(
        self,
        registry,
        premium_discount,
        event_discount,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                premium_discount=premium_discount,
                event_discount=event_discount,
            ),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_discount_mapping"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "overrides",
        (
            {"scheduled_accrual": "0.0"},
            {"event_accrual": "0.0"},
            {"event_accrual": "period.accrual_fraction"},
            {"event_accrual": "interval.period_fraction_elapsed"},
        ),
    )
    def test_rejects_credit_default_swap_unbound_coupon_accruals(
        self,
        registry,
        overrides,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(**overrides),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_accrual_mapping"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "overrides",
        (
            {"schedule_start": "self._spec.end_date"},
            {"schedule_end": "self._spec.start_date"},
            {"schedule_frequency": "Frequency.ANNUAL"},
            {"schedule_day_count": "DayCountConvention.ACT_365F"},
            {"schedule_calendar": "other_calendar"},
            {"schedule_bda": "BusinessDayAdjustment.PRECEDING"},
            {"schedule_roll": "RollConvention.IMM"},
            {"schedule_stub": "StubType.LONG_FIRST"},
            {"schedule_payment_lag": "30"},
        ),
    )
    def test_rejects_credit_default_swap_unbound_schedule_fields(
        self,
        registry,
        overrides,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(**overrides),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_schedule_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_opaque_schedule_keywords(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def spin():\n"
            "    while True:\n"
            "        pass\n\n"
            + _cds_composition_source().replace(
                "        payment_lag_days=0,\n",
                "        payment_lag_days=0,\n"
                "        **((self._spec.notional == 0 and spin()) or {}),\n",
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_schedule_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "overrides",
        (
            {"conditional_credit_curve": "other_credit_curve"},
            {
                "initial_survival": (
                    "other_credit_curve.survival_probability("
                    "grid.intervals[0].start_time)"
                )
            },
        ),
    )
    def test_rejects_credit_default_swap_unbound_credit_curve(
        self,
        registry,
        overrides,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(**overrides),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_credit_curve_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "overrides",
        (
            {"premium_notional": "1.0"},
            {"protection_notional": "1.0"},
            {"event_notional": "1.0"},
            {"premium_rate": "0.01"},
            {"event_rate": "0.01"},
            {"recovery": "0.4"},
            {"valuation_adjustment": "0.0"},
            {
                "valuation_adjustment": (
                    "1.0 * spread * period.accrual_fraction "
                    "* grid.elapsed_period_fractions[period_index]"
                )
            },
        ),
    )
    def test_rejects_credit_default_swap_unbound_economic_terms(
        self,
        registry,
        overrides,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]

        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(**overrides),
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("anchor", "sign_line"),
        (
            (
                "            weight=survival_weight,\n",
                "            sign=-1.0,\n",
            ),
            (
                "                discount_factor=event_discount,\n",
                "                sign=-1.0,\n",
            ),
            (
                "                weight=event_weight,\n",
                "                sign=-1.0,\n",
            ),
        ),
    )
    def test_rejects_credit_default_swap_negative_constructor_sign(
        self,
        registry,
        anchor,
        sign_line,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            anchor,
            anchor + sign_line,
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    def test_accepts_credit_default_swap_positive_constructor_sign_alias(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(extra_setup="leg_sign = 1.0")
        for anchor, sign_line in (
            (
                "            weight=survival_weight,\n",
                "            sign=leg_sign,\n",
            ),
            (
                "                discount_factor=event_discount,\n",
                "                sign=leg_sign,\n",
            ),
            (
                "                weight=event_weight,\n",
                "                sign=leg_sign,\n",
            ),
        ):
            source = source.replace(anchor, anchor + sign_line, 1)

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert not any(f.severity == "error" for f in findings)

    @pytest.mark.parametrize(
        "extra_setup",
        (
            "leg_sign = 1.0\n    leg_sign = -1.0",
            "leg_sign = 1.0\n    leg_sign *= -1.0",
        ),
    )
    def test_rejects_credit_default_swap_reassigned_constructor_sign_alias(
        self,
        registry,
        extra_setup,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(
            extra_setup=extra_setup,
        ).replace(
            "            weight=survival_weight,\n",
            "            weight=survival_weight,\n            sign=leg_sign,\n",
            1,
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_economic_binding"
            for finding in findings
        )

    def test_accepts_credit_default_swap_explicit_first_event_composition(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        validator = AlgorithmContractValidator()
        findings = validator.validate(
            _cds_composition_source(),
            _make_plan("credit_default_swap"),
            spec,
        )
        assert not any(f.severity == "error" for f in findings)

    @pytest.mark.parametrize(
        ("extra_setup", "replacements", "expected_category"),
        (
            (
                "if False:\n        discount_curve = market_state.discount",
                (
                    ("market_state.discount.discount(", "discount_curve.discount("),
                ),
                "credit_default_swap_discount_mapping",
            ),
            (
                "if False:\n        credit_curve = market_state.credit_curve",
                (
                    ("market_state.credit_curve", "credit_curve"),
                ),
                "credit_default_swap_credit_curve_binding",
            ),
            (
                "if False:\n        grid_alias = grid",
                (
                    ("grid.", "grid_alias."),
                ),
                "credit_default_swap_valuation_origin",
            ),
            (
                "if False:\n        weights_alias = weights",
                (
                    ("weights.survival_weights", "weights_alias.survival_weights"),
                    ("weights.event_weights", "weights_alias.event_weights"),
                ),
                "credit_default_swap_weight_mapping",
            ),
            (
                "if False:\n        spec_alias = self._spec",
                (
                    ("self._spec.", "spec_alias."),
                ),
                "credit_default_swap_economic_binding",
            ),
        ),
    )
    def test_rejects_credit_default_swap_conditionally_assigned_aliases(
        self,
        registry,
        extra_setup,
        replacements,
        expected_category,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(extra_setup=extra_setup)
        for original, replacement in replacements:
            source = source.replace(original, replacement)

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == expected_category
            for finding in findings
        )

    def test_rejects_credit_default_swap_evaluate_on_unrelated_class(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "class UnrelatedPayoff:\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
            + "\n\nclass CDSPayoff:\n    pass\n"
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "definition_time_setup",
        (
            "while True:\n    pass\n\n",
            "class ImportTrap:\n    while True:\n        pass\n\n",
            (
                "def import_trap(value: tuple(iter(int, 1))):\n"
                "    return value\n\n"
            ),
            (
                "def annotation_trap(value: MissingType):\n"
                "    return value\n\n"
            ),
            (
                "def annotation_trap(value: int | 1):\n"
                "    return value\n\n"
            ),
            (
                "from trellis.core.types import Frequency\n\n"
                "def annotation_trap(value: Frequency.MISSING):\n"
                "    return value\n\n"
            ),
            (
                "def default_trap(value=MISSING):\n"
                "    return value\n\n"
            ),
            (
                "def default_trap(value={[]: 1}):\n"
                "    return value\n\n"
            ),
            (
                "def default_trap(value={((), []): 1}):\n"
                "    return value\n\n"
            ),
            (
                "def default_trap(value={[]}):\n"
                "    return value\n\n"
            ),
            (
                "def default_trap(value=Frequency.QUARTERLY):\n"
                "    return value\n\n"
            ),
            (
                "def default_trap(value=Frequency.QUARTERLY):\n"
                "    return value\n\n"
                "from trellis.core.types import Frequency\n\n"
            ),
            (
                "from trellis.core.types import Frequency\n\n"
                "def default_trap(value=Frequency.MISSING):\n"
                "    return value\n\n"
            ),
            (
                "def annotation_trap(value: PricingValue):\n"
                "    return value\n\n"
                "from trellis.core.payoff import PricingValue\n\n"
            ),
            (
                "from trellis.core.payoff import PricingValue\n\n"
                "def PricingValue():\n"
                "    return float\n\n"
                "def annotation_trap(value: PricingValue):\n"
                "    return value\n\n"
            ),
            (
                "class Spin:\n"
                "    def __class_getitem__(cls, item):\n"
                "        while True:\n"
                "            pass\n\n"
                "def annotation_trap(value: Spin[int]):\n"
                "    return value\n\n"
            ),
            (
                "from dataclasses import dataclass\n\n"
                "def dataclass(*args, **kwargs):\n"
                "    def replace_spec(spec_class):\n"
                "        return spec_class\n"
                "    return replace_spec\n\n"
                "@dataclass(frozen=True)\n"
                "class CDSSpec:\n"
                "    notional: float = 1.0\n\n"
            ),
            (
                "from dataclasses import dataclass\n\n"
                "class SpecNamespace:\n"
                "    def dataclass(*args, **kwargs):\n"
                "        def replace_spec(spec_class):\n"
                "            return spec_class\n"
                "        return replace_spec\n\n"
                "    @dataclass(frozen=True)\n"
                "    class CDSSpec:\n"
                "        notional: float = 1.0\n\n"
            ),
        ),
    )
    def test_rejects_credit_default_swap_definition_time_control_flow(
        self,
        registry,
        definition_time_setup,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = definition_time_setup + _cds_composition_source()

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        "evaluate_definition",
        (
            "class Trap:\n        while True:\n            pass",
            (
                "def trap(value=tuple(iter(int, 1))):\n"
                "        return value"
            ),
            (
                "def spin():\n"
                "        while True:\n"
                "            pass\n"
                "    @spin()\n"
                "    def decorated():\n"
                "        pass"
            ),
            "trap = lambda value=tuple(iter(int, 1)): value",
        ),
    )
    def test_rejects_credit_default_swap_evaluate_definition_time_control_flow(
        self,
        registry,
        evaluate_definition,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source(extra_setup=evaluate_definition)

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize("field_default", ("[]", "{}", "{1}"))
    def test_rejects_credit_default_swap_local_dataclass_mutable_default(
        self,
        registry,
        field_default,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = "from dataclasses import dataclass\n\n" + _cds_composition_source(
            extra_setup=(
                "@dataclass(frozen=True)\n"
                "    class Trap:\n"
                f"        items: list = {field_default}"
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    @pytest.mark.parametrize(
        ("module_setup", "class_definition"),
        (
            (
                "",
                "@dataclass(frozen=True)\n"
                "    class Trap:\n"
                "        value: int = 1",
            ),
            (
                "from dataclasses import dataclass\n\n",
                "@dataclass(frozen=True)\n"
                "    class Trap:\n"
                "        optional: int = 1\n"
                "        required: int",
            ),
        ),
    )
    def test_rejects_credit_default_swap_invalid_local_dataclass_definition(
        self,
        registry,
        module_setup,
        class_definition,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = module_setup + _cds_composition_source(
            extra_setup=class_definition
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_extra_parameter_shadowed_decorator(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def spin():\n"
            "    while True:\n"
            "        pass\n\n"
            + _cds_composition_source(
                extra_setup="@property\n    def trap():\n        return 1"
            ).replace(
                "def evaluate(self, market_state):",
                (
                    "def evaluate(\n"
                    "    self, market_state, property=lambda fn: spin(),\n"
                    "):"
                ),
                1,
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_local_shadowed_decorator(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def spin():\n"
            "    while True:\n"
            "        pass\n\n"
            + _cds_composition_source(
                extra_setup=(
                    "def property(fn):\n"
                    "        return spin()\n"
                    "    @property\n"
                    "    def trap():\n"
                    "        return 1"
                )
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_accepts_credit_default_swap_authoritative_enum_function_default(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from trellis.core.types import Frequency\n\n"
            "def inert_default(value=Frequency.QUARTERLY):\n"
            "    return value\n\n"
            "def inert_container_default(\n"
            "    value={'frequencies': (Frequency.QUARTERLY,)},\n"
            "):\n"
            "    return value\n\n"
            "def inert_union_annotation(value: int | None):\n"
            "    return value\n\n"
            + _cds_composition_source()
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    @pytest.mark.parametrize(
        ("constructor_body", "property_body"),
        (
            ("self._spec = object()", "return self._spec"),
            ("self._spec = spec", "return object()"),
        ),
    )
    def test_rejects_credit_default_swap_substituted_payoff_spec(
        self,
        registry,
        constructor_body,
        property_body,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "class CDSPayoff:\n"
            "    def __init__(self, spec):\n"
            f"        {constructor_body}\n\n"
            "    @property\n"
            "    def spec(self):\n"
            f"        {property_body}\n\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_shadowed_property_decorator(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def property(function):\n"
            "    while True:\n"
            "        pass\n\n"
            "class CDSPayoff:\n"
            "    def __init__(self, spec):\n"
            "        self._spec = spec\n\n"
            "    @property\n"
            "    def spec(self):\n"
            "        return self._spec\n\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_mutable_spec_dataclass(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class CDSSpec:\n"
            "    notional: float = 1.0\n\n"
            + _cds_composition_source()
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_accepts_credit_default_swap_frozen_spec_dataclass(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class CDSSpec:\n"
            "    notional: float = 1.0\n\n"
            + _cds_composition_source()
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert not any(finding.severity == "error" for finding in findings)

    def test_rejects_credit_default_swap_changed_planned_spec_default(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class CDSSpec:\n"
            "    recovery: float = 0.9\n\n"
            + _cds_composition_source()
        )
        plan = _make_plan(
            "credit_default_swap",
            payoff_spec_name="CDSSpec",
            payoff_spec_fields=(("recovery", "float", "0.4"),),
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_spec_schema_binding"
            for finding in findings
        )

    def test_accepts_credit_default_swap_exact_planned_spec_schema(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class CDSSpec:\n"
            "    recovery: float = 0.4\n\n"
            + _cds_composition_source()
        )
        plan = _make_plan(
            "credit_default_swap",
            payoff_spec_name="CDSSpec",
            payoff_spec_fields=(("recovery", "float", "0.4"),),
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert not any(finding.severity == "error" for finding in findings)

    def test_rejects_credit_default_swap_duplicate_planned_spec_class(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        spec_class = (
            "@dataclass(frozen=True)\n"
            "class CDSSpec:\n"
            "    recovery: float = 0.4\n\n"
        )
        source = (
            "from dataclasses import dataclass\n\n"
            + spec_class
            + spec_class
            + _cds_composition_source()
        )
        plan = _make_plan(
            "credit_default_swap",
            payoff_spec_name="CDSSpec",
            payoff_spec_fields=(("recovery", "float", "0.4"),),
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_spec_schema_binding"
            for finding in findings
        )

    def test_rejects_credit_default_swap_spec_behavior_override(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class CDSSpec:\n"
            "    notional: float = 1.0\n\n"
            "    def __getattribute__(self, name):\n"
            "        return 1.0\n\n"
            + _cds_composition_source()
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_unsafe_requirements_property(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "class CDSPayoff:\n"
            "    def __init__(self, spec):\n"
            "        self._spec = spec\n\n"
            "    @property\n"
            "    def spec(self):\n"
            "        return self._spec\n\n"
            "    @property\n"
            "    def requirements(self):\n"
            "        while True:\n"
            "            pass\n\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_payoff_destructor(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "class CDSPayoff:\n"
            "    def __init__(self, spec):\n"
            "        self._spec = spec\n\n"
            "    @property\n"
            "    def spec(self):\n"
            "        return self._spec\n\n"
            "    @property\n"
            "    def requirements(self):\n"
            "        return {'credit_curve', 'discount_curve'}\n\n"
            "    def __del__(self):\n"
            "        while True:\n"
            "            pass\n\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_rejects_credit_default_swap_annotated_payoff_behavior_hook(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "def spin():\n"
            "    while True:\n"
            "        pass\n\n"
            "class CDSPayoff:\n"
            "    __getattribute__: object = lambda self, name: spin()\n\n"
            "    def __init__(self, spec):\n"
            "        self._spec = spec\n\n"
            "    @property\n"
            "    def spec(self):\n"
            "        return self._spec\n\n"
            "    @property\n"
            "    def requirements(self):\n"
            "        return {'credit_curve', 'discount_curve'}\n\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert any(
            finding.category == "credit_default_swap_incomplete_event_grid"
            for finding in findings
        )

    def test_accepts_credit_default_swap_authoritative_payoff_spec(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "class CDSPayoff:\n"
            "    def __init__(self, spec):\n"
            "        self._spec = spec\n\n"
            "    @property\n"
            "    def spec(self):\n"
            "        return self._spec\n\n"
            "    @property\n"
            "    def requirements(self):\n"
            "        return {'credit_curve', 'discount_curve'}\n\n"
            + textwrap.indent(_cds_composition_source().strip(), "    ")
        )
        plan = replace(
            _make_plan("credit_default_swap"),
            payoff_class_name="CDSPayoff",
        )

        findings = AlgorithmContractValidator().validate(source, plan, spec)

        assert not any(finding.severity == "error" for finding in findings)

    def test_accepts_credit_default_swap_optional_valuation_date_fallback(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        findings = AlgorithmContractValidator().validate(
            _cds_composition_source(
                time_origin=(
                    'getattr(self._spec, "valuation_date", None) '
                    "or self._spec.start_date"
                ),
            ),
            _make_plan("credit_default_swap"),
            spec,
        )

        errors = [finding for finding in findings if finding.severity == "error"]
        assert not errors, errors

    def test_rejects_credit_default_swap_shadowed_optional_valuation_getattr(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            "getattr = lambda instance, field, default: instance.start_date\n"
            + _cds_composition_source(
                time_origin=(
                    'getattr(self._spec, "valuation_date", None) '
                    "or self._spec.start_date"
                ),
            )
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert any(
            finding.category == "credit_default_swap_valuation_origin"
            for finding in findings
        )

    def test_accepts_credit_default_swap_semantic_leg_names(self, registry):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = (
            _cds_composition_source()
            .replace("protection_leg", "protection")
            .replace("premium_leg", "premium")
        )

        findings = AlgorithmContractValidator().validate(
            source,
            _make_plan("credit_default_swap"),
            spec,
        )

        assert not any(f.severity == "error" for f in findings)

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

    def test_credit_default_swap_contract_failure_is_blocking_by_default(
        self,
        registry,
    ):
        spec = [r for r in registry.routes if r.id == "credit_default_swap"][0]
        source = _cds_composition_source().replace(
            "    schedule = build_period_schedule(",
            "    coupon_cashflow_pv = lambda cashflow: 0.0\n"
            "    schedule = build_period_schedule(",
            1,
        )

        report = validate_generated_semantics(
            source,
            _make_plan("credit_default_swap"),
            route_spec=spec,
        )

        assert not report.ok
        assert report.mode == "blocking"
        assert any(
            finding.category == "credit_default_swap_cashflow_primitive_binding"
            for finding in report.errors
        )

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

    @pytest.mark.parametrize(
        "forbidden_symbol",
        [
            "price_basket_option_analytical",
            "price_basket_option_monte_carlo",
            "price_basket_option_transform_proxy",
            "price_ranked_observation_basket_monte_carlo",
            "build_ranked_observation_basket_state_payoff",
        ],
    )
    def test_terminal_basket_authority_rejects_helpers_and_ranked_substitution(
        self,
        registry,
        forbidden_symbol,
    ):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        plan = _make_plan(
            "analytical_black76",
            "analytical",
            instrument_type="basket_option",
            primitives=(
                PrimitiveRef(
                    module="trellis.models.resolution.terminal_basket",
                    symbol="resolve_terminal_basket_inputs",
                    role="market_binding",
                ),
                PrimitiveRef(
                    module="trellis.models.analytical.terminal_basket",
                    symbol="two_asset_extremum_option_stulz",
                    role="pricing_kernel",
                ),
            ),
        )
        source = (
            "def evaluate(self, market_state):\n"
            f"    return {forbidden_symbol}(market_state, self._spec)\n"
        )

        report = validate_generated_semantics(
            source,
            plan,
            route_spec=spec,
        )

        assert not report.ok
        assert report.mode == "blocking"
        assert any(
            finding.category == "terminal_basket_forbidden_helper"
            for finding in report.findings
        )
