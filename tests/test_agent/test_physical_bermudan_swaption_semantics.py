"""Strict semantic contract tests for physical dual-curve Bermudan swaptions."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from math import exp
from types import MappingProxyType
from types import SimpleNamespace

import pytest


def _strict_contract(**overrides):
    from trellis.agent.semantic_contracts import (
        make_physical_bermudan_swaption_contract,
    )

    fields = {
        "description": "Physical payer Bermudan swaption with co-terminal USD swap tails",
        "exercise_dates": ("2027-06-15", "2028-06-15", "2029-06-15"),
        "exercise_to_swap_start": (
            ("2027-06-15", "2027-06-17"),
            ("2028-06-15", "2028-06-19"),
            ("2029-06-15", "2029-06-19"),
        ),
        "swap_maturity": "2032-06-17",
        "notional": 10_000_000.0,
        "fixed_rate": 0.031,
        "payer_receiver": "payer",
        "settlement_type": "physical",
        "currency": "USD",
        "discount_curve_id": "USD-OIS",
        "forecast_curve_id": "USD-SOFR-3M",
        "hull_white_parameter_source": "USD-HW1F-2026Q4",
        "fixed_frequency": "semiannual",
        "fixed_day_count": "30/360",
        "fixed_calendar_name": "USSettlement",
        "fixed_business_day_adjustment": "modified_following",
        "fixed_stub_rule": "short_final",
        "fixed_roll_convention": "none",
        "fixed_payment_lag_business_days": 2,
        "floating_frequency": "quarterly",
        "floating_day_count": "ACT/360",
        "floating_calendar_name": "USSettlement",
        "floating_business_day_adjustment": "modified_following",
        "floating_stub_rule": "short_final",
        "floating_roll_convention": "none",
        "floating_fixing_lag_business_days": 2,
        "floating_reset_lag_business_days": 2,
        "floating_payment_lag_business_days": 2,
        "floating_rate_index": "USD-SOFR-3M",
        "floating_compounding": "simple",
        "floating_gearing": 1.0,
        "floating_spread": 0.0,
        "model_time_day_count": "ACT/365F",
        "model_time_calendar_name": "USSettlement",
        "projection_policy": "static_additive_forward_basis",
        "lattice_steps": 240,
        "lattice_date_tolerance_days": 2,
    }
    fields.update(overrides)
    return make_physical_bermudan_swaption_contract(**fields)


def _with_term(contract, key: str, value):
    terms = dict(contract.product.term_fields)
    terms[key] = value
    return replace(
        contract,
        product=replace(contract.product, term_fields=MappingProxyType(terms)),
    )


def test_strict_physical_bermudan_contract_compiles_without_legacy_fallbacks():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _strict_contract()
    report = validate_semantic_contract(contract)

    assert report.ok, report.errors
    assert contract.semantic_id == "physical_bermudan_swaption"
    assert contract.product.instrument_class == "physical_bermudan_swaption"
    assert contract.product.exercise_style == "bermudan"
    assert contract.product.settlement_rule == "physical_swap_delivery_at_exercise"
    assert contract.product.payoff_traits == (
        "physical_settlement",
        "co_terminal_swap_tails",
        "dual_curve_projection",
        "strict_conventions",
        "named_hull_white_parameters",
        "bermudan_exercise",
    )
    terms = contract.product.term_fields
    assert terms["exercise_to_swap_start"] == (
        ("2027-06-15", "2027-06-17"),
        ("2028-06-15", "2028-06-19"),
        ("2029-06-15", "2029-06-19"),
    )
    assert terms["fixed_stub_rule"] == "short_final"
    assert terms["fixed_roll_convention"] == "none"
    assert terms["floating_reset_lag_business_days"] == 2
    assert terms["model_time_calendar_name"] == "USSettlement"
    assert terms["lattice_steps"] == 240
    assert tuple(item.input_id for item in contract.market_data.required_inputs) == (
        "discount_curve",
        "forecast_curve",
        "hull_white_parameters",
    )
    assert contract.market_data.derivable_inputs == ()
    assert contract.methods.candidate_methods == ("rate_tree",)
    assert contract.blueprint.primitive_families == (
        "physical_bermudan_swaption_lattice",
    )
    assert contract.blueprint.spec_schema_hints == ("physical_bermudan_swaption",)
    assert "analytical_black76" not in repr(contract)
    assert "black_vol_surface" not in repr(contract.market_data)

    compiled = compile_semantic_contract(contract)
    assert compiled.preferred_method == "rate_tree"
    assert compiled.primitive_routes == ("physical_bermudan_swaption_lattice",)
    assert compiled.spec_schema_hint == "physical_bermudan_swaption"
    assert compiled.required_market_data == (
        "discount_curve",
        "forecast_curve",
        "hull_white_parameters",
    )
    assert compiled.dsl_lowering.admissibility_errors == ()
    assert compiled.dsl_lowering.route_id == "physical_bermudan_swaption_lattice"
    from trellis.agent.dsl_algebra import ControlStyle, collect_primitive_refs

    assert compiled.dsl_lowering.control_styles == (ControlStyle.HOLDER_MAX,)
    assert set(collect_primitive_refs(compiled.dsl_lowering.normalized_expr)) == {
        "trellis.models.hull_white_parameters.resolve_named_hull_white_parameter_set",
        "trellis.models.rate_swap_tail.PhysicalBermudanSwapTailSpec",
        "trellis.models.rate_swap_tail.compile_physical_bermudan_swap_tail_spec",
        "trellis.models.rate_swap_tail.NamedRateCurve",
        "trellis.models.rate_swap_tail.resolve_co_terminal_swap_tails",
        "trellis.models.trees.models.MODEL_REGISTRY",
        "trellis.models.trees.algebra.BINOMIAL_1F_TOPOLOGY",
        "trellis.models.trees.algebra.UNIFORM_ADDITIVE_MESH",
        "trellis.models.trees.algebra.TERM_STRUCTURE_TARGET",
        "trellis.models.trees.algebra.build_lattice",
        "trellis.models.rate_swap_tail.price_physical_bermudan_swaption_lattice",
    }
    assert all("rate_style_swaption" not in module for module in compiled.target_modules)
    assert all("black" not in module for module in compiled.target_modules)

    with pytest.raises(ValueError, match="analytical.*not a candidate"):
        compile_semantic_contract(contract, preferred_method="analytical")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("discount_curve_id", None),
        ("forecast_curve_id", 7),
        ("hull_white_parameter_source", " USD-HW1F-2026Q4"),
        ("fixed_frequency", "semiannual "),
        ("currency", "usd"),
        ("notional", "10000000"),
        ("fixed_rate", float("nan")),
        ("floating_gearing", True),
        ("floating_spread", object()),
        ("lattice_steps", True),
        ("lattice_date_tolerance_days", 1.5),
    ),
)
def test_constructor_rejects_malformed_authored_values_before_coercion(
    field,
    bad_value,
):
    with pytest.raises((TypeError, ValueError), match=field):
        _strict_contract(**{field: bad_value})


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("fixed_frequency", "weekly"),
        ("floating_frequency", "3M"),
        ("fixed_day_count", "ACT/ACT ICMA"),
        ("floating_day_count", "ACT360"),
        ("fixed_calendar_name", "US-NY"),
        ("floating_calendar_name", "target"),
        ("model_time_calendar_name", "NewYork"),
        ("fixed_business_day_adjustment", "mod_follow"),
        ("floating_business_day_adjustment", "adjusted"),
        ("fixed_stub_rule", "short"),
        ("floating_stub_rule", "stub"),
        ("fixed_roll_convention", "day_of_month_17"),
        ("floating_roll_convention", "monthly"),
        ("model_time_day_count", "ACT/365"),
        ("floating_compounding", "continuous"),
        ("projection_policy", "dynamic_basis"),
    ),
)
def test_semantic_validation_rejects_tokens_outside_the_executable_vocabulary(
    field,
    bad_value,
):
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    report = validate_semantic_contract(_with_term(_strict_contract(), field, bad_value))

    assert not report.ok
    assert field in "; ".join(report.errors)


def test_semantic_terms_compile_and_resolve_on_the_strict_numerical_schedule():
    from trellis.models.rate_swap_tail import (
        NamedRateCurve,
        compile_physical_bermudan_swap_tail_spec,
        resolve_co_terminal_swap_tails,
    )

    class FlatCurve:
        def __init__(self, rate: float):
            self.rate = rate

        def discount(self, time: float) -> float:
            return exp(-self.rate * time)

    contract = _strict_contract()
    compiled = compile_physical_bermudan_swap_tail_spec(
        SimpleNamespace(**dict(contract.product.term_fields)),
        valuation_date=date(2026, 11, 15),
    )
    resolved = resolve_co_terminal_swap_tails(
        compiled,
        discount_curve=NamedRateCurve("USD-OIS", FlatCurve(0.02)),
        forecast_curve=NamedRateCurve("USD-SOFR-3M", FlatCurve(0.03)),
    )

    assert len(resolved.tails) == 3
    assert all(tail.fixed_periods for tail in resolved.tails)
    assert all(tail.floating_periods for tail in resolved.tails)


def test_strict_physical_bermudan_semantic_identity_does_not_alias_legacy_family():
    from trellis.agent.semantic_concepts import resolve_semantic_concept

    resolution = resolve_semantic_concept(
        "physical_bermudan_swaption",
        instrument_type="physical_bermudan_swaption",
    )

    assert resolution.concept_id == "physical_bermudan_swaption"
    assert resolution.resolution_kind == "reuse_existing_concept"


def test_strict_physical_bermudan_schema_preserves_all_authored_terms():
    from trellis.agent.planner import plan_build

    plan = plan_build(
        "Strict physical Bermudan swaption",
        {"discount_curve", "forward_curve", "model_parameters"},
        instrument_type="physical_bermudan_swaption",
        preferred_method="rate_tree",
        spec_schema_hint="physical_bermudan_swaption",
    )

    assert plan.payoff_class_name == "PhysicalBermudanSwaptionPayoff"
    assert plan.spec_schema is not None
    assert plan.spec_schema.requirements == [
        "discount_curve",
        "forward_curve",
        "model_parameters",
    ]
    fields = {field.name: field for field in plan.spec_schema.fields}
    assert tuple(fields) == (
        "notional",
        "fixed_rate",
        "exercise_dates",
        "exercise_to_swap_start",
        "swap_maturity",
        "payer_receiver",
        "settlement_type",
        "currency",
        "discount_curve_id",
        "forecast_curve_id",
        "hull_white_parameter_source",
        "fixed_frequency",
        "fixed_day_count",
        "fixed_calendar_name",
        "fixed_business_day_adjustment",
        "fixed_stub_rule",
        "fixed_roll_convention",
        "fixed_payment_lag_business_days",
        "floating_frequency",
        "floating_day_count",
        "floating_calendar_name",
        "floating_business_day_adjustment",
        "floating_stub_rule",
        "floating_roll_convention",
        "floating_fixing_lag_business_days",
        "floating_reset_lag_business_days",
        "floating_payment_lag_business_days",
        "floating_rate_index",
        "floating_compounding",
        "floating_gearing",
        "floating_spread",
        "model_time_day_count",
        "model_time_calendar_name",
        "projection_policy",
        "lattice_steps",
        "lattice_date_tolerance_days",
    )
    assert all(field.default is None for field in fields.values())


@pytest.mark.parametrize(
    ("term_name", "bad_value", "message"),
    (
        ("settlement_type", "cash", "physical settlement_type"),
        ("settlement_type", "annuity", "physical settlement_type"),
        ("discount_curve_id", "", "discount_curve_id"),
        ("forecast_curve_id", "", "forecast_curve_id"),
        ("forecast_curve_id", "USD-OIS", "must identify separate curves"),
        ("hull_white_parameter_source", "", "hull_white_parameter_source"),
        ("fixed_frequency", "", "fixed_frequency"),
        ("floating_day_count", "", "floating_day_count"),
        ("projection_policy", "single_curve", "static_additive_forward_basis"),
        ("lattice_steps", 0, "lattice_steps must be a positive integer"),
    ),
)
def test_strict_physical_bermudan_validation_rejects_missing_or_unsupported_terms(
    term_name,
    bad_value,
    message,
):
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    report = validate_semantic_contract(_with_term(_strict_contract(), term_name, bad_value))

    assert not report.ok
    assert message in "; ".join(report.errors)


def test_strict_physical_bermudan_validation_rejects_non_co_terminal_mapping():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _with_term(
        _strict_contract(),
        "exercise_to_swap_start",
        (
            ("2027-06-15", "2027-06-17"),
            ("2029-06-15", "2029-06-18"),
        ),
    )
    report = validate_semantic_contract(contract)

    assert not report.ok
    assert "exercise_to_swap_start" in "; ".join(report.errors)


def test_strict_validation_rejects_adjusted_fixed_accrual_before_exercise():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _strict_contract(
        exercise_dates=("2027-06-19", "2028-06-17"),
        exercise_to_swap_start=(
            ("2027-06-19", "2027-06-19"),
            ("2028-06-17", "2028-06-17"),
        ),
        fixed_calendar_name="WeekendOnly",
        fixed_business_day_adjustment="preceding",
        floating_calendar_name="WeekendOnly",
        floating_business_day_adjustment="following",
        floating_fixing_lag_business_days=0,
        floating_reset_lag_business_days=0,
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert "first fixed accrual start on or after exercise" in "; ".join(report.errors)


def test_strict_physical_bermudan_validation_rejects_legacy_route_or_black_input():
    from trellis.agent.semantic_contracts import SemanticMarketInputSpec
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _strict_contract()
    legacy_route = replace(
        contract,
        blueprint=replace(contract.blueprint, primitive_families=("exercise_lattice",)),
    )
    black_input = replace(
        contract,
        market_data=replace(
            contract.market_data,
            required_inputs=(
                *contract.market_data.required_inputs,
                SemanticMarketInputSpec(
                    input_id="black_vol_surface",
                    capability="black_vol_surface",
                ),
            ),
        ),
    )

    legacy_report = validate_semantic_contract(legacy_route)
    black_report = validate_semantic_contract(black_input)

    assert not legacy_report.ok
    assert "physical_bermudan_swaption_lattice" in "; ".join(legacy_report.errors)
    assert not black_report.ok
    assert "black_vol_surface" in "; ".join(black_report.errors)


def test_strict_physical_bermudan_validation_rejects_schema_or_obligation_fallback():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _strict_contract()
    legacy_schema = replace(
        contract,
        blueprint=replace(contract.blueprint, spec_schema_hints=("bermudan_swaption",)),
    )
    cash_obligation = replace(
        contract,
        product=replace(
            contract.product,
            obligations=(
                replace(contract.product.obligations[0], settlement_kind="cash"),
            ),
        ),
    )

    schema_report = validate_semantic_contract(legacy_schema)
    obligation_report = validate_semantic_contract(cash_obligation)

    assert not schema_report.ok
    assert "physical_bermudan_swaption spec schema" in "; ".join(schema_report.errors)
    assert not obligation_report.ok
    assert "physical swap-delivery obligation" in "; ".join(obligation_report.errors)
