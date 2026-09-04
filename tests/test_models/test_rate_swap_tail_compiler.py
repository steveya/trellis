"""Strict semantic-spec compilation for physical Bermudan swap tails."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from types import SimpleNamespace

import pytest

from trellis.conventions.calendar import (
    BRAZIL,
    SYDNEY,
    BusinessDayAdjustment,
    TARGET,
    TOKYO,
    TORONTO,
    UK_SETTLEMENT,
    US_SETTLEMENT,
    WEEKEND_ONLY,
    ZURICH,
)
from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.types import Frequency
from trellis.models.rate_swap_tail import (
    compile_physical_bermudan_swap_tail_spec,
)


def _semantic_spec(**overrides):
    fields = {
        "notional": 10_000_000.0,
        "fixed_rate": 0.031,
        "exercise_dates": ("2027-06-15", "2028-06-15"),
        "exercise_to_swap_start": (
            ("2027-06-15", "2027-06-17"),
            ("2028-06-15", "2028-06-19"),
        ),
        "swap_maturity": "2032-06-17",
        "payer_receiver": "payer",
        "settlement_type": "physical",
        "discount_curve_id": "USD-OIS",
        "forecast_curve_id": "USD-SOFR-3M",
        "currency": "USD",
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
        "floating_business_day_adjustment": "following",
        "floating_stub_rule": "long_last",
        "floating_roll_convention": "eom",
        "floating_fixing_lag_business_days": 2,
        "floating_reset_lag_business_days": 1,
        "floating_payment_lag_business_days": 2,
        "floating_rate_index": "USD-SOFR-3M",
        "floating_compounding": "simple",
        "floating_gearing": 1.0,
        "floating_spread": 0.001,
        "model_time_day_count": "ACT/365F",
        "model_time_calendar_name": "TARGET",
        "projection_policy": "static_additive_forward_basis",
        "lattice_date_tolerance_days": 2,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_compiler_consumes_every_supported_semantic_convention_exactly():
    compiled = compile_physical_bermudan_swap_tail_spec(
        _semantic_spec(),
        valuation_date=date(2026, 11, 15),
    )

    assert compiled.valuation_date == date(2026, 11, 15)
    assert tuple(
        (item.exercise_date, item.swap_start_date)
        for item in compiled.exercise_swap_starts
    ) == (
        (date(2027, 6, 15), date(2027, 6, 17)),
        (date(2028, 6, 15), date(2028, 6, 19)),
    )
    assert compiled.common_maturity_date == date(2032, 6, 17)
    assert compiled.option_side == "payer"
    assert compiled.settlement_style == "physical"
    assert compiled.discount_curve_name == "USD-OIS"
    assert compiled.forecast_curve_name == "USD-SOFR-3M"
    assert compiled.currency == "USD"
    assert compiled.projection_policy == "static_additive_forward_basis"
    assert compiled.max_lattice_date_error_days == 2.0

    fixed = compiled.fixed_convention
    assert fixed.frequency is Frequency.SEMI_ANNUAL
    assert fixed.day_count is DayCountConvention.THIRTY_360
    assert fixed.calendar is US_SETTLEMENT
    assert fixed.business_day_adjustment is BusinessDayAdjustment.MODIFIED_FOLLOWING
    assert fixed.stub_type is StubType.SHORT_LAST
    assert fixed.roll_convention is RollConvention.NONE
    assert fixed.payment_lag_business_days == 2

    floating = compiled.floating_convention
    assert floating.frequency is Frequency.QUARTERLY
    assert floating.day_count is DayCountConvention.ACT_360
    assert floating.calendar is US_SETTLEMENT
    assert floating.business_day_adjustment is BusinessDayAdjustment.FOLLOWING
    assert floating.stub_type is StubType.LONG_LAST
    assert floating.roll_convention is RollConvention.EOM
    assert floating.reset_lag_business_days == 1
    assert floating.fixing_lag_business_days == 2
    assert floating.payment_lag_business_days == 2
    assert floating.rate_index == "USD-SOFR-3M"
    assert floating.compounding == "simple"
    assert floating.gearing == 1.0
    assert floating.spread == 0.001

    assert compiled.model_day_count is DayCountConvention.ACT_365
    assert compiled.model_time_calendar is TARGET

    with pytest.raises(FrozenInstanceError):
        compiled.option_side = "receiver"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("authored_name", "expected"),
    (
        ("WeekendOnly", WEEKEND_ONLY),
        ("weekend_only", WEEKEND_ONLY),
        ("USSettlement", US_SETTLEMENT),
        ("UKSettlement", UK_SETTLEMENT),
        ("TARGET", TARGET),
        ("Tokyo", TOKYO),
        ("Sydney", SYDNEY),
        ("Toronto", TORONTO),
        ("Zurich", ZURICH),
        ("Brazil", BRAZIL),
    ),
)
def test_compiler_resolves_only_declared_builtin_calendar_names(
    authored_name,
    expected,
):
    compiled = compile_physical_bermudan_swap_tail_spec(
        _semantic_spec(
            fixed_calendar_name=authored_name,
            floating_calendar_name=authored_name,
            model_time_calendar_name=authored_name,
        ),
        valuation_date=date(2026, 11, 15),
    )

    assert compiled.fixed_convention.calendar is expected
    assert compiled.floating_convention.calendar is expected
    assert compiled.model_time_calendar is expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fixed_frequency", "SemiAnnual", "fixed_frequency"),
        ("fixed_day_count", "actual/360", "fixed_day_count"),
        ("fixed_calendar_name", "NewYork", "fixed_calendar_name"),
        ("fixed_calendar_name", "US-NY", "fixed_calendar_name"),
        ("fixed_business_day_adjustment", "mod_follow", "fixed_business_day_adjustment"),
        ("fixed_stub_rule", "short", "fixed_stub_rule"),
        ("fixed_roll_convention", "day_of_month_17", "fixed_roll_convention"),
        ("floating_frequency", "3M", "floating_frequency"),
        ("floating_day_count", "ACT360", "floating_day_count"),
        ("floating_calendar_name", "target", "floating_calendar_name"),
        ("floating_business_day_adjustment", "adjusted", "floating_business_day_adjustment"),
        ("floating_stub_rule", "stub", "floating_stub_rule"),
        ("floating_roll_convention", "monthly", "floating_roll_convention"),
        ("model_time_day_count", "365", "model_time_day_count"),
        ("model_time_calendar_name", "US", "model_time_calendar_name"),
        ("payer_receiver", "pay", "payer_receiver"),
        ("settlement_type", "cash", "settlement_type"),
        ("projection_policy", "single_curve", "projection_policy"),
    ),
)
def test_compiler_rejects_unknown_or_inexact_semantic_tokens(field, value, message):
    with pytest.raises(ValueError, match=message):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(**{field: value}),
            valuation_date=date(2026, 11, 15),
        )


def test_compiler_requires_mapping_to_match_authored_exercise_dates_exactly():
    with pytest.raises(ValueError, match="exercise_to_swap_start.*exactly"):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(
                exercise_to_swap_start=(("2027-06-15", "2027-06-17"),),
            ),
            valuation_date=date(2026, 11, 15),
        )

    with pytest.raises(ValueError, match="exercise_to_swap_start.*in order"):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(
                exercise_to_swap_start=(
                    ("2028-06-15", "2028-06-19"),
                    ("2027-06-15", "2027-06-17"),
                ),
            ),
            valuation_date=date(2026, 11, 15),
        )


def test_compiler_rejects_an_adjusted_fixed_accrual_start_before_exercise():
    with pytest.raises(ValueError, match="fixed accrual start cannot precede exercise"):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(
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
            ),
            valuation_date=date(2026, 11, 15),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fixed_payment_lag_business_days", True),
        ("floating_fixing_lag_business_days", 1.5),
        ("floating_reset_lag_business_days", -1),
        ("floating_payment_lag_business_days", "2"),
        ("lattice_date_tolerance_days", True),
    ),
)
def test_compiler_rejects_implicit_or_invalid_numeric_convention_values(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(**{field: value}),
            valuation_date=date(2026, 11, 15),
        )


def test_compiler_rejects_missing_required_convention_field():
    spec = _semantic_spec()
    delattr(spec, "floating_reset_lag_business_days")

    with pytest.raises(ValueError, match="floating_reset_lag_business_days is required"):
        compile_physical_bermudan_swap_tail_spec(
            spec,
            valuation_date=date(2026, 11, 15),
        )


@pytest.mark.parametrize(
    "authored_day_count",
    (
        "ACT/360",
        "ACT/365",
        "ACT/ACT",
        "ACT/ACT ISDA",
        "ACT/ACT ICMA",
        "30/360",
        "30E/360",
        "30E/360 ISDA",
        "ACT/365.25",
        "BUS/252",
        "1/1",
    ),
)
def test_compiler_model_time_admits_only_exact_act_365f(authored_day_count):
    with pytest.raises(ValueError, match="model_time_day_count"):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(model_time_day_count=authored_day_count),
            valuation_date=date(2026, 11, 15),
        )


@pytest.mark.parametrize("field", ("fixed_day_count", "floating_day_count"))
def test_compiler_rejects_act_act_icma_coupon_basis(field):
    with pytest.raises(ValueError, match="ACT/ACT ICMA.*not supported"):
        compile_physical_bermudan_swap_tail_spec(
            _semantic_spec(**{field: "ACT/ACT ICMA"}),
            valuation_date=date(2026, 11, 15),
        )
