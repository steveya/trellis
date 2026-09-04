from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from math import exp

import pytest

from trellis.conventions.calendar import BusinessDayAdjustment, Calendar, US_SETTLEMENT
from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.types import Frequency
from trellis.models.rate_swap_tail import (
    ExerciseSwapStart,
    FixedLegConvention,
    FloatingLegConvention,
    NamedRateCurve,
    PhysicalBermudanSwapTailSpec,
    build_bermudan_swaption_exercise_values,
    map_swap_tail_dates_to_lattice,
    observe_conditional_discount_bonds,
    price_physical_bermudan_swaption_lattice,
    resolve_co_terminal_swap_tails,
)
from trellis.models.trees.lattice import RecombiningLattice


class FlatCurve:
    def __init__(self, rate: float):
        self.rate = rate

    def discount(self, time: float) -> float:
        return exp(-self.rate * time)


def _fixed_convention(
    *,
    frequency: Frequency = Frequency.SEMI_ANNUAL,
    day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    calendar: Calendar | None = None,
    bda: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING,
) -> FixedLegConvention:
    return FixedLegConvention(
        frequency=frequency,
        day_count=day_count,
        calendar=calendar or Calendar("WeekendOnly"),
        business_day_adjustment=bda,
        stub_type=StubType.SHORT_LAST,
        roll_convention=RollConvention.NONE,
        payment_lag_business_days=0,
    )


def _floating_convention(
    *,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Calendar | None = None,
    bda: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING,
    reset_lag: int = 0,
    fixing_lag: int = 0,
    payment_lag: int = 0,
    rate_index: str = "USD-SOFR-3M",
) -> FloatingLegConvention:
    return FloatingLegConvention(
        frequency=frequency,
        day_count=day_count,
        calendar=calendar or Calendar("WeekendOnly"),
        business_day_adjustment=bda,
        stub_type=StubType.SHORT_LAST,
        roll_convention=RollConvention.NONE,
        reset_lag_business_days=reset_lag,
        fixing_lag_business_days=fixing_lag,
        payment_lag_business_days=payment_lag,
        rate_index=rate_index,
        compounding="simple",
        gearing=1.0,
        spread=0.0,
    )


def _spec(
    *,
    option_side: str = "payer",
    exercises: tuple[ExerciseSwapStart, ...] | None = None,
    maturity: date = date(2027, 1, 1),
    fixed: FixedLegConvention | None = None,
    floating: FloatingLegConvention | None = None,
    max_error_days: float = 3.0,
) -> PhysicalBermudanSwapTailSpec:
    return PhysicalBermudanSwapTailSpec(
        valuation_date=date(2025, 1, 1),
        exercise_swap_starts=exercises
        or (
            ExerciseSwapStart(date(2025, 7, 1), date(2025, 7, 1)),
            ExerciseSwapStart(date(2026, 1, 1), date(2026, 1, 1)),
        ),
        common_maturity_date=maturity,
        notional=1_000_000.0,
        fixed_rate=0.025,
        currency="USD",
        option_side=option_side,
        settlement_style="physical",
        fixed_convention=fixed or _fixed_convention(),
        floating_convention=floating or _floating_convention(),
        discount_curve_name="USD-OIS",
        forecast_curve_name="USD-SOFR-3M",
        projection_policy="static_additive_forward_basis",
        model_day_count=DayCountConvention.ACT_365,
        model_time_calendar=Calendar("WeekendOnly"),
        max_lattice_date_error_days=max_error_days,
    )


def _resolve(
    spec: PhysicalBermudanSwapTailSpec,
    *,
    discount_rate: float = 0.02,
    forecast_rate: float = 0.03,
):
    return resolve_co_terminal_swap_tails(
        spec,
        discount_curve=NamedRateCurve("USD-OIS", FlatCurve(discount_rate)),
        forecast_curve=NamedRateCurve("USD-SOFR-3M", FlatCurve(forecast_rate)),
    )


def _deterministic_lattice(*, rate: float = 0.02, n_steps: int = 24) -> RecombiningLattice:
    lattice = RecombiningLattice(n_steps=n_steps, dt=2.0 / n_steps, branching=2)
    for step in range(n_steps):
        for node in range(lattice.n_nodes(step)):
            lattice.set_probabilities(step, node, [0.5, 0.5])
            lattice.set_discount(step, node, exp(-rate * lattice.dt))
    return lattice


def test_contract_is_frozen_and_fails_closed_outside_physical_settlement():
    spec = _spec()
    with pytest.raises(FrozenInstanceError):
        spec.fixed_rate = 0.03  # type: ignore[misc]

    with pytest.raises(ValueError, match="physical settlement"):
        _spec().__class__(**{**spec.__dict__, "settlement_style": "cash"})


@pytest.mark.parametrize(
    "model_day_count",
    tuple(
        convention
        for convention in DayCountConvention
        if convention is not DayCountConvention.ACT_365
    ),
)
def test_contract_rejects_non_act_365f_model_time_bases(model_day_count):
    spec = _spec()

    with pytest.raises(ValueError, match="model-time day count.*ACT/365F"):
        spec.__class__(**{**spec.__dict__, "model_day_count": model_day_count})


@pytest.mark.parametrize("leg", ("fixed", "floating"))
def test_contract_rejects_act_act_icma_coupon_day_counts_without_quasi_periods(leg):
    with pytest.raises(ValueError, match="ACT/ACT ICMA.*not supported"):
        if leg == "fixed":
            _fixed_convention(day_count=DayCountConvention.ACT_ACT_ICMA)
        else:
            _floating_convention(day_count=DayCountConvention.ACT_ACT_ICMA)


@pytest.mark.parametrize("bad_name", ("", "  ", " USD-OIS", "USD-OIS ", 7, None))
def test_typed_contract_requires_exact_nonblank_curve_and_index_names(bad_name):
    curve = FlatCurve(0.02)
    with pytest.raises(ValueError, match="name must be an exact non-blank string"):
        NamedRateCurve(bad_name, curve)  # type: ignore[arg-type]

    spec = _spec()
    with pytest.raises(ValueError, match="discount curve name.*exact non-blank string"):
        spec.__class__(**{**spec.__dict__, "discount_curve_name": bad_name})
    with pytest.raises(ValueError, match="forecast curve name.*exact non-blank string"):
        spec.__class__(**{**spec.__dict__, "forecast_curve_name": bad_name})

    floating = _floating_convention()
    with pytest.raises(ValueError, match="floating rate index.*exact non-blank string"):
        floating.__class__(**{**floating.__dict__, "rate_index": bad_name})


@pytest.mark.parametrize("bad_currency", ("", "  ", "usd", "USD ", "US", "USDD", 7, None))
def test_typed_contract_requires_an_exact_uppercase_three_letter_currency(bad_currency):
    spec = _spec()

    with pytest.raises(ValueError, match="currency.*three uppercase ASCII letters"):
        spec.__class__(**{**spec.__dict__, "currency": bad_currency})


def test_resolution_requires_exact_separate_named_curves_and_policy():
    spec = _spec()
    with pytest.raises(ValueError, match="discount curve name"):
        resolve_co_terminal_swap_tails(
            spec,
            discount_curve=NamedRateCurve("USD-WRONG", FlatCurve(0.02)),
            forecast_curve=NamedRateCurve("USD-SOFR-3M", FlatCurve(0.03)),
        )
    with pytest.raises(ValueError, match="forecast curve name"):
        resolve_co_terminal_swap_tails(
            spec,
            discount_curve=NamedRateCurve("USD-OIS", FlatCurve(0.02)),
            forecast_curve=NamedRateCurve("USD-WRONG", FlatCurve(0.03)),
        )
    with pytest.raises(ValueError, match="static_additive_forward_basis"):
        spec.__class__(**{**spec.__dict__, "projection_policy": "dynamic_basis"})

    mismatched_index = _spec(
        floating=_floating_convention(rate_index="USD-LIBOR-3M")
    )
    with pytest.raises(ValueError, match="floating rate index.*forecast curve name"):
        _resolve(mismatched_index)


def test_lazy_builtin_calendar_is_admitted_by_the_structural_contract():
    spec = _spec(
        fixed=_fixed_convention(calendar=US_SETTLEMENT),
        floating=_floating_convention(calendar=US_SETTLEMENT),
    )
    spec = spec.__class__(**{**spec.__dict__, "model_time_calendar": US_SETTLEMENT})

    resolved = _resolve(spec)

    assert resolved.tails[0].fixed_periods
    assert resolved.tails[0].floating_periods


def test_each_exercise_builds_its_own_co_terminal_fixed_and_floating_schedules():
    resolved = _resolve(_spec())

    assert len(resolved.tails) == 2
    first, second = resolved.tails
    assert first.exercise_date == date(2025, 7, 1)
    assert second.exercise_date == date(2026, 1, 1)
    assert first.common_maturity_date == second.common_maturity_date == date(2027, 1, 1)
    assert len(first.fixed_periods) == 3
    assert len(first.floating_periods) == 6
    assert len(second.fixed_periods) == 2
    assert len(second.floating_periods) == 4
    assert first.fixed_periods[0].authored_start_date == date(2025, 7, 1)
    assert second.fixed_periods[0].authored_start_date == date(2026, 1, 1)


def test_schedule_consumes_business_day_lags_and_leg_day_counts():
    calendar = Calendar("WeekendOnly")
    spec = _spec(
        exercises=(ExerciseSwapStart(date(2025, 8, 27), date(2025, 8, 30)),),
        maturity=date(2026, 2, 28),
        fixed=_fixed_convention(
            frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.THIRTY_360,
            calendar=calendar,
        ),
        floating=_floating_convention(
            frequency=Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_360,
            calendar=calendar,
            reset_lag=1,
            fixing_lag=2,
            payment_lag=2,
        ),
    )
    tail = _resolve(spec).tails[0]
    first_float = tail.floating_periods[0]

    assert first_float.authored_start_date == date(2025, 8, 30)
    assert first_float.accrual_start_date == date(2025, 9, 1)
    assert first_float.reset_date == date(2025, 8, 29)
    assert first_float.fixing_date == date(2025, 8, 28)
    assert first_float.payment_date.weekday() < 5
    assert first_float.payment_date > first_float.accrual_end_date
    assert first_float.accrual_fraction == pytest.approx(
        (first_float.accrual_end_date - first_float.accrual_start_date).days / 360.0
    )
    assert tail.fixed_periods[0].accrual_fraction == pytest.approx(181.0 / 360.0)


def test_contract_rejects_an_adjusted_fixed_accrual_start_before_exercise():
    with pytest.raises(ValueError, match="fixed accrual start cannot precede exercise"):
        _spec(
            exercises=(
                ExerciseSwapStart(date(2025, 8, 30), date(2025, 8, 30)),
            ),
            maturity=date(2026, 2, 28),
            fixed=_fixed_convention(bda=BusinessDayAdjustment.PRECEDING),
            floating=_floating_convention(bda=BusinessDayAdjustment.FOLLOWING),
        )


@pytest.mark.parametrize(
    ("reset_lag", "fixing_lag", "message"),
    ((2, 0, "reset date cannot precede exercise"), (0, 2, "fixing date cannot precede exercise")),
)
def test_resolution_rejects_a_first_coupon_reset_or_fixed_before_exercise(
    reset_lag,
    fixing_lag,
    message,
):
    spec = _spec(
        exercises=(ExerciseSwapStart(date(2025, 8, 29), date(2025, 8, 30)),),
        maturity=date(2026, 2, 28),
        floating=_floating_convention(reset_lag=reset_lag, fixing_lag=fixing_lag),
    )
    with pytest.raises(ValueError, match=message):
        _resolve(spec)


def test_reset_and_fixing_dates_are_mapped_as_economic_lattice_events():
    resolved = _resolve(
        _spec(
            exercises=(ExerciseSwapStart(date(2025, 8, 27), date(2025, 8, 30)),),
            maturity=date(2026, 2, 28),
            floating=_floating_convention(reset_lag=1, fixing_lag=2),
            max_error_days=0.0,
        )
    )

    mapped = map_swap_tail_dates_to_lattice(
        resolved,
        _deterministic_lattice(n_steps=730),
    )

    first_period = resolved.tails[0].floating_periods[0]
    reset_points = [point for point in mapped.date_points if point.role == "floating_reset"]
    fixing_points = [point for point in mapped.date_points if point.role == "floating_fixing"]
    assert reset_points[0].adjusted_date == first_period.reset_date == date(2025, 8, 29)
    assert fixing_points[0].adjusted_date == first_period.fixing_date == date(2025, 8, 28)
    assert mapped.step_for(first_period.reset_date) != mapped.step_for(first_period.fixing_date)


def test_date_mapping_preserves_authored_and_adjusted_dates_and_rejects_collisions():
    resolved = _resolve(
        _spec(
            exercises=(ExerciseSwapStart(date(2025, 8, 27), date(2025, 8, 30)),),
            maturity=date(2026, 2, 28),
            max_error_days=20.0,
        )
    )
    mapped = map_swap_tail_dates_to_lattice(resolved, _deterministic_lattice(n_steps=120))
    start_points = [point for point in mapped.date_points if point.role == "floating_accrual_start"]
    assert start_points[0].authored_date == date(2025, 8, 30)
    assert start_points[0].adjusted_date == date(2025, 9, 1)
    assert start_points[0].error_days <= resolved.spec.max_lattice_date_error_days

    collision_resolved = _resolve(
        _spec(
            exercises=(ExerciseSwapStart(date(2025, 8, 27), date(2025, 8, 30)),),
            maturity=date(2026, 2, 28),
            max_error_days=400.0,
        )
    )
    with pytest.raises(ValueError, match="distinct adjusted dates.*same lattice step"):
        map_swap_tail_dates_to_lattice(collision_resolved, _deterministic_lattice(n_steps=2))


def test_date_mapping_rejects_dates_outside_error_tolerance():
    resolved = _resolve(_spec(max_error_days=0.01))
    with pytest.raises(ValueError, match="maximum lattice date error"):
        map_swap_tail_dates_to_lattice(resolved, _deterministic_lattice(n_steps=24))


def test_conditional_bond_observations_are_nodewise_and_single_curve_float_telescopes():
    resolved = _resolve(_spec(), discount_rate=0.02, forecast_rate=0.02)
    lattice = _deterministic_lattice(rate=0.02)
    mapped = map_swap_tail_dates_to_lattice(resolved, lattice)
    observations = observe_conditional_discount_bonds(lattice, mapped)
    exercise_values = build_bermudan_swaption_exercise_values(resolved, mapped, observations)

    assert observations.observations
    assert all(
        len(observation.node_values) == lattice.n_nodes(observation.exercise_step)
        for observation in observations.observations
    )
    first = exercise_values.by_exercise[0]
    tail = resolved.tails[0]
    start_step = mapped.step_for(tail.floating_periods[0].accrual_start_date)
    end_step = mapped.step_for(tail.floating_periods[-1].accrual_end_date)
    expected_float_unit = exp(-0.02 * (start_step - first.exercise_step) * lattice.dt) - exp(
        -0.02 * (end_step - first.exercise_step) * lattice.dt
    )
    assert first.floating_leg_values[0] / resolved.spec.notional == pytest.approx(
        expected_float_unit
    )


def test_forecast_curve_shock_moves_payer_and_receiver_values_in_opposite_directions():
    lattice = _deterministic_lattice(rate=0.02)

    def values(option_side: str, forecast_rate: float):
        resolved = _resolve(_spec(option_side=option_side), forecast_rate=forecast_rate)
        mapped = map_swap_tail_dates_to_lattice(resolved, lattice)
        observations = observe_conditional_discount_bonds(lattice, mapped)
        return build_bermudan_swaption_exercise_values(
            resolved, mapped, observations
        ).by_exercise[0]

    payer_low = values("payer", 0.01)
    payer_high = values("payer", 0.04)
    receiver_low = values("receiver", 0.01)
    receiver_high = values("receiver", 0.04)

    assert payer_high.signed_swap_values[0] > payer_low.signed_swap_values[0]
    assert receiver_high.signed_swap_values[0] < receiver_low.signed_swap_values[0]
    assert all(value >= 0.0 for value in payer_high.exercise_values)
    assert all(value >= 0.0 for value in receiver_low.exercise_values)


def test_exact_pricing_helper_composes_generic_rollback_and_more_rights_do_not_hurt():
    lattice = _deterministic_lattice(rate=0.02)
    discount = NamedRateCurve("USD-OIS", FlatCurve(0.02))
    forecast = NamedRateCurve("USD-SOFR-3M", FlatCurve(0.03))
    bermudan = _spec()
    european_right = _spec(exercises=(bermudan.exercise_swap_starts[-1],))

    bermudan_value = price_physical_bermudan_swaption_lattice(
        lattice,
        bermudan,
        discount_curve=discount,
        forecast_curve=forecast,
    )
    european_value = price_physical_bermudan_swaption_lattice(
        lattice,
        european_right,
        discount_curve=discount,
        forecast_curve=forecast,
    )

    assert bermudan_value >= european_value
    assert bermudan_value > 0.0


@pytest.mark.parametrize(
    ("option_side", "forecast_rate"),
    (("payer", 0.03), ("receiver", 0.01)),
)
def test_single_exercise_helper_matches_manual_zero_sigma_rollback(
    option_side: str,
    forecast_rate: float,
):
    """A deterministic lattice is the zero-short-rate-volatility oracle."""
    lattice = _deterministic_lattice(rate=0.02)
    spec = _spec(
        option_side=option_side,
        exercises=(ExerciseSwapStart(date(2026, 1, 1), date(2026, 1, 1)),),
    )
    discount = NamedRateCurve("USD-OIS", FlatCurve(0.02))
    forecast = NamedRateCurve("USD-SOFR-3M", FlatCurve(forecast_rate))
    resolved = resolve_co_terminal_swap_tails(
        spec,
        discount_curve=discount,
        forecast_curve=forecast,
    )
    mapped = map_swap_tail_dates_to_lattice(resolved, lattice)
    observations = observe_conditional_discount_bonds(lattice, mapped)
    node_values = build_bermudan_swaption_exercise_values(
        resolved,
        mapped,
        observations,
    ).by_exercise[0]
    expected = node_values.exercise_values[0] * exp(
        -0.02 * node_values.exercise_step * lattice.dt
    )

    actual = price_physical_bermudan_swaption_lattice(
        lattice,
        spec,
        discount_curve=discount,
        forecast_curve=forecast,
    )

    assert actual == pytest.approx(expected)
    assert actual > 0.0
