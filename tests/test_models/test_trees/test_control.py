"""Tests for lattice control contracts."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.date_utils import build_exercise_timeline_from_dates
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.trees.control import (
    ExerciseObjective,
    build_exercise_timeline_from_dates,
    build_payment_timeline,
    lattice_step_from_time,
    lattice_steps_from_timeline,
    resolve_lattice_exercise_policy,
    resolve_lattice_exercise_policy_from_control_style,
)


def test_resolve_lattice_exercise_policy_for_issuer_call():
    policy = resolve_lattice_exercise_policy("issuer_call", exercise_steps=[10, 25, 10])

    assert policy.exercise_type == "bermudan"
    assert policy.exercise_steps == (10, 25)
    assert policy.objective == ExerciseObjective.ISSUER_MINIMIZE
    assert policy.objective_name == "min"
    assert policy.exercise_fn is min


def test_resolve_lattice_exercise_policy_from_control_style_for_issuer_call():
    policy = resolve_lattice_exercise_policy_from_control_style(
        "issuer_min",
        exercise_steps=[10, 25, 10],
    )

    assert policy.exercise_style == "issuer_call"
    assert policy.exercise_type == "bermudan"
    assert policy.exercise_steps == (10, 25)
    assert policy.objective == ExerciseObjective.ISSUER_MINIMIZE


def test_lattice_steps_from_timeline_maps_exercise_dates():
    timeline = build_exercise_timeline_from_dates(
        [date(2027, 1, 15), date(2029, 1, 15), date(2032, 1, 15)],
        day_count=DayCountConvention.ACT_365,
        time_origin=date(2025, 1, 15),
        label="call_dates",
    )

    steps = lattice_steps_from_timeline(
        timeline,
        dt=1.0,
        n_steps=6,
    )

    assert steps == (2, 4)


def test_lattice_steps_from_timeline_supports_positional_dt_and_n_steps_for_numeric_times():
    steps = lattice_steps_from_timeline([2.0, 4.0, 7.0], 1.0, 6)

    assert steps == (2, 4)


def test_lattice_step_from_time_includes_terminal_step_by_default():
    step = lattice_step_from_time(6.0, dt=1.0, n_steps=6)

    assert step == 6


def test_lattice_steps_from_timeline_keeps_singleton_terminal_compatibility():
    steps = lattice_steps_from_timeline([6.0], 1.0, 6)

    assert steps == (6,)


def test_lattice_step_from_time_accepts_schedule_period():
    from trellis.core.types import SchedulePeriod

    period = SchedulePeriod(
        start_date=date(2025, 1, 15),
        end_date=date(2025, 7, 15),
        payment_date=date(2025, 7, 15),
        accrual_fraction=0.5,
        t_start=0.0,
        t_end=0.5,
        t_payment=0.5,
    )

    step = lattice_step_from_time(period, dt=0.25, n_steps=8)

    assert step == 2


def test_control_module_reexports_timeline_builders():
    payment_timeline = build_payment_timeline(
        date(2025, 1, 15),
        date(2026, 1, 15),
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
        time_origin=date(2025, 1, 15),
    )
    exercise_timeline = build_exercise_timeline_from_dates(
        [date(2025, 7, 15), date(2026, 1, 15)],
        day_count=DayCountConvention.ACT_365,
        time_origin=date(2025, 1, 15),
    )

    assert len(payment_timeline.periods) == 2
    assert len(exercise_timeline.periods) == 2


def test_lattice_steps_from_timeline_without_lattice_spacing_returns_ordinal_map():
    mapping = lattice_steps_from_timeline(
        [date(2027, 1, 15), date(2029, 1, 15), date(2032, 1, 15)]
    )

    assert mapping == {
        date(2027, 1, 15): 1,
        date(2029, 1, 15): 2,
        date(2032, 1, 15): 3,
    }


def test_lattice_steps_from_timeline_requires_positive_dt():
    timeline = build_exercise_timeline_from_dates(
        [date(2027, 1, 15)],
        day_count=DayCountConvention.ACT_365,
        time_origin=date(2025, 1, 15),
    )

    with pytest.raises(ValueError, match="dt must be positive"):
        lattice_steps_from_timeline(timeline, dt=0.0, n_steps=10)
