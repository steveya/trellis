"""Tests for the agent-side bridge into event-aware Monte Carlo runtime specs."""

from __future__ import annotations

from datetime import date

import numpy as raw_np

from trellis.core.types import DayCountConvention

from trellis.agent.family_lowering_ir import (
    EventAwareMonteCarloIR,
    MCCalibrationBindingSpec,
    MCControlSpec,
    MCEventSpec,
    MCEventTimeSpec,
    MCMeasureSpec,
    MCPathRequirementSpec,
    MCPayoffReducerSpec,
    MCProcessSpec,
    MCStateSpec,
)


def test_build_event_aware_monte_carlo_problem_from_family_ir_maps_typed_contract():
    from trellis.agent.assembly_tools import build_event_aware_monte_carlo_problem_from_family_ir
    from trellis.models.monte_carlo.event_aware import (
        EventAwareMonteCarloProcessSpec,
        build_event_aware_monte_carlo_problem_from_family_ir as build_runtime_problem_from_family_ir,
    )

    family_ir = EventAwareMonteCarloIR(
        route_id="monte_carlo_paths",
        route_family="monte_carlo",
        product_instrument="swaption",
        payoff_family="swaption",
        state_spec=MCStateSpec(
            state_variable="short_rate",
            dimension=1,
            state_tags=("terminal_markov", "schedule_state"),
        ),
        process_spec=MCProcessSpec(
            process_family="hull_white_1f",
            simulation_scheme="exact_ou",
        ),
        path_requirement_spec=MCPathRequirementSpec(
            requirement_kind="event_replay",
            reducer_kinds=("discounted_swap_pv",),
            stored_fields=("exercise_state",),
        ),
        payoff_reducer_spec=MCPayoffReducerSpec(
            reducer_kind="compiled_schedule_payoff",
            output_semantics="swaption_exercise_payoff",
            event_dependencies=("exercise", "settlement"),
        ),
        control_spec=MCControlSpec(
            control_style="identity",
            controller_role="holder",
        ),
        measure_spec=MCMeasureSpec(
            measure_family="risk_neutral",
            numeraire_binding="discount_curve",
        ),
        calibration_binding=MCCalibrationBindingSpec(
            model_family="hull_white_1f",
            quote_family="black_swaption_vol",
            required_parameters=("mean_reversion", "sigma"),
            requires_quote_normalization=True,
        ),
        event_timeline=(
            MCEventTimeSpec(
                event_date="2027-03-15",
                schedule_roles=("observation_dates", "settlement_dates"),
                phase_sequence=("observation", "settlement"),
                events=(
                    MCEventSpec(
                        event_name="exercise",
                        event_kind="observation",
                        schedule_role="observation_dates",
                        phase="observation",
                        value_semantics="forward_swap_rate",
                    ),
                    MCEventSpec(
                        event_name="settlement",
                        event_kind="settlement",
                        schedule_role="settlement_dates",
                        phase="settlement",
                        value_semantics="cash_settlement",
                    ),
                ),
            ),
        ),
    )

    problem_spec = build_event_aware_monte_carlo_problem_from_family_ir(
        family_ir,
        process_spec=EventAwareMonteCarloProcessSpec(
            family="hull_white_1f",
            mean_reversion=0.10,
            sigma=0.01,
            theta=0.03,
        ),
        initial_state=0.03,
        maturity=1.0,
        event_time_map={"2027-03-15": 1.0},
        event_payloads={
            "settlement": {"rule": "terminal_value"},
        },
        state_payoff=lambda state: raw_np.maximum(state.settlement_value("settlement"), 0.0),
    )

    assert problem_spec.path_requirement_kind == "event_replay"
    assert problem_spec.reducer_kind == "compiled_schedule_payoff"
    assert problem_spec.settlement_event == "settlement"
    assert tuple(event.name for event in problem_spec.event_specs) == ("exercise", "settlement")
    assert problem_spec.event_specs[-1].payload["rule"] == "terminal_value"
    assert problem_spec.process_spec.family == "hull_white_1f"

    runtime_problem_spec = build_runtime_problem_from_family_ir(
        family_ir,
        process_spec=EventAwareMonteCarloProcessSpec(
            family="hull_white_1f",
            mean_reversion=0.10,
            sigma=0.01,
            theta=0.03,
        ),
        initial_state=0.03,
        maturity=1.0,
        event_time_map={"2027-03-15": 1.0},
        event_payloads={
            "settlement": {"rule": "terminal_value"},
        },
        state_payoff=lambda state: raw_np.maximum(state.settlement_value("settlement"), 0.0),
    )

    assert runtime_problem_spec.path_requirement_kind == problem_spec.path_requirement_kind
    assert runtime_problem_spec.reducer_kind == problem_spec.reducer_kind
    assert runtime_problem_spec.settlement_event == problem_spec.settlement_event
    assert runtime_problem_spec.process_spec == problem_spec.process_spec
    assert runtime_problem_spec.event_specs == problem_spec.event_specs


def test_build_timed_event_aware_monte_carlo_problem_from_family_ir_derives_event_times():
    from trellis.models.monte_carlo.event_aware import (
        EventAwareMonteCarloProcessSpec,
        build_timed_event_aware_monte_carlo_problem_from_family_ir,
    )

    family_ir = EventAwareMonteCarloIR(
        route_id="monte_carlo_paths",
        route_family="monte_carlo",
        product_instrument="swaption",
        payoff_family="swaption",
        state_spec=MCStateSpec(state_variable="short_rate", dimension=1),
        process_spec=MCProcessSpec(process_family="hull_white_1f"),
        path_requirement_spec=MCPathRequirementSpec(requirement_kind="event_replay"),
        payoff_reducer_spec=MCPayoffReducerSpec(reducer_kind="compiled_schedule_payoff"),
        control_spec=MCControlSpec(control_style="identity", controller_role="holder"),
        measure_spec=MCMeasureSpec(measure_family="risk_neutral", numeraire_binding="discount_curve"),
        event_timeline=(
            MCEventTimeSpec(
                event_date="2027-03-15",
                schedule_roles=("observation_dates", "settlement_dates"),
                phase_sequence=("observation", "settlement"),
                events=(
                    MCEventSpec(
                        event_name="exercise",
                        event_kind="observation",
                        schedule_role="observation_dates",
                        phase="observation",
                    ),
                    MCEventSpec(
                        event_name="settlement",
                        event_kind="settlement",
                        schedule_role="settlement_dates",
                        phase="settlement",
                    ),
                ),
            ),
        ),
    )

    problem_spec = build_timed_event_aware_monte_carlo_problem_from_family_ir(
        family_ir,
        process_spec=EventAwareMonteCarloProcessSpec(
            family="hull_white_1f",
            mean_reversion=0.10,
            sigma=0.01,
            theta=0.03,
        ),
        initial_state=0.03,
        maturity=1.0,
        time_origin=date(2026, 3, 15),
        day_count=DayCountConvention.ACT_365,
        event_payloads={"settlement": {"rule": "terminal_value"}},
        state_payoff=lambda state: raw_np.maximum(state.settlement_value("settlement"), 0.0),
    )

    assert tuple(event.name for event in problem_spec.event_specs) == ("exercise", "settlement")
    assert tuple(round(event.time, 10) for event in problem_spec.event_specs) == (1.0, 1.0)
    assert problem_spec.settlement_event == "settlement"
