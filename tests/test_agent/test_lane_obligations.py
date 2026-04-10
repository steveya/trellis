"""Tests for compiler-emitted lane obligations."""

from __future__ import annotations

from trellis.agent.dsl_lowering import SemanticDslLowering
from trellis.agent.family_lowering_ir import (
    ControlProgramIR,
    EventProgramIR,
    EventAwareMonteCarloIR,
    MCControlSpec,
    MCEventSpec,
    MCEventTimeSpec,
    MCMeasureSpec,
    MCPathRequirementSpec,
    MCPayoffReducerSpec,
    MCProcessSpec,
    MCStateSpec,
    SemanticEventSpec,
    SemanticEventTimeSpec,
    TransformCharacteristicSpec,
    TransformControlSpec,
    TransformPricingIR,
    TransformStateSpec,
)
from trellis.agent.lane_obligations import compile_lane_construction_plan


def test_event_aware_monte_carlo_lane_plan_emits_process_path_and_event_contracts():
    family_ir = EventAwareMonteCarloIR(
        route_id="monte_carlo_paths",
        route_family="monte_carlo",
        product_instrument="swaption",
        payoff_family="swaption",
        state_spec=MCStateSpec(
            state_variable="short_rate",
            state_tags=("terminal_markov", "schedule_state"),
        ),
        process_spec=MCProcessSpec(
            process_family="hull_white_1f",
            simulation_scheme="exact_ou",
        ),
        event_program=EventProgramIR(
            timeline=(
                SemanticEventTimeSpec(
                    event_date="2027-03-15",
                    schedule_roles=("observation_dates", "decision_dates"),
                    phase_sequence=("observation", "decision"),
                    events=(
                        SemanticEventSpec(
                            event_name="exercise",
                            event_kind="exercise",
                            schedule_role="decision_dates",
                            phase="decision",
                            value_semantics="holder_exercise_projection",
                            transform_kind="project_max",
                        ),
                    ),
                ),
            ),
        ),
        control_program=ControlProgramIR(
            control_style="holder_max",
            controller_role="holder",
            decision_phase="decision",
            schedule_role="decision_dates",
            exercise_style="european",
        ),
        path_requirement_spec=MCPathRequirementSpec(
            requirement_kind="event_replay",
            reducer_kinds=("discounted_swap_pv",),
        ),
        payoff_reducer_spec=MCPayoffReducerSpec(
            reducer_kind="positive_part_at_exercise",
            output_semantics="swaption_exercise_payoff",
        ),
        control_spec=MCControlSpec(
            control_style="identity",
            controller_role="holder",
        ),
        measure_spec=MCMeasureSpec(
            measure_family="risk_neutral",
            numeraire_binding="discount_curve",
        ),
        event_timeline=(
            MCEventTimeSpec(
                event_date="2027-03-15",
                schedule_roles=("observation_dates",),
                phase_sequence=("observation",),
                events=(
                    MCEventSpec(
                        event_name="exercise",
                        event_kind="observation",
                        schedule_role="observation_dates",
                        phase="observation",
                        value_semantics="forward_swap_rate",
                    ),
                ),
            ),
        ),
    )
    lowering = SemanticDslLowering(
        route_id="monte_carlo_paths",
        route_family="monte_carlo",
        family_ir=family_ir,
        expr=None,
        normalized_expr=None,
    )

    plan = compile_lane_construction_plan(
        preferred_method="monte_carlo",
        required_market_data=("discount_curve", "forward_curve", "black_vol_surface"),
        dsl_lowering=lowering,
    )

    assert plan is not None
    assert plan.lane_family == "monte_carlo"
    assert "short_rate" in plan.state_obligations
    assert "event_replay" in plan.state_obligations
    assert "discounted_swap_pv" in plan.state_obligations
    assert "control_style:identity" in plan.control_obligations
    assert "semantic_control_style:holder_max" in plan.control_obligations
    assert "controller_role:holder" in plan.control_obligations
    assert "semantic_event_kind:exercise" in plan.control_obligations
    assert "event_kind:observation" in plan.control_obligations
    assert "measure_family:risk_neutral" in plan.control_obligations
    assert any("hull_white_1f" in step for step in plan.construction_steps)
    assert any("event_replay" in step for step in plan.construction_steps)


def test_transform_lane_plan_emits_terminal_characteristic_contract():
    family_ir = TransformPricingIR(
        route_id="transform_fft",
        route_family="fft_pricing",
        product_instrument="european_option",
        payoff_family="vanilla_option",
        state_spec=TransformStateSpec(
            state_variable="spot",
            state_tags=("terminal_markov",),
        ),
        characteristic_spec=TransformCharacteristicSpec(
            model_family="equity_diffusion",
            characteristic_family="gbm_log_spot",
            supported_methods=("fft", "cos"),
            backend_capability="helper_backed",
        ),
        control_program=ControlProgramIR(
            control_style="holder_max",
            controller_role="holder",
            decision_phase="decision",
            schedule_role="decision_dates",
            exercise_style="european",
        ),
        control_spec=TransformControlSpec(
            control_style="identity",
            controller_role="holder",
        ),
        terminal_payoff_kind="vanilla_terminal_payoff",
        strike_semantics="vanilla_strike",
        quote_semantics="equity_black_vol_surface",
        helper_symbol="price_vanilla_equity_option_transform",
        market_mapping="single_state_diffusion_transform_inputs",
    )
    lowering = SemanticDslLowering(
        route_id="transform_fft",
        route_family="fft_pricing",
        family_ir=family_ir,
        expr=None,
        normalized_expr=None,
    )

    plan = compile_lane_construction_plan(
        preferred_method="fft_pricing",
        required_market_data=("discount_curve", "black_vol_surface"),
        dsl_lowering=lowering,
    )

    assert plan is not None
    assert plan.lane_family == "fft_pricing"
    assert "spot" in plan.state_obligations
    assert "gbm_log_spot" in plan.state_obligations
    assert "vanilla_terminal_payoff" in plan.state_obligations
    assert "control_style:identity" in plan.control_obligations
    assert "semantic_control_style:holder_max" in plan.control_obligations
    assert "quote_semantics:equity_black_vol_surface" in plan.control_obligations
    assert any("terminal-only transform contract" in step for step in plan.construction_steps)
    assert any("helper_backed" in step for step in plan.construction_steps)
