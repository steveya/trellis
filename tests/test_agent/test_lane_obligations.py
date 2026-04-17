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


def test_waterfall_lane_plan_emits_cashflow_engine_construction_steps():
    """QUA-816 slice 1 round-1 Codex P1: removing the route-card adapter for
    ``waterfall_cashflows`` must not leave the lane with empty construction
    steps.  The lane-obligation surface now carries the "map collateral
    cashflows onto the tranche structure" guidance that used to live as an
    adapter on the route card.
    """
    lowering = SemanticDslLowering(
        route_id="waterfall_cashflows",
        route_family="waterfall",
        family_ir=None,
        expr=None,
        normalized_expr=None,
    )

    plan = compile_lane_construction_plan(
        preferred_method="waterfall",
        required_market_data=("discount_curve",),
        dsl_lowering=lowering,
    )

    assert plan is not None
    assert plan.lane_family == "waterfall"
    assert plan.construction_steps, (
        "waterfall lane plan must emit construction steps even without a "
        "specialized family IR; otherwise removing route-card adapters "
        "leaves build-time guidance empty."
    )
    assert any("cashflow_engine" in step for step in plan.construction_steps)
    assert any("tranche" in step.lower() for step in plan.construction_steps)


def test_exercise_monte_carlo_lane_plan_emits_policy_summaries_and_build_steps():
    """QUA-880: early-exercise MC lanes migrate dynamic policy summaries +
    3 construction adapters onto the typed lane-obligation surface.

    ``exercise_monte_carlo`` used to carry three adapters (exercise-date
    derivation, payoff callback, continuation estimator selection), three
    static notes (LSM naming invariant), and two dynamic notes that called
    ``render_early_exercise_policy_summary()`` at render time.  All of that
    now flows through ``compile_lane_construction_plan`` when the control
    spec indicates ``holder_max`` / ``issuer_min``.
    """
    family_ir = EventAwareMonteCarloIR(
        route_id="exercise_monte_carlo",
        route_family="exercise",
        product_instrument="american_option",
        payoff_family="vanilla_option",
        state_spec=MCStateSpec(
            state_variable="spot",
            state_tags=("terminal_markov", "schedule_state"),
        ),
        process_spec=MCProcessSpec(
            process_family="gbm_1d",
            simulation_scheme="exact_gbm",
        ),
        event_program=EventProgramIR(
            timeline=(
                SemanticEventTimeSpec(
                    event_date="2027-06-15",
                    schedule_roles=("exercise_dates",),
                    phase_sequence=("decision",),
                    events=(
                        SemanticEventSpec(
                            event_name="exercise",
                            event_kind="decision",
                            schedule_role="exercise_dates",
                            phase="decision",
                            value_semantics="spot_minus_strike_positive_part",
                        ),
                    ),
                ),
            ),
        ),
        control_program=ControlProgramIR(
            control_style="holder_max",
            controller_role="holder",
            decision_phase="decision",
        ),
        path_requirement_spec=MCPathRequirementSpec(
            requirement_kind="full_path",
        ),
        payoff_reducer_spec=MCPayoffReducerSpec(
            reducer_kind="exercise_policy",
        ),
        control_spec=MCControlSpec(
            control_style="holder_max",
            controller_role="holder",
        ),
        measure_spec=MCMeasureSpec(
            measure_family="risk_neutral",
            numeraire_binding="money_market_account",
        ),
        event_timeline=(
            MCEventTimeSpec(
                event_date="2027-06-15",
                schedule_roles=("exercise_dates",),
                phase_sequence=("decision",),
                events=(
                    MCEventSpec(
                        event_name="exercise",
                        event_kind="decision",
                        schedule_role="exercise_dates",
                        phase="decision",
                        value_semantics="spot_minus_strike_positive_part",
                    ),
                ),
            ),
        ),
        event_specs=(
            MCEventSpec(
                event_name="exercise",
                event_kind="decision",
                schedule_role="exercise_dates",
                phase="decision",
                value_semantics="spot_minus_strike_positive_part",
            ),
        ),
    )
    lowering = SemanticDslLowering(
        route_id="exercise_monte_carlo",
        route_family="exercise",
        family_ir=family_ir,
        expr=None,
        normalized_expr=None,
    )

    plan = compile_lane_construction_plan(
        preferred_method="monte_carlo",
        required_market_data=("discount_curve", "black_vol_surface"),
        dsl_lowering=lowering,
    )

    assert plan is not None
    assert plan.lane_family == "monte_carlo"

    # Control obligations carry the approved / implemented policy
    # summaries + the LSM naming invariant.
    control_text = " ".join(plan.control_obligations)
    assert "approved_exercise_policies:" in control_text
    assert "implemented_exercise_policies:" in control_text
    assert "longstaff_schwartz" in control_text

    # Construction steps carry the three adapter equivalents.
    steps_text = " ".join(plan.construction_steps)
    assert "exercise-date grid" in steps_text
    assert "spot-to-exercise payoff callback" in steps_text
    assert "continuation estimator" in steps_text.lower() or (
        "regression basis" in steps_text.lower()
    )


def test_pde_theta_1d_lane_plan_emits_kernel_contract_construction_steps():
    """QUA-880: PDE theta-method lane migrates its 5 pricing-kernel notes
    onto the lane-obligation surface as ``Pricing-kernel contract:`` steps.
    """
    from trellis.agent.family_lowering_ir import (
        EventAwarePDEIR,
        PDEBoundarySpec,
        PDEControlSpec,
        PDEOperatorSpec,
        PDEStateSpec,
    )

    family_ir = EventAwarePDEIR(
        route_id="pde_theta_1d",
        route_family="pde_solver",
        product_instrument="american_option",
        payoff_family="vanilla_option",
        state_spec=PDEStateSpec(state_variable="spot"),
        operator_spec=PDEOperatorSpec(operator_family="black_scholes_1d"),
        boundary_spec=PDEBoundarySpec(terminal_condition_kind="vanilla_payoff"),
        control_spec=PDEControlSpec(control_style="holder_max"),
    )
    lowering = SemanticDslLowering(
        route_id="pde_theta_1d",
        route_family="pde_solver",
        family_ir=family_ir,
        expr=None,
        normalized_expr=None,
    )

    plan = compile_lane_construction_plan(
        preferred_method="pde_solver",
        required_market_data=("discount_curve", "black_vol_surface"),
        dsl_lowering=lowering,
    )

    assert plan is not None
    assert plan.lane_family == "pde_solver"
    # Five kernel-contract invariants from the route-card notes now flow
    # through the lane plan as `Pricing-kernel contract:` construction
    # steps.  Each corresponds to one of the retired notes.
    kernel_steps = [
        step for step in plan.construction_steps
        if step.startswith("Pricing-kernel contract:")
    ]
    assert len(kernel_steps) >= 5
    joined = " ".join(kernel_steps)
    assert "theta_method_1d" in joined
    assert "BlackScholesOperator" in joined
    assert "exercise_fn=max" in joined
    assert "rannacher_timesteps" in joined
    assert "ndarray" in joined


def test_pde_theta_1d_lane_plan_omits_kernel_contracts_on_helper_backed_path():
    """QUA-880 round-1 Codex P1: helper-backed PDE routes must not emit
    the manual-kernel-contract instructions.

    ``price_event_aware_equity_option_pde`` and ``price_callable_bond_pde``
    wrap the Grid / BlackScholesOperator / theta_method_1d assembly
    themselves.  Emitting the kernel contracts alongside those helpers
    would tell the generated adapter to reassemble the kernel manually,
    directly conflicting with the exact-helper contract.  The IR's
    ``helper_symbol`` field is the signal.
    """
    from trellis.agent.family_lowering_ir import (
        EventAwarePDEIR,
        PDEBoundarySpec,
        PDEControlSpec,
        PDEOperatorSpec,
        PDEStateSpec,
    )

    family_ir = EventAwarePDEIR(
        route_id="pde_theta_1d",
        route_family="pde_solver",
        product_instrument="american_option",
        payoff_family="vanilla_option",
        state_spec=PDEStateSpec(state_variable="spot"),
        operator_spec=PDEOperatorSpec(operator_family="black_scholes_1d"),
        boundary_spec=PDEBoundarySpec(terminal_condition_kind="vanilla_payoff"),
        control_spec=PDEControlSpec(control_style="holder_max"),
        helper_symbol="price_event_aware_equity_option_pde",
    )
    lowering = SemanticDslLowering(
        route_id="pde_theta_1d",
        route_family="pde_solver",
        family_ir=family_ir,
        expr=None,
        normalized_expr=None,
    )

    plan = compile_lane_construction_plan(
        preferred_method="pde_solver",
        required_market_data=("discount_curve", "black_vol_surface"),
        dsl_lowering=lowering,
    )

    assert plan is not None
    kernel_steps = [
        step for step in plan.construction_steps
        if step.startswith("Pricing-kernel contract:")
    ]
    assert kernel_steps == [], (
        "helper-backed PDE routes must NOT emit kernel-contract steps; "
        f"saw {kernel_steps}"
    )


def test_fallback_pde_lane_plan_omits_kernel_contracts_when_helper_primitive_present():
    """Parallel helper-backed gate for the fallback (non-semantic) path.

    When the primitive_plan resolves to a helper like
    ``price_event_aware_equity_option_pde``, the fallback construction
    steps must not carry the kernel contracts either.
    """
    from types import SimpleNamespace

    from trellis.agent.lane_obligations import (
        compile_fallback_lane_construction_plan,
    )

    primitive_plan = SimpleNamespace(
        route="pde_theta_1d",
        route_family="pde_solver",
        primitives=(
            SimpleNamespace(
                module="trellis.models.equity_option_pde",
                symbol="price_event_aware_equity_option_pde",
                role="route_helper",
                required=True,
            ),
        ),
        blockers=(),
    )
    plan = compile_fallback_lane_construction_plan(
        preferred_method="pde_solver",
        required_market_data=("discount_curve", "black_vol_surface"),
        primitive_plan=primitive_plan,
        product_ir=None,
        instrument_type="american_option",
    )
    assert plan is not None
    kernel_steps = [
        step for step in plan.construction_steps
        if step.startswith("Pricing-kernel contract:")
    ]
    assert kernel_steps == [], (
        "helper-backed PDE fallback plan must NOT emit kernel contracts; "
        f"saw {kernel_steps}"
    )


def test_rendered_route_card_surfaces_exercise_policy_control_obligations():
    """QUA-880 round-1 Codex P1: rendered route cards must expose the
    approved/implemented early-exercise policy obligations, not just the
    first 3 control fields.
    """
    from trellis.agent.codegen_guardrails import GenerationPlan, render_generation_route_card

    plan = GenerationPlan(
        method="monte_carlo",
        instrument_type="american_option",
        inspected_modules=(),
        approved_modules=(),
        symbols_to_reuse=(),
        proposed_tests=(),
        lane_family="monte_carlo",
        lane_plan_kind="constructive_synthesis",
        lane_control_obligations=(
            "control_style:identity",
            "controller_role:holder",
            "measure_family:risk_neutral",
            "numeraire_binding:money_market_account",
            "semantic_control_style:holder_max",
            "semantic_controller_role:holder",
            "event_kind:decision",
            "approved_exercise_policies:`longstaff_schwartz` [implemented]",
            "implemented_exercise_policies:`longstaff_schwartz`",
            "exercise_policy_name_invariant:longstaff_schwartz_not_lsm_mc",
        ),
    )
    card = render_generation_route_card(plan)
    assert "approved_exercise_policies" in card
    assert "implemented_exercise_policies" in card
    assert "exercise_policy_name_invariant" in card
