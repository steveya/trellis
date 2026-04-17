"""Compiler-emitted lane obligations for construction-first generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trellis.agent.early_exercise_policy import (
    render_early_exercise_policy_summary,
    render_implemented_early_exercise_policy_summary,
)
from trellis.agent.family_lowering_ir import (
    AnalyticalBlack76IR,
    CorrelatedBasketMonteCarloIR,
    CreditDefaultSwapIR,
    EventAwareMonteCarloIR,
    EventAwarePDEIR,
    ExerciseLatticeIR,
    NthToDefaultIR,
    TransformPricingIR,
    VanillaEquityPDEIR,
)
from trellis.agent.knowledge.methods import normalize_method

if TYPE_CHECKING:
    from trellis.agent.codegen_guardrails import PrimitivePlan


@dataclass(frozen=True)
class LaneBinding:
    """One reusable checked primitive or exact backend exposed by the compiler."""

    primitive_ref: str
    role: str
    required: bool = True
    binding_kind: str = "reusable_primitive"


@dataclass(frozen=True)
class LaneConstructionPlan:
    """Constructive lane-level obligations emitted from semantic compilation."""

    lane_family: str
    plan_kind: str
    route_id: str | None = None
    route_family: str | None = None
    expr_kind: str | None = None
    timeline_roles: tuple[str, ...] = ()
    market_requirements: tuple[str, ...] = ()
    state_obligations: tuple[str, ...] = ()
    control_obligations: tuple[str, ...] = ()
    construction_steps: tuple[str, ...] = ()
    reusable_bindings: tuple[LaneBinding, ...] = ()
    exact_target_refs: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()

    @property
    def has_exact_target_binding(self) -> bool:
        """Return ``True`` when the compiler found a safe exact backend target."""
        return bool(self.exact_target_refs)

    @property
    def has_constructive_plan(self) -> bool:
        """Return ``True`` when the compiler can still articulate a lane build plan."""
        return bool(self.construction_steps)


def compile_lane_construction_plan(
    *,
    preferred_method: str,
    required_market_data: tuple[str, ...],
    dsl_lowering=None,
    unsupported_paths: tuple[str, ...] = (),
) -> LaneConstructionPlan | None:
    """Emit a constructive lane plan from lowering and semantic metadata."""
    lane_family = _lane_family_for_method(preferred_method, dsl_lowering=dsl_lowering)
    if not lane_family:
        return None

    family_ir = getattr(dsl_lowering, "family_ir", None)
    route_id = str(getattr(dsl_lowering, "route_id", "") or "") or None
    route_family = str(getattr(dsl_lowering, "route_family", "") or "") or None
    expr_kind = (
        None
        if dsl_lowering is None or getattr(dsl_lowering, "normalized_expr", None) is None
        else type(dsl_lowering.normalized_expr).__name__
    )
    reusable_bindings = tuple(
        LaneBinding(
            primitive_ref=f"{binding.module}.{binding.symbol}",
            role=str(binding.role),
            required=bool(binding.required),
            binding_kind=_binding_kind_for_role(str(binding.role)),
        )
        for binding in getattr(dsl_lowering, "target_bindings", ()) or ()
    )
    exact_target_refs = tuple(
        binding.primitive_ref
        for binding in reusable_bindings
        if binding.role in {"route_helper", "pricing_kernel", "schedule_builder"}
    )
    timeline_roles = _timeline_roles_for(family_ir)
    state_obligations = _state_obligations_for(family_ir)
    control_obligations = _control_obligations_for(family_ir)
    construction_steps = _construction_steps_for(
        lane_family=lane_family,
        family_ir=family_ir,
    )
    unresolved_primitives = _tuple_unique(
        list(getattr(dsl_lowering, "admissibility_errors", ()) or ())
        + list(unsupported_paths or ())
    )
    plan_kind = (
        "exact_target_binding"
        if exact_target_refs
        else "constructive_synthesis"
        if construction_steps
        else "unsupported"
    )

    return LaneConstructionPlan(
        lane_family=lane_family,
        plan_kind=plan_kind,
        route_id=route_id,
        route_family=route_family,
        expr_kind=expr_kind,
        timeline_roles=timeline_roles,
        market_requirements=_tuple_unique(required_market_data),
        state_obligations=state_obligations,
        control_obligations=control_obligations,
        construction_steps=construction_steps,
        reusable_bindings=reusable_bindings,
        exact_target_refs=exact_target_refs,
        unresolved_primitives=unresolved_primitives,
    )


def compile_fallback_lane_construction_plan(
    *,
    preferred_method: str,
    required_market_data: tuple[str, ...] | list[str] | set[str] = (),
    primitive_plan: PrimitivePlan | None = None,
    product_ir=None,
    instrument_type: str | None = None,
) -> LaneConstructionPlan | None:
    """Emit a constructive lane plan when semantic compilation is unavailable.

    This keeps knowledge-light simple-derivative builds on a compiler-shaped
    surface even when we only have a ProductIR + primitive route, not a full
    semantic contract / DSL lowering.
    """
    lane_family = _lane_family_for_method(preferred_method)
    if not lane_family:
        return None

    required_market = _tuple_unique(required_market_data)
    route_id = getattr(primitive_plan, "route", None)
    route_family = getattr(primitive_plan, "route_family", None)
    reusable_bindings = _fallback_reusable_bindings(primitive_plan)
    exact_target_refs = tuple(
        binding.primitive_ref
        for binding in reusable_bindings
        if binding.role in {"route_helper", "pricing_kernel", "schedule_builder"}
    )
    timeline_roles = _fallback_timeline_roles(product_ir)
    state_obligations = _fallback_state_obligations(
        product_ir=product_ir,
        instrument_type=instrument_type,
        required_market_data=required_market,
    )
    control_obligations = _fallback_control_obligations(
        product_ir=product_ir,
        instrument_type=instrument_type,
        lane_family=lane_family,
    )
    construction_steps = _fallback_construction_steps(
        lane_family=lane_family,
        product_ir=product_ir,
        instrument_type=instrument_type,
        required_market_data=required_market,
        primitive_plan=primitive_plan,
    )
    unresolved_primitives = tuple(getattr(primitive_plan, "blockers", ()) or ())
    plan_kind = (
        "exact_target_binding"
        if exact_target_refs
        else "constructive_synthesis"
        if construction_steps
        else "unsupported"
    )
    expr_kind = None
    if product_ir is not None:
        expr_kind = str(getattr(product_ir, "payoff_family", "") or "") or None

    return LaneConstructionPlan(
        lane_family=lane_family,
        plan_kind=plan_kind,
        route_id=str(route_id or "") or None,
        route_family=str(route_family or "") or None,
        expr_kind=expr_kind,
        timeline_roles=timeline_roles,
        market_requirements=required_market,
        state_obligations=state_obligations,
        control_obligations=control_obligations,
        construction_steps=construction_steps,
        reusable_bindings=reusable_bindings,
        exact_target_refs=exact_target_refs,
        unresolved_primitives=unresolved_primitives,
    )


def lane_construction_plan_summary(plan: LaneConstructionPlan | None) -> dict[str, object] | None:
    """Project a lane plan onto YAML-safe primitives."""
    if plan is None:
        return None
    return {
        "lane_family": plan.lane_family,
        "plan_kind": plan.plan_kind,
        "route_id": plan.route_id,
        "route_family": plan.route_family,
        "expr_kind": plan.expr_kind,
        "timeline_roles": list(plan.timeline_roles),
        "market_requirements": list(plan.market_requirements),
        "state_obligations": list(plan.state_obligations),
        "control_obligations": list(plan.control_obligations),
        "construction_steps": list(plan.construction_steps),
        "reusable_bindings": [
            {
                "primitive_ref": binding.primitive_ref,
                "role": binding.role,
                "required": binding.required,
                "binding_kind": binding.binding_kind,
            }
            for binding in plan.reusable_bindings
        ],
        "exact_target_refs": list(plan.exact_target_refs),
        "unresolved_primitives": list(plan.unresolved_primitives),
    }


def _lane_family_for_method(preferred_method: str, *, dsl_lowering=None) -> str:
    """Normalize methods and route families onto construction lanes."""
    method = normalize_method(preferred_method or "")
    if method == "rate_tree":
        return "lattice"
    if method in {"monte_carlo", "qmc"}:
        return "monte_carlo"
    if method:
        return method
    route_family = str(getattr(dsl_lowering, "route_family", "") or "").lower()
    if "lattice" in route_family or "tree" in route_family:
        return "lattice"
    if route_family:
        return route_family
    return ""


def _binding_kind_for_role(role: str) -> str:
    """Classify lowering bindings for prompt rendering."""
    if role == "market_binding":
        return "market_binding"
    if role in {"route_helper", "pricing_kernel", "schedule_builder"}:
        return "exact_backend"
    return "reusable_primitive"


def _timeline_roles_for(family_ir) -> tuple[str, ...]:
    """Return normalized timeline-role obligations from a family IR."""
    return _tuple_unique(
        getattr(item, "value", str(item))
        for item in getattr(family_ir, "timeline_roles", ()) or ()
    )


def _state_obligations_for(family_ir) -> tuple[str, ...]:
    """Return state obligations implied by the family IR."""
    if isinstance(family_ir, EventAwareMonteCarloIR):
        return _tuple_unique(
            (
                family_ir.state_spec.state_variable,
                *(family_ir.state_spec.state_tags or ()),
                family_ir.path_requirement_spec.requirement_kind,
                *(family_ir.path_requirement_spec.reducer_kinds or ()),
                *(family_ir.path_requirement_spec.stored_fields or ()),
                *(f"semantic_transform:{kind}" for kind in getattr(getattr(family_ir, "event_program", None), "transform_kinds", ()) or ()),
            )
        )
    if isinstance(family_ir, EventAwarePDEIR):
        return _tuple_unique(
            (
                family_ir.state_spec.state_variable,
                *(family_ir.state_spec.state_tags or ()),
                family_ir.boundary_spec.terminal_condition_kind,
                *(f"semantic_event_kind:{kind}" for kind in getattr(getattr(family_ir, "event_program", None), "event_kinds", ()) or ()),
            )
        )
    if isinstance(family_ir, TransformPricingIR):
        return _tuple_unique(
            (
                family_ir.state_spec.state_variable,
                *(family_ir.state_spec.state_tags or ()),
                family_ir.characteristic_spec.characteristic_family,
                family_ir.terminal_payoff_kind,
            )
        )
    if isinstance(family_ir, ExerciseLatticeIR):
        return _tuple_unique(
            (
                *family_ir.state_field_names,
                *family_ir.derived_quantities,
            )
        )
    if isinstance(family_ir, CorrelatedBasketMonteCarloIR):
        return _tuple_unique(
            (
                family_ir.path_requirement_kind,
                *family_ir.state_field_names,
                *family_ir.state_tags,
            )
        )
    if isinstance(family_ir, CreditDefaultSwapIR):
        return _tuple_unique(family_ir.state_field_names)
    if isinstance(family_ir, NthToDefaultIR):
        return _tuple_unique(
            (
                *family_ir.state_field_names,
                *family_ir.state_tags,
            )
        )
    return ()


def _control_obligations_for(family_ir) -> tuple[str, ...]:
    """Return control semantics implied by the family IR."""
    if isinstance(family_ir, EventAwareMonteCarloIR):
        semantic_control = getattr(family_ir, "control_program", None)
        obligations = [
            f"control_style:{family_ir.control_spec.control_style}",
            f"controller_role:{family_ir.control_spec.controller_role}",
            f"measure_family:{family_ir.measure_spec.measure_family}",
            f"numeraire_binding:{family_ir.measure_spec.numeraire_binding}",
        ]
        if semantic_control is not None:
            obligations.extend(
                (
                    f"semantic_control_style:{semantic_control.control_style}",
                    f"semantic_controller_role:{semantic_control.controller_role}",
                )
            )
        obligations.extend(
            f"semantic_event_kind:{kind}"
            for kind in getattr(getattr(family_ir, "event_program", None), "event_kinds", ()) or ()
        )
        obligations.extend(f"event_kind:{kind}" for kind in family_ir.event_kinds)
        calibration_binding = getattr(family_ir, "calibration_binding", None)
        if calibration_binding is not None:
            model_family = str(getattr(calibration_binding, "model_family", "") or "").strip()
            quote_family = str(getattr(calibration_binding, "quote_family", "") or "").strip()
            if model_family:
                obligations.append(f"calibration_model:{model_family}")
            if quote_family:
                obligations.append(f"quote_family:{quote_family}")
            if bool(getattr(calibration_binding, "requires_quote_normalization", False)):
                obligations.append("requires_quote_normalization:true")
        # QUA-880: migrate the `exercise_monte_carlo` route card's
        # `dynamic_notes` (approved / implemented early-exercise policy
        # summaries) and static notes (LSM naming invariants) onto the
        # typed obligation surface.  Control-style ``holder_max`` or
        # ``issuer_min`` on an MC family indicates an early-exercise MC
        # route; the typed obligation then carries the same guidance
        # previously held as route-card prose.
        control_style = str(family_ir.control_spec.control_style or "").strip().lower()
        if control_style in {"holder_max", "issuer_min"}:
            obligations.append(
                f"approved_exercise_policies:{render_early_exercise_policy_summary()}"
            )
            obligations.append(
                f"implemented_exercise_policies:"
                f"{render_implemented_early_exercise_policy_summary()}"
            )
            # Preserve the LSM naming invariant as an obligation so
            # generated adapters don't rename `longstaff_schwartz` to
            # `lsm_mc` or surface a `method=\"lsm\"` engine mode.
            obligations.append(
                "exercise_policy_name_invariant:longstaff_schwartz_not_lsm_mc"
            )
        return _tuple_unique(obligations)
    if isinstance(family_ir, EventAwarePDEIR):
        semantic_control = getattr(family_ir, "control_program", None)
        return _tuple_unique(
            (
                f"control_style:{family_ir.control_spec.control_style}",
                f"controller_role:{family_ir.control_spec.controller_role}",
                *(
                    (
                        f"semantic_control_style:{semantic_control.control_style}",
                        f"semantic_controller_role:{semantic_control.controller_role}",
                    )
                    if semantic_control is not None
                    else ()
                ),
                *(f"semantic_event_kind:{kind}" for kind in getattr(getattr(family_ir, "event_program", None), "event_kinds", ()) or ()),
                *(f"event_transform:{kind}" for kind in family_ir.event_transform_kinds),
            )
        )
    if isinstance(family_ir, TransformPricingIR):
        semantic_control = getattr(family_ir, "control_program", None)
        return _tuple_unique(
            (
                f"control_style:{family_ir.control_spec.control_style}",
                f"controller_role:{family_ir.control_spec.controller_role}",
                f"quote_semantics:{family_ir.quote_semantics}",
                f"strike_semantics:{family_ir.strike_semantics}",
                *(
                    (
                        f"semantic_control_style:{semantic_control.control_style}",
                        f"semantic_controller_role:{semantic_control.controller_role}",
                    )
                    if semantic_control is not None
                    else ()
                ),
            )
        )
    if isinstance(family_ir, ExerciseLatticeIR):
        return _tuple_unique(
            (
                f"control_style:{family_ir.control_style}",
                f"controller_role:{family_ir.controller_role}",
                f"exercise_style:{family_ir.exercise_style}",
                f"schedule_role:{family_ir.schedule_role}",
                f"decision_phase:{family_ir.decision_phase}",
                *(f"semantic_event_kind:{kind}" for kind in getattr(getattr(family_ir, "event_program", None), "event_kinds", ()) or ()),
            )
        )
    if isinstance(family_ir, CorrelatedBasketMonteCarloIR):
        return _tuple_unique(
            (
                f"controller_style:{family_ir.controller_style}",
                f"selection_rule:{family_ir.selection_rule}",
                f"lock_rule:{family_ir.lock_rule}",
                f"aggregation_rule:{family_ir.aggregation_rule}",
            )
        )
    if isinstance(family_ir, CreditDefaultSwapIR):
        return _tuple_unique(
            (
                f"pricing_mode:{family_ir.pricing_mode}",
                f"schedule_role:{family_ir.schedule_role}",
            )
        )
    if isinstance(family_ir, NthToDefaultIR):
        return _tuple_unique(
            (
                f"trigger_rank:{family_ir.trigger_rank}",
                f"schedule_role:{family_ir.schedule_role}",
                f"copula:{family_ir.copula_symbol}",
            )
        )
    return ()


def _construction_steps_for(*, lane_family: str, family_ir) -> tuple[str, ...]:
    """Emit constructive build steps for one lane/family pair."""
    if isinstance(family_ir, AnalyticalBlack76IR):
        return (
            f"Bind analytical market inputs via `{family_ir.market_mapping}`.",
            f"Construct the terminal `{family_ir.option_type}` claim with `{family_ir.kernel_symbol}`.",
            "Keep the implementation path-static and discount-consistent; do not introduce simulation or rollback.",
        )
    if isinstance(family_ir, EventAwareMonteCarloIR):
        process_family = family_ir.process_spec.process_family or "generic_state_process"
        path_kind = family_ir.path_requirement_spec.requirement_kind or "terminal_only"
        reducer_kind = family_ir.payoff_reducer_spec.reducer_kind or "path_payoff"
        steps = [
            f"Assemble a one-factor `{process_family}` simulation over `{family_ir.state_spec.state_variable or 'state'}`.",
            f"Bind the reduced-state requirement `{path_kind}` and event kinds `{', '.join(family_ir.event_kinds) or 'none'}` before path generation.",
            f"Aggregate the payoff through `{reducer_kind}` under the `{family_ir.measure_spec.measure_family}` measure before discount-consistent readout.",
        ]
        # QUA-880: early-exercise MC routes (control_style == holder_max /
        # issuer_min) used to carry three adapters on the route card --
        # exercise-date derivation, spot-to-exercise-payoff callback, and
        # continuation-estimator selection.  Emit those as typed
        # construction steps when the control spec requests them.
        control_style = str(family_ir.control_spec.control_style or "").strip().lower()
        if control_style in {"holder_max", "issuer_min"}:
            steps.extend(
                (
                    "Derive the exercise-date grid from the declared schedule role or time grid; do not improvise exercise dates in the adapter.",
                    "Build the spot-to-exercise payoff callback explicitly and bind it to the typed `exercise_control` primitive.",
                    "Select a continuation estimator or regression basis (e.g. Longstaff-Schwartz) from the available `exercise_control` primitives; keep the estimator choice a binding detail, not a bespoke engine mode.",
                )
            )
        return tuple(steps)
    if isinstance(family_ir, VanillaEquityPDEIR):
        return (
            "Assemble a one-dimensional terminal payoff on the expiry timeline.",
            f"Build the PDE operator through `{family_ir.helper_symbol}` with theta={family_ir.theta:.1f}.",
            "Apply backward stepping and boundary conditions on the pricing grid before interpolating back to spot.",
        )
    if isinstance(family_ir, EventAwarePDEIR):
        operator_family = family_ir.operator_spec.operator_family or "generic_1d"
        terminal_kind = family_ir.boundary_spec.terminal_condition_kind or "terminal_condition"
        control_style = family_ir.control_spec.control_style or "identity"
        steps = [
            f"Assemble a one-dimensional `{operator_family}` rollback state over `{family_ir.state_spec.state_variable or 'state'}`.",
            f"Apply `{terminal_kind}` at maturity and schedule the typed event transforms: {', '.join(family_ir.event_transform_kinds) or 'none'}.",
            f"Enforce `{control_style}` control semantics during backward stepping before reading out the price.",
        ]
        # QUA-880 round-1 Codex P1: the kernel-contract invariants apply
        # only to the default-branch (non-helper) PDE path.  Helper-backed
        # routes like ``price_event_aware_equity_option_pde`` or
        # ``price_callable_bond_pde`` wrap the Grid / BlackScholesOperator /
        # theta_method_1d assembly themselves; emitting the kernel
        # contracts alongside a helper ref would tell the adapter to
        # manually reassemble the kernel -- directly conflicting with the
        # exact-helper-only contract.  Gate the kernel contracts on the
        # absence of a ``helper_symbol`` on the IR.
        helper_symbol = str(getattr(family_ir, "helper_symbol", "") or "").strip()
        if not helper_symbol:
            steps.extend(
                (
                    "Pricing-kernel contract: construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
                    "Pricing-kernel contract: use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
                    "Pricing-kernel contract: for American-style exercise, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
                    "Pricing-kernel contract: for barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
                    "Pricing-kernel contract: Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.",
                )
            )
        return tuple(steps)
    if isinstance(family_ir, TransformPricingIR):
        characteristic_family = (
            family_ir.characteristic_spec.characteristic_family or "generic_transform_characteristic"
        )
        backend_capability = family_ir.characteristic_spec.backend_capability or "raw_kernel_only"
        binding_target = family_ir.helper_symbol or "raw FFT/COS kernels"
        return (
            f"Assemble a terminal-only transform contract over `{family_ir.state_spec.state_variable or 'state'}` with `{characteristic_family}`.",
            f"Normalize the payoff to `{family_ir.terminal_payoff_kind}` with `{family_ir.quote_semantics}` inputs before transform evaluation.",
            f"Dispatch through `{binding_target}` according to backend capability `{backend_capability}` instead of widening semantic admissibility.",
        )
    if isinstance(family_ir, ExerciseLatticeIR):
        return (
            f"Map the `{family_ir.schedule_role}` schedule onto lattice decision steps.",
            f"Apply `{family_ir.control_style}` control at `{family_ir.decision_phase}` across continuation and exercise branches.",
            f"Delegate the stable embedded-value rollback to `{family_ir.helper_symbol}` unless the contract truly requires a new kernel.",
        )
    if isinstance(family_ir, CorrelatedBasketMonteCarloIR):
        return (
            f"Resolve basket semantics with `{family_ir.market_binding_symbol}` before path generation.",
            f"Simulate observation-state paths that satisfy `{family_ir.path_requirement_kind}`.",
            f"Apply `{family_ir.selection_rule}` and `{family_ir.lock_rule}` before discounting the `{family_ir.aggregation_rule}` payoff.",
        )
    if isinstance(family_ir, CreditDefaultSwapIR):
        return (
            f"Build the CDS schedule with `{family_ir.schedule_builder_symbol}` and keep the leg semantics explicit: {', '.join(family_ir.leg_semantics)}.",
            f"Bind discount and credit-curve inputs to the `{family_ir.pricing_mode}` lane without routing through equity-option kernels.",
            f"Price the schedule-driven contract through `{family_ir.helper_symbol}` unless the request introduces a genuinely new credit kernel.",
        )
    if isinstance(family_ir, NthToDefaultIR):
        return (
            f"Keep the reference-entity pool explicit and preserve nth-default rank={family_ir.trigger_rank} across the route.",
            f"Bind marginal credit inputs and dependence assumptions through `{family_ir.copula_symbol}` before payoff aggregation.",
            f"Delegate the checked-in basket-credit payoff assembly to `{family_ir.helper_symbol}` unless the request introduces a new kernel.",
        )
    if lane_family == "analytical":
        return (
            "Bind resolved market inputs onto a closed-form or quasi-closed-form kernel.",
            "Keep the implementation path-static and reuse analytical support primitives before writing new math.",
        )
    if lane_family == "lattice":
        return (
            "Build a recombining lattice over the contract timeline.",
            "Keep state, continuation, and control semantics explicit during rollback.",
        )
    if lane_family == "monte_carlo":
        return (
            "Resolve the observation/event timeline before path generation.",
            "Keep state propagation and payoff aggregation explicit over the simulated paths.",
        )
    if lane_family == "pde_solver":
        # QUA-880: include the theta-method kernel-contract invariants on
        # the generic PDE lane fallback as well, so plans that don't
        # classify as ``EventAwarePDEIR`` still surface the guidance the
        # retired route-card notes used to provide.
        return (
            "Discretize the contract onto a PDE grid with explicit terminal and boundary conditions.",
            "Evolve the grid backward with the selected stepping scheme before reading out the PV.",
            "Pricing-kernel contract: construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
            "Pricing-kernel contract: use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
            "Pricing-kernel contract: for American-style exercise, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
            "Pricing-kernel contract: for barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
            "Pricing-kernel contract: Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.",
        )
    if lane_family == "waterfall":
        # QUA-816 round-1 Codex P1: the `waterfall_cashflows` route card used
        # to carry `map_collateral_cashflows_into_structure` as an adapter.
        # Emit the equivalent constructive guidance here so removing the
        # route-card prose does not leave the lane with empty steps.
        return (
            "Map collateral cashflows onto the declared tranche structure via the cashflow_engine primitives.",
            "Keep deal-schedule, locked-cashflow, and remaining-pool state explicit across the waterfall rollup.",
        )
    return ()


def _fallback_reusable_bindings(
    primitive_plan: PrimitivePlan | None,
) -> tuple[LaneBinding, ...]:
    """Project primitive-plan bindings onto lane bindings."""
    if primitive_plan is None:
        return ()
    return tuple(
        LaneBinding(
            primitive_ref=f"{primitive.module}.{primitive.symbol}",
            role=str(primitive.role),
            required=bool(primitive.required),
            binding_kind=_binding_kind_for_role(str(primitive.role)),
        )
        for primitive in (primitive_plan.primitives or ())
    )


def _fallback_timeline_roles(product_ir) -> tuple[str, ...]:
    """Infer lightweight timeline roles from the ProductIR fallback surface."""
    if product_ir is None:
        return ()
    roles: list[str] = []
    if bool(getattr(product_ir, "schedule_dependence", False)):
        roles.append("payment")
    exercise_style = str(getattr(product_ir, "exercise_style", "") or "").strip().lower()
    if exercise_style in {"american", "bermudan", "issuer_call", "holder_put"}:
        roles.append("exercise")
    return _tuple_unique(roles)


def _fallback_state_obligations(
    *,
    product_ir,
    instrument_type: str | None,
    required_market_data: tuple[str, ...],
) -> tuple[str, ...]:
    """Infer coarse state obligations when no typed family IR exists yet."""
    obligations: list[str] = []
    instrument = str(getattr(product_ir, "instrument", "") or instrument_type or "").strip().lower()
    if instrument in {"european_option", "american_option"}:
        obligations.extend(("spot", "strike", "expiry"))
    if "fx_rates" in required_market_data:
        obligations.append("fx_rate_scalar_spot")
    if "forward_curve" in required_market_data:
        obligations.append("foreign_discount_curve")
    if "black_vol_surface" in required_market_data:
        obligations.append("expiry_black_vol")
    return _tuple_unique(obligations)


def _fallback_control_obligations(
    *,
    product_ir,
    instrument_type: str | None,
    lane_family: str,
) -> tuple[str, ...]:
    """Infer coarse control obligations when no typed family IR exists yet."""
    obligations: list[str] = []
    instrument = str(getattr(product_ir, "instrument", "") or instrument_type or "").strip().lower()
    exercise_style = str(getattr(product_ir, "exercise_style", "") or "").strip().lower()
    if instrument in {"american_option", "american_put"} and exercise_style in {"american", "bermudan"}:
        obligations.append(f"exercise_style:{exercise_style}")
        if lane_family == "lattice":
            obligations.append("control_style:holder_max")
    return _tuple_unique(obligations)


def _fallback_construction_steps(
    *,
    lane_family: str,
    product_ir,
    instrument_type: str | None,
    required_market_data: tuple[str, ...],
    primitive_plan: PrimitivePlan | None,
) -> tuple[str, ...]:
    """Emit constructive fallback steps for non-semantic builds."""
    helper_symbols = {
        str(primitive.symbol)
        for primitive in (getattr(primitive_plan, "primitives", ()) or ())
        if bool(getattr(primitive, "required", True))
    }
    instrument = str(getattr(product_ir, "instrument", "") or instrument_type or "").strip().lower()

    if "price_vanilla_equity_option_tree" in helper_symbols:
        return (
            "Resolve a spec-like contract with `spot`, `strike`, `expiry_date`, and the optional fields `option_type`, `exercise_style`, and `day_count`.",
            "Call `price_vanilla_equity_option_tree(market_state, spec_like, model=\"crr\"|\"jarrow_rudd\", n_steps=...)`; do not invent `underlying=`, `exercise=`, or other bespoke helper keywords.",
            "Let the checked equity-tree helper own rate, discounting, and vol resolution; keep the adapter thin and market-state explicit.",
        )

    if lane_family == "monte_carlo" and "fx_rates" in required_market_data:
        return (
            "Resolve scalar FX spot from `market_state.fx_rates[spec.fx_pair].spot` or the bridged scalar `market_state.spot`; do not use the `FXRate` wrapper directly in arithmetic.",
            "Bind domestic and foreign discount factors from `market_state.discount` and `market_state.forecast_curves[...]` before deriving drift or discounting.",
            "Simulate the FX spot under a GBM-style terminal law with drift `r_d - r_f`, then discount the terminal payoff with the domestic curve.",
        )

    if lane_family == "lattice" and instrument in {"american_option", "american_put"}:
        return (
            "Resolve spot, strike, expiry, and exercise style explicitly before building the early-exercise contract.",
            "Keep discounting and volatility on the market-state side; do not invent alternate curve or vol accessors.",
            "Prefer the smallest exact helper-backed adapter when a stable lattice backend exists; otherwise keep the rollback and early-exercise contract explicit.",
        )

    if lane_family == "monte_carlo":
        return (
            "Resolve the observation/event timeline before path generation.",
            "Propagate the state under the selected process and aggregate the payoff explicitly on terminal or scheduled events.",
        )
    if lane_family == "lattice":
        return (
            "Resolve the pricing state and early-exercise contract before building the lattice.",
            "Keep continuation, discounting, and exercise semantics explicit during backward induction.",
        )
    if lane_family == "analytical":
        return (
            "Resolve scalar market inputs and contract terms before applying the closed-form kernel.",
            "Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.",
        )
    if lane_family == "waterfall":
        # QUA-816 round-1 Codex P1: keep waterfall guidance non-empty even
        # on the fallback (non-semantic) path.
        return (
            "Resolve the deal-schedule, locked-cashflow, and remaining-pool state explicitly before walking the waterfall.",
            "Route collateral cashflows through the declared tranche structure via the cashflow_engine primitives.",
        )
    if lane_family == "pde_solver":
        # QUA-880: the `pde_theta_1d` route card's five kernel-contract
        # notes now flow through the lane-obligation surface.  Even on
        # fallback (non-semantic) PDE plans, surface those invariants so
        # the generated adapter prompt retains the guidance that used to
        # live in `routes.yaml` (`rannacher_timesteps` smoothing, the
        # ndarray-vs-callable terminal-payoff contract, etc.).
        steps = [
            "Resolve the pricing state, terminal-condition payoff array, and boundary conditions before calling the theta-method solver.",
            "Keep discounting and volatility semantics on the market-state side; the theta-method PDE expects callable `sigma_fn` / `r_fn` inputs, not scalars.",
        ]
        # QUA-880 round-1 Codex P1: omit the kernel-contract invariants
        # when the primitive plan resolves to a helper-backed PDE route
        # (``price_event_aware_equity_option_pde``,
        # ``price_callable_bond_pde``, ``price_vanilla_equity_option_pde``).
        # Those helpers wrap the Grid / BlackScholesOperator assembly
        # themselves, so adapters should stay thin and call the helper
        # rather than reassemble the kernel manually.
        helper_symbols = {
            str(primitive.symbol)
            for primitive in (getattr(primitive_plan, "primitives", ()) or ())
            if bool(getattr(primitive, "required", True))
            and str(getattr(primitive, "role", "") or "").strip() == "route_helper"
        }
        pde_helper_wrappers = {
            "price_event_aware_equity_option_pde",
            "price_callable_bond_pde",
            "price_vanilla_equity_option_pde",
        }
        if helper_symbols & pde_helper_wrappers:
            return tuple(steps)
        steps.extend(
            (
                "Pricing-kernel contract: construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
                "Pricing-kernel contract: use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
                "Pricing-kernel contract: for American-style exercise, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
                "Pricing-kernel contract: for barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
                "Pricing-kernel contract: Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.",
            )
        )
        return tuple(steps)
    return ()


def _tuple_unique(values) -> tuple[str, ...]:
    """Return a stable tuple of unique non-empty strings."""
    normalized: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)
