"""Lower normalized DSL fragments onto checked-in helper-backed route targets.

This module is the first executable bridge from the semiring/Bellman DSL
algebra to the checked-in semantic compiler outputs. It is intentionally
conservative:

- it only lowers routes that already have stable route IDs and helper targets
- it returns structured admissibility errors instead of guessing
- it treats helper bindings as targets of the DSL, not as the DSL itself
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from trellis.agent.dsl_algebra import (
    ChoiceExpr,
    ContractAtom,
    ContractExpr,
    ContractSignature,
    ControlStyle,
    ThenExpr,
    collect_control_styles,
    normalize_contract_expr,
    validate_contract_expr,
)
from trellis.agent.family_lowering_ir import (
    AnalyticalBlack76IR,
    CorrelatedBasketMonteCarloIR,
    CreditDefaultSwapIR,
    ExerciseLatticeIR,
    NthToDefaultIR,
    VanillaEquityPDEIR,
    build_family_lowering_ir,
)
from trellis.agent.route_registry import (
    find_route_by_id,
    resolve_route_adapters,
    resolve_route_family,
    resolve_route_notes,
    resolve_route_primitives,
)
from trellis.core.types import TimelineRole


@dataclass(frozen=True)
class DslTargetBinding:
    """One checked-in implementation target selected by DSL lowering."""

    module: str
    symbol: str
    role: str
    required: bool = True

    @property
    def primitive_ref(self) -> str:
        """Return the stable module.symbol reference."""
        return f"{self.module}.{self.symbol}"


@dataclass(frozen=True)
class DslLoweringError:
    """Structured lowering rejection recorded without raising."""

    route_id: str | None
    stage: str
    code: str
    message: str


@dataclass(frozen=True)
class SemanticDslLowering:
    """Deterministic lowering result for one compiled semantic contract."""

    route_id: str | None
    route_family: str | None
    family_ir: object | None
    expr: ContractExpr | None
    normalized_expr: ContractExpr | None
    target_bindings: tuple[DslTargetBinding, ...] = ()
    adapters: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    errors: tuple[DslLoweringError, ...] = ()

    @property
    def control_styles(self) -> tuple[ControlStyle, ...]:
        """Return the explicit control styles present after lowering."""
        if self.normalized_expr is None:
            return ()
        return collect_control_styles(self.normalized_expr)

    @property
    def helper_modules(self) -> tuple[str, ...]:
        """Return the distinct modules referenced by the lowering targets."""
        modules: list[str] = []
        for binding in self.target_bindings:
            if binding.module not in modules:
                modules.append(binding.module)
        return tuple(modules)

    @property
    def helper_refs(self) -> tuple[str, ...]:
        """Return the distinct module.symbol targets referenced by the lowering."""
        refs: list[str] = []
        for binding in self.target_bindings:
            if binding.primitive_ref not in refs:
                refs.append(binding.primitive_ref)
        return tuple(refs)

    @property
    def admissibility_errors(self) -> tuple[str, ...]:
        """Backward-compatible lowering error messages."""
        return tuple(item.message for item in self.errors)


def lower_semantic_blueprint(
    contract,
    *,
    product_ir,
    pricing_plan,
    primitive_routes: tuple[str, ...],
    valuation_context=None,
    market_binding_spec=None,
) -> SemanticDslLowering:
    """Lower one compiled semantic contract onto a DSL expression and targets.

    The result always returns a structured object. Unsupported or inadmissible
    routes populate ``admissibility_errors`` instead of guessing a lowering.
    """
    if not primitive_routes:
        return SemanticDslLowering(
            route_id=None,
            route_family=None,
            family_ir=None,
            expr=None,
            normalized_expr=None,
            errors=(
                _lowering_error(
                    route_id=None,
                    stage="route_selection",
                    code="missing_primitive_routes",
                    message="No primitive routes declared for DSL lowering.",
                ),
            ),
        )

    errors: list[DslLoweringError] = []
    for route_id in primitive_routes:
        route = find_route_by_id(route_id)
        if route is None:
            errors.append(
                _lowering_error(
                    route_id=route_id,
                    stage="route_selection",
                    code="unknown_primitive_route",
                    message=f"Unknown primitive route for DSL lowering: '{route_id}'",
                )
            )
            continue

        bindings = tuple(
            DslTargetBinding(
                module=primitive.module,
                symbol=primitive.symbol,
                role=primitive.role,
                required=primitive.required,
            )
            for primitive in resolve_route_primitives(route, product_ir)
        )

        route_family = resolve_route_family(route, product_ir)
        adapters = resolve_route_adapters(route, product_ir)
        notes = resolve_route_notes(route, product_ir)
        try:
            family_ir = build_family_lowering_ir(
                contract,
                route_id=route_id,
                route_family=route_family,
                product_ir=product_ir,
                valuation_context=valuation_context,
                market_binding_spec=market_binding_spec,
            )
        except ValueError as exc:
            errors.append(
                _lowering_error(
                    route_id=route_id,
                    stage="family_ir",
                    code=_infer_error_code(str(exc)),
                    message=f"Route '{route_id}' family lowering rejected the semantic contract: {exc}",
                )
            )
            continue
        if family_ir is not None:
            expr, lowering_errors = _build_expr_for_family_ir(
                route_id=route_id,
                family_ir=family_ir,
                bindings=bindings,
            )
        else:
            expr, lowering_errors = _build_expr_for_route(
                contract,
                product_ir=product_ir,
                pricing_plan=pricing_plan,
                route_id=route_id,
                route_family=route_family,
                bindings=bindings,
            )
        if lowering_errors:
            errors.extend(
                _lowering_error(
                    route_id=route_id,
                    stage="dsl_expr",
                    code=_infer_error_code(error),
                    message=error,
                )
                for error in lowering_errors
            )
            continue
        assert expr is not None

        type_errors = validate_contract_expr(expr)
        if type_errors:
            errors.extend(
                _lowering_error(
                    route_id=route_id,
                    stage="dsl_typecheck",
                    code="invalid_dsl_fragment",
                    message=f"Route '{route_id}' lowered to an invalid DSL fragment: {error}",
                )
                for error in type_errors
            )
            continue

        return SemanticDslLowering(
            route_id=route_id,
            route_family=route_family,
            family_ir=family_ir,
            expr=expr,
            normalized_expr=normalize_contract_expr(expr),
            target_bindings=bindings,
            adapters=adapters,
            notes=notes,
            errors=(),
        )

    return SemanticDslLowering(
        route_id=primitive_routes[0],
        route_family=None,
        family_ir=None,
        expr=None,
        normalized_expr=None,
        errors=tuple(errors),
    )


def _build_expr_for_family_ir(
    *,
    route_id: str,
    family_ir,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a lowering expression from a typed family IR."""
    if isinstance(family_ir, AnalyticalBlack76IR):
        return _build_black76_expr_from_family_ir(
            route_id=route_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, VanillaEquityPDEIR):
        return _build_vanilla_equity_pde_expr_from_family_ir(
            route_id=route_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, ExerciseLatticeIR):
        return _build_exercise_lattice_expr_from_family_ir(
            route_id=route_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, CorrelatedBasketMonteCarloIR):
        return _build_correlated_basket_mc_expr_from_family_ir(
            route_id=route_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, CreditDefaultSwapIR):
        return _build_credit_default_swap_expr_from_family_ir(
            route_id=route_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, NthToDefaultIR):
        return _build_nth_to_default_expr_from_family_ir(
            route_id=route_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    return None, (
        f"Route '{route_id}' produced an unsupported family lowering IR '{type(family_ir).__name__}'.",
    )


def _build_expr_for_route(
    contract,
    *,
    product_ir,
    pricing_plan,
    route_id: str,
    route_family: str,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build the semantic DSL fragment for one resolved route."""
    market_signature = _market_signature(contract)
    if route_id == "analytical_black76":
        return _build_black76_expr(
            contract,
            route_id=route_id,
            market_signature=market_signature,
            bindings=bindings,
        )
    route_helper = next(
        (binding for binding in bindings if binding.role == "route_helper"),
        None,
    )
    if route_helper is None:
        return None, (
            f"Route '{route_id}' has no helper-backed lowering target.",
        )

    control_style = _control_style_for_product(product_ir)
    if control_style is not None:
        return _build_control_expr(
            product_ir=product_ir,
            pricing_plan=pricing_plan,
            route_id=route_id,
            route_family=route_family,
            market_signature=market_signature,
            bindings=bindings,
        )

    market_binding = next(
        (binding for binding in bindings if binding.role == "market_binding"),
        None,
    )
    if market_binding is not None:
        binding_atom = ContractAtom(
            atom_id=f"{route_id}:market_binding",
            primitive_ref=market_binding.primitive_ref,
            description="Resolve market inputs into a route-local state bundle.",
            signature=ContractSignature(
                inputs=market_signature.inputs,
                outputs=("resolved_state:state",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        helper_atom = ContractAtom(
            atom_id=f"{route_id}:route_helper",
            primitive_ref=route_helper.primitive_ref,
            description="Delegate the priced route to the checked-in helper.",
            signature=ContractSignature(
                inputs=("resolved_state:state",),
                outputs=("price:scalar",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        return ThenExpr(terms=(binding_atom, helper_atom)), ()

    helper_atom = ContractAtom(
        atom_id=f"{route_id}:route_helper",
        primitive_ref=route_helper.primitive_ref,
        description="Delegate the priced route to the checked-in helper.",
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return helper_atom, ()


def _build_black76_expr(
    contract,
    *,
    route_id: str,
    market_signature: ContractSignature,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a direct kernel lowering for Black76-style analytical routes."""
    payoff_family = getattr(contract.product, "payoff_family", "")
    if payoff_family not in {"vanilla_option", "swaption"}:
        return None, (
            "Route 'analytical_black76' only has an explicit DSL lowering for "
            "plain vanilla-option and rate-style swaption semantics in this slice.",
        )

    route_helper = next(
        (binding for binding in bindings if binding.role == "route_helper"),
        None,
    )
    if (
        payoff_family == "swaption"
        and getattr(contract.product, "exercise_style", "") == "bermudan"
        and route_helper is not None
    ):
        return (
            ContractAtom(
                atom_id=f"{route_id}:route_helper",
                primitive_ref=route_helper.primitive_ref,
                description=(
                    "Checked-in analytical lower-bound helper for a Bermudan "
                    "rate-style swaption."
                ),
                signature=ContractSignature(
                    inputs=market_signature.inputs,
                    outputs=("price:scalar",),
                    timeline_roles=market_signature.timeline_roles,
                    market_data_requirements=market_signature.market_data_requirements,
                ),
            ),
            (),
        )

    if (
        payoff_family == "swaption"
        and getattr(contract.product, "exercise_style", "") == "european"
        and route_helper is not None
    ):
        market_binding = next(
            (binding for binding in bindings if binding.role == "market_binding"),
            None,
        )
        if market_binding is None:
            return None, (
                f"Route '{route_id}' is missing the required market binding for European rate-style swaption lowering.",
            )

        binding_atom = ContractAtom(
            atom_id=f"{route_id}:market_binding",
            primitive_ref=market_binding.primitive_ref,
            description="Resolve market inputs for a European rate-style swaption.",
            signature=ContractSignature(
                inputs=market_signature.inputs,
                outputs=("resolved_state:state",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        helper_atom = ContractAtom(
            atom_id=f"{route_id}:route_helper",
            primitive_ref=route_helper.primitive_ref,
            description="Delegate European rate-style swaption pricing to the checked-in Black76 raw helper.",
            signature=ContractSignature(
                inputs=("resolved_state:state",),
                outputs=("price:scalar",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        return ThenExpr(terms=(binding_atom, helper_atom)), ()

    option_type = _option_type_for_contract(contract)
    kernel_name = "black76_put" if option_type == "put" else "black76_call"
    kernel = next(
        (
            binding
            for binding in bindings
            if binding.role == "pricing_kernel" and binding.symbol == kernel_name
        ),
        None,
    )
    if kernel is None:
        return None, (
            f"Route '{route_id}' is missing the required pricing kernel '{kernel_name}'.",
        )

    kernel_atom = ContractAtom(
        atom_id=f"{route_id}:{kernel_name}",
        primitive_ref=kernel.primitive_ref,
        description=(
            f"Direct Black76 {option_type} kernel for "
            f"{'rate-style swaption' if payoff_family == 'swaption' else 'plain vanilla payoff'}."
        ),
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return kernel_atom, ()


def _build_black76_expr_from_family_ir(
    *,
    route_id: str,
    family_ir: AnalyticalBlack76IR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a direct vanilla analytical-kernel lowering from typed family IR."""
    kernel = next(
        (
            binding
            for binding in bindings
            if binding.role == "pricing_kernel" and binding.symbol == family_ir.kernel_symbol
        ),
        None,
    )
    if kernel is None:
        return None, (
            f"Route '{route_id}' is missing the required pricing kernel '{family_ir.kernel_symbol}'.",
        )

    signature = _market_signature_from_family_ir(family_ir)
    kernel_atom = ContractAtom(
        atom_id=f"{route_id}:{family_ir.kernel_symbol}",
        primitive_ref=kernel.primitive_ref,
        description=(
            f"Typed Black76 {family_ir.option_type} kernel for plain vanilla payoff "
            f"with {family_ir.market_mapping} market binding."
        ),
        signature=signature,
    )
    return kernel_atom, ()


def _build_vanilla_equity_pde_expr_from_family_ir(
    *,
    route_id: str,
    family_ir: VanillaEquityPDEIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a vanilla PDE helper lowering from typed family IR."""
    route_helper = next(
        (
            binding
            for binding in bindings
            if binding.role == "route_helper" and binding.symbol == family_ir.helper_symbol
        ),
        None,
    )
    if route_helper is None:
        return None, (
            f"Route '{route_id}' is missing the required route helper '{family_ir.helper_symbol}'.",
        )

    helper_atom = ContractAtom(
        atom_id=f"{route_id}:route_helper",
        primitive_ref=route_helper.primitive_ref,
        description=(
            f"Typed theta-method PDE helper for vanilla {family_ir.option_type} payoff "
            f"with theta={family_ir.theta:g}."
        ),
        signature=_market_signature_from_family_ir(family_ir),
    )
    return helper_atom, ()


def _build_exercise_lattice_expr_from_family_ir(
    *,
    route_id: str,
    family_ir: ExerciseLatticeIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build an exercise-lattice lowering from typed family IR."""
    route_helper = next(
        (
            binding
            for binding in bindings
            if binding.role == "route_helper" and binding.symbol == family_ir.helper_symbol
        ),
        None,
    )
    if route_helper is None:
        return None, (
            f"Route '{route_id}' is missing the required route helper '{family_ir.helper_symbol}'.",
        )

    control_binding = next(
        (
            binding
            for binding in bindings
            if binding.role == "control_policy" and binding.symbol == family_ir.control_symbol
        ),
        None,
    )
    market_signature = _market_signature_from_family_ir(family_ir)
    control_style = _control_style_from_family_ir(family_ir)
    if control_style is None:
        helper_atom = ContractAtom(
            atom_id=f"{route_id}:route_helper",
            primitive_ref=route_helper.primitive_ref,
            description=(
                f"Typed exercise-lattice helper for {family_ir.product_instrument} "
                "without strategic control."
            ),
            signature=market_signature,
        )
        return helper_atom, ()

    branch_signature = ContractSignature(
        inputs=market_signature.inputs,
        outputs=("value:scalar",),
        timeline_roles=market_signature.timeline_roles | {TimelineRole.EXERCISE},
        market_data_requirements=market_signature.market_data_requirements,
    )
    continuation = ContractAtom(
        atom_id=f"{family_ir.route_family}:continuation",
        signature=branch_signature,
        description=(
            f"Continuation branch for typed {family_ir.exercise_style} exercise-lattice route."
        ),
    )
    branch_label = "exercise_now" if control_style is ControlStyle.ISSUER_MIN else "exercise_or_keep"
    exercise = ContractAtom(
        atom_id=f"{family_ir.route_family}:{branch_label}",
        signature=branch_signature,
        primitive_ref=control_binding.primitive_ref if control_binding is not None else None,
        description=(
            f"Immediate exercise/call branch for typed {family_ir.exercise_style} "
            "exercise-lattice route."
        ),
    )
    return ChoiceExpr(
        style=control_style,
        branches=(continuation, exercise),
        label=family_ir.exercise_style or route_id,
    ), ()


def _build_correlated_basket_mc_expr_from_family_ir(
    *,
    route_id: str,
    family_ir: CorrelatedBasketMonteCarloIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a ranked-observation basket MC lowering from typed family IR."""
    market_binding = next(
        (
            binding
            for binding in bindings
            if binding.role == "market_binding" and binding.symbol == family_ir.market_binding_symbol
        ),
        None,
    )
    if market_binding is None:
        return None, (
            f"Route '{route_id}' is missing the required market binding '{family_ir.market_binding_symbol}'.",
        )

    route_helper = next(
        (
            binding
            for binding in bindings
            if binding.role == "route_helper" and binding.symbol == family_ir.helper_symbol
        ),
        None,
    )
    if route_helper is None:
        return None, (
            f"Route '{route_id}' is missing the required route helper '{family_ir.helper_symbol}'.",
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    binding_atom = ContractAtom(
        atom_id=f"{route_id}:market_binding",
        primitive_ref=market_binding.primitive_ref,
        description=(
            "Typed ranked-observation basket binding that resolves constituent market data, "
            "correlation, and schedule semantics into a reusable resolved basket state."
        ),
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("resolved_state:state",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    helper_atom = ContractAtom(
        atom_id=f"{route_id}:route_helper",
        primitive_ref=route_helper.primitive_ref,
        description=(
            f"Typed ranked-observation basket Monte Carlo helper with "
            f"{family_ir.path_requirement_kind} path-state requirement."
        ),
        signature=ContractSignature(
            inputs=("resolved_state:state",),
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return ThenExpr(terms=(binding_atom, helper_atom)), ()


def _build_credit_default_swap_expr_from_family_ir(
    *,
    route_id: str,
    family_ir: CreditDefaultSwapIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a typed CDS schedule-builder and helper lowering."""
    schedule_builder = next(
        (
            binding
            for binding in bindings
            if binding.role == "schedule_builder" and binding.symbol == family_ir.schedule_builder_symbol
        ),
        None,
    )
    if schedule_builder is None:
        return None, (
            f"Route '{route_id}' is missing the required schedule builder '{family_ir.schedule_builder_symbol}'.",
        )

    route_helper = next(
        (
            binding
            for binding in bindings
            if binding.role == "route_helper" and binding.symbol == family_ir.helper_symbol
        ),
        None,
    )
    if route_helper is None:
        return None, (
            f"Route '{route_id}' is missing the required route helper '{family_ir.helper_symbol}'.",
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    schedule_atom = ContractAtom(
        atom_id=f"{route_id}:schedule_builder",
        primitive_ref=schedule_builder.primitive_ref,
        description="Build the canonical CDS premium schedule shared by the checked-in route helpers.",
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("payment_schedule:schedule", *market_signature.inputs),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    helper_atom = ContractAtom(
        atom_id=f"{route_id}:route_helper",
        primitive_ref=route_helper.primitive_ref,
        description=(
            f"Typed single-name CDS {family_ir.pricing_mode} helper using the checked-in "
            "premium/protection-leg pricing contract."
        ),
        signature=ContractSignature(
            inputs=("payment_schedule:schedule", *market_signature.inputs),
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return ThenExpr(terms=(schedule_atom, helper_atom)), ()


def _build_nth_to_default_expr_from_family_ir(
    *,
    route_id: str,
    family_ir: NthToDefaultIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a typed nth-to-default helper lowering."""
    route_helper = next(
        (
            binding
            for binding in bindings
            if binding.role == "route_helper" and binding.symbol == family_ir.helper_symbol
        ),
        None,
    )
    if route_helper is None:
        return None, (
            f"Route '{route_id}' is missing the required route helper '{family_ir.helper_symbol}'.",
        )

    copula_binding = next(
        (
            binding
            for binding in bindings
            if binding.role == "default_time_sampler" and binding.symbol == family_ir.copula_symbol
        ),
        None,
    )
    if copula_binding is None:
        return None, (
            f"Route '{route_id}' is missing the required copula primitive '{family_ir.copula_symbol}'.",
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    helper_atom = ContractAtom(
        atom_id=f"{route_id}:route_helper",
        primitive_ref=route_helper.primitive_ref,
        description=(
            f"Typed nth-to-default helper backed by the checked-in "
            f"{family_ir.copula_symbol} dependence route."
        ),
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return helper_atom, ()


def _build_control_expr(
    *,
    product_ir,
    pricing_plan,
    route_id: str,
    route_family: str,
    market_signature: ContractSignature,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build an explicit Bellman/choice expression for a control route."""
    if pricing_plan.method not in {"rate_tree", "monte_carlo"}:
        return None, (
            f"Route '{route_id}' is a control route but method '{pricing_plan.method}' "
            "is not Bellman-compatible.",
        )

    control_binding = next(
        (binding for binding in bindings if binding.role == "control_policy"),
        None,
    )
    route_helper = next(
        (binding for binding in bindings if binding.role == "route_helper"),
        None,
    )
    if route_helper is None:
        return None, (f"Route '{route_id}' has no helper target for control lowering.",)

    # The semantic DSL keeps the Bellman choice explicit and treats helper/control
    # bindings as the checked-in implementation targets of that choice.
    control_style = _control_style_for_product(product_ir)
    assert control_style is not None
    branch_signature = ContractSignature(
        inputs=market_signature.inputs,
        outputs=("value:scalar",),
        timeline_roles=market_signature.timeline_roles | {TimelineRole.EXERCISE},
        market_data_requirements=market_signature.market_data_requirements,
    )
    continuation = ContractAtom(
        atom_id=f"{route_family}:continuation",
        signature=branch_signature,
        description="Continuation branch for the control route.",
    )
    if control_style is ControlStyle.ISSUER_MIN:
        branch_label = "exercise_now"
    else:
        branch_label = "exercise_or_keep"
    exercise = ContractAtom(
        atom_id=f"{route_family}:{branch_label}",
        signature=branch_signature,
        primitive_ref=control_binding.primitive_ref if control_binding is not None else None,
        description="Immediate exercise/call branch for the control route.",
    )
    expr = ChoiceExpr(
        style=control_style,
        branches=(continuation, exercise),
        label=getattr(product_ir, "exercise_style", "") or route_id,
    )
    return expr, ()


def _market_signature(contract) -> ContractSignature:
    """Build a coarse typed interface for the route-level contract inputs."""
    input_ports = tuple(
        f"market:{item.input_id}" for item in contract.market_data.required_inputs
    )
    return ContractSignature(
        inputs=input_ports,
        outputs=("price:scalar",),
        timeline_roles=_timeline_roles_for_contract(contract),
        market_data_requirements=frozenset(item.input_id for item in contract.market_data.required_inputs),
    )


def _market_signature_from_family_ir(family_ir) -> ContractSignature:
    """Build a coarse typed interface directly from family lowering IR."""
    return ContractSignature(
        inputs=tuple(f"market:{input_id}" for input_id in family_ir.required_input_ids),
        outputs=("price:scalar",),
        timeline_roles=family_ir.timeline_roles,
        market_data_requirements=family_ir.market_data_requirements,
    )


def _control_style_from_family_ir(family_ir: ExerciseLatticeIR) -> ControlStyle | None:
    """Map typed family-IR controller style onto the DSL control enum."""
    style = str(family_ir.control_style or "identity").strip().lower()
    if style == "holder_max":
        return ControlStyle.HOLDER_MAX
    if style == "issuer_min":
        return ControlStyle.ISSUER_MIN
    if style == "identity":
        return None
    raise ValueError(f"Unsupported exercise-lattice family control style '{style}'.")


def _timeline_roles_for_contract(contract) -> frozenset[TimelineRole]:
    """Infer coarse timeline roles from semantic contract metadata."""
    roles: set[TimelineRole] = set()
    product = contract.product
    if (
        product.schedule_dependence
        or product.observation_schedule
        or tuple(getattr(getattr(product, "timeline", None), "observation_dates", ()) or ())
        or tuple(getattr(product, "observables", ()) or ())
    ):
        roles.add(TimelineRole.OBSERVATION)
    if (
        product.exercise_style
        and product.exercise_style != "none"
    ) or tuple(getattr(getattr(product, "timeline", None), "decision_dates", ()) or ()):
        roles.add(TimelineRole.EXERCISE)
    if _typed_settlement_rules(product) or tuple(getattr(getattr(product, "timeline", None), "settlement_dates", ()) or ()):
        roles.add(TimelineRole.SETTLEMENT)
    if _has_typed_payment_semantics(product):
        roles.add(TimelineRole.PAYMENT)
    return frozenset(roles)


def _control_style_for_product(product_ir) -> ControlStyle | None:
    """Map ProductIR exercise style onto the explicit Bellman control style."""
    exercise_style = getattr(product_ir, "exercise_style", "none")
    if exercise_style in {"american", "bermudan", "holder_put"}:
        return ControlStyle.HOLDER_MAX
    if exercise_style == "issuer_call":
        return ControlStyle.ISSUER_MIN
    return None


def _option_type_for_contract(contract) -> str:
    """Resolve a coarse call/put label from the checked-in semantic contract."""
    product = contract.product
    if getattr(product, "payoff_family", "") == "swaption":
        description = str(getattr(contract, "description", "")).lower()
        if "receiver" in description:
            return "put"
        return "call"
    if hasattr(product, "option_type"):
        option_type = str(getattr(product, "option_type")).strip().lower()
        if option_type in {"call", "put"}:
            return option_type
    description = str(getattr(contract, "description", "")).lower()
    if re.search(r"\bput\b", description):
        return "put"
    return "call"


def _typed_settlement_rules(product) -> tuple[str, ...]:
    """Return typed settlement rules emitted by obligations, deduplicated in order."""
    rules: list[str] = []
    for obligation in getattr(product, "obligations", ()) or ():
        rule = str(getattr(obligation, "settle_date_rule", "")).strip()
        if rule and rule not in rules:
            rules.append(rule)
    return tuple(rules)


def _has_typed_payment_semantics(product) -> bool:
    """Return whether typed observables or obligations imply payment timing."""
    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in getattr(product, "observables", ()) or ()
    }
    if "cashflow_schedule" in observable_types:
        return True
    payoff_traits = {
        str(item).strip().lower()
        for item in getattr(product, "payoff_traits", ()) or ()
    }
    if "fixed_coupons" in payoff_traits or "floating_coupons" in payoff_traits:
        return True
    return any(
        "coupon" in str(getattr(obligation, "amount_expression", "")).strip().lower()
        for obligation in getattr(product, "obligations", ()) or ()
    )


def _lowering_error(
    *,
    route_id: str | None,
    stage: str,
    code: str,
    message: str,
) -> DslLoweringError:
    """Build one structured lowering error record."""
    return DslLoweringError(
        route_id=route_id,
        stage=stage,
        code=code,
        message=message,
    )


def _infer_error_code(message: str) -> str:
    """Map a lowering rejection message onto a stable machine-readable code."""
    lower = message.lower()
    if "unknown primitive route" in lower:
        return "unknown_primitive_route"
    if "no primitive routes declared" in lower:
        return "missing_primitive_routes"
    if "invalid dsl fragment" in lower:
        return "invalid_dsl_fragment"
    if "unsupported family lowering ir" in lower:
        return "unsupported_family_ir"
    if "missing required market inputs" in lower:
        return "missing_market_inputs"
    if "missing required typed observables" in lower:
        return "missing_observables"
    if "missing required state tags" in lower:
        return "missing_state_tags"
    if "missing the required market binding" in lower:
        return "missing_market_binding"
    if "missing the required pricing kernel" in lower:
        return "missing_pricing_kernel"
    if "missing the required schedule builder" in lower:
        return "missing_schedule_builder"
    if "missing the required route helper" in lower or "has no helper target" in lower or "has no helper-backed lowering target" in lower:
        return "missing_route_helper"
    if "family lowering rejected the semantic contract" in lower:
        return "family_ir_rejected"
    return "lowering_rejected"
