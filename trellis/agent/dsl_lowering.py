"""Lower normalized DSL fragments onto checked-in primitive binding targets.

This module is the first executable bridge from the semiring/Bellman DSL
algebra to the checked-in semantic compiler outputs. It is intentionally
conservative:

- it only lowers bindings that already have stable exact targets
- it returns structured admissibility errors instead of guessing
- it treats exact backend bindings as targets of the DSL, not as the DSL itself
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
    EventTriggeredTwoLeggedContractIR,
    EventAwareMonteCarloIR,
    EventAwarePDEIR,
    ExerciseLatticeIR,
    NthToDefaultIR,
    TransformPricingIR,
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
    binding_id: str | None = None


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
    binding_id: str = ""

    @property
    def control_styles(self) -> tuple[ControlStyle, ...]:
        """Return the explicit control styles present after lowering."""
        if self.normalized_expr is None:
            return ()
        return collect_control_styles(self.normalized_expr)

    @property
    def target_modules(self) -> tuple[str, ...]:
        """Return the distinct modules referenced by the lowering targets."""
        modules: list[str] = []
        for binding in self.target_bindings:
            if binding.module not in modules:
                modules.append(binding.module)
        return tuple(modules)

    @property
    def target_refs(self) -> tuple[str, ...]:
        """Return the distinct module.symbol targets referenced by the lowering."""
        refs: list[str] = []
        for binding in self.target_bindings:
            if binding.primitive_ref not in refs:
                refs.append(binding.primitive_ref)
        return tuple(refs)

    @property
    def route_helper_refs(self) -> tuple[str, ...]:
        """Return only targets explicitly admitted as product/route helpers."""
        return tuple(
            binding.primitive_ref
            for binding in self.target_bindings
            if binding.role == "route_helper"
        )

    @property
    def helper_modules(self) -> tuple[str, ...]:
        """Backward-compatible alias for :attr:`target_modules`."""
        return self.target_modules

    @property
    def helper_refs(self) -> tuple[str, ...]:
        """Backward-compatible alias for :attr:`target_refs`."""
        return self.target_refs

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

    method = str(getattr(pricing_plan, "method", "") or "").strip() or None
    errors: list[DslLoweringError] = []
    fallback_result: SemanticDslLowering | None = None
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

        method = str(getattr(pricing_plan, "method", "") or "").strip() or None
        binding_spec = _resolve_backend_binding_for_route(
            route_id,
            product_ir=product_ir,
            method=method,
        )
        binding_id = str(getattr(binding_spec, "binding_id", "") or "").strip()
        bindings = _target_bindings_for_route(
            route,
            product_ir=product_ir,
            binding_spec=binding_spec,
            method=method,
        )

        route_family = str(
            getattr(binding_spec, "route_family", "")
            or resolve_route_family(
                route,
                product_ir,
                binding_spec=binding_spec,
                method=method,
            )
        )
        adapters = resolve_route_adapters(route, product_ir, method=method)
        notes = resolve_route_notes(route, product_ir, method=method)
        try:
            family_ir = build_family_lowering_ir(
                contract,
                route_id=route_id,
                route_family=route_family,
                product_ir=product_ir,
                method=method,
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
                binding_id=binding_id,
                family_ir=family_ir,
                bindings=bindings,
            )
        else:
            expr, lowering_errors = _build_expr_for_route(
                contract,
                product_ir=product_ir,
                pricing_plan=pricing_plan,
                route_id=route_id,
                binding_id=binding_id,
                route_family=route_family,
                bindings=bindings,
            )
        if lowering_errors:
            fallback_result = SemanticDslLowering(
                route_id=route_id,
                route_family=route_family,
                family_ir=family_ir,
                expr=None,
                normalized_expr=None,
                target_bindings=bindings,
                adapters=adapters,
                notes=notes,
                errors=tuple(
                    _lowering_error(
                        route_id=route_id,
                        stage="dsl_expr",
                        code=_infer_error_code(error),
                        message=error,
                        binding_id=binding_id or None,
                    )
                    for error in lowering_errors
                ),
                binding_id=binding_id,
            )
            errors.extend(
                fallback_result.errors
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
            binding_id=binding_id,
        )

    if fallback_result is not None:
        return SemanticDslLowering(
            route_id=fallback_result.route_id,
            route_family=fallback_result.route_family,
            family_ir=fallback_result.family_ir,
            expr=None,
            normalized_expr=None,
            target_bindings=fallback_result.target_bindings,
            adapters=fallback_result.adapters,
            notes=fallback_result.notes,
            errors=tuple(errors),
            binding_id=fallback_result.binding_id,
        )

    return SemanticDslLowering(
        route_id=primitive_routes[0],
        route_family=None,
        family_ir=None,
        expr=None,
        normalized_expr=None,
        errors=tuple(errors),
    )


def _resolve_backend_binding_for_route(
    route_id: str,
    *,
    product_ir,
    method: str | None = None,
):
    """Resolve the canonical backend binding for one route when available."""
    try:
        from trellis.agent.backend_bindings import resolve_backend_binding_by_route_id

        return resolve_backend_binding_by_route_id(
            route_id,
            product_ir=product_ir,
            method=method,
        )
    except Exception:
        return None


def _target_bindings_for_route(
    route,
    *,
    product_ir,
    binding_spec,
    method: str | None = None,
) -> tuple[DslTargetBinding, ...]:
    """Resolve DSL bindings from the binding catalog before falling back to route cards."""
    primitives = tuple(
        getattr(binding_spec, "primitives", ())
        or resolve_route_primitives(
            route,
            product_ir,
            binding_spec=binding_spec,
            method=method,
        )
    )
    return tuple(
        DslTargetBinding(
            module=primitive.module,
            symbol=primitive.symbol,
            role=primitive.role,
            required=primitive.required,
        )
        for primitive in primitives
        if not primitive.excluded
    )


def _build_expr_for_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a lowering expression from a typed family IR."""
    if isinstance(family_ir, AnalyticalBlack76IR):
        return _build_black76_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, EventAwarePDEIR):
        return _build_event_aware_pde_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, EventAwareMonteCarloIR):
        return _build_event_aware_monte_carlo_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, TransformPricingIR):
        return _build_transform_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, ExerciseLatticeIR):
        return _build_exercise_lattice_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, CorrelatedBasketMonteCarloIR):
        return _build_correlated_basket_mc_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, EventTriggeredTwoLeggedContractIR):
        return _build_event_triggered_two_legged_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
        )
    if isinstance(family_ir, NthToDefaultIR):
        return _build_nth_to_default_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
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
    binding_id: str,
    route_family: str,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build the semantic DSL fragment for one resolved route."""
    market_signature = _market_signature(contract)
    if (
        route_id == "rate_tree_backward_induction"
        and getattr(contract.product, "payoff_family", "") == "swaption"
        and getattr(contract.product, "exercise_style", "") == "european"
    ):
        return _build_european_swaption_rate_lattice_expr(
            route_id=route_id,
            binding_id=binding_id,
            market_signature=market_signature,
            bindings=bindings,
        )
    route_helper = next(
        (binding for binding in bindings if binding.role == "route_helper"),
        None,
    )
    if _can_build_black76_expr(contract, bindings):
        return _build_black76_expr(
            contract,
            route_id=route_id,
            binding_id=binding_id,
            market_signature=market_signature,
            bindings=bindings,
        )
    resolved_kernel_expr = _build_resolved_pricing_kernel_expr(
        route_id=route_id,
        binding_id=binding_id,
        market_signature=market_signature,
        bindings=bindings,
    )
    if resolved_kernel_expr is not None:
        return resolved_kernel_expr, ()
    if route_helper is None:
        return None, (_missing_helper_target_message(route_id, binding_id),)

    control_style = _control_style_for_product(product_ir)
    if control_style is not None and any(
        binding.role == "control_policy" for binding in bindings
    ):
        return _build_control_expr(
            product_ir=product_ir,
            pricing_plan=pricing_plan,
            route_id=route_id,
            binding_id=binding_id,
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
            atom_id=_binding_atom_id(route_id, binding_id, "market_binding"),
            primitive_ref=market_binding.primitive_ref,
            description="Resolve market inputs into a binding-local state bundle.",
            signature=ContractSignature(
                inputs=market_signature.inputs,
                outputs=("resolved_state:state",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        helper_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
            primitive_ref=route_helper.primitive_ref,
            description="Delegate the priced binding to the checked-in helper.",
            signature=ContractSignature(
                inputs=("resolved_state:state",),
                outputs=("price:scalar",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        return ThenExpr(terms=(binding_atom, helper_atom)), ()

    helper_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
        primitive_ref=route_helper.primitive_ref,
        description="Delegate the priced binding to the checked-in helper.",
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return helper_atom, ()


def _build_european_swaption_rate_lattice_expr(
    *,
    route_id: str,
    binding_id: str,
    market_signature: ContractSignature,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build the explicit one-exercise swaption lattice composition."""
    stages = (
        (
            "contract_spec",
            "BermudanSwaptionTreeSpec",
            market_signature.inputs,
            ("tree_contract:state", *market_signature.inputs),
            "Narrow the European swaption into a one-exercise tree contract.",
        ),
        (
            "curve_basis_binding",
            "resolve_swaption_curve_basis_spread",
            ("tree_contract:state", *market_signature.inputs),
            ("basis_adjusted_contract:state", *market_signature.inputs),
            "Project the discount/forecast curve basis into the tree strike.",
        ),
        (
            "tree_input_binding",
            "resolve_bermudan_swaption_tree_inputs",
            ("basis_adjusted_contract:state", *market_signature.inputs),
            (
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
                *market_signature.inputs,
            ),
            "Resolve schedule, volatility, mean reversion, and lattice horizon inputs.",
        ),
        (
            "topology",
            "BINOMIAL_1F_TOPOLOGY",
            (
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
                *market_signature.inputs,
            ),
            (
                "topology:state",
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
                *market_signature.inputs,
            ),
            "Select the admitted one-factor binomial topology.",
        ),
        (
            "mesh",
            "UNIFORM_ADDITIVE_MESH",
            (
                "topology:state",
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
                *market_signature.inputs,
            ),
            (
                "mesh:state",
                "topology:state",
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
                *market_signature.inputs,
            ),
            "Select the admitted uniform additive mesh.",
        ),
        (
            "calibration_target",
            "TERM_STRUCTURE_TARGET",
            (
                "mesh:state",
                "topology:state",
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
                *market_signature.inputs,
            ),
            (
                "calibration_target:state",
                "mesh:state",
                "topology:state",
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
            ),
            "Bind the discount curve as the term-structure calibration target.",
        ),
        (
            "lattice_builder",
            "build_lattice",
            (
                "calibration_target:state",
                "mesh:state",
                "topology:state",
                "resolved_tree_inputs:state",
                "basis_adjusted_contract:state",
            ),
            (
                "rate_lattice:state",
                "basis_adjusted_contract:state",
                "resolved_tree_inputs:state",
            ),
            "Build and calibrate the generic short-rate lattice.",
        ),
        (
            "contract_compiler",
            "compile_bermudan_swaption_contract_spec",
            (
                "rate_lattice:state",
                "basis_adjusted_contract:state",
                "resolved_tree_inputs:state",
            ),
            ("rate_lattice:state", "lattice_contract:state"),
            "Compile the one-exercise swap claim and holder control contract.",
        ),
        (
            "pricing_kernel",
            "price_on_lattice",
            ("rate_lattice:state", "lattice_contract:state"),
            ("price:scalar",),
            "Roll the compiled contract back on the calibrated lattice.",
        ),
    )
    by_symbol = {binding.symbol: binding for binding in bindings if binding.required}
    missing = tuple(symbol for _, symbol, *_ in stages if symbol not in by_symbol)
    if missing:
        return None, tuple(
            _missing_primitive_message(route_id, binding_id, "lattice composition", symbol)
            for symbol in missing
        )
    atoms = tuple(
        ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, role),
            primitive_ref=by_symbol[symbol].primitive_ref,
            description=description,
            signature=ContractSignature(
                inputs=inputs,
                outputs=outputs,
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        for role, symbol, inputs, outputs, description in stages
    )
    return ThenExpr(terms=atoms), ()


def _build_resolved_pricing_kernel_expr(
    *,
    route_id: str,
    binding_id: str,
    market_signature: ContractSignature,
    bindings: tuple[DslTargetBinding, ...],
) -> ContractExpr | None:
    """Compose one required market resolver with one required pricing kernel."""
    market_bindings = tuple(
        binding
        for binding in bindings
        if binding.required and binding.role == "market_binding"
    )
    pricing_kernels = tuple(
        binding
        for binding in bindings
        if binding.required and binding.role == "pricing_kernel"
    )
    required_bindings = tuple(binding for binding in bindings if binding.required)
    if (
        len(market_bindings) != 1
        or len(pricing_kernels) != 1
        or len(required_bindings) != 2
    ):
        return None

    market_binding = market_bindings[0]
    pricing_kernel = pricing_kernels[0]
    binding_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "market_binding"),
        primitive_ref=market_binding.primitive_ref,
        description="Resolve contractual and market inputs into a typed pricing basis.",
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("resolved_state:state",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    kernel_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "pricing_kernel"),
        primitive_ref=pricing_kernel.primitive_ref,
        description="Evaluate the resolved pricing basis with the selected raw kernel.",
        signature=ContractSignature(
            inputs=("resolved_state:state",),
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return ThenExpr(terms=(binding_atom, kernel_atom))


def _build_black76_expr(
    contract,
    *,
    route_id: str,
    binding_id: str,
    market_signature: ContractSignature,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a direct kernel lowering for Black76-style analytical routes."""
    payoff_family = getattr(contract.product, "payoff_family", "")
    if payoff_family not in {"vanilla_option", "swaption"}:
        return None, (
            f"{_binding_subject(route_id, binding_id)} only has an explicit DSL lowering for "
            "plain vanilla-option and rate-style swaption semantics in this slice.",
        )

    route_helper = next(
        (binding for binding in bindings if binding.role == "route_helper"),
        None,
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
            return (
                ContractAtom(
                    atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
                    primitive_ref=route_helper.primitive_ref,
                    description="Delegate European rate-style swaption pricing to the checked-in Black76 family helper.",
                    signature=ContractSignature(
                        inputs=market_signature.inputs,
                        outputs=("price:scalar",),
                        timeline_roles=market_signature.timeline_roles,
                        market_data_requirements=market_signature.market_data_requirements,
                    ),
                ),
                (),
            )

        binding_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "market_binding"),
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
            atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
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
        return None, (_missing_primitive_message(route_id, binding_id, "pricing kernel", kernel_name),)

    kernel_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "pricing_kernel"),
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
    binding_id: str,
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
            _missing_primitive_message(
                route_id,
                binding_id,
                "pricing kernel",
                family_ir.kernel_symbol,
            ),
        )

    signature = _market_signature_from_family_ir(family_ir)
    kernel_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "pricing_kernel"),
        primitive_ref=kernel.primitive_ref,
        description=(
            f"Typed Black76 {family_ir.option_type} kernel for plain vanilla payoff "
            f"with {family_ir.market_mapping} market binding."
        ),
        signature=signature,
    )
    return kernel_atom, ()


def _build_event_aware_pde_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir: EventAwarePDEIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build an event-aware PDE lowering from typed family IR."""
    market_signature = _market_signature_from_family_ir(family_ir)
    if family_ir.helper_symbol:
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
                _missing_primitive_message(
                    route_id,
                    binding_id,
                    "helper",
                    family_ir.helper_symbol,
                ),
            )
        helper_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
            primitive_ref=route_helper.primitive_ref,
            description=_event_aware_pde_helper_description(family_ir),
            signature=market_signature,
        )
        return helper_atom, ()

    required_bindings = {
        symbol: next(
            (binding for binding in bindings if binding.symbol == symbol),
            None,
        )
        for symbol in (
            "resolve_single_state_diffusion_inputs",
            "terminal_intrinsic_from_resolved",
            "build_event_aware_pde_problem",
            "solve_event_aware_pde",
            "interpolate_pde_values",
        )
    }
    missing = tuple(
        symbol for symbol, binding in required_bindings.items() if binding is None
    )
    if missing:
        return None, tuple(
            _missing_primitive_message(route_id, binding_id, "PDE", symbol)
            for symbol in missing
        )

    stages = (
        (
            "market_binding",
            "resolve_single_state_diffusion_inputs",
            market_signature.inputs,
            ("resolved_state:state",),
            "Resolve the single-state diffusion market and contract inputs.",
        ),
        (
            "payoff_primitive",
            "terminal_intrinsic_from_resolved",
            ("resolved_state:state",),
            ("terminal_contract:state",),
            "Bind terminal payoff and boundary semantics to resolved state.",
        ),
        (
            "problem_builder",
            "build_event_aware_pde_problem",
            ("terminal_contract:state",),
            ("pde_problem:state",),
            "Assemble the typed grid, operator, boundary, and theta-method problem.",
        ),
        (
            "pricing_kernel",
            "solve_event_aware_pde",
            ("pde_problem:state",),
            ("pde_surface:state",),
            "Run the generic event-aware backward rollback.",
        ),
        (
            "interpolation",
            "interpolate_pde_values",
            ("pde_surface:state",),
            ("price:scalar",),
            "Interpolate the solved surface at the resolved spot.",
        ),
    )
    atoms = tuple(
        ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, role),
            primitive_ref=required_bindings[symbol].primitive_ref,
            description=description,
            signature=ContractSignature(
                inputs=inputs,
                outputs=outputs,
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        for role, symbol, inputs, outputs, description in stages
    )
    return ThenExpr(terms=atoms), ()


def _build_event_aware_monte_carlo_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir: EventAwareMonteCarloIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a bounded event-aware Monte Carlo lowering from typed family IR."""
    market_signature = _market_signature_from_family_ir(family_ir)
    if family_ir.helper_symbol:
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
                _missing_primitive_message(route_id, binding_id, "helper", family_ir.helper_symbol),
            )
        helper_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
            primitive_ref=route_helper.primitive_ref,
            description=(
                f"Typed event-aware Monte Carlo family helper for {family_ir.product_instrument or 'compiled'} "
                f"with process={family_ir.process_spec.process_family or 'generic_state_process'} "
                f"and reducer={family_ir.payoff_reducer_spec.reducer_kind or 'compiled_payoff'}."
            ),
            signature=market_signature,
        )
        return helper_atom, ()

    if family_ir.payoff_family == "swaption":
        required_bindings = {
            symbol: next(
                (binding for binding in bindings if binding.symbol == symbol),
                None,
            )
            for symbol in (
                "resolve_swaption_black76_inputs",
                "build_payment_timeline",
                "resolve_hull_white_monte_carlo_process_inputs",
                "build_discounted_swap_pv_payload",
                "build_short_rate_discount_reducer",
                "EventAwareMonteCarloEvent",
                "EventAwareMonteCarloProblemSpec",
                "build_event_aware_monte_carlo_problem",
                "price_event_aware_monte_carlo",
            )
        }
        missing = tuple(
            symbol for symbol, binding in required_bindings.items() if binding is None
        )
        if missing:
            return None, tuple(
                _missing_primitive_message(
                    route_id,
                    binding_id,
                    "European swaption Monte Carlo",
                    symbol,
                )
                for symbol in missing
            )

        stages = (
            (
                "market_binding",
                "resolve_swaption_black76_inputs",
                market_signature.inputs,
                ("resolved_swaption:state",),
                "Resolve the European expiry, schedule basis, and market conventions.",
            ),
            (
                "schedule_builder",
                "build_payment_timeline",
                ("resolved_swaption:state",),
                ("payment_timeline:state",),
                "Build the underlying swap payment timeline from explicit swap start.",
            ),
            (
                "process_binding",
                "resolve_hull_white_monte_carlo_process_inputs",
                ("payment_timeline:state",),
                ("mc_process_binding:state",),
                "Bind the Hull-White process to market or explicit comparison parameters.",
            ),
            (
                "settlement_payload",
                "build_discounted_swap_pv_payload",
                ("mc_process_binding:state",),
                ("settlement_payload:state",),
                "Build the discounted swap-PV settlement payload.",
            ),
            (
                "path_reducer",
                "build_short_rate_discount_reducer",
                ("settlement_payload:state",),
                ("discount_reducer:state",),
                "Build the reduced-state short-rate discount accumulator.",
            ),
            (
                "event_contract",
                "EventAwareMonteCarloEvent",
                ("discount_reducer:state",),
                ("event_specs:state",),
                "Declare the expiry observation and settlement events.",
            ),
            (
                "problem_spec",
                "EventAwareMonteCarloProblemSpec",
                ("event_specs:state",),
                ("mc_problem_spec:state",),
                "Declare the typed event-aware Monte Carlo problem.",
            ),
            (
                "problem_builder",
                "build_event_aware_monte_carlo_problem",
                ("mc_problem_spec:state",),
                ("mc_problem:state",),
                "Compile process, events, reducer, and payoff into the runtime problem.",
            ),
            (
                "monte_carlo_estimator",
                "price_event_aware_monte_carlo",
                ("mc_problem:state",),
                ("price:scalar",),
                "Evaluate the compiled problem with explicit path, step, and seed controls.",
            ),
        )
        atoms = tuple(
            ContractAtom(
                atom_id=_binding_atom_id(route_id, binding_id, role),
                primitive_ref=required_bindings[symbol].primitive_ref,
                description=description,
                signature=ContractSignature(
                    inputs=inputs,
                    outputs=outputs,
                    timeline_roles=market_signature.timeline_roles,
                    market_data_requirements=market_signature.market_data_requirements,
                ),
            )
            for role, symbol, inputs, outputs, description in stages
        )
        return ThenExpr(terms=atoms), ()

    terminal_estimator = next(
        (
            binding
            for binding in bindings
            if binding.role == "monte_carlo_estimator"
        ),
        None,
    )
    terminal_payoff = next(
        (
            binding
            for binding in bindings
            if binding.role in {"payoff_primitive", "terminal_payoff"}
        ),
        None,
    )
    path_requirement = str(
        family_ir.path_requirement_spec.requirement_kind or ""
    ).strip().lower()
    if path_requirement == "terminal_only" and terminal_estimator is not None:
        if terminal_payoff is None:
            return None, (
                _missing_primitive_message(
                    route_id,
                    binding_id,
                    "terminal payoff",
                ),
            )
        payoff_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "payoff_primitive"),
            primitive_ref=terminal_payoff.primitive_ref,
            description=(
                "Bind the derivative-specific terminal payoff as a callback over "
                "resolved single-state diffusion inputs."
            ),
            signature=ContractSignature(
                inputs=market_signature.inputs,
                outputs=("terminal_payoff_callback:state",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        estimator_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "monte_carlo_estimator"),
            primitive_ref=terminal_estimator.primitive_ref,
            description=(
                "Evaluate the terminal claim with explicit simulation scheme and "
                "variance-reduction controls."
            ),
            signature=ContractSignature(
                inputs=("terminal_payoff_callback:state",),
                outputs=("price:scalar",),
                timeline_roles=market_signature.timeline_roles,
                market_data_requirements=market_signature.market_data_requirements,
            ),
        )
        return ThenExpr(terms=(payoff_atom, estimator_atom)), ()

    process_binding = next(
        (
            binding
            for binding in bindings
            if binding.role == "state_process"
            and _binding_supports_mc_process(binding, family_ir.process_spec.process_family)
        ),
        None,
    )
    path_simulation = next(
        (binding for binding in bindings if binding.role == "path_simulation"),
        None,
    )
    if path_simulation is None:
        return None, (
            _missing_primitive_message(route_id, binding_id, "Monte Carlo path simulation"),
        )

    reducer_binding = next(
        (
            binding
            for binding in bindings
            if binding.role in {"pricing_kernel", "route_helper"}
            and (
                not family_ir.helper_symbol
                or binding.symbol == family_ir.helper_symbol
            )
        ),
        None,
    )

    process_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "state_process"),
        primitive_ref=process_binding.primitive_ref if process_binding is not None else None,
        description=(
            f"Compile the typed {family_ir.process_spec.process_family or 'event-aware'} "
            f"state process over {family_ir.state_spec.state_variable or 'state'}."
        ),
        signature=ContractSignature(
            inputs=market_signature.inputs,
            outputs=("mc_process:state",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    simulation_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "path_simulation"),
        primitive_ref=path_simulation.primitive_ref,
        description=(
            f"Simulate paths under the typed {family_ir.path_requirement_spec.requirement_kind or 'terminal_only'} "
            f"contract with event kinds {', '.join(family_ir.event_kinds) or 'none'}."
        ),
        signature=ContractSignature(
            inputs=("mc_process:state",),
            outputs=("path_state:state",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    reducer_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "payoff_reducer"),
        primitive_ref=reducer_binding.primitive_ref if reducer_binding is not None else None,
        description=(
            f"Reduce simulated path state through {family_ir.payoff_reducer_spec.reducer_kind or 'compiled_payoff'} "
            "into a price scalar."
        ),
        signature=ContractSignature(
            inputs=("path_state:state",),
            outputs=("price:scalar",),
            timeline_roles=market_signature.timeline_roles,
            market_data_requirements=market_signature.market_data_requirements,
        ),
    )
    return ThenExpr(terms=(process_atom, simulation_atom, reducer_atom)), ()


def _build_transform_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir: TransformPricingIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a bounded transform-pricing lowering from typed family IR."""
    market_signature = _market_signature_from_family_ir(family_ir)
    if family_ir.helper_symbol:
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
                _missing_primitive_message(route_id, binding_id, "helper", family_ir.helper_symbol),
            )
        helper_atom = ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
            primitive_ref=route_helper.primitive_ref,
            description=(
                f"Typed transform helper for {family_ir.product_instrument or 'compiled'} "
                f"with characteristic={family_ir.characteristic_spec.characteristic_family or 'generic_cf'}."
            ),
            signature=market_signature,
        )
        return helper_atom, ()

    transform_pricer = next(
        (
            binding
            for binding in bindings
            if binding.role == "transform_pricer"
        ),
        None,
    )
    if transform_pricer is None:
        return None, (
            _missing_primitive_message(route_id, binding_id, "raw transform pricing"),
        )

    kernel_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "transform_pricer"),
        primitive_ref=transform_pricer.primitive_ref,
        description=(
            f"Typed raw transform pricing kernel for {family_ir.product_instrument or 'compiled'} "
            f"with characteristic={family_ir.characteristic_spec.characteristic_family or 'generic_cf'}."
        ),
        signature=market_signature,
    )
    return kernel_atom, ()


def _can_build_black76_expr(contract, bindings: tuple[DslTargetBinding, ...]) -> bool:
    """Return whether fallback lowering can use the Black76 family surface."""
    payoff_family = str(getattr(contract.product, "payoff_family", "") or "").strip()
    if payoff_family not in {"vanilla_option", "swaption"}:
        return False
    return any(
        binding.role == "pricing_kernel"
        and binding.symbol in {"black76_call", "black76_put"}
        for binding in bindings
    )


def _binding_subject(route_id: str, binding_id: str) -> str:
    """Return a readable binding-first label with route fallback."""
    if str(binding_id or "").strip():
        return f"Binding '{binding_id}'"
    return f"Route '{route_id}'"


def _binding_atom_id(
    route_id: str,
    binding_id: str,
    role: str,
) -> str:
    """Return a stable DSL atom id rooted in binding identity.

    ``role`` identifies the kind of atom within the binding (e.g. 'route_helper',
    'pricing_kernel', 'market_binding', 'schedule_builder', 'state_process').
    """
    identity = str(binding_id or "").strip() or route_id
    if not identity:
        identity = "unbound"
    return f"{identity}:{role}"


def _missing_helper_target_message(
    route_id: str,
    binding_id: str,
    context: str = "DSL lowering",
) -> str:
    """Return a binding-first missing-helper message."""
    return (
        f"{_binding_subject(route_id, binding_id)} has no helper-backed lowering target"
        f" for {context}."
    )


def _missing_primitive_message(
    route_id: str,
    binding_id: str,
    primitive_kind: str,
    symbol: str | None = None,
) -> str:
    """Return a binding-first missing-primitive message."""
    suffix = f" primitive '{symbol}'" if str(symbol or "").strip() else " primitive"
    return (
        f"{_binding_subject(route_id, binding_id)} is missing the required "
        f"{primitive_kind}{suffix}."
    )


def _binding_supports_mc_process(binding: DslTargetBinding, process_family: str) -> bool:
    """Return whether one route binding can satisfy the typed MC process family."""
    family = str(process_family or "").strip()
    if not family:
        return True
    if family == "gbm_1d":
        return binding.symbol == "GBM"
    if family == "local_vol_1d":
        return binding.symbol == "LocalVol"
    if family == "hull_white_1f":
        return binding.symbol == "HullWhite"
    return False


def _event_aware_pde_helper_description(family_ir: EventAwarePDEIR) -> str:
    """Return a readable helper description for typed event-aware PDE routes."""
    if isinstance(family_ir, VanillaEquityPDEIR):
        return (
            f"Typed theta-method PDE helper for vanilla {family_ir.option_type} payoff "
            f"with theta={family_ir.theta:g}."
        )
    operator_family = str(family_ir.operator_spec.operator_family or "generic_1d").strip()
    control_style = str(family_ir.control_spec.control_style or "identity").strip()
    return (
        f"Typed event-aware PDE helper for {family_ir.product_instrument or 'compiled'} "
        f"rollback with operator={operator_family} and control={control_style}."
    )


def _build_exercise_lattice_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir: ExerciseLatticeIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build an exercise-lattice lowering from typed family IR."""
    if (
        family_ir.product_instrument == "swaption"
        and family_ir.exercise_style == "bermudan"
        and not family_ir.helper_symbol
    ):
        return _build_bermudan_swaption_lattice_expr_from_family_ir(
            route_id=route_id,
            binding_id=binding_id,
            family_ir=family_ir,
            bindings=bindings,
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
            _missing_primitive_message(route_id, binding_id, "helper", family_ir.helper_symbol),
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
            atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
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


def _build_bermudan_swaption_lattice_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir: ExerciseLatticeIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build the explicit generic-lattice composition for a Bermudan swaption."""
    symbols = (
        "normalize_explicit_dates",
        "year_fraction",
        "build_payment_timeline",
        "resolve_bermudan_swaption_tree_inputs",
        "BINOMIAL_1F_TOPOLOGY",
        "UNIFORM_ADDITIVE_MESH",
        "TERM_STRUCTURE_TARGET",
        "build_lattice",
        "lattice_step_from_time",
        "LatticeLinearClaimSpec",
        "LatticeContractSpec",
        "value_on_lattice",
        "LatticeControlSpec",
        "price_on_lattice",
    )
    required_bindings = {
        symbol: next(
            (binding for binding in bindings if binding.symbol == symbol),
            None,
        )
        for symbol in symbols
    }
    missing = tuple(
        symbol for symbol, binding in required_bindings.items() if binding is None
    )
    if missing:
        return None, tuple(
            _missing_primitive_message(
                route_id,
                binding_id,
                "Bermudan lattice",
                symbol,
            )
            for symbol in missing
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    timeline_roles = market_signature.timeline_roles | {
        TimelineRole.EXERCISE,
        TimelineRole.PAYMENT,
    }
    requirements = market_signature.market_data_requirements

    def atom(
        role: str,
        symbol: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        description: str,
    ) -> ContractAtom:
        binding = required_bindings[symbol]
        assert binding is not None
        return ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, role),
            primitive_ref=binding.primitive_ref,
            description=description,
            signature=ContractSignature(
                inputs=inputs,
                outputs=outputs,
                timeline_roles=timeline_roles,
                market_data_requirements=requirements,
            ),
        )

    ports = (
        market_signature.inputs,
        ("normalized_exercise_schedule:schedule",),
        ("measured_exercise_schedule:schedule",),
        ("exercise_payment_schedules:schedule",),
        ("resolved_lattice_inputs:state",),
        ("lattice_topology:state",),
        ("lattice_mesh:state",),
        ("calibrated_lattice_target:state",),
        ("built_rate_lattice:state",),
        ("mapped_lattice_schedules:state",),
        ("fixed_leg_claim:contract",),
        ("fixed_leg_contract:contract",),
        ("fixed_leg_observations:state",),
        ("signed_swap_values:state",),
        ("exercise_decision:state",),
        ("option_claim:contract",),
        ("option_contract:contract",),
        ("price:scalar",),
    )
    stages = (
        atom(
            "schedule_normalizer",
            "normalize_explicit_dates",
            ports[0],
            ports[1],
            "Normalize contractual Bermudan exercise dates before lattice mapping.",
        ),
        atom(
            "timeline_mapping",
            "year_fraction",
            ports[1],
            ports[2],
            "Measure and quantize live exercise dates from settlement.",
        ),
        atom(
            "payment_timeline_builder",
            "build_payment_timeline",
            ports[2],
            ports[3],
            "Build the underlying fixed-leg payment timeline from the first exercise date.",
        ),
        atom(
            "market_binding",
            "resolve_bermudan_swaption_tree_inputs",
            ports[3],
            ports[4],
            "Resolve curve, volatility, Hull-White parameters, horizon, and step controls.",
        ),
        atom(
            "topology",
            "BINOMIAL_1F_TOPOLOGY",
            ports[4],
            ports[5],
            "Select the admitted one-factor recombining topology.",
        ),
        atom(
            "mesh",
            "UNIFORM_ADDITIVE_MESH",
            ports[5],
            ports[6],
            "Select the additive short-rate mesh.",
        ),
        atom(
            "calibration_target",
            "TERM_STRUCTURE_TARGET",
            ports[6],
            ports[7],
            "Bind the discount curve as the lattice calibration target.",
        ),
        atom(
            "lattice_builder",
            "build_lattice",
            ports[7],
            ports[8],
            "Build and calibrate the generic one-factor rate lattice.",
        ),
        atom(
            "schedule_mapping",
            "lattice_step_from_time",
            ports[8],
            ports[9],
            "Map exercise and payment times to bounded lattice steps.",
        ),
        atom(
            "fixed_leg_claim",
            "LatticeLinearClaimSpec",
            ports[9],
            ports[10],
            "Represent fixed coupons and principal as a generic linear lattice claim.",
        ),
        atom(
            "fixed_leg_contract",
            "LatticeContractSpec",
            ports[10],
            ports[11],
            "Wrap the fixed leg in the generic lattice contract surface.",
        ),
        atom(
            "observation_rollback",
            "value_on_lattice",
            ports[11],
            ports[12],
            "Capture fixed-leg continuation values only at Bermudan exercise steps.",
        ),
        ContractAtom(
            atom_id=_binding_atom_id(route_id, binding_id, "payer_receiver_algebra"),
            description=(
                "Form payer or receiver swap values from par notional and fixed-leg "
                "continuation observations in adapter-owned algebra."
            ),
            signature=ContractSignature(
                inputs=ports[12],
                outputs=ports[13],
                timeline_roles=timeline_roles,
                market_data_requirements=requirements,
            ),
        ),
    )
    choice_signature = ContractSignature(
        inputs=ports[13],
        outputs=ports[14],
        timeline_roles=timeline_roles,
        market_data_requirements=requirements,
    )
    holder_choice = ChoiceExpr(
        style=ControlStyle.HOLDER_MAX,
        branches=(
            ContractAtom(
                atom_id=f"{family_ir.route_family}:continuation",
                description="Continue the option rollback without exercising.",
                signature=choice_signature,
            ),
            ContractAtom(
                atom_id=f"{family_ir.route_family}:exercise_now",
                primitive_ref=required_bindings["LatticeControlSpec"].primitive_ref,
                description=(
                    "Exercise into the positive payer/receiver swap value under holder-max control."
                ),
                signature=choice_signature,
            ),
        ),
        label="bermudan_swaption_holder_max",
    )
    suffix = (
        atom(
            "option_claim",
            "LatticeLinearClaimSpec",
            ports[14],
            ports[15],
            "Represent the option continuation claim with zero terminal payoff.",
        ),
        atom(
            "option_contract",
            "LatticeContractSpec",
            ports[15],
            ports[16],
            "Attach the holder-max control to the generic option contract.",
        ),
        atom(
            "pricing_kernel",
            "price_on_lattice",
            ports[16],
            ports[17],
            "Run the generic controlled rollback and return the root price.",
        ),
    )
    return ThenExpr(terms=(*stages, holder_choice, *suffix)), ()


def _build_correlated_basket_mc_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
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
            _missing_primitive_message(
                route_id,
                binding_id,
                "market binding",
                family_ir.market_binding_symbol,
            ),
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
            _missing_primitive_message(route_id, binding_id, "helper", family_ir.helper_symbol),
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    binding_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "market_binding"),
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
        atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
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


def _build_event_triggered_two_legged_expr_from_family_ir(
    *,
    route_id: str,
    binding_id: str,
    family_ir: EventTriggeredTwoLeggedContractIR,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build a typed event-triggered two-legged schedule-builder and helper lowering."""
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
            _missing_primitive_message(
                route_id,
                binding_id,
                "schedule builder",
                family_ir.schedule_builder_symbol,
            ),
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
            _missing_primitive_message(route_id, binding_id, "helper", family_ir.helper_symbol),
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    schedule_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "schedule_builder"),
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
        atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
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
    binding_id: str,
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
            _missing_primitive_message(route_id, binding_id, "helper", family_ir.helper_symbol),
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
            _missing_primitive_message(route_id, binding_id, "copula", family_ir.copula_symbol),
        )

    market_signature = _market_signature_from_family_ir(family_ir)
    helper_atom = ContractAtom(
        atom_id=_binding_atom_id(route_id, binding_id, "route_helper"),
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
    binding_id: str,
    route_family: str,
    market_signature: ContractSignature,
    bindings: tuple[DslTargetBinding, ...],
) -> tuple[ContractExpr | None, tuple[str, ...]]:
    """Build an explicit Bellman/choice expression for a control route."""
    if pricing_plan.method not in {"rate_tree", "monte_carlo"}:
        return None, (
            f"{_binding_subject(route_id, binding_id)} is a control binding but method '{pricing_plan.method}' "
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
        return None, (_missing_helper_target_message(route_id, binding_id, "control lowering"),)

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
    binding_id: str | None = None,
) -> DslLoweringError:
    """Build one structured lowering error record."""
    return DslLoweringError(
        route_id=route_id,
        stage=stage,
        code=code,
        message=message,
        binding_id=binding_id,
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
    if (
        "missing the required route helper" in lower
        or "missing the required helper primitive" in lower
        or "has no helper target" in lower
        or "has no helper-backed lowering target" in lower
    ):
        return "missing_route_helper"
    if "family lowering rejected the semantic contract" in lower:
        return "family_ir_rejected"
    return "lowering_rejected"
