"""Selection-only lowering admission for bounded quoted-observable contracts.

This slice deliberately stops at structural admission. It reuses the shared
Phase 3 declaration substrate so quoted-observable contracts can prove their
route-free authority surface without pretending that a checked pricer already
exists for arbitrary future curve/surface quote products.
"""

from __future__ import annotations

from trellis.agent.contract_pattern import (
    ConstantPattern,
    ContractPattern,
    ExercisePattern,
    ObservationPattern,
    PayoffPattern,
    UnderlyingPattern,
    Wildcard,
)
from trellis.agent.contract_ir import ContractIR
from trellis.agent.contract_ir_solver_compiler import (
    ContractIRSolverSelection,
    select_contract_ir_solver,
)
from trellis.agent.contract_ir_solver_registry import (
    ContractIRSolverDeclaration,
    ContractIRSolverMarketRequirements,
    ContractIRSolverMaterialization,
    ContractIRSolverProvenance,
    ContractIRSolverRegistry,
    ContractIRSolverSelectionAuthority,
    build_contract_ir_solver_registry,
)


def _unimplemented_quoted_observable_lowering(**kwargs):
    raise NotImplementedError(
        "Quoted-observable admission is selection-only in QUA-928; no checked executable lowering is landed yet."
    )


def _quote_admission_adapter(**kwargs) -> dict[str, object]:
    return {"call_kwargs": {}}


def _curve_spread_pattern() -> ContractPattern:
    return ContractPattern(
        payoff=PayoffPattern(
            kind="scaled",
            args=(
                ConstantPattern(value=Wildcard("notional")),
                PayoffPattern(
                    kind="sub",
                    args=(
                        PayoffPattern(
                            kind="curve_quote",
                            args=(
                                Wildcard("curve_id"),
                                Wildcard("lhs_coordinate"),
                                Wildcard("convention"),
                            ),
                        ),
                        PayoffPattern(
                            kind="curve_quote",
                            args=(
                                Wildcard("curve_id"),
                                Wildcard("rhs_coordinate"),
                                Wildcard("convention"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        exercise=ExercisePattern(style="european"),
        observation=ObservationPattern(kind="terminal"),
        underlying=UnderlyingPattern(kind="quoted_observable_curve"),
    )


def _surface_spread_pattern() -> ContractPattern:
    return ContractPattern(
        payoff=PayoffPattern(
            kind="scaled",
            args=(
                ConstantPattern(value=Wildcard("notional")),
                PayoffPattern(
                    kind="sub",
                    args=(
                        PayoffPattern(
                            kind="surface_quote",
                            args=(
                                Wildcard("surface_id"),
                                Wildcard("lhs_coordinate"),
                                Wildcard("convention"),
                            ),
                        ),
                        PayoffPattern(
                            kind="surface_quote",
                            args=(
                                Wildcard("surface_id"),
                                Wildcard("rhs_coordinate"),
                                Wildcard("convention"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        exercise=ExercisePattern(style="european"),
        observation=ObservationPattern(kind="terminal"),
        underlying=UnderlyingPattern(kind="quoted_observable_surface"),
    )


def _default_registry() -> ContractIRSolverRegistry:
    declarations = (
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=_curve_spread_pattern(),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.agent.quoted_observable_admission._unimplemented_quoted_observable_lowering",
                call_style="helper_call",
                adapter_ref="trellis.agent.quoted_observable_admission._quote_admission_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="quoted_observable_curve_spread_linear",
                validation_bundle_id="quoted_observable_admission_curve_linear",
            ),
            market_requirements=ContractIRSolverMarketRequirements(
                required_coordinate_kinds=("curve_quote",),
            ),
            precedence=20,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=_surface_spread_pattern(),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.agent.quoted_observable_admission._unimplemented_quoted_observable_lowering",
                call_style="helper_call",
                adapter_ref="trellis.agent.quoted_observable_admission._quote_admission_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="quoted_observable_surface_spread_linear",
                validation_bundle_id="quoted_observable_admission_surface_linear",
            ),
            market_requirements=ContractIRSolverMarketRequirements(
                required_coordinate_kinds=("surface_quote",),
            ),
            precedence=19,
        ),
    )
    return build_contract_ir_solver_registry(declarations)


_DEFAULT_REGISTRY: ContractIRSolverRegistry | None = None


def default_quoted_observable_admission_registry() -> ContractIRSolverRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _default_registry()
    return _DEFAULT_REGISTRY


def select_quoted_observable_lowering(
    contract_ir: ContractIR,
    *,
    term_environment=None,
    valuation_context=None,
    preferred_method: str | None = None,
    requested_outputs=None,
    registry: ContractIRSolverRegistry | None = None,
) -> ContractIRSolverSelection:
    """Select one bounded quoted-observable lowering admission declaration."""

    return select_contract_ir_solver(
        contract_ir,
        term_environment=term_environment,
        valuation_context=valuation_context,
        preferred_method=preferred_method,
        requested_outputs=requested_outputs,
        registry=registry or default_quoted_observable_admission_registry(),
    )


__all__ = [
    "default_quoted_observable_admission_registry",
    "select_quoted_observable_lowering",
]
