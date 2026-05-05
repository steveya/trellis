"""Executable lowering admission for bounded quoted-observable contracts.

This slice reuses the shared Phase 3 declaration substrate so quoted-observable
contracts can prove executable route-free authority for the first bounded
cohort: terminal linear curve-spread and surface-spread payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.agent.contract_pattern import (
    ConstantPattern,
    ContractPattern,
    ExercisePattern,
    ObservationPattern,
    PayoffPattern,
    UnderlyingPattern,
    Wildcard,
)
from trellis.agent.contract_ir import ContractIR, Singleton
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


@dataclass(frozen=True)
class _CurveSpreadSpec:
    notional: float
    curve_id: str
    lhs_coordinate: object
    rhs_coordinate: object
    convention: str
    expiry_date: date
    day_count: object


@dataclass(frozen=True)
class _SurfaceSpreadSpec:
    notional: float
    surface_id: str
    lhs_coordinate: object
    rhs_coordinate: object
    convention: str
    expiry_date: date
    day_count: object


def _curve_quote_adapter(
    *,
    contract_ir,
    term_environment,
    valuation_context,
    market_state,
    bindings: dict[str, object],
) -> dict[str, object]:
    exercise = contract_ir.exercise.schedule
    observation = contract_ir.observation.schedule
    if getattr(contract_ir.observation, "kind", "") != "terminal":
        raise ValueError("Quoted-observable curve spread lowering requires terminal observation")
    if not isinstance(exercise, Singleton) or not isinstance(observation, Singleton):
        raise ValueError("Quoted-observable curve spread lowering requires European singleton exercise")
    if observation.t != exercise.t:
        raise ValueError("Quoted-observable curve spread lowering requires expiry-aligned observation")
    spec = _CurveSpreadSpec(
        notional=float(bindings["notional"]),
        curve_id=str(bindings["curve_id"]),
        lhs_coordinate=bindings["lhs_coordinate"],
        rhs_coordinate=bindings["rhs_coordinate"],
        convention=str(bindings["convention"]),
        expiry_date=observation.t,
        day_count=term_environment.accrual_conventions.day_count,
    )
    return {
        "call_kwargs": {
            "market_state": market_state,
            "spec": spec,
        },
        "value_scale": 1.0,
        "resolved_market_coordinates": ("discount_curve",),
    }


def _surface_quote_adapter(
    *,
    contract_ir,
    term_environment,
    valuation_context,
    market_state,
    bindings: dict[str, object],
) -> dict[str, object]:
    exercise = contract_ir.exercise.schedule
    observation = contract_ir.observation.schedule
    if getattr(contract_ir.observation, "kind", "") != "terminal":
        raise ValueError("Quoted-observable surface spread lowering requires terminal observation")
    if not isinstance(exercise, Singleton) or not isinstance(observation, Singleton):
        raise ValueError("Quoted-observable surface spread lowering requires European singleton exercise")
    if observation.t != exercise.t:
        raise ValueError("Quoted-observable surface spread lowering requires expiry-aligned observation")
    spec = _SurfaceSpreadSpec(
        notional=float(bindings["notional"]),
        surface_id=str(bindings["surface_id"]),
        lhs_coordinate=bindings["lhs_coordinate"],
        rhs_coordinate=bindings["rhs_coordinate"],
        convention=str(bindings["convention"]),
        expiry_date=observation.t,
        day_count=term_environment.accrual_conventions.day_count,
    )
    return {
        "call_kwargs": {
            "market_state": market_state,
            "spec": spec,
        },
        "value_scale": 1.0,
        "resolved_market_coordinates": ("discount_curve", "black_vol_surface", "spot"),
    }


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


def quoted_observable_solver_declarations() -> tuple[ContractIRSolverDeclaration, ...]:
    """Return the bounded quoted-observable structural solver declarations."""

    return (
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=_curve_spread_pattern(),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.quoted_observable.price_curve_quote_spread_analytical",
                call_style="helper_call",
                adapter_ref="trellis.agent.quoted_observable_admission._curve_quote_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="quoted_observable_curve_spread_linear",
                validation_bundle_id="quoted_observable_admission_curve_linear",
                helper_refs=("trellis.models.quoted_observable.price_curve_quote_spread_analytical",),
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
                callable_ref="trellis.models.quoted_observable.price_surface_quote_spread_analytical",
                call_style="helper_call",
                adapter_ref="trellis.agent.quoted_observable_admission._surface_quote_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="quoted_observable_surface_spread_linear",
                validation_bundle_id="quoted_observable_admission_surface_linear",
                helper_refs=("trellis.models.quoted_observable.price_surface_quote_spread_analytical",),
            ),
            market_requirements=ContractIRSolverMarketRequirements(
                required_coordinate_kinds=("surface_quote",),
            ),
            precedence=19,
        ),
    )


def _default_registry() -> ContractIRSolverRegistry:
    declarations = quoted_observable_solver_declarations()
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
    "quoted_observable_solver_declarations",
    "select_quoted_observable_lowering",
]
