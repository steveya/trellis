"""Structural solver declaration substrate for ContractIR selection (QUA-925).

This module is intentionally narrow. It introduces the typed declaration and
registry surface that later Phase 3 slices will use, but it does not yet
compile or execute solver calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from trellis.agent.contract_pattern import (
    AtomPattern,
    ConstantPattern,
    ContractPattern,
    ExercisePattern,
    ObservationPattern,
    PayoffPattern,
    SchedulePattern,
    SpotPattern,
    StrikePattern,
    UnderlyingPattern,
    Wildcard,
)
from trellis.agent.knowledge.methods import normalize_method


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _normalize_methods(methods: Iterable[str]) -> tuple[str, ...]:
    return _unique_strings(normalize_method(method) for method in methods)


class ContractIRSolverRegistryError(ValueError):
    """Base error for malformed declaration registries."""


class ContractIRSolverOverlapError(ContractIRSolverRegistryError):
    """Raised when two declarations conservatively overlap without resolution."""


@dataclass(frozen=True)
class ContractIRSolverSelectionAuthority:
    """The structural selection contract for a solver declaration."""

    contract_pattern: ContractPattern
    admissible_methods: tuple[str, ...] = ()
    required_term_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "admissible_methods", _normalize_methods(self.admissible_methods))
        object.__setattr__(self, "required_term_groups", _unique_strings(self.required_term_groups))


@dataclass(frozen=True)
class ContractIRSolverOutputSupport:
    """Requested-output support declared by a solver binding."""

    supported_outputs: tuple[str, ...] = ("price",)
    supported_measures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_outputs", _unique_strings(self.supported_outputs))
        object.__setattr__(self, "supported_measures", _unique_strings(self.supported_measures))


@dataclass(frozen=True)
class ContractIRSolverMarketRequirements:
    """Market capability requirements for a solver declaration."""

    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    required_coordinate_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_capabilities", _unique_strings(self.required_capabilities))
        object.__setattr__(self, "optional_capabilities", _unique_strings(self.optional_capabilities))
        object.__setattr__(
            self,
            "required_coordinate_kinds",
            _unique_strings(self.required_coordinate_kinds),
        )


@dataclass(frozen=True)
class ContractIRSolverMaterialization:
    """How a selected declaration will eventually materialize a callable."""

    callable_ref: str
    call_style: str
    adapter_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.callable_ref:
            raise ContractIRSolverRegistryError("materialization.callable_ref must be non-empty")
        if self.call_style not in {"helper_call", "raw_kernel_kwargs"}:
            raise ContractIRSolverRegistryError(
                "materialization.call_style must be 'helper_call' or 'raw_kernel_kwargs'"
            )


@dataclass(frozen=True)
class ContractIRSolverProvenance:
    """Stable identifier and binding provenance for one declaration."""

    declaration_id: str
    validation_bundle_id: str = ""
    compatibility_alias_policy: str = "operator_visible"
    helper_refs: tuple[str, ...] = ()
    pricing_kernel_refs: tuple[str, ...] = ()
    schedule_builder_refs: tuple[str, ...] = ()
    cashflow_engine_refs: tuple[str, ...] = ()
    market_binding_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.declaration_id:
            raise ContractIRSolverRegistryError("provenance.declaration_id must be non-empty")
        object.__setattr__(self, "helper_refs", _unique_strings(self.helper_refs))
        object.__setattr__(self, "pricing_kernel_refs", _unique_strings(self.pricing_kernel_refs))
        object.__setattr__(
            self,
            "schedule_builder_refs",
            _unique_strings(self.schedule_builder_refs),
        )
        object.__setattr__(
            self,
            "cashflow_engine_refs",
            _unique_strings(self.cashflow_engine_refs),
        )
        object.__setattr__(
            self,
            "market_binding_refs",
            _unique_strings(self.market_binding_refs),
        )


@dataclass(frozen=True)
class ContractIRSolverDeclaration:
    """A single structural solver declaration."""

    authority: ContractIRSolverSelectionAuthority
    materialization: ContractIRSolverMaterialization
    provenance: ContractIRSolverProvenance
    outputs: ContractIRSolverOutputSupport = field(default_factory=ContractIRSolverOutputSupport)
    market_requirements: ContractIRSolverMarketRequirements = field(
        default_factory=ContractIRSolverMarketRequirements
    )
    precedence: int = 0
    subordinates_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subordinates_to", _unique_strings(self.subordinates_to))


@dataclass(frozen=True)
class RegisteredContractIRSolverDeclaration:
    """A declaration paired with its deterministic registration index."""

    declaration: ContractIRSolverDeclaration
    registration_index: int

    @property
    def declaration_id(self) -> str:
        return self.declaration.provenance.declaration_id


@dataclass(frozen=True)
class ContractIRSolverRegistry:
    """Validated structural declarations in deterministic registration order."""

    declarations: tuple[RegisteredContractIRSolverDeclaration, ...]

    def selection_order(self) -> tuple[RegisteredContractIRSolverDeclaration, ...]:
        return tuple(
            sorted(
                self.declarations,
                key=lambda item: (-item.declaration.precedence, item.registration_index),
            )
        )

    def get(self, declaration_id: str) -> RegisteredContractIRSolverDeclaration:
        for item in self.declarations:
            if item.declaration_id == declaration_id:
                return item
        raise KeyError(declaration_id)


def build_contract_ir_solver_registry(
    declarations: Iterable[ContractIRSolverDeclaration],
) -> ContractIRSolverRegistry:
    """Validate and register structural solver declarations."""

    registered = tuple(
        RegisteredContractIRSolverDeclaration(declaration=declaration, registration_index=index)
        for index, declaration in enumerate(declarations)
    )
    _validate_unique_ids(registered)
    by_id = {item.declaration_id: item for item in registered}
    _validate_subordination_graph(registered, by_id)
    _validate_overlap_resolution(registered)
    return ContractIRSolverRegistry(declarations=registered)


def _validate_unique_ids(
    declarations: tuple[RegisteredContractIRSolverDeclaration, ...],
) -> None:
    seen: set[str] = set()
    for item in declarations:
        declaration_id = item.declaration_id
        if declaration_id in seen:
            raise ContractIRSolverRegistryError(
                f"duplicate ContractIR solver declaration id {declaration_id!r}"
            )
        seen.add(declaration_id)


def _validate_subordination_graph(
    declarations: tuple[RegisteredContractIRSolverDeclaration, ...],
    by_id: dict[str, RegisteredContractIRSolverDeclaration],
) -> None:
    for item in declarations:
        for superior_id in item.declaration.subordinates_to:
            if superior_id not in by_id:
                raise ContractIRSolverRegistryError(
                    f"declaration {item.declaration_id!r} subordinates_to unknown id {superior_id!r}"
                )
            superior = by_id[superior_id]
            if item.declaration_id == superior_id:
                raise ContractIRSolverRegistryError(
                    f"declaration {item.declaration_id!r} cannot subordinate to itself"
                )
            if item.declaration.precedence >= superior.declaration.precedence:
                raise ContractIRSolverRegistryError(
                    "subordinate declarations "
                    f"{item.declaration_id!r} -> {superior_id!r} require strictly lower precedence"
                )
    _validate_subordination_acyclic(declarations)


def _validate_subordination_acyclic(
    declarations: tuple[RegisteredContractIRSolverDeclaration, ...],
) -> None:
    graph = {
        item.declaration_id: tuple(item.declaration.subordinates_to) for item in declarations
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ContractIRSolverRegistryError(
                f"cyclic subordination detected at declaration {node!r}"
            )
        visiting.add(node)
        for parent in graph[node]:
            _visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        _visit(node)


def _validate_overlap_resolution(
    declarations: tuple[RegisteredContractIRSolverDeclaration, ...],
) -> None:
    for left, right in combinations(declarations, 2):
        if not _declarations_may_overlap(left.declaration, right.declaration):
            continue
        left_subordinate = right.declaration_id in left.declaration.subordinates_to
        right_subordinate = left.declaration_id in right.declaration.subordinates_to
        if left_subordinate and right_subordinate:
            raise ContractIRSolverRegistryError(
                "mutual subordination is not allowed for overlapping declarations "
                f"{left.declaration_id!r} and {right.declaration_id!r}"
            )
        if left_subordinate or right_subordinate:
            continue
        if left.declaration.precedence == right.declaration.precedence:
            raise ContractIRSolverOverlapError(
                "equal-precedence overlapping ContractIR solver declarations "
                f"{left.declaration_id!r} and {right.declaration_id!r} require "
                "an explicit precedence split or subordinate relation"
            )


def _declarations_may_overlap(
    left: ContractIRSolverDeclaration,
    right: ContractIRSolverDeclaration,
) -> bool:
    if not _method_sets_intersect(
        left.authority.admissible_methods,
        right.authority.admissible_methods,
    ):
        return False
    left_signature = _contract_pattern_overlap_signature(left.authority.contract_pattern)
    right_signature = _contract_pattern_overlap_signature(right.authority.contract_pattern)
    for key in left_signature:
        if _selectors_conflict(left_signature[key], right_signature[key]):
            return False
    return True


def _method_sets_intersect(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if not left or not right:
        return True
    return bool(set(left) & set(right))


def _selectors_conflict(left: object | None, right: object | None) -> bool:
    return left is not None and right is not None and left != right


def _contract_pattern_overlap_signature(pattern: ContractPattern) -> dict[str, object | None]:
    return {
        "payoff_kind": _payoff_selector(pattern.payoff),
        "exercise_style": _literal_selector(pattern.exercise.style if pattern.exercise else None),
        "exercise_frequency": _schedule_frequency_selector(
            pattern.exercise.schedule if pattern.exercise else None
        ),
        "observation_kind": _literal_selector(
            pattern.observation.kind if pattern.observation else None
        ),
        "underlying_kind": _literal_selector(
            pattern.underlying.kind if pattern.underlying else None
        ),
        "underlying_dynamics": _literal_selector(
            pattern.underlying.dynamics if pattern.underlying else None
        ),
    }


def _payoff_selector(node: object | None) -> object | None:
    if node is None:
        return None
    if isinstance(node, PayoffPattern):
        return node.kind
    if isinstance(node, SpotPattern):
        return "spot"
    if isinstance(node, StrikePattern):
        return "strike"
    if isinstance(node, ConstantPattern):
        return "constant"
    return None


def _schedule_frequency_selector(node: SchedulePattern | None) -> object | None:
    if node is None:
        return None
    return _literal_selector(node.frequency)


def _literal_selector(node: object | None) -> object | None:
    if node is None or isinstance(node, Wildcard):
        return None
    if isinstance(node, AtomPattern):
        return node.value
    if isinstance(node, (str, int, float, bool)):
        return node
    if isinstance(node, (ExercisePattern, ObservationPattern, UnderlyingPattern, SchedulePattern)):
        return None
    return None


__all__ = [
    "ContractIRSolverDeclaration",
    "ContractIRSolverMaterialization",
    "ContractIRSolverMarketRequirements",
    "ContractIRSolverOutputSupport",
    "ContractIRSolverOverlapError",
    "ContractIRSolverProvenance",
    "ContractIRSolverRegistry",
    "ContractIRSolverRegistryError",
    "ContractIRSolverSelectionAuthority",
    "RegisteredContractIRSolverDeclaration",
    "build_contract_ir_solver_registry",
]
