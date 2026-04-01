"""Composition algebra: typed payoff components and composite contracts.

QUA-413 proof-of-concept. This module defines the core dataclasses for
expressing structured products as DAGs of typed sub-contracts. It is NOT
wired into the build pipeline — that's a future implementation ticket.

Design document: docs/developer/composition_calibration_design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trellis.agent.dsl_algebra import (
    ChoiceExpr,
    ContractAtom,
    ContractExpr,
    ContractSignature,
    ControlStyle,
    normalize_contract_expr,
)
from trellis.core.types import TimelineRole


# ---------------------------------------------------------------------------
# Component interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentPort:
    """A named, typed input or output of a component."""

    name: str
    port_type: str  # "scalar", "array", "schedule", "state", "mask"
    description: str = ""
    optional: bool = False

    @property
    def signature_label(self) -> str:
        """Return the stable typed label used by the DSL algebra."""
        return f"{self.name}:{self.port_type}"


# ---------------------------------------------------------------------------
# PayoffComponent: the atom
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PayoffComponent:
    """A single typed payoff building block.

    Components are the atoms; products are molecules built from component
    DAGs.  Each component declares its type, interface (typed ports),
    constraints (compatible methods, market data), and validation hooks.
    """

    component_id: str
    component_type: str  # e.g., "barrier", "coupon_stream", "exercise_policy"

    # Typed interface
    inputs: tuple[ComponentPort, ...] = ()
    outputs: tuple[ComponentPort, ...] = ()

    # Constraints
    compatible_methods: tuple[str, ...] = ()
    market_data_requirements: frozenset[str] = frozenset()
    timeline_roles: frozenset[TimelineRole] = frozenset()

    # Validation hooks
    semantic_validators: tuple[str, ...] = ()
    financial_invariants: tuple[str, ...] = ()

    # Metadata
    description: str = ""
    proven_primitive: str | None = None  # module.symbol if reusable

    @property
    def signature(self) -> ContractSignature:
        """Return the typed DSL signature for this component."""
        return ContractSignature(
            inputs=tuple(port.signature_label for port in self.inputs),
            outputs=tuple(port.signature_label for port in self.outputs),
            timeline_roles=self.timeline_roles,
            market_data_requirements=self.market_data_requirements,
        )

    def to_contract_atom(self) -> ContractAtom:
        """Bridge the component into the executable DSL algebra."""
        return ContractAtom(
            atom_id=self.component_id,
            signature=self.signature,
            primitive_ref=self.proven_primitive,
            description=self.description,
        )


# ---------------------------------------------------------------------------
# Composition edges
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionEdge:
    """A typed relationship between two components in the DAG."""

    source: str        # component_id
    target: str        # component_id
    edge_type: str     # "sequential", "conditional", "parallel", "override"
    condition: str = ""  # for conditional edges: when does the edge activate?


@dataclass(frozen=True)
class ControlBoundary:
    """Explicit Bellman-style choice boundary over compatible component branches.

    The controller component carries the exercise or control semantics while the
    branches carry the priced continuation/exercise alternatives. Keeping this
    separate from generic DAG edges avoids smuggling choice semantics through
    stringly ``edge_type`` conventions.
    """

    boundary_id: str
    controller_component: str
    style: ControlStyle
    branches: tuple[str, ...]
    label: str = ""


# ---------------------------------------------------------------------------
# Method conflict resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodResolution:
    """Result of resolving method conflicts across components."""

    resolved_method: str
    resolution_kind: str  # "intersection", "dominance", "conflict"
    dominant_component: str | None = None
    overridden_components: tuple[str, ...] = ()
    reason: str = ""


def resolve_method_conflicts(
    components: tuple[PayoffComponent, ...],
) -> MethodResolution:
    """Resolve method conflicts across components.

    1. Intersect compatible_methods across all components.
    2. If intersection is non-empty, use it (pick first from intersection).
    3. If empty, find the component with exercise_policy type — it dominates.
    4. If no exercise component, flag as unresolvable conflict.
    """
    if not components:
        return MethodResolution(
            resolved_method="analytical",
            resolution_kind="empty",
            reason="No components",
        )

    # Compute intersection
    method_sets = [set(c.compatible_methods) for c in components if c.compatible_methods]
    if not method_sets:
        return MethodResolution(
            resolved_method="analytical",
            resolution_kind="empty",
            reason="No method constraints declared",
        )

    intersection = method_sets[0]
    for ms in method_sets[1:]:
        intersection = intersection & ms

    if intersection:
        # Preference order
        preference = ("analytical", "rate_tree", "monte_carlo", "pde_solver", "fft_pricing")
        for pref in preference:
            if pref in intersection:
                return MethodResolution(
                    resolved_method=pref,
                    resolution_kind="intersection",
                    reason=f"Method intersection: {sorted(intersection)}",
                )
        return MethodResolution(
            resolved_method=sorted(intersection)[0],
            resolution_kind="intersection",
            reason=f"Method intersection: {sorted(intersection)}",
        )

    # No intersection — find dominant component (exercise_policy wins)
    exercise_components = [c for c in components if c.component_type == "exercise_policy"]
    if exercise_components:
        dominant = exercise_components[0]
        dominant_method = dominant.compatible_methods[0] if dominant.compatible_methods else "rate_tree"
        overridden = tuple(
            c.component_id for c in components
            if c.component_id != dominant.component_id and dominant_method not in c.compatible_methods
        )
        return MethodResolution(
            resolved_method=dominant_method,
            resolution_kind="dominance",
            dominant_component=dominant.component_id,
            overridden_components=overridden,
            reason=f"Exercise policy '{dominant.component_id}' dominates; "
                   f"overrides {overridden}",
        )

    return MethodResolution(
        resolved_method="",
        resolution_kind="conflict",
        reason="No method intersection and no exercise_policy component to resolve dominance",
    )


# ---------------------------------------------------------------------------
# CompositeSemanticContract: the molecule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositeSemanticContract:
    """A structured product expressed as a DAG of typed components."""

    composite_id: str
    description: str
    components: tuple[PayoffComponent, ...]
    edges: tuple[CompositionEdge, ...]
    control_boundaries: tuple[ControlBoundary, ...] = ()

    # Derived (computed at construction or by compiler)
    market_data_union: frozenset[str] = frozenset()
    method_resolution: MethodResolution | None = None
    dominant_component: str | None = None

    # Validation
    composite_validators: tuple[str, ...] = ()

    def compute_market_data_union(self) -> frozenset[str]:
        """Union market data requirements across all components."""
        return frozenset().union(*(c.market_data_requirements for c in self.components))

    def compute_method_resolution(self) -> MethodResolution:
        """Resolve method conflicts across all components."""
        return resolve_method_conflicts(self.components)

    def validate_dag(self) -> tuple[str, ...]:
        """Check DAG validity: acyclic, connected, ports wired."""
        errors: list[str] = []
        component_ids = {c.component_id for c in self.components}
        components_by_id = {c.component_id: c for c in self.components}

        # Check all edge endpoints exist
        for edge in self.edges:
            if edge.source not in component_ids:
                errors.append(f"Edge source '{edge.source}' not in components")
            if edge.target not in component_ids:
                errors.append(f"Edge target '{edge.target}' not in components")

        # Check for cycles (simple DFS)
        adj: dict[str, list[str]] = {cid: [] for cid in component_ids}
        for edge in self.edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        visited: set[str] = set()
        in_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj.get(node, []):
                if _has_cycle(neighbor):
                    return True
            in_stack.discard(node)
            return False

        for cid in component_ids:
            if _has_cycle(cid):
                errors.append(f"Cycle detected involving component '{cid}'")
                break

        # Check connectivity (all components reachable from some root)
        if self.edges and component_ids:
            reachable: set[str] = set()
            sources = {e.source for e in self.edges}
            targets = {e.target for e in self.edges}
            roots = sources - targets
            if not roots:
                roots = {next(iter(component_ids))}

            queue = list(roots)
            while queue:
                node = queue.pop()
                if node in reachable:
                    continue
                reachable.add(node)
                queue.extend(adj.get(node, []))

            unreachable = component_ids - reachable
            if unreachable:
                errors.append(f"Disconnected components: {sorted(unreachable)}")

        errors.extend(self.validate_edge_signatures())
        errors.extend(self.validate_control_boundaries())
        return tuple(errors)

    def validate_edge_signatures(self) -> tuple[str, ...]:
        """Check edge-level signature compatibility where the edge type implies it."""
        errors: list[str] = []
        components_by_id = {c.component_id: c for c in self.components}

        for edge in self.edges:
            source = components_by_id.get(edge.source)
            target = components_by_id.get(edge.target)
            if source is None or target is None:
                continue

            source_sig = source.signature
            target_sig = target.signature

            if edge.edge_type in {"sequential", "conditional"}:
                if not source_sig.sequential_compatible(target_sig):
                    errors.append(
                        "Edge signature mismatch "
                        f"({edge.edge_type}) {edge.source}->{edge.target}: "
                        f"{source_sig.outputs} does not feed {target_sig.inputs}"
                    )
                continue

            if edge.edge_type in {"parallel", "override"}:
                if not source_sig.additive_compatible(target_sig):
                    errors.append(
                        "Edge signature mismatch "
                        f"({edge.edge_type}) {edge.source}<->{edge.target}: "
                        f"{source_sig.inputs}->{source_sig.outputs} vs "
                        f"{target_sig.inputs}->{target_sig.outputs}"
                    )

        return tuple(errors)

    def component_atoms(self) -> tuple[ContractAtom, ...]:
        """Return every component as a DSL atom."""
        return tuple(component.to_contract_atom() for component in self.components)

    def validate_control_boundaries(self) -> tuple[str, ...]:
        """Check explicit Bellman/control boundaries for structural validity."""
        errors: list[str] = []
        components_by_id = {c.component_id: c for c in self.components}

        seen_ids: set[str] = set()
        for boundary in self.control_boundaries:
            if boundary.boundary_id in seen_ids:
                errors.append(
                    f"Duplicate control boundary '{boundary.boundary_id}'"
                )
                continue
            seen_ids.add(boundary.boundary_id)

            controller = components_by_id.get(boundary.controller_component)
            if controller is None:
                errors.append(
                    "Control boundary references undefined controller "
                    f"'{boundary.controller_component}'"
                )
                continue
            if controller.component_type != "exercise_policy":
                errors.append(
                    "Control boundary controller must be an exercise_policy component: "
                    f"'{boundary.controller_component}' is '{controller.component_type}'"
                )

            if len(boundary.branches) < 2:
                errors.append(
                    f"Control boundary '{boundary.boundary_id}' must list at least two branches"
                )
                continue

            duplicate_branches = {
                branch_id
                for branch_id in boundary.branches
                if boundary.branches.count(branch_id) > 1
            }
            if duplicate_branches:
                errors.append(
                    f"Control boundary '{boundary.boundary_id}' repeats branches: "
                    f"{sorted(duplicate_branches)}"
                )

            if boundary.controller_component in boundary.branches:
                errors.append(
                    f"Control boundary '{boundary.boundary_id}' must not include the "
                    "controller as a priced branch"
                )

            branch_components: list[PayoffComponent] = []
            for branch_id in boundary.branches:
                branch = components_by_id.get(branch_id)
                if branch is None:
                    errors.append(
                        "Control boundary references undefined branch "
                        f"'{branch_id}'"
                    )
                    continue
                branch_components.append(branch)

            if len(branch_components) < 2:
                continue

            first_signature = branch_components[0].signature
            for branch in branch_components[1:]:
                if not first_signature.additive_compatible(branch.signature):
                    errors.append(
                        "Control boundary branch mismatch "
                        f"('{boundary.boundary_id}'): "
                        f"{branch_components[0].component_id} "
                        f"{first_signature.inputs}->{first_signature.outputs} vs "
                        f"{branch.component_id} "
                        f"{branch.signature.inputs}->{branch.signature.outputs}"
                    )

        return tuple(errors)

    def control_boundary_expr(self, boundary_id: str) -> ContractExpr:
        """Lower one explicit control boundary onto the DSL Bellman layer."""
        boundary = next(
            (item for item in self.control_boundaries if item.boundary_id == boundary_id),
            None,
        )
        if boundary is None:
            raise KeyError(f"Unknown control boundary: {boundary_id}")

        errors = tuple(
            error
            for error in self.validate_control_boundaries()
            if f"'{boundary_id}'" in error or boundary.controller_component in error
        )
        if errors:
            raise ValueError("; ".join(errors))

        components_by_id = {c.component_id: c for c in self.components}
        expr = ChoiceExpr(
            style=boundary.style,
            branches=tuple(
                components_by_id[branch_id].to_contract_atom()
                for branch_id in boundary.branches
            ),
            label=boundary.label or boundary.boundary_id,
        )
        return normalize_contract_expr(expr)

    def collect_control_styles(self) -> tuple[ControlStyle, ...]:
        """Return the distinct explicit control styles declared on this composite."""
        styles: list[ControlStyle] = []
        seen: set[ControlStyle] = set()
        for boundary in self.control_boundaries:
            if boundary.style in seen:
                continue
            seen.add(boundary.style)
            styles.append(boundary.style)
        return tuple(styles)

    def proven_components(self) -> tuple[PayoffComponent, ...]:
        """Return components with existing reusable primitives."""
        return tuple(c for c in self.components if c.proven_primitive is not None)

    def generation_required_components(self) -> tuple[PayoffComponent, ...]:
        """Return components that need LLM code generation."""
        return tuple(c for c in self.components if c.proven_primitive is None)


# ---------------------------------------------------------------------------
# CalibrationContract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationAcceptanceCriteria:
    """When is calibration good enough?"""

    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    stability_check: bool = True
    max_fitting_error_bps: float = 5.0


@dataclass(frozen=True)
class CalibrationTarget:
    """What parameter to calibrate."""

    parameter: str         # "hw_mean_reversion", "sabr_alpha", "local_vol_surface"
    output_capability: str  # MarketState capability name


@dataclass(frozen=True)
class CalibrationContract:
    """Typed calibration step executed before pricing.

    Output binding: calibrated parameters are materialized as MarketState
    capabilities (e.g., ``market_state.hw_short_rate_params``).
    """

    calibration_id: str
    target: CalibrationTarget
    fitting_instruments: tuple[str, ...]
    optimizer: str  # "analytical", "least_squares", "differential_evolution"
    acceptance_criteria: CalibrationAcceptanceCriteria = CalibrationAcceptanceCriteria()
    output_binding: str = ""  # MarketState capability name
    proven_primitive: str | None = None  # existing calibration module.symbol
    description: str = ""
