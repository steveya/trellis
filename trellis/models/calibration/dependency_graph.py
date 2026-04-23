"""Generic calibration dependency graph primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from types import MappingProxyType
from typing import Mapping, Sequence


def _normalize_str(value: object) -> str:
    """Return a stripped string representation."""
    if value is None:
        return ""
    return str(value).strip()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy for user metadata."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_str_tuple(values: Sequence[object] | None) -> tuple[str, ...]:
    """Return a stable, non-empty tuple of stripped strings."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    normalized: list[str] = []
    for value in values:
        normalized_value = _normalize_str(value)
        if normalized_value:
            normalized.append(normalized_value)
    return tuple(normalized)


def _normalize_edge(edge: Sequence[object]) -> tuple[str, str]:
    """Return a normalized dependency edge ``(source, target)``."""
    if len(edge) != 2:
        raise ValueError("Dependency edges must contain exactly two node ids")
    source, target = edge
    normalized_source = _normalize_str(source)
    normalized_target = _normalize_str(target)
    if not normalized_source or not normalized_target:
        raise ValueError("Dependency edges require non-empty source and target node ids")
    return normalized_source, normalized_target


@dataclass(frozen=True)
class CalibrationDependencyNode:
    """Typed calibration workflow node metadata."""

    node_id: str
    object_kind: str
    object_name: str
    source_ref: str = ""
    required: bool = True
    depends_on: tuple[str, ...] = ()
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_id = _normalize_str(self.node_id)
        object_kind = _normalize_str(self.object_kind)
        object_name = _normalize_str(self.object_name)
        source_ref = _normalize_str(self.source_ref)
        description = _normalize_str(self.description)
        depends_on = _normalize_str_tuple(self.depends_on)

        if not node_id:
            raise ValueError("CalibrationDependencyNode requires a non-empty node_id")
        if not object_kind:
            raise ValueError(f"CalibrationDependencyNode {node_id!r} requires a non-empty object_kind")
        if not object_name:
            raise ValueError(f"CalibrationDependencyNode {node_id!r} requires a non-empty object_name")

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "object_kind", object_kind)
        object.__setattr__(self, "object_name", object_name)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "depends_on", depends_on)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload for workflow assembly."""
        return {
            "node_id": self.node_id,
            "object_kind": self.object_kind,
            "object_name": self.object_name,
            "source_ref": self.source_ref,
            "required": bool(self.required),
            "depends_on": list(self.depends_on),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


class CalibrationDependencyGraphError(ValueError):
    """Base error for calibration dependency graph validation."""


class DuplicateCalibrationDependencyNodeError(CalibrationDependencyGraphError):
    """Raised when a workflow reuses the same node_id more than once."""


class MissingCalibrationDependencyNodeError(CalibrationDependencyGraphError):
    """Raised when a dependency edge or required link references a missing node."""


class CalibrationDependencyCycleError(CalibrationDependencyGraphError):
    """Raised when the workflow contains a dependency cycle."""


@dataclass(frozen=True)
class CalibrationDependencyGraph:
    """Generic dependency graph for calibration workflows.

    Edge direction is ``(source, target)`` where ``source`` depends on
    ``target`` and therefore ``target`` must appear first in topological order.
    """

    workflow_id: str
    nodes: tuple[CalibrationDependencyNode, ...]
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        workflow_id = _normalize_str(self.workflow_id)
        if not workflow_id:
            raise ValueError("CalibrationDependencyGraph requires a non-empty workflow_id")
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(_normalize_edge(edge) for edge in self.edges))
        self.validate()

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return the node ids in input order."""
        return tuple(node.node_id for node in self.nodes)

    def dependency_order(self) -> tuple[str, ...]:
        """Return a dependency-first node ordering."""
        return self._topological_order_ids()

    @property
    def topological_order(self) -> tuple[CalibrationDependencyNode, ...]:
        """Return the nodes in dependency-first order."""
        node_lookup = self._node_lookup()
        return tuple(node_lookup[node_id] for node_id in self._topological_order_ids())

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload for workflow assembly."""
        return {
            "workflow_id": self.workflow_id,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [list(edge) for edge in self._dependency_pairs()],
            "dependency_order": list(self.dependency_order()),
        }

    def validate(self) -> None:
        """Validate node uniqueness, edge references, and cycle freedom."""
        self._validate_node_ids()
        node_lookup = self._node_lookup()
        missing_messages = self._missing_reference_messages(node_lookup)
        if missing_messages:
            raise MissingCalibrationDependencyNodeError(
                f"CalibrationDependencyGraph workflow {self.workflow_id!r} has unresolved dependency reference(s): "
                + "; ".join(missing_messages)
            )
        cycle = self.detect_cycle()
        if cycle is not None:
            cycle_text = " -> ".join(cycle)
            raise CalibrationDependencyCycleError(
                f"CalibrationDependencyGraph workflow {self.workflow_id!r} contains a cycle: {cycle_text}"
            )

    def detect_cycle(self) -> tuple[str, ...] | None:
        """Return one cycle path if the graph is cyclic, otherwise ``None``."""
        node_lookup = self._node_lookup()
        adjacency = self._dependency_adjacency(node_lookup)
        node_position = self._node_position()
        visited: set[str] = set()
        active: set[str] = set()
        stack: list[str] = []

        def visit(node_id: str) -> tuple[str, ...] | None:
            visited.add(node_id)
            active.add(node_id)
            stack.append(node_id)
            for dependent in sorted(adjacency.get(node_id, ()), key=node_position.__getitem__):
                if dependent not in visited:
                    cycle = visit(dependent)
                    if cycle is not None:
                        return cycle
                elif dependent in active:
                    start_index = stack.index(dependent)
                    return tuple(stack[start_index:] + [dependent])
            stack.pop()
            active.remove(node_id)
            return None

        for node_id in self.node_ids:
            if node_id not in visited:
                cycle = visit(node_id)
                if cycle is not None:
                    return cycle
        return None

    def _validate_node_ids(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for node in self.nodes:
            if node.node_id in seen and node.node_id not in duplicates:
                duplicates.append(node.node_id)
            seen.add(node.node_id)
        if duplicates:
            duplicate_text = ", ".join(repr(node_id) for node_id in duplicates)
            raise DuplicateCalibrationDependencyNodeError(
                f"CalibrationDependencyGraph workflow {self.workflow_id!r} contains duplicate node_id(s): {duplicate_text}"
            )

    def _node_lookup(self) -> dict[str, CalibrationDependencyNode]:
        return {node.node_id: node for node in self.nodes}

    def _node_position(self) -> dict[str, int]:
        return {node.node_id: position for position, node in enumerate(self.nodes)}

    def _missing_reference_messages(self, node_lookup: Mapping[str, CalibrationDependencyNode]) -> list[str]:
        node_ids = set(node_lookup)
        messages: list[str] = []

        for source, target in self._dependency_pairs():
            missing_parts: list[str] = []
            if source not in node_ids:
                missing_parts.append(f"missing source {source!r}")
            if target not in node_ids:
                missing_parts.append(f"missing target {target!r}")
            if missing_parts:
                messages.append(f"edge {source!r} -> {target!r} has {', '.join(missing_parts)}")

        for node in self.nodes:
            missing_dependencies = [
                dependency for dependency in node.depends_on if dependency not in node_ids
            ]
            if missing_dependencies:
                missing_text = ", ".join(repr(dependency) for dependency in missing_dependencies)
                messages.append(f"node {node.node_id!r} depends on missing node(s): {missing_text}")

        return messages

    def _dependency_adjacency(self, node_lookup: Mapping[str, CalibrationDependencyNode]) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_lookup}
        for source, target in self._dependency_pairs():
            if source in node_lookup and target in node_lookup:
                adjacency[target].add(source)
        return adjacency

    def _dependency_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = list(self.edges)
        for node in self.nodes:
            for target in node.depends_on:
                pairs.append((node.node_id, target))
        return tuple(dict.fromkeys(pairs))

    def _topological_order_ids(self) -> tuple[str, ...]:
        node_lookup = self._node_lookup()
        adjacency = self._dependency_adjacency(node_lookup)
        node_position = self._node_position()
        indegree: dict[str, int] = {node_id: 0 for node_id in node_lookup}

        for source, target in self._dependency_pairs():
            if source in node_lookup and target in node_lookup:
                indegree[source] += 1

        zero_indegree: list[tuple[int, str]] = []
        for node_id, count in indegree.items():
            if count == 0:
                heappush(zero_indegree, (node_position[node_id], node_id))

        ordered: list[str] = []
        working_indegree = dict(indegree)
        while zero_indegree:
            _, node_id = heappop(zero_indegree)
            ordered.append(node_id)
            for dependent in sorted(adjacency.get(node_id, ()), key=node_position.__getitem__):
                working_indegree[dependent] -= 1
                if working_indegree[dependent] == 0:
                    heappush(zero_indegree, (node_position[dependent], dependent))

        if len(ordered) != len(node_lookup):
            cycle = self.detect_cycle()
            cycle_text = " -> ".join(cycle) if cycle is not None else "unknown cycle"
            raise CalibrationDependencyCycleError(
                f"CalibrationDependencyGraph workflow {self.workflow_id!r} contains a cycle: {cycle_text}"
            )
        return tuple(ordered)


__all__ = [
    "CalibrationDependencyCycleError",
    "CalibrationDependencyGraph",
    "CalibrationDependencyGraphError",
    "CalibrationDependencyNode",
    "DuplicateCalibrationDependencyNodeError",
    "MissingCalibrationDependencyNodeError",
]
