"""Small MCP-facing schema helpers for the Trellis server shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class ToolDefinition:
    """One transport-neutral MCP tool definition."""

    name: str
    description: str
    input_schema: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "input_schema", _freeze_mapping(self.input_schema))


__all__ = [
    "ToolDefinition",
]
