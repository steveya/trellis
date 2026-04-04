"""Structured errors for the thin Trellis MCP adapter layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class TrellisMcpError(RuntimeError):
    """Stable structured MCP-side error for tool/resource/prompt adapters."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "code", str(self.code or "").strip())
        object.__setattr__(self, "message", str(self.message or "").strip())
        object.__setattr__(self, "details", _freeze_mapping(self.details))
        RuntimeError.__init__(self, self.message)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


__all__ = [
    "TrellisMcpError",
]
