"""Canonical governed execution result envelope."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from trellis.platform.runs import ArtifactReference


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable tuple of unique strings."""
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _coerce_artifacts(values) -> tuple[ArtifactReference, ...]:
    """Normalize nested artifact payloads into canonical references."""
    if not values:
        return ()
    return tuple(
        item if isinstance(item, ArtifactReference) else ArtifactReference.from_dict(item)
        for item in values
    )


@dataclass(frozen=True)
class ExecutionResult:
    """Stable internal result envelope for governed execution."""

    run_id: str
    request_id: str
    status: str
    action: str
    output_mode: str
    result_payload: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[ArtifactReference, ...] = ()
    audit_summary: Mapping[str, object] = field(default_factory=dict)
    trace_path: str = ""
    policy_outcome: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize nested fields into stable immutable primitives."""
        object.__setattr__(self, "run_id", str(self.run_id or "").strip())
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())
        object.__setattr__(self, "status", str(self.status or "").strip())
        object.__setattr__(self, "action", str(self.action or "").strip())
        object.__setattr__(self, "output_mode", str(self.output_mode or "").strip())
        object.__setattr__(self, "result_payload", _freeze_mapping(self.result_payload))
        object.__setattr__(self, "warnings", _string_tuple(self.warnings))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "artifacts", _coerce_artifacts(self.artifacts))
        object.__setattr__(self, "audit_summary", _freeze_mapping(self.audit_summary))
        object.__setattr__(self, "trace_path", str(self.trace_path or "").strip())
        object.__setattr__(self, "policy_outcome", _freeze_mapping(self.policy_outcome))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "status": self.status,
            "action": self.action,
            "output_mode": self.output_mode,
            "result_payload": dict(self.result_payload),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "audit_summary": dict(self.audit_summary),
            "trace_path": self.trace_path,
            "policy_outcome": dict(self.policy_outcome),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExecutionResult:
        """Rehydrate one execution result."""
        return cls(
            run_id=str(payload.get("run_id", "")).strip(),
            request_id=str(payload.get("request_id", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            action=str(payload.get("action", "")).strip(),
            output_mode=str(payload.get("output_mode", "")).strip(),
            result_payload=payload.get("result_payload") or {},
            warnings=payload.get("warnings") or (),
            provenance=payload.get("provenance") or {},
            artifacts=payload.get("artifacts") or (),
            audit_summary=payload.get("audit_summary") or {},
            trace_path=str(payload.get("trace_path", "")).strip(),
            policy_outcome=payload.get("policy_outcome") or {},
        )


def execution_result_exception(
    result: ExecutionResult,
    *,
    default_message: str | None = None,
) -> Exception:
    """Map one failed or blocked execution result back onto a Python exception."""
    if result.status == "blocked":
        message = (
            default_message
            or str(result.result_payload.get("error") or "").strip()
            or str(result.result_payload.get("reason") or "").strip()
            or "Governed execution was blocked."
        )
        return RuntimeError(message)

    error_type = str(result.result_payload.get("error_type") or "").strip()
    message = (
        default_message
        or str(result.result_payload.get("error") or "").strip()
        or str(result.result_payload.get("reason") or "").strip()
        or "Governed execution failed."
    )
    exc_cls = getattr(builtins, error_type, None)
    if isinstance(exc_cls, type) and issubclass(exc_cls, Exception):
        return exc_cls(message)
    return RuntimeError(message)


def project_execution_result_value(
    result: ExecutionResult,
    *,
    key: str = "result",
    default_message: str | None = None,
):
    """Project a successful execution result into the requested payload value."""
    if result.status != "succeeded":
        raise execution_result_exception(result, default_message=default_message)
    if not key:
        return result.result_payload
    if key not in result.result_payload:
        raise KeyError(
            f"ExecutionResult payload has no {key!r}. "
            f"Available: {sorted(result.result_payload)}"
        )
    return result.result_payload[key]


def execution_result_trace_details(result: ExecutionResult) -> dict[str, object]:
    """Return compact trace-friendly metadata for one execution result."""
    details: dict[str, object] = {
        "run_id": result.run_id,
        "action": result.action,
        "status": result.status,
    }
    if result.warnings:
        details["warnings"] = list(result.warnings)
    if result.trace_path:
        details["trace_path"] = result.trace_path
    return details


def is_pending_execution_result(result: ExecutionResult) -> bool:
    """Return whether one result represents a temporary pending-adapter envelope."""
    return str(result.result_payload.get("reason") or "").strip() == "route_adapter_pending"
