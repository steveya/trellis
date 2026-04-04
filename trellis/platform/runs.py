"""Canonical run-ledger records and local persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from trellis.platform.context import ExecutionContext


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_LEDGER_ROOT = _REPO_ROOT / ".trellis_state" / "runs"
_RUN_SCHEMA_VERSION = 1


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable tuple of unique strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class ArtifactReference:
    """Canonical reference to one persisted run artifact."""

    artifact_id: str
    artifact_kind: str
    uri: str
    role: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize artifact metadata into immutable primitives."""
        object.__setattr__(self, "artifact_id", str(self.artifact_id or "").strip())
        object.__setattr__(self, "artifact_kind", str(self.artifact_kind or "").strip())
        object.__setattr__(self, "uri", str(self.uri or "").strip())
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "uri": self.uri,
            "role": self.role,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArtifactReference:
        """Rehydrate one artifact reference."""
        return cls(
            artifact_id=str(payload.get("artifact_id", "")).strip(),
            artifact_kind=str(payload.get("artifact_kind", "")).strip(),
            uri=str(payload.get("uri", "")).strip(),
            role=str(payload.get("role", "")).strip(),
            metadata=payload.get("metadata") or {},
        )


def _coerce_artifact_refs(values) -> tuple[ArtifactReference, ...]:
    """Convert dict payloads into artifact-reference records."""
    refs = []
    for value in values or ():
        refs.append(value if isinstance(value, ArtifactReference) else ArtifactReference.from_dict(value))
    deduped: dict[tuple[str, str], ArtifactReference] = {
        (item.artifact_kind, item.uri): item for item in refs if item.uri
    }
    return tuple(sorted(deduped.values(), key=lambda item: (item.artifact_kind, item.uri)))


@dataclass(frozen=True)
class RunRecord:
    """Canonical governed run record."""

    run_id: str
    request_id: str
    status: str
    action: str
    run_mode: str
    session_id: str
    policy_id: str
    trade_identity: Mapping[str, object] = field(default_factory=dict)
    selected_model: Mapping[str, object] = field(default_factory=dict)
    selected_engine: Mapping[str, object] = field(default_factory=dict)
    provider_bindings: Mapping[str, object] = field(default_factory=dict)
    market_snapshot_id: str = ""
    valuation_timestamp: str = ""
    warnings: tuple[str, ...] = ()
    result_summary: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    validation_summary: Mapping[str, object] = field(default_factory=dict)
    policy_outcome: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[ArtifactReference, ...] = ()
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: int = _RUN_SCHEMA_VERSION

    def __post_init__(self):
        """Normalize nested run metadata into immutable primitives."""
        object.__setattr__(self, "run_id", str(self.run_id or "").strip())
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())
        object.__setattr__(self, "status", str(self.status or "").strip())
        object.__setattr__(self, "action", str(self.action or "").strip())
        object.__setattr__(self, "run_mode", str(self.run_mode or "").strip())
        object.__setattr__(self, "session_id", str(self.session_id or "").strip())
        object.__setattr__(self, "policy_id", str(self.policy_id or "").strip())
        object.__setattr__(self, "trade_identity", _freeze_mapping(self.trade_identity))
        object.__setattr__(self, "selected_model", _freeze_mapping(self.selected_model))
        object.__setattr__(self, "selected_engine", _freeze_mapping(self.selected_engine))
        object.__setattr__(self, "provider_bindings", _freeze_mapping(self.provider_bindings))
        object.__setattr__(self, "market_snapshot_id", str(self.market_snapshot_id or "").strip())
        object.__setattr__(self, "valuation_timestamp", str(self.valuation_timestamp or "").strip())
        object.__setattr__(self, "warnings", _string_tuple(self.warnings))
        object.__setattr__(self, "result_summary", _freeze_mapping(self.result_summary))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "validation_summary", _freeze_mapping(self.validation_summary))
        object.__setattr__(self, "policy_outcome", _freeze_mapping(self.policy_outcome))
        object.__setattr__(self, "artifacts", _coerce_artifact_refs(self.artifacts))
        object.__setattr__(self, "created_at", str(self.created_at or _utc_now()).strip())
        object.__setattr__(self, "updated_at", str(self.updated_at or self.created_at).strip())

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        """Return the ordered artifact uris for convenience projections."""
        return tuple(item.uri for item in self.artifacts)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "status": self.status,
            "action": self.action,
            "run_mode": self.run_mode,
            "session_id": self.session_id,
            "policy_id": self.policy_id,
            "trade_identity": dict(self.trade_identity),
            "selected_model": dict(self.selected_model),
            "selected_engine": dict(self.selected_engine),
            "provider_bindings": dict(self.provider_bindings),
            "market_snapshot_id": self.market_snapshot_id,
            "valuation_timestamp": self.valuation_timestamp,
            "warnings": list(self.warnings),
            "result_summary": dict(self.result_summary),
            "provenance": dict(self.provenance),
            "validation_summary": dict(self.validation_summary),
            "policy_outcome": dict(self.policy_outcome),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "artifact_paths": list(self.artifact_paths),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunRecord:
        """Rehydrate one run record."""
        return cls(
            schema_version=int(payload.get("schema_version", _RUN_SCHEMA_VERSION)),
            run_id=str(payload.get("run_id", "")).strip(),
            request_id=str(payload.get("request_id", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            action=str(payload.get("action", "")).strip(),
            run_mode=str(payload.get("run_mode", "")).strip(),
            session_id=str(payload.get("session_id", "")).strip(),
            policy_id=str(payload.get("policy_id", "")).strip(),
            trade_identity=payload.get("trade_identity") or {},
            selected_model=payload.get("selected_model") or {},
            selected_engine=payload.get("selected_engine") or {},
            provider_bindings=payload.get("provider_bindings") or {},
            market_snapshot_id=str(payload.get("market_snapshot_id", "")).strip(),
            valuation_timestamp=str(payload.get("valuation_timestamp", "")).strip(),
            warnings=payload.get("warnings") or (),
            result_summary=payload.get("result_summary") or {},
            provenance=payload.get("provenance") or {},
            validation_summary=payload.get("validation_summary") or {},
            policy_outcome=payload.get("policy_outcome") or {},
            artifacts=payload.get("artifacts") or (),
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
        )


def build_run_record(
    *,
    run_id: str,
    request_id: str,
    status: str,
    action: str,
    execution_context: ExecutionContext,
    trade_identity: Mapping[str, object] | None = None,
    selected_model: Mapping[str, object] | None = None,
    selected_engine: Mapping[str, object] | None = None,
    market_snapshot_id: str = "",
    valuation_timestamp: str = "",
    warnings=(),
    result_summary: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
    validation_summary: Mapping[str, object] | None = None,
    policy_outcome: Mapping[str, object] | None = None,
    artifacts=(),
) -> RunRecord:
    """Build one canonical run record from execution context plus run outputs."""
    return RunRecord(
        run_id=run_id,
        request_id=request_id,
        status=status,
        action=action,
        run_mode=execution_context.run_mode.value,
        session_id=execution_context.session_id,
        policy_id=execution_context.policy_bundle_id,
        trade_identity=trade_identity or {},
        selected_model=selected_model or {},
        selected_engine=selected_engine or {},
        provider_bindings=execution_context.provider_bindings.to_dict(),
        market_snapshot_id=market_snapshot_id,
        valuation_timestamp=valuation_timestamp,
        warnings=warnings,
        result_summary=result_summary or {},
        provenance=provenance or {},
        validation_summary=validation_summary or {},
        policy_outcome=policy_outcome or {},
        artifacts=artifacts,
    )


class RunLedgerStore:
    """Local-first persistence for canonical governed run records."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else _RUN_LEDGER_ROOT

    def _run_path(self, run_id: str) -> Path:
        return self.base_dir / f"{run_id}.json"

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text())

    def create_run(self, record: RunRecord) -> RunRecord:
        """Persist one new canonical run record."""
        path = self._run_path(record.run_id)
        if path.exists():
            raise FileExistsError(f"Run already exists: {record.run_id}")
        self._write_json(path, record.to_dict())
        return self.get_run(record.run_id) or record

    def save_run(self, record: RunRecord) -> RunRecord:
        """Upsert one canonical run record."""
        self._write_json(self._run_path(record.run_id), record.to_dict())
        return self.get_run(record.run_id) or record

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load one canonical run record."""
        path = self._run_path(run_id)
        if not path.exists():
            return None
        return RunRecord.from_dict(self._read_json(path))

    def list_runs(self) -> list[RunRecord]:
        """Return all persisted run records."""
        if not self.base_dir.exists():
            return []
        return [
            RunRecord.from_dict(self._read_json(path))
            for path in sorted(self.base_dir.glob("*.json"))
        ]

    def build_audit_bundle(self, run_id: str):
        """Build the canonical governed audit bundle for one persisted run."""
        from trellis.platform.audits import build_run_audit_bundle

        record = self.get_run(run_id)
        if record is None:
            raise FileNotFoundError(f"Unknown run id: {run_id}")
        return build_run_audit_bundle(record)

    def attach_artifacts(self, run_id: str, artifacts) -> RunRecord:
        """Attach artifact references to one persisted run record."""
        record = self.get_run(run_id)
        if record is None:
            raise FileNotFoundError(f"Unknown run id: {run_id}")
        updated = replace(
            record,
            artifacts=record.artifacts + _coerce_artifact_refs(artifacts),
            updated_at=_utc_now(),
        )
        return self.save_run(updated)


def legacy_artifact_refs_from_paths(
    *,
    platform_trace_path: str | None = None,
    model_audit_path: str | None = None,
    task_run_history_path: str | None = None,
    task_run_latest_path: str | None = None,
    diagnosis_packet_path: str | None = None,
    diagnosis_dossier_path: str | None = None,
) -> tuple[ArtifactReference, ...]:
    """Normalize current repo artifact paths into canonical run-ledger refs."""
    refs = []
    if diagnosis_dossier_path:
        refs.append(
            ArtifactReference(
                artifact_id="diagnosis_dossier",
                artifact_kind="diagnosis_dossier",
                uri=diagnosis_dossier_path,
            )
        )
    if diagnosis_packet_path:
        refs.append(
            ArtifactReference(
                artifact_id="diagnosis_packet",
                artifact_kind="diagnosis_packet",
                uri=diagnosis_packet_path,
            )
        )
    if model_audit_path:
        refs.append(
            ArtifactReference(
                artifact_id="model_audit",
                artifact_kind="model_audit",
                uri=model_audit_path,
            )
        )
    if platform_trace_path:
        refs.append(
            ArtifactReference(
                artifact_id="platform_trace",
                artifact_kind="platform_trace",
                uri=platform_trace_path,
            )
        )
    if task_run_history_path:
        refs.append(
            ArtifactReference(
                artifact_id="task_run_history",
                artifact_kind="task_run_history",
                uri=task_run_history_path,
            )
        )
    if task_run_latest_path:
        refs.append(
            ArtifactReference(
                artifact_id="task_run_latest",
                artifact_kind="task_run_latest",
                uri=task_run_latest_path,
            )
        )
    return _coerce_artifact_refs(refs)
