"""Governed model-registry records and local persistence."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_REGISTRY_ROOT = _REPO_ROOT / ".trellis_state" / "models"
_MODEL_SCHEMA_VERSION = 1
_VERSION_ARTIFACT_FILES: dict[str, tuple[str, str]] = {
    "contract": ("contract.json", "json"),
    "code": ("implementation.py", "text"),
    "methodology": ("methodology.json", "json"),
    "validation-plan": ("validation-plan.json", "json"),
    "validation-report": ("validation-report.json", "json"),
    "lineage": ("lineage.json", "json"),
}


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_token(value: str | None, *, fallback: str = "") -> str:
    """Normalize one identifier token."""
    text = str(value or "").strip()
    return text or fallback


def _string_tuple(values, *, sort_values: bool = False) -> tuple[str, ...]:
    """Return a stable tuple of unique strings."""
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    if sort_values:
        items.sort()
    return tuple(items)


class InvalidLifecycleTransitionError(ValueError):
    """Raised when one model-version lifecycle transition is invalid."""


class ModelLifecycleStatus(str, Enum):
    """Governed model lifecycle states."""

    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"
    DEPRECATED = "deprecated"

    @classmethod
    def normalize(cls, value: ModelLifecycleStatus | str | None) -> ModelLifecycleStatus:
        """Coerce one lifecycle-status value into the canonical enum."""
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower()
        for member in cls:
            if member.value == token:
                return member
        raise ValueError(f"Unknown lifecycle status: {value!r}")


_TRANSITION_RULES: dict[ModelLifecycleStatus, frozenset[ModelLifecycleStatus]] = {
    ModelLifecycleStatus.DRAFT: frozenset(
        {ModelLifecycleStatus.VALIDATED, ModelLifecycleStatus.DEPRECATED}
    ),
    ModelLifecycleStatus.VALIDATED: frozenset(
        {ModelLifecycleStatus.APPROVED, ModelLifecycleStatus.DEPRECATED}
    ),
    ModelLifecycleStatus.APPROVED: frozenset({ModelLifecycleStatus.DEPRECATED}),
    ModelLifecycleStatus.DEPRECATED: frozenset(),
}


@dataclass(frozen=True)
class ModelLineage:
    """Lineage metadata for one governed model version."""

    parent_model_id: str = ""
    parent_version: str = ""
    source_run_id: str = ""
    source_request_id: str = ""
    source_audit_id: str = ""
    source_audit_path: str = ""
    derived_from: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize lineage identifiers into immutable primitives."""
        object.__setattr__(self, "parent_model_id", _normalize_token(self.parent_model_id))
        object.__setattr__(self, "parent_version", _normalize_token(self.parent_version))
        object.__setattr__(self, "source_run_id", _normalize_token(self.source_run_id))
        object.__setattr__(self, "source_request_id", _normalize_token(self.source_request_id))
        object.__setattr__(self, "source_audit_id", _normalize_token(self.source_audit_id))
        object.__setattr__(self, "source_audit_path", _normalize_token(self.source_audit_path))
        object.__setattr__(self, "derived_from", _string_tuple(self.derived_from))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "parent_model_id": self.parent_model_id,
            "parent_version": self.parent_version,
            "source_run_id": self.source_run_id,
            "source_request_id": self.source_request_id,
            "source_audit_id": self.source_audit_id,
            "source_audit_path": self.source_audit_path,
            "derived_from": list(self.derived_from),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelLineage:
        """Rehydrate one lineage payload."""
        return cls(
            parent_model_id=str(payload.get("parent_model_id", "")).strip(),
            parent_version=str(payload.get("parent_version", "")).strip(),
            source_run_id=str(payload.get("source_run_id", "")).strip(),
            source_request_id=str(payload.get("source_request_id", "")).strip(),
            source_audit_id=str(payload.get("source_audit_id", "")).strip(),
            source_audit_path=str(payload.get("source_audit_path", "")).strip(),
            derived_from=payload.get("derived_from") or (),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ModelLifecycleTransition:
    """One auditable lifecycle transition for a model version."""

    transition_id: str
    from_status: ModelLifecycleStatus | None
    to_status: ModelLifecycleStatus
    changed_at: str
    changed_by: str
    reason: str
    notes: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize transition metadata into immutable primitives."""
        object.__setattr__(self, "from_status", None if self.from_status is None else ModelLifecycleStatus.normalize(self.from_status))
        object.__setattr__(self, "to_status", ModelLifecycleStatus.normalize(self.to_status))
        object.__setattr__(self, "transition_id", _normalize_token(self.transition_id))
        object.__setattr__(self, "changed_at", _normalize_token(self.changed_at, fallback=_utc_now()))
        object.__setattr__(self, "changed_by", _normalize_token(self.changed_by))
        object.__setattr__(self, "reason", _normalize_token(self.reason))
        object.__setattr__(self, "notes", str(self.notes or "").strip())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "transition_id": self.transition_id,
            "from_status": None if self.from_status is None else self.from_status.value,
            "to_status": self.to_status.value,
            "changed_at": self.changed_at,
            "changed_by": self.changed_by,
            "reason": self.reason,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelLifecycleTransition:
        """Rehydrate one lifecycle-transition payload."""
        from_status = payload.get("from_status")
        return cls(
            transition_id=str(payload.get("transition_id", "")).strip(),
            from_status=None if from_status in {None, ""} else from_status,
            to_status=payload.get("to_status") or ModelLifecycleStatus.DRAFT.value,
            changed_at=str(payload.get("changed_at", "")).strip(),
            changed_by=str(payload.get("changed_by", "")).strip(),
            reason=str(payload.get("reason", "")).strip(),
            notes=str(payload.get("notes", "")).strip(),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ModelRecord:
    """Persistent model identity and typed match basis."""

    model_id: str
    semantic_id: str
    semantic_version: str
    product_family: str
    instrument_class: str = ""
    payoff_family: str = ""
    exercise_style: str = ""
    underlier_structure: str = ""
    payout_currency: str = ""
    reporting_currency: str = ""
    required_market_data: tuple[str, ...] = ()
    supported_method_families: tuple[str, ...] = ()
    status: ModelLifecycleStatus = ModelLifecycleStatus.DRAFT
    latest_version: str = ""
    latest_validated_version: str = ""
    latest_approved_version: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = _MODEL_SCHEMA_VERSION

    def __post_init__(self):
        """Normalize typed match-basis fields."""
        object.__setattr__(self, "model_id", _normalize_token(self.model_id))
        object.__setattr__(self, "semantic_id", _normalize_token(self.semantic_id))
        object.__setattr__(self, "semantic_version", _normalize_token(self.semantic_version))
        object.__setattr__(self, "product_family", _normalize_token(self.product_family))
        object.__setattr__(self, "instrument_class", _normalize_token(self.instrument_class))
        object.__setattr__(self, "payoff_family", _normalize_token(self.payoff_family))
        object.__setattr__(self, "exercise_style", _normalize_token(self.exercise_style))
        object.__setattr__(self, "underlier_structure", _normalize_token(self.underlier_structure))
        object.__setattr__(self, "payout_currency", _normalize_token(self.payout_currency))
        object.__setattr__(self, "reporting_currency", _normalize_token(self.reporting_currency))
        object.__setattr__(self, "required_market_data", _string_tuple(self.required_market_data, sort_values=True))
        object.__setattr__(self, "supported_method_families", _string_tuple(self.supported_method_families, sort_values=True))
        object.__setattr__(self, "status", ModelLifecycleStatus.normalize(self.status))
        object.__setattr__(self, "latest_version", _normalize_token(self.latest_version))
        object.__setattr__(self, "latest_validated_version", _normalize_token(self.latest_validated_version))
        object.__setattr__(self, "latest_approved_version", _normalize_token(self.latest_approved_version))
        object.__setattr__(self, "created_at", _normalize_token(self.created_at, fallback=_utc_now()))
        object.__setattr__(self, "updated_at", _normalize_token(self.updated_at, fallback=self.created_at))
        object.__setattr__(self, "tags", _string_tuple(self.tags, sort_values=True))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def match_basis(self) -> dict[str, object]:
        """Return the typed match basis used by later selection work."""
        return {
            "semantic_id": self.semantic_id,
            "semantic_version": self.semantic_version,
            "product_family": self.product_family,
            "instrument_class": self.instrument_class,
            "payoff_family": self.payoff_family,
            "exercise_style": self.exercise_style,
            "underlier_structure": self.underlier_structure,
            "payout_currency": self.payout_currency,
            "reporting_currency": self.reporting_currency,
            "required_market_data": self.required_market_data,
            "supported_method_families": self.supported_method_families,
            "status": self.status.value,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "semantic_id": self.semantic_id,
            "semantic_version": self.semantic_version,
            "product_family": self.product_family,
            "instrument_class": self.instrument_class,
            "payoff_family": self.payoff_family,
            "exercise_style": self.exercise_style,
            "underlier_structure": self.underlier_structure,
            "payout_currency": self.payout_currency,
            "reporting_currency": self.reporting_currency,
            "required_market_data": list(self.required_market_data),
            "supported_method_families": list(self.supported_method_families),
            "status": self.status.value,
            "latest_version": self.latest_version,
            "latest_validated_version": self.latest_validated_version,
            "latest_approved_version": self.latest_approved_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelRecord:
        """Rehydrate one model record."""
        return cls(
            schema_version=int(payload.get("schema_version", _MODEL_SCHEMA_VERSION)),
            model_id=str(payload.get("model_id", "")).strip(),
            semantic_id=str(payload.get("semantic_id", "")).strip(),
            semantic_version=str(payload.get("semantic_version", "")).strip(),
            product_family=str(payload.get("product_family", "")).strip(),
            instrument_class=str(payload.get("instrument_class", "")).strip(),
            payoff_family=str(payload.get("payoff_family", "")).strip(),
            exercise_style=str(payload.get("exercise_style", "")).strip(),
            underlier_structure=str(payload.get("underlier_structure", "")).strip(),
            payout_currency=str(payload.get("payout_currency", "")).strip(),
            reporting_currency=str(payload.get("reporting_currency", "")).strip(),
            required_market_data=payload.get("required_market_data") or (),
            supported_method_families=payload.get("supported_method_families") or (),
            status=payload.get("status") or ModelLifecycleStatus.DRAFT.value,
            latest_version=str(payload.get("latest_version", "")).strip(),
            latest_validated_version=str(payload.get("latest_validated_version", "")).strip(),
            latest_approved_version=str(payload.get("latest_approved_version", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            tags=payload.get("tags") or (),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ModelVersionRecord:
    """Persistent metadata for one version of one governed model."""

    model_id: str
    version: str
    status: ModelLifecycleStatus = ModelLifecycleStatus.DRAFT
    contract_summary: Mapping[str, object] = field(default_factory=dict)
    methodology_summary: Mapping[str, object] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    engine_binding: Mapping[str, object] = field(default_factory=dict)
    validation_summary: Mapping[str, object] = field(default_factory=dict)
    validation_refs: tuple[str, ...] = ()
    lineage: ModelLineage = field(default_factory=ModelLineage)
    artifacts: Mapping[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    transitions: tuple[ModelLifecycleTransition, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = _MODEL_SCHEMA_VERSION

    def __post_init__(self):
        """Normalize nested version metadata into immutable primitives."""
        object.__setattr__(self, "model_id", _normalize_token(self.model_id))
        object.__setattr__(self, "version", _normalize_token(self.version))
        object.__setattr__(self, "status", ModelLifecycleStatus.normalize(self.status))
        object.__setattr__(self, "contract_summary", _freeze_mapping(self.contract_summary))
        object.__setattr__(self, "methodology_summary", _freeze_mapping(self.methodology_summary))
        object.__setattr__(self, "assumptions", _string_tuple(self.assumptions))
        object.__setattr__(self, "engine_binding", _freeze_mapping(self.engine_binding))
        object.__setattr__(self, "validation_summary", _freeze_mapping(self.validation_summary))
        object.__setattr__(self, "validation_refs", _string_tuple(self.validation_refs))
        object.__setattr__(
            self,
            "lineage",
            self.lineage if isinstance(self.lineage, ModelLineage) else ModelLineage.from_dict(self.lineage),
        )
        object.__setattr__(self, "artifacts", _freeze_mapping(self.artifacts))
        object.__setattr__(self, "created_at", _normalize_token(self.created_at, fallback=_utc_now()))
        object.__setattr__(self, "updated_at", _normalize_token(self.updated_at, fallback=self.created_at))
        object.__setattr__(
            self,
            "transitions",
            tuple(
                item if isinstance(item, ModelLifecycleTransition) else ModelLifecycleTransition.from_dict(item)
                for item in (self.transitions or ())
            ),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "contract_summary": dict(self.contract_summary),
            "methodology_summary": dict(self.methodology_summary),
            "assumptions": list(self.assumptions),
            "engine_binding": dict(self.engine_binding),
            "validation_summary": dict(self.validation_summary),
            "validation_refs": list(self.validation_refs),
            "lineage": self.lineage.to_dict(),
            "artifacts": dict(self.artifacts),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "transitions": [item.to_dict() for item in self.transitions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelVersionRecord:
        """Rehydrate one model-version record."""
        return cls(
            schema_version=int(payload.get("schema_version", _MODEL_SCHEMA_VERSION)),
            model_id=str(payload.get("model_id", "")).strip(),
            version=str(payload.get("version", "")).strip(),
            status=payload.get("status") or ModelLifecycleStatus.DRAFT.value,
            contract_summary=payload.get("contract_summary") or {},
            methodology_summary=payload.get("methodology_summary") or {},
            assumptions=payload.get("assumptions") or (),
            engine_binding=payload.get("engine_binding") or {},
            validation_summary=payload.get("validation_summary") or {},
            validation_refs=payload.get("validation_refs") or (),
            lineage=payload.get("lineage") or {},
            artifacts=payload.get("artifacts") or {},
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            transitions=payload.get("transitions") or (),
            metadata=payload.get("metadata") or {},
        )


def _is_lifecycle_transition_allowed(
    from_status: ModelLifecycleStatus,
    to_status: ModelLifecycleStatus,
) -> bool:
    """Return whether one lifecycle transition is allowed."""
    if from_status == to_status:
        return True
    return to_status in _TRANSITION_RULES[from_status]


class ModelRegistryStore:
    """Local-first persistence for governed model and version metadata."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else _MODEL_REGISTRY_ROOT

    def _model_dir(self, model_id: str) -> Path:
        return self.base_dir / model_id

    def _model_record_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "model.json"

    def _versions_dir(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "versions"

    def _version_dir(self, model_id: str, version: str) -> Path:
        return self._versions_dir(model_id) / version

    def _legacy_version_record_path(self, model_id: str, version: str) -> Path:
        return self._versions_dir(model_id) / f"{version}.json"

    def _version_record_path(self, model_id: str, version: str) -> Path:
        return self._version_dir(model_id, version) / "manifest.json"

    def _version_artifact_path(self, model_id: str, version: str, artifact_name: str) -> Path:
        filename, _ = _VERSION_ARTIFACT_FILES[artifact_name]
        return self._version_dir(model_id, version) / filename

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text())

    def create_model(self, record: ModelRecord) -> ModelRecord:
        """Persist one new governed model record."""
        path = self._model_record_path(record.model_id)
        if path.exists():
            raise FileExistsError(f"Model already exists: {record.model_id}")
        self._write_json(path, record.to_dict())
        return self.get_model(record.model_id) or record

    def save_model(self, record: ModelRecord) -> ModelRecord:
        """Upsert one governed model record."""
        self._write_json(self._model_record_path(record.model_id), record.to_dict())
        return self.get_model(record.model_id) or record

    def get_model(self, model_id: str) -> ModelRecord | None:
        """Load one model record from disk."""
        path = self._model_record_path(model_id)
        if not path.exists():
            return None
        return ModelRecord.from_dict(self._read_json(path))

    def list_models(self) -> list[ModelRecord]:
        """Return all stored model records."""
        if not self.base_dir.exists():
            return []
        records: list[ModelRecord] = []
        for model_path in sorted(self.base_dir.glob("*/model.json")):
            records.append(ModelRecord.from_dict(self._read_json(model_path)))
        return records

    def create_version(
        self,
        record: ModelVersionRecord,
        *,
        actor: str,
        reason: str,
        notes: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ModelVersionRecord:
        """Persist one new model-version record with an auditable creation transition."""
        if self.get_model(record.model_id) is None:
            raise FileNotFoundError(f"Unknown model id: {record.model_id}")
        path = self._version_record_path(record.model_id, record.version)
        if path.exists() or self._legacy_version_record_path(record.model_id, record.version).exists():
            raise FileExistsError(f"Version already exists: {record.model_id}:{record.version}")

        transition = ModelLifecycleTransition(
            transition_id=f"{record.model_id}:{record.version}:1",
            from_status=None,
            to_status=record.status,
            changed_at=record.created_at,
            changed_by=actor,
            reason=reason,
            notes=notes,
            metadata=metadata or {},
        )
        stored = replace(record, transitions=(transition,))
        self._write_json(path, stored.to_dict())
        self._refresh_model_summary(record.model_id)
        return self.get_version(record.model_id, record.version) or stored

    def save_version(self, record: ModelVersionRecord) -> ModelVersionRecord:
        """Upsert one governed model-version record."""
        if self.get_model(record.model_id) is None:
            raise FileNotFoundError(f"Unknown model id: {record.model_id}")
        self._write_json(self._version_record_path(record.model_id, record.version), record.to_dict())
        self._refresh_model_summary(record.model_id)
        return self.get_version(record.model_id, record.version) or record

    def get_version(self, model_id: str, version: str) -> ModelVersionRecord | None:
        """Load one stored model version."""
        path = self._version_record_path(model_id, version)
        if not path.exists():
            path = self._legacy_version_record_path(model_id, version)
            if not path.exists():
                return None
        return ModelVersionRecord.from_dict(self._read_json(path))

    def list_versions(self, model_id: str) -> list[ModelVersionRecord]:
        """Return all stored versions for one model."""
        versions_dir = self._versions_dir(model_id)
        if not versions_dir.exists():
            return []
        records: list[ModelVersionRecord] = []
        seen: set[str] = set()
        for path in sorted(versions_dir.glob("*/manifest.json")):
            record = ModelVersionRecord.from_dict(self._read_json(path))
            records.append(record)
            seen.add(record.version)
        for path in sorted(versions_dir.glob("*.json")):
            record = ModelVersionRecord.from_dict(self._read_json(path))
            if record.version in seen:
                continue
            records.append(record)
        return sorted(records, key=lambda item: (item.created_at, item.version))

    def transition_version(
        self,
        model_id: str,
        version: str,
        to_status: ModelLifecycleStatus | str,
        *,
        actor: str,
        reason: str,
        notes: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ModelVersionRecord:
        """Apply one auditable lifecycle transition to a model version."""
        record = self.get_version(model_id, version)
        if record is None:
            raise FileNotFoundError(f"Unknown model version: {model_id}:{version}")
        normalized_status = ModelLifecycleStatus.normalize(to_status)
        if not _is_lifecycle_transition_allowed(record.status, normalized_status):
            raise InvalidLifecycleTransitionError(
                f"Invalid transition: {record.status.value} -> {normalized_status.value}"
            )
        if record.status == normalized_status:
            return record

        transition = ModelLifecycleTransition(
            transition_id=f"{model_id}:{version}:{len(record.transitions) + 1}",
            from_status=record.status,
            to_status=normalized_status,
            changed_at=_utc_now(),
            changed_by=actor,
            reason=reason,
            notes=notes,
            metadata=metadata or {},
        )
        updated = replace(
            record,
            status=normalized_status,
            updated_at=transition.changed_at,
            transitions=record.transitions + (transition,),
        )
        self._write_json(self._version_record_path(model_id, version), updated.to_dict())
        self._refresh_model_summary(model_id)
        return self.get_version(model_id, version) or updated

    def version_artifact_uri(self, model_id: str, version: str, artifact_name: str) -> str:
        """Return the canonical resource URI for one version artifact."""
        if artifact_name not in _VERSION_ARTIFACT_FILES:
            raise KeyError(f"Unknown model-version artifact: {artifact_name}")
        return f"trellis://models/{model_id}/versions/{version}/{artifact_name}"

    def write_version_artifact(self, model_id: str, version: str, artifact_name: str, payload) -> str:
        """Persist one canonical version artifact and return its stable URI."""
        try:
            _, mode = _VERSION_ARTIFACT_FILES[artifact_name]
        except KeyError as exc:
            raise KeyError(f"Unknown model-version artifact: {artifact_name}") from exc
        path = self._version_artifact_path(model_id, version, artifact_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "json":
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            path.write_text(str(payload or ""))
        return self.version_artifact_uri(model_id, version, artifact_name)

    def load_version_artifact(self, model_id: str, version: str, artifact_name: str):
        """Load one persisted version artifact when present."""
        try:
            _, mode = _VERSION_ARTIFACT_FILES[artifact_name]
        except KeyError as exc:
            raise KeyError(f"Unknown model-version artifact: {artifact_name}") from exc
        path = self._version_artifact_path(model_id, version, artifact_name)
        if not path.exists():
            return None
        if mode == "json":
            return json.loads(path.read_text())
        return path.read_text()

    def diff_versions(self, model_id: str, left_version: str, right_version: str) -> dict[str, object]:
        """Return a governed diff across the main review surfaces for two versions."""
        left = self.get_version(model_id, left_version)
        right = self.get_version(model_id, right_version)
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append(left_version)
            if right is None:
                missing.append(right_version)
            raise FileNotFoundError(
                f"Unknown model version(s): {model_id}:{', '.join(missing)}"
            )

        left_code = self.load_version_artifact(model_id, left_version, "code")
        right_code = self.load_version_artifact(model_id, right_version, "code")
        return {
            "model_id": model_id,
            "left_version": left_version,
            "right_version": right_version,
            "status_changed": left.status.value != right.status.value,
            "status": {"left": left.status.value, "right": right.status.value},
            "contract_summary_changed": left.contract_summary != right.contract_summary,
            "contract_summary": {"left": dict(left.contract_summary), "right": dict(right.contract_summary)},
            "methodology_summary_changed": left.methodology_summary != right.methodology_summary,
            "methodology_summary": {
                "left": dict(left.methodology_summary),
                "right": dict(right.methodology_summary),
            },
            "engine_binding_changed": left.engine_binding != right.engine_binding,
            "engine_binding": {"left": dict(left.engine_binding), "right": dict(right.engine_binding)},
            "assumptions_changed": left.assumptions != right.assumptions,
            "assumptions": {"left": list(left.assumptions), "right": list(right.assumptions)},
            "validation_summary_changed": left.validation_summary != right.validation_summary,
            "validation_summary": {
                "left": dict(left.validation_summary),
                "right": dict(right.validation_summary),
            },
            "policy_metadata_changed": left.metadata != right.metadata,
            "policy_metadata": {"left": dict(left.metadata), "right": dict(right.metadata)},
            "lineage_changed": left.lineage.to_dict() != right.lineage.to_dict(),
            "lineage": {"left": left.lineage.to_dict(), "right": right.lineage.to_dict()},
            "artifacts_changed": dict(left.artifacts) != dict(right.artifacts),
            "artifacts": {"left": dict(left.artifacts), "right": dict(right.artifacts)},
            "code_changed": left_code != right_code,
            "code_diff": list(
                difflib.unified_diff(
                    str(left_code or "").splitlines(),
                    str(right_code or "").splitlines(),
                    fromfile=f"{left_version}/implementation.py",
                    tofile=f"{right_version}/implementation.py",
                    lineterm="",
                )
            ),
        }

    def _refresh_model_summary(self, model_id: str) -> ModelRecord:
        """Recompute one model manifest from its persisted versions."""
        model = self.get_model(model_id)
        if model is None:
            raise FileNotFoundError(f"Unknown model id: {model_id}")
        versions = self.list_versions(model_id)
        latest_version = versions[-1].version if versions else ""

        validated_versions = [
            item.version
            for item in versions
            if item.status in {ModelLifecycleStatus.VALIDATED, ModelLifecycleStatus.APPROVED}
        ]
        approved_versions = [
            item.version
            for item in versions
            if item.status is ModelLifecycleStatus.APPROVED
        ]

        if approved_versions:
            model_status = ModelLifecycleStatus.APPROVED
        elif validated_versions:
            model_status = ModelLifecycleStatus.VALIDATED
        elif any(item.status is ModelLifecycleStatus.DRAFT for item in versions):
            model_status = ModelLifecycleStatus.DRAFT
        elif versions:
            model_status = ModelLifecycleStatus.DEPRECATED
        else:
            model_status = model.status

        updated = replace(
            model,
            status=model_status,
            latest_version=latest_version,
            latest_validated_version=validated_versions[-1] if validated_versions else "",
            latest_approved_version=approved_versions[-1] if approved_versions else "",
            updated_at=_utc_now(),
        )
        self._write_json(self._model_record_path(model_id), updated.to_dict())
        return updated


def _status_rank(status: ModelLifecycleStatus) -> int:
    """Return the deterministic execution preference rank for one lifecycle status."""
    return {
        ModelLifecycleStatus.DRAFT: 1,
        ModelLifecycleStatus.VALIDATED: 2,
        ModelLifecycleStatus.APPROVED: 3,
        ModelLifecycleStatus.DEPRECATED: 0,
    }[status]


@dataclass(frozen=True)
class ModelExecutionRejection:
    """Structured lifecycle rejection for one model version."""

    code: str
    message: str
    version: str = ""
    status: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize rejection metadata into immutable primitives."""
        object.__setattr__(self, "code", _normalize_token(self.code))
        object.__setattr__(self, "message", str(self.message or "").strip())
        object.__setattr__(self, "version", _normalize_token(self.version))
        object.__setattr__(self, "status", _normalize_token(self.status))
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "code": self.code,
            "message": self.message,
            "version": self.version,
            "status": self.status,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ModelExecutionGateResult:
    """Deterministic lifecycle gate result for one governed model id."""

    model_id: str
    run_mode: str
    policy_id: str
    allowed: bool
    allowed_statuses: tuple[str, ...] = ()
    selected_model: Mapping[str, object] = field(default_factory=dict)
    rejections: tuple[ModelExecutionRejection, ...] = ()

    def __post_init__(self):
        """Normalize gate metadata into immutable primitives."""
        object.__setattr__(self, "model_id", _normalize_token(self.model_id))
        object.__setattr__(self, "run_mode", _normalize_token(self.run_mode))
        object.__setattr__(self, "policy_id", _normalize_token(self.policy_id))
        object.__setattr__(self, "allowed_statuses", _string_tuple(self.allowed_statuses))
        object.__setattr__(self, "selected_model", _freeze_mapping(self.selected_model))
        object.__setattr__(
            self,
            "rejections",
            tuple(
                rejection
                if isinstance(rejection, ModelExecutionRejection)
                else ModelExecutionRejection(**rejection)
                for rejection in self.rejections
            ),
        )

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        """Return the ordered rejection codes."""
        return tuple(rejection.code for rejection in self.rejections)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "model_id": self.model_id,
            "run_mode": self.run_mode,
            "policy_id": self.policy_id,
            "allowed": self.allowed,
            "allowed_statuses": list(self.allowed_statuses),
            "selected_model": dict(self.selected_model),
            "rejection_codes": list(self.rejection_codes),
            "rejections": [rejection.to_dict() for rejection in self.rejections],
        }


class ModelExecutionGateError(RuntimeError):
    """Raised when no lifecycle-eligible model version exists for execution."""

    def __init__(self, result: ModelExecutionGateResult):
        self.result = result
        summary = ", ".join(result.rejection_codes) or "unknown_model_execution_rejection"
        super().__init__(f"Governed model execution blocked: {summary}")


def evaluate_model_execution_gate(
    *,
    registry: ModelRegistryStore,
    model_id: str,
    execution_context,
    policy_bundle=None,
) -> ModelExecutionGateResult:
    """Evaluate lifecycle eligibility for one governed model id."""
    from trellis.agent.cycle_surface import build_cycle_result_surface
    from trellis.platform.policies import get_policy_bundle

    bundle = policy_bundle or get_policy_bundle(
        execution_context.policy_bundle_id,
        run_mode=execution_context.run_mode,
    )
    allowed_statuses = tuple(status for status in bundle.allowed_model_statuses)
    allowed_status_values = tuple(status.value for status in allowed_statuses)
    model = registry.get_model(model_id)
    if model is None:
        return ModelExecutionGateResult(
            model_id=model_id,
            run_mode=execution_context.run_mode.value,
            policy_id=bundle.policy_id,
            allowed=False,
            allowed_statuses=allowed_status_values,
            rejections=(
                ModelExecutionRejection(
                    code="model_not_found",
                    message=f"No governed model record exists for {model_id!r}",
                ),
            ),
        )

    versions = registry.list_versions(model_id)
    if not versions:
        return ModelExecutionGateResult(
            model_id=model.model_id,
            run_mode=execution_context.run_mode.value,
            policy_id=bundle.policy_id,
            allowed=False,
            allowed_statuses=allowed_status_values,
            rejections=(
                ModelExecutionRejection(
                    code="no_registered_versions",
                    message=f"No governed model versions exist for {model_id!r}",
                ),
            ),
        )

    eligible_versions = [
        version for version in versions if version.status in allowed_statuses
    ]
    rejections = tuple(
        ModelExecutionRejection(
            code="lifecycle_not_allowed",
            message=(
                f"Model version {version.version!r} with lifecycle {version.status.value!r} "
                f"is not executable in {execution_context.run_mode.value!r}"
            ),
            version=version.version,
            status=version.status.value,
            details={"allowed_statuses": list(allowed_status_values)},
        )
        for version in versions
        if version.status not in allowed_statuses
    )
    if not eligible_versions:
        return ModelExecutionGateResult(
            model_id=model.model_id,
            run_mode=execution_context.run_mode.value,
            policy_id=bundle.policy_id,
            allowed=False,
            allowed_statuses=allowed_status_values,
            rejections=rejections,
        )

    selected = max(
        eligible_versions,
        key=lambda version: (_status_rank(version.status), version.updated_at, version.version),
    )
    return ModelExecutionGateResult(
        model_id=model.model_id,
        run_mode=execution_context.run_mode.value,
        policy_id=bundle.policy_id,
        allowed=True,
        allowed_statuses=allowed_status_values,
        selected_model={
            "model_id": model.model_id,
            "version": selected.version,
            "status": selected.status.value,
            "engine_binding": dict(selected.engine_binding),
            "contract_summary": dict(selected.contract_summary),
            "methodology_summary": dict(selected.methodology_summary),
            "agent_cycle": build_cycle_result_surface(
                _cycle_report_for_version(selected),
                promotion_governance=_cycle_governance_for_version(selected),
            ),
        },
        rejections=rejections,
    )


def _cycle_governance_for_version(version: ModelVersionRecord) -> Mapping[str, object] | None:
    for transition in reversed(version.transitions):
        metadata = dict(transition.metadata)
        governance = metadata.get("cycle_promotion_governance")
        if isinstance(governance, Mapping):
            return governance
    metadata_governance = version.metadata.get("cycle_promotion_governance")
    if isinstance(metadata_governance, Mapping):
        return metadata_governance
    return None


def _cycle_report_for_version(version: ModelVersionRecord) -> Mapping[str, object] | None:
    for transition in reversed(version.transitions):
        metadata = dict(transition.metadata)
        report = metadata.get("cycle_report")
        if isinstance(report, Mapping):
            return report
        governance = metadata.get("cycle_promotion_governance")
        if isinstance(governance, Mapping) and isinstance(governance.get("cycle_report"), Mapping):
            return governance.get("cycle_report")
    metadata_report = version.metadata.get("cycle_report")
    if isinstance(metadata_report, Mapping):
        return metadata_report
    return None


def enforce_model_execution_gate(
    *,
    registry: ModelRegistryStore,
    model_id: str,
    execution_context,
    policy_bundle=None,
) -> ModelExecutionGateResult:
    """Evaluate lifecycle eligibility and raise when execution is blocked."""
    result = evaluate_model_execution_gate(
        registry=registry,
        model_id=model_id,
        execution_context=execution_context,
        policy_bundle=policy_bundle,
    )
    if not result.allowed:
        raise ModelExecutionGateError(result)
    return result
