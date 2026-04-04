"""Filesystem-backed state-root config and durable MCP-oriented records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from trellis.platform.context import ExecutionContext, ProviderBindings, RunMode


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STATE_ROOT = _REPO_ROOT / ".trellis_state"
_DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "server.yaml"
_SCHEMA_VERSION = 1


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_token(value: str | None, *, fallback: str = "") -> str:
    """Normalize one identifier-like token."""
    text = str(value or "").strip()
    return text or fallback


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable ordered tuple of unique strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _json_safe(value):
    """Convert nested values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return str(value)


@dataclass(frozen=True)
class ServerIdentityConfig:
    """Top-level server identity and transport settings."""

    name: str = "trellis"
    transport: str = "streamable_http"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ServerIdentityConfig:
        payload = payload or {}
        return cls(
            name=str(payload.get("name", "trellis")).strip() or "trellis",
            transport=str(payload.get("transport", "streamable_http")).strip() or "streamable_http",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "transport": self.transport,
        }


@dataclass(frozen=True)
class RuntimeDefaultsConfig:
    """Governed runtime defaults loaded from the server config."""

    run_mode: str = "research"
    output_mode: str = "structured"
    audit_mode: str = "summary"
    allow_mock_data: bool = False
    require_explicit_provider_binding: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> RuntimeDefaultsConfig:
        payload = payload or {}
        return cls(
            run_mode=str(payload.get("run_mode", "research")).strip() or "research",
            output_mode=str(payload.get("output_mode", "structured")).strip() or "structured",
            audit_mode=str(payload.get("audit_mode", "summary")).strip() or "summary",
            allow_mock_data=bool(payload.get("allow_mock_data", False)),
            require_explicit_provider_binding=bool(
                payload.get("require_explicit_provider_binding", True)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_mode": self.run_mode,
            "output_mode": self.output_mode,
            "audit_mode": self.audit_mode,
            "allow_mock_data": self.allow_mock_data,
            "require_explicit_provider_binding": self.require_explicit_provider_binding,
        }


@dataclass(frozen=True)
class ProviderDefaultConfig:
    """Default provider bindings for one provider family."""

    primary: str | None = None
    fallback: str | None = None
    mock: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ProviderDefaultConfig:
        payload = payload or {}
        return cls(
            primary=None if payload.get("primary") in {None, ""} else str(payload.get("primary")).strip(),
            fallback=None if payload.get("fallback") in {None, ""} else str(payload.get("fallback")).strip(),
            mock=None if payload.get("mock") in {None, ""} else str(payload.get("mock")).strip(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary,
            "fallback": self.fallback,
            "mock": self.mock,
        }


@dataclass(frozen=True)
class ProviderDefaultsConfig:
    """Default provider-family wiring from bootstrap config."""

    market_data: ProviderDefaultConfig = field(default_factory=ProviderDefaultConfig)
    pricing_engine: ProviderDefaultConfig = field(default_factory=ProviderDefaultConfig)
    model_store: ProviderDefaultConfig = field(default_factory=ProviderDefaultConfig)
    validation_engine: ProviderDefaultConfig = field(default_factory=ProviderDefaultConfig)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ProviderDefaultsConfig:
        payload = payload or {}
        return cls(
            market_data=ProviderDefaultConfig.from_dict(payload.get("market_data")),
            pricing_engine=ProviderDefaultConfig.from_dict(payload.get("pricing_engine")),
            model_store=ProviderDefaultConfig.from_dict(payload.get("model_store")),
            validation_engine=ProviderDefaultConfig.from_dict(payload.get("validation_engine")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "market_data": self.market_data.to_dict(),
            "pricing_engine": self.pricing_engine.to_dict(),
            "model_store": self.model_store.to_dict(),
            "validation_engine": self.validation_engine.to_dict(),
        }


@dataclass(frozen=True)
class PolicyDefaultsConfig:
    """Active policy defaults and production requirements."""

    active: str = "quant-safe-v1"
    production_requires: tuple[str, ...] = (
        "approved_model",
        "explicit_market_data_provider",
        "persisted_market_snapshot",
        "full_run_ledger",
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> PolicyDefaultsConfig:
        payload = payload or {}
        return cls(
            active=str(payload.get("active", "quant-safe-v1")).strip() or "quant-safe-v1",
            production_requires=_string_tuple(
                payload.get("production_requires")
                or (
                    "approved_model",
                    "explicit_market_data_provider",
                    "persisted_market_snapshot",
                    "full_run_ledger",
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "production_requires": list(self.production_requires),
        }


@dataclass(frozen=True)
class LifecycleDefaultsConfig:
    """Lifecycle-state config for governed model records."""

    statuses: tuple[str, ...] = ("draft", "validated", "approved", "deprecated")
    auto_promote: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> LifecycleDefaultsConfig:
        payload = payload or {}
        return cls(
            statuses=_string_tuple(
                payload.get("statuses")
                or ("draft", "validated", "approved", "deprecated")
            ),
            auto_promote=bool(payload.get("auto_promote", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "statuses": list(self.statuses),
            "auto_promote": self.auto_promote,
        }


@dataclass(frozen=True)
class ObservabilityConfig:
    """Observability toggles for run persistence and event emission."""

    persist_run_ledgers: bool = True
    persist_logs: bool = True
    emit_progress_events: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ObservabilityConfig:
        payload = payload or {}
        return cls(
            persist_run_ledgers=bool(payload.get("persist_run_ledgers", True)),
            persist_logs=bool(payload.get("persist_logs", True)),
            emit_progress_events=bool(payload.get("emit_progress_events", True)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "persist_run_ledgers": self.persist_run_ledgers,
            "persist_logs": self.persist_logs,
            "emit_progress_events": self.emit_progress_events,
        }


@dataclass(frozen=True)
class TrellisServerConfig:
    """Normalized server bootstrap config plus resolved filesystem state root."""

    state_root: Path
    config_path: Path
    server: ServerIdentityConfig = field(default_factory=ServerIdentityConfig)
    defaults: RuntimeDefaultsConfig = field(default_factory=RuntimeDefaultsConfig)
    providers: ProviderDefaultsConfig = field(default_factory=ProviderDefaultsConfig)
    policies: PolicyDefaultsConfig = field(default_factory=PolicyDefaultsConfig)
    lifecycle: LifecycleDefaultsConfig = field(default_factory=LifecycleDefaultsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def to_dict(self) -> dict[str, object]:
        return {
            "server": self.server.to_dict(),
            "defaults": self.defaults.to_dict(),
            "providers": self.providers.to_dict(),
            "policies": self.policies.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "observability": self.observability.to_dict(),
        }


@dataclass(frozen=True)
class TrellisStatePaths:
    """Resolved paths for the governed MCP state root."""

    state_root: Path
    config_dir: Path
    config_path: Path
    sessions_dir: Path
    providers_dir: Path
    policies_dir: Path
    models_dir: Path
    runs_dir: Path
    snapshots_dir: Path
    validations_dir: Path

    def ensure(self) -> TrellisStatePaths:
        """Create the directory layout if it does not already exist."""
        for path in (
            self.state_root,
            self.config_dir,
            self.sessions_dir,
            self.providers_dir,
            self.policies_dir,
            self.models_dir,
            self.runs_dir,
            self.snapshots_dir,
            self.validations_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def resolve_state_root(
    *,
    state_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the active Trellis state root with explicit args taking precedence."""
    if state_root is not None:
        return Path(state_root)
    env = dict(os.environ) if env is None else dict(env)
    env_state_root = str(env.get("TRELLIS_STATE_ROOT", "")).strip()
    if env_state_root:
        return Path(env_state_root)
    return _DEFAULT_STATE_ROOT


def build_state_paths(
    *,
    state_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> TrellisStatePaths:
    """Return the resolved directory layout for the active state root."""
    root = resolve_state_root(state_root=state_root, env=env)
    return TrellisStatePaths(
        state_root=root,
        config_dir=root / "config",
        config_path=root / _DEFAULT_CONFIG_RELATIVE_PATH,
        sessions_dir=root / "sessions",
        providers_dir=root / "providers",
        policies_dir=root / "policies",
        models_dir=root / "models",
        runs_dir=root / "runs",
        snapshots_dir=root / "snapshots",
        validations_dir=root / "validations",
    )


def _load_raw_config(config_path: Path) -> dict[str, object]:
    """Load one YAML config file when present."""
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {config_path}")
    return payload


def _write_default_config(config_path: Path, config: TrellisServerConfig) -> None:
    """Persist one default YAML config."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))


def load_trellis_server_config(
    *,
    config_path: Path | str | None = None,
    state_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> TrellisServerConfig:
    """Load the Trellis server bootstrap config and ensure the state root exists."""
    paths = build_state_paths(state_root=state_root, env=env).ensure()
    resolved_config_path = Path(config_path) if config_path is not None else paths.config_path
    raw = _load_raw_config(resolved_config_path)
    config = TrellisServerConfig(
        state_root=paths.state_root,
        config_path=resolved_config_path,
        server=ServerIdentityConfig.from_dict(raw.get("server")),
        defaults=RuntimeDefaultsConfig.from_dict(raw.get("defaults")),
        providers=ProviderDefaultsConfig.from_dict(raw.get("providers")),
        policies=PolicyDefaultsConfig.from_dict(raw.get("policies")),
        lifecycle=LifecycleDefaultsConfig.from_dict(raw.get("lifecycle")),
        observability=ObservabilityConfig.from_dict(raw.get("observability")),
    )
    if not resolved_config_path.exists():
        _write_default_config(resolved_config_path, config)
    return config


@dataclass(frozen=True)
class SessionContextRecord:
    """Durable governed session context used by MCP tools and services."""

    session_id: str
    run_mode: RunMode = RunMode.RESEARCH
    default_output_mode: str = "structured"
    default_audit_mode: str = "summary"
    connected_providers: tuple[str, ...] = ()
    provider_bindings: ProviderBindings = field(default_factory=ProviderBindings)
    active_policy: str = ""
    allow_mock_data: bool = False
    require_provider_disclosure: bool = True
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(self, "session_id", _normalize_token(self.session_id))
        object.__setattr__(self, "run_mode", RunMode.normalize(self.run_mode))
        object.__setattr__(
            self,
            "provider_bindings",
            self.provider_bindings
            if isinstance(self.provider_bindings, ProviderBindings)
            else ProviderBindings.from_dict(self.provider_bindings),
        )
        object.__setattr__(self, "connected_providers", _string_tuple(self.connected_providers))
        active_policy = _normalize_token(self.active_policy)
        if not active_policy:
            active_policy = f"policy_bundle.{self.run_mode.value}.default"
        object.__setattr__(self, "active_policy", active_policy)
        object.__setattr__(self, "created_at", _normalize_token(self.created_at, fallback=_utc_now()))
        object.__setattr__(self, "updated_at", _normalize_token(self.updated_at, fallback=self.created_at))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def execution_context(self) -> ExecutionContext:
        """Project this durable session record onto the runtime execution context."""
        return ExecutionContext(
            session_id=self.session_id,
            run_mode=self.run_mode,
            provider_bindings=self.provider_bindings,
            policy_bundle_id=self.active_policy,
            allow_mock_data=self.allow_mock_data,
            require_provider_disclosure=self.require_provider_disclosure,
            default_output_mode=self.default_output_mode,
            default_audit_mode=self.default_audit_mode,
            metadata=self.metadata,
        )

    @classmethod
    def from_execution_context(
        cls,
        context: ExecutionContext,
        *,
        connected_providers=(),
    ) -> SessionContextRecord:
        return cls(
            session_id=context.session_id,
            run_mode=context.run_mode,
            default_output_mode=context.default_output_mode,
            default_audit_mode=context.default_audit_mode,
            connected_providers=connected_providers,
            provider_bindings=context.provider_bindings,
            active_policy=context.policy_bundle_id,
            allow_mock_data=context.allow_mock_data,
            require_provider_disclosure=context.require_provider_disclosure,
            metadata=context.metadata,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "run_mode": self.run_mode.value,
            "default_output_mode": self.default_output_mode,
            "default_audit_mode": self.default_audit_mode,
            "connected_providers": list(self.connected_providers),
            "provider_bindings": self.provider_bindings.to_dict(),
            "active_policy": self.active_policy,
            "allow_mock_data": self.allow_mock_data,
            "require_provider_disclosure": self.require_provider_disclosure,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SessionContextRecord:
        return cls(
            schema_version=int(payload.get("schema_version", _SCHEMA_VERSION)),
            session_id=str(payload.get("session_id", "")).strip(),
            run_mode=payload.get("run_mode", RunMode.RESEARCH.value),
            default_output_mode=str(payload.get("default_output_mode", "structured")).strip(),
            default_audit_mode=str(payload.get("default_audit_mode", "summary")).strip(),
            connected_providers=payload.get("connected_providers") or (),
            provider_bindings=payload.get("provider_bindings") or {},
            active_policy=str(payload.get("active_policy", "")).strip(),
            allow_mock_data=bool(payload.get("allow_mock_data", False)),
            require_provider_disclosure=bool(payload.get("require_provider_disclosure", True)),
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class SnapshotRecord:
    """Durable JSON-safe market snapshot record for governed provenance."""

    snapshot_id: str
    provider_id: str = ""
    as_of: str = ""
    source: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(self, "snapshot_id", _normalize_token(self.snapshot_id))
        object.__setattr__(self, "provider_id", _normalize_token(self.provider_id))
        object.__setattr__(self, "as_of", _normalize_token(self.as_of))
        object.__setattr__(self, "source", _normalize_token(self.source))
        object.__setattr__(self, "payload", _freeze_mapping(_json_safe(self.payload)))
        object.__setattr__(self, "provenance", _freeze_mapping(_json_safe(self.provenance)))
        object.__setattr__(self, "created_at", _normalize_token(self.created_at, fallback=_utc_now()))
        object.__setattr__(self, "updated_at", _normalize_token(self.updated_at, fallback=self.created_at))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "provider_id": self.provider_id,
            "as_of": self.as_of,
            "source": self.source,
            "payload": dict(self.payload),
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SnapshotRecord:
        return cls(
            schema_version=int(payload.get("schema_version", _SCHEMA_VERSION)),
            snapshot_id=str(payload.get("snapshot_id", "")).strip(),
            provider_id=str(payload.get("provider_id", "")).strip(),
            as_of=str(payload.get("as_of", "")).strip(),
            source=str(payload.get("source", "")).strip(),
            payload=payload.get("payload") or {},
            provenance=payload.get("provenance") or {},
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
        )


@dataclass(frozen=True)
class ValidationRecord:
    """Durable validation record for governed model lifecycle workflows."""

    validation_id: str
    model_id: str
    version: str
    status: str
    summary: Mapping[str, object] = field(default_factory=dict)
    refs: tuple[str, ...] = ()
    policy_outcome: Mapping[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(self, "validation_id", _normalize_token(self.validation_id))
        object.__setattr__(self, "model_id", _normalize_token(self.model_id))
        object.__setattr__(self, "version", _normalize_token(self.version))
        object.__setattr__(self, "status", _normalize_token(self.status))
        object.__setattr__(self, "summary", _freeze_mapping(_json_safe(self.summary)))
        object.__setattr__(self, "refs", _string_tuple(self.refs))
        object.__setattr__(self, "policy_outcome", _freeze_mapping(_json_safe(self.policy_outcome)))
        object.__setattr__(self, "created_at", _normalize_token(self.created_at, fallback=_utc_now()))
        object.__setattr__(self, "updated_at", _normalize_token(self.updated_at, fallback=self.created_at))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status,
            "summary": dict(self.summary),
            "refs": list(self.refs),
            "policy_outcome": dict(self.policy_outcome),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ValidationRecord:
        return cls(
            schema_version=int(payload.get("schema_version", _SCHEMA_VERSION)),
            validation_id=str(payload.get("validation_id", "")).strip(),
            model_id=str(payload.get("model_id", "")).strip(),
            version=str(payload.get("version", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            summary=payload.get("summary") or {},
            refs=payload.get("refs") or (),
            policy_outcome=payload.get("policy_outcome") or {},
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
        )


class _JsonRecordStore:
    """Small JSON-record store helper for filesystem-backed persistent records."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text())


class SessionContextStore(_JsonRecordStore):
    """Filesystem-backed persistence for governed session records."""

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def save_session(self, record: SessionContextRecord) -> SessionContextRecord:
        self._write_json(self._path(record.session_id), record.to_dict())
        return self.get_session(record.session_id) or record

    def get_session(self, session_id: str) -> SessionContextRecord | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return SessionContextRecord.from_dict(self._read_json(path))

    def list_sessions(self) -> list[SessionContextRecord]:
        if not self.base_dir.exists():
            return []
        return [
            SessionContextRecord.from_dict(self._read_json(path))
            for path in sorted(self.base_dir.glob("*.json"))
        ]


class SnapshotStore(_JsonRecordStore):
    """Filesystem-backed persistence for JSON-safe governed snapshot records."""

    def _path(self, snapshot_id: str) -> Path:
        return self.base_dir / f"{snapshot_id}.json"

    def save_snapshot(self, record: SnapshotRecord) -> SnapshotRecord:
        self._write_json(self._path(record.snapshot_id), record.to_dict())
        return self.get_snapshot(record.snapshot_id) or record

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord | None:
        path = self._path(snapshot_id)
        if not path.exists():
            return None
        return SnapshotRecord.from_dict(self._read_json(path))

    def list_snapshots(self) -> list[SnapshotRecord]:
        if not self.base_dir.exists():
            return []
        return [
            SnapshotRecord.from_dict(self._read_json(path))
            for path in sorted(self.base_dir.glob("*.json"))
        ]


class ValidationStore(_JsonRecordStore):
    """Filesystem-backed persistence for governed validation records."""

    def _path(self, validation_id: str) -> Path:
        return self.base_dir / f"{validation_id}.json"

    def save_validation(self, record: ValidationRecord) -> ValidationRecord:
        self._write_json(self._path(record.validation_id), record.to_dict())
        return self.get_validation(record.validation_id) or record

    def get_validation(self, validation_id: str) -> ValidationRecord | None:
        path = self._path(validation_id)
        if not path.exists():
            return None
        return ValidationRecord.from_dict(self._read_json(path))

    def list_validations(
        self,
        *,
        model_id: str | None = None,
        version: str | None = None,
    ) -> list[ValidationRecord]:
        if not self.base_dir.exists():
            return []
        records = [
            ValidationRecord.from_dict(self._read_json(path))
            for path in sorted(self.base_dir.glob("*.json"))
        ]
        if model_id is not None:
            records = [record for record in records if record.model_id == model_id]
        if version is not None:
            records = [record for record in records if record.version == version]
        return sorted(records, key=lambda record: (record.created_at, record.validation_id))

    def latest_validation(
        self,
        *,
        model_id: str,
        version: str,
    ) -> ValidationRecord | None:
        """Return the most recent validation record for one exact model version."""
        records = self.list_validations(model_id=model_id, version=version)
        if not records:
            return None
        return records[-1]


__all__ = [
    "LifecycleDefaultsConfig",
    "ObservabilityConfig",
    "PolicyDefaultsConfig",
    "ProviderDefaultConfig",
    "ProviderDefaultsConfig",
    "RuntimeDefaultsConfig",
    "ServerIdentityConfig",
    "SessionContextRecord",
    "SessionContextStore",
    "SnapshotRecord",
    "SnapshotStore",
    "TrellisServerConfig",
    "TrellisStatePaths",
    "ValidationRecord",
    "ValidationStore",
    "build_state_paths",
    "load_trellis_server_config",
    "resolve_state_root",
]
