"""Governed runtime context records for the platform boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_token(value: str | None, *, fallback: str) -> str:
    """Normalize one record token to a stable lowercase identifier."""
    text = str(value or "").strip().lower()
    for char in (" ", "-", "/"):
        text = text.replace(char, "_")
    return text or fallback


def _coerce_provider_binding(value) -> ProviderBinding | None:
    """Convert nested dict payloads into provider-binding records."""
    if value is None or isinstance(value, ProviderBinding):
        return value
    if isinstance(value, Mapping):
        return ProviderBinding.from_dict(value)
    raise TypeError(f"Unsupported provider binding payload: {type(value)!r}")


def _coerce_binding_set(value) -> ProviderBindingSet:
    """Convert nested dict payloads into provider-binding-set records."""
    if isinstance(value, ProviderBindingSet):
        return value
    if isinstance(value, Mapping):
        return ProviderBindingSet.from_dict(value)
    if value is None:
        return ProviderBindingSet()
    raise TypeError(f"Unsupported provider binding set payload: {type(value)!r}")


def _default_session_id(prefix: str = "session") -> str:
    """Generate a stable-looking session identifier for runtime contexts."""
    return f"{prefix}_{uuid4().hex[:12]}"


def _snapshot_source(snapshot) -> str | None:
    """Infer one market-source label from a MarketSnapshot-like object."""
    if snapshot is None:
        return None
    source = getattr(snapshot, "source", None)
    if source:
        return _normalize_token(source, fallback="unbound_market_snapshot")
    provenance = getattr(snapshot, "provenance", None) or {}
    source_kind = provenance.get("source_kind")
    if source_kind:
        return _normalize_token(source_kind, fallback="unbound_market_snapshot")
    return _normalize_token(type(snapshot).__name__, fallback="unbound_market_snapshot")


def _snapshot_provider_id(snapshot) -> str:
    """Return the canonical provider id for one snapshot when available."""
    if snapshot is None:
        return ""
    provider_id = getattr(snapshot, "provider_id", None)
    if provider_id:
        return _normalize_token(provider_id, fallback="")
    provenance = getattr(snapshot, "provenance", None) or {}
    return _normalize_token(provenance.get("provider_id"), fallback="")


def _snapshot_source_kind(snapshot) -> str:
    """Infer the source-kind label from a MarketSnapshot-like object."""
    if snapshot is None:
        return ""
    provenance = getattr(snapshot, "provenance", None) or {}
    return _normalize_token(provenance.get("source_kind"), fallback="")


def _market_source_label(
    *,
    market_snapshot=None,
    market_source: str | None = None,
) -> str:
    """Return the canonical market-source label for one runtime surface."""
    snapshot_source = _snapshot_source(market_snapshot)
    if snapshot_source is not None:
        source_kind = _snapshot_source_kind(market_snapshot)
        if source_kind == "explicit_input":
            return "explicit_input"
        return snapshot_source
    return _normalize_token(market_source, fallback="unbound_market_snapshot")


class RunMode(str, Enum):
    """Governed runtime modes for execution policy and provenance."""

    SANDBOX = "sandbox"
    RESEARCH = "research"
    PRODUCTION = "production"

    @classmethod
    def normalize(cls, value: RunMode | str | None) -> RunMode:
        """Coerce one runtime-mode input into the canonical enum."""
        if isinstance(value, cls):
            return value
        token = _normalize_token(value, fallback=cls.SANDBOX.value)
        for member in cls:
            if member.value == token:
                return member
        raise ValueError(f"Unknown run mode: {value!r}")


@dataclass(frozen=True)
class ProviderBinding:
    """One explicit provider binding record."""

    provider_id: str
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize binding ids and freeze metadata for persistence."""
        provider_id = _normalize_token(self.provider_id, fallback="")
        if not provider_id:
            raise ValueError("provider_id is required")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload for persistence and transport."""
        return {
            "provider_id": self.provider_id,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProviderBinding:
        """Rehydrate one provider binding from a serialized payload."""
        return cls(
            provider_id=str(payload.get("provider_id", "")).strip(),
            label=str(payload.get("label", "")).strip(),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ProviderBindingSet:
    """Typed primary/fallback slots for one provider family."""

    primary: ProviderBinding | None = None
    fallback: ProviderBinding | None = None

    def __post_init__(self):
        """Coerce nested dict payloads into binding records."""
        object.__setattr__(self, "primary", _coerce_provider_binding(self.primary))
        object.__setattr__(self, "fallback", _coerce_provider_binding(self.fallback))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload for this provider family."""
        return {
            "primary": None if self.primary is None else self.primary.to_dict(),
            "fallback": None if self.fallback is None else self.fallback.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProviderBindingSet:
        """Rehydrate one binding set from a serialized payload."""
        return cls(
            primary=payload.get("primary"),
            fallback=payload.get("fallback"),
        )


@dataclass(frozen=True)
class ProviderBindings:
    """Explicit provider ids grouped by runtime concern."""

    market_data: ProviderBindingSet = field(default_factory=ProviderBindingSet)
    pricing_engine: ProviderBindingSet = field(default_factory=ProviderBindingSet)
    model_store: ProviderBindingSet = field(default_factory=ProviderBindingSet)
    validation_engine: ProviderBindingSet = field(default_factory=ProviderBindingSet)

    def __post_init__(self):
        """Coerce nested dict payloads into canonical binding sets."""
        object.__setattr__(self, "market_data", _coerce_binding_set(self.market_data))
        object.__setattr__(self, "pricing_engine", _coerce_binding_set(self.pricing_engine))
        object.__setattr__(self, "model_store", _coerce_binding_set(self.model_store))
        object.__setattr__(self, "validation_engine", _coerce_binding_set(self.validation_engine))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload for persistence and transport."""
        return {
            "market_data": self.market_data.to_dict(),
            "pricing_engine": self.pricing_engine.to_dict(),
            "model_store": self.model_store.to_dict(),
            "validation_engine": self.validation_engine.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProviderBindings:
        """Rehydrate provider bindings from a serialized payload."""
        return cls(
            market_data=payload.get("market_data"),
            pricing_engine=payload.get("pricing_engine"),
            model_store=payload.get("model_store"),
            validation_engine=payload.get("validation_engine"),
        )


def infer_run_mode(
    *,
    market_snapshot=None,
    market_source: str | None = None,
) -> RunMode:
    """Infer a default governed run mode from the current market surface."""
    source = _market_source_label(
        market_snapshot=market_snapshot,
        market_source=market_source,
    )
    source_kind = _snapshot_source_kind(market_snapshot)
    if source in {"mock", "explicit_input", "unit", "test"}:
        return RunMode.SANDBOX
    if source_kind in {"synthetic_snapshot", "explicit_input"}:
        return RunMode.SANDBOX
    return RunMode.RESEARCH


def default_policy_bundle_id(run_mode: RunMode | str | None) -> str:
    """Return the default policy bundle identifier for one run mode."""
    normalized = RunMode.normalize(run_mode)
    return f"policy_bundle.{normalized.value}.default"


def default_provider_bindings(
    *,
    market_snapshot=None,
    market_source: str | None = None,
) -> ProviderBindings:
    """Build the default provider-binding scaffold for one runtime surface."""
    provider_id = _snapshot_provider_id(market_snapshot)
    source = _market_source_label(
        market_snapshot=market_snapshot,
        market_source=market_source,
    )
    return ProviderBindings(
        market_data=ProviderBindingSet(
            primary=ProviderBinding(provider_id or f"market_data.{source}"),
        ),
        pricing_engine=ProviderBindingSet(
            primary=ProviderBinding("pricing_engine.local"),
        ),
        model_store=ProviderBindingSet(
            primary=ProviderBinding("model_store.local"),
        ),
        validation_engine=ProviderBindingSet(
            primary=ProviderBinding("validation_engine.local"),
        ),
    )


@dataclass(frozen=True)
class ExecutionContext:
    """Canonical governed runtime state carried separately from request intent."""

    session_id: str
    run_mode: RunMode = RunMode.SANDBOX
    provider_bindings: ProviderBindings = field(default_factory=ProviderBindings)
    policy_bundle_id: str = ""
    allow_mock_data: bool = False
    require_provider_disclosure: bool = True
    default_output_mode: str = "result_only"
    default_audit_mode: str = "summary"
    requested_persistence: str = "ephemeral"
    requested_snapshot_policy: str = "prefer_bound_snapshot"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize nested records and freeze metadata for persistence."""
        object.__setattr__(
            self,
            "session_id",
            str(self.session_id or _default_session_id()).strip(),
        )
        object.__setattr__(self, "run_mode", RunMode.normalize(self.run_mode))
        object.__setattr__(
            self,
            "provider_bindings",
            self.provider_bindings
            if isinstance(self.provider_bindings, ProviderBindings)
            else ProviderBindings.from_dict(self.provider_bindings),
        )
        policy_bundle_id = str(self.policy_bundle_id or "").strip()
        if not policy_bundle_id:
            policy_bundle_id = default_policy_bundle_id(self.run_mode)
        object.__setattr__(self, "policy_bundle_id", policy_bundle_id)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload for persistence and transport."""
        return {
            "session_id": self.session_id,
            "run_mode": self.run_mode.value,
            "provider_bindings": self.provider_bindings.to_dict(),
            "policy_bundle_id": self.policy_bundle_id,
            "allow_mock_data": self.allow_mock_data,
            "require_provider_disclosure": self.require_provider_disclosure,
            "default_output_mode": self.default_output_mode,
            "default_audit_mode": self.default_audit_mode,
            "requested_persistence": self.requested_persistence,
            "requested_snapshot_policy": self.requested_snapshot_policy,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExecutionContext:
        """Rehydrate one execution context from a serialized payload."""
        return cls(
            session_id=str(payload.get("session_id", "")).strip(),
            run_mode=payload.get("run_mode"),
            provider_bindings=payload.get("provider_bindings") or {},
            policy_bundle_id=str(payload.get("policy_bundle_id", "")).strip(),
            allow_mock_data=bool(payload.get("allow_mock_data", False)),
            require_provider_disclosure=bool(
                payload.get("require_provider_disclosure", True)
            ),
            default_output_mode=str(
                payload.get("default_output_mode", "result_only")
            ).strip(),
            default_audit_mode=str(
                payload.get("default_audit_mode", "summary")
            ).strip(),
            requested_persistence=str(
                payload.get("requested_persistence", "ephemeral")
            ).strip(),
            requested_snapshot_policy=str(
                payload.get("requested_snapshot_policy", "prefer_bound_snapshot")
            ).strip(),
            metadata=payload.get("metadata") or {},
        )


def build_execution_context(
    *,
    session_id: str | None = None,
    market_snapshot=None,
    market_source: str | None = None,
    run_mode: RunMode | str | None = None,
    provider_bindings: ProviderBindings | Mapping[str, object] | None = None,
    policy_bundle_id: str | None = None,
    allow_mock_data: bool | None = None,
    require_provider_disclosure: bool | None = None,
    default_output_mode: str = "result_only",
    default_audit_mode: str = "summary",
    requested_persistence: str = "ephemeral",
    requested_snapshot_policy: str = "prefer_bound_snapshot",
    metadata: Mapping[str, object] | None = None,
) -> ExecutionContext:
    """Build one explicit governed runtime context from a current request surface."""
    resolved_run_mode = RunMode.normalize(
        run_mode
        if run_mode is not None
        else infer_run_mode(
            market_snapshot=market_snapshot,
            market_source=market_source,
        )
    )
    resolved_provider_bindings = (
        provider_bindings
        if provider_bindings is not None
        else default_provider_bindings(
            market_snapshot=market_snapshot,
            market_source=market_source,
        )
    )
    resolved_allow_mock_data = (
        allow_mock_data
        if allow_mock_data is not None
        else resolved_run_mode is RunMode.SANDBOX
    )
    resolved_require_provider_disclosure = (
        require_provider_disclosure
        if require_provider_disclosure is not None
        else resolved_run_mode is not RunMode.SANDBOX
    )
    return ExecutionContext(
        session_id=str(session_id or _default_session_id()).strip(),
        run_mode=resolved_run_mode,
        provider_bindings=resolved_provider_bindings,
        policy_bundle_id=str(policy_bundle_id or "").strip(),
        allow_mock_data=resolved_allow_mock_data,
        require_provider_disclosure=resolved_require_provider_disclosure,
        default_output_mode=default_output_mode,
        default_audit_mode=default_audit_mode,
        requested_persistence=requested_persistence,
        requested_snapshot_policy=requested_snapshot_policy,
        metadata=metadata or {},
    )


def execution_context_from_session(
    session,
    *,
    run_mode: RunMode | str | None = None,
    provider_bindings: ProviderBindings | Mapping[str, object] | None = None,
    policy_bundle_id: str | None = None,
    allow_mock_data: bool | None = None,
    require_provider_disclosure: bool | None = None,
    default_output_mode: str = "result_only",
    default_audit_mode: str = "summary",
    requested_persistence: str = "ephemeral",
    requested_snapshot_policy: str = "prefer_bound_snapshot",
    metadata: Mapping[str, object] | None = None,
) -> ExecutionContext:
    """Normalize one Session into the governed runtime context record."""
    return build_execution_context(
        session_id=getattr(session, "session_id", None),
        market_snapshot=getattr(session, "market_snapshot", None),
        run_mode=run_mode,
        provider_bindings=provider_bindings,
        policy_bundle_id=policy_bundle_id,
        allow_mock_data=allow_mock_data,
        require_provider_disclosure=require_provider_disclosure,
        default_output_mode=default_output_mode,
        default_audit_mode=default_audit_mode,
        requested_persistence=requested_persistence,
        requested_snapshot_policy=requested_snapshot_policy,
        metadata={
            "surface": "session",
            "agent_enabled": bool(getattr(session, "agent_enabled", False)),
            **dict(metadata or {}),
        },
    )


def execution_context_from_pipeline(
    pipeline,
    *,
    run_mode: RunMode | str | None = None,
    provider_bindings: ProviderBindings | Mapping[str, object] | None = None,
    policy_bundle_id: str | None = None,
    allow_mock_data: bool | None = None,
    require_provider_disclosure: bool | None = None,
    default_output_mode: str = "result_only",
    default_audit_mode: str = "summary",
    requested_persistence: str = "ephemeral",
    requested_snapshot_policy: str = "prefer_bound_snapshot",
    metadata: Mapping[str, object] | None = None,
) -> ExecutionContext:
    """Normalize one Pipeline configuration into governed runtime context."""
    return build_execution_context(
        session_id=getattr(pipeline, "_session_id", None) or _default_session_id("pipeline"),
        market_snapshot=getattr(pipeline, "_market_snapshot", None),
        market_source=getattr(pipeline, "_data_source", None),
        run_mode=run_mode,
        provider_bindings=provider_bindings,
        policy_bundle_id=policy_bundle_id,
        allow_mock_data=allow_mock_data,
        require_provider_disclosure=require_provider_disclosure,
        default_output_mode=default_output_mode,
        default_audit_mode=default_audit_mode,
        requested_persistence=requested_persistence,
        requested_snapshot_policy=requested_snapshot_policy,
        metadata={
            "surface": "pipeline",
            "scenario_count": len(getattr(pipeline, "_scenarios", None) or [{"name": "base"}]),
            **dict(metadata or {}),
        },
    )


def execution_context_summary(context: ExecutionContext) -> dict[str, object]:
    """Return a YAML-safe summary of the governed runtime context."""
    return context.to_dict()
