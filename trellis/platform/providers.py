"""Governed provider registry and snapshot resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Callable, Mapping

from trellis.data.resolver import resolve_market_snapshot as resolve_market_snapshot_from_source
from trellis.data.schema import MarketSnapshot
from trellis.platform.context import ExecutionContext


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_token(value: str | None) -> str:
    """Normalize one provider token to a stable lowercase identifier."""
    text = str(value or "").strip().lower()
    for char in (" ", "-", "/"):
        text = text.replace(char, "_")
    return text


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable tuple of unique strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _json_safe(value):
    """Convert nested values into deterministic JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    attributes = {}
    for attr in ("spot", "domestic", "foreign", "tenors", "rates", "expiries", "strikes", "vols"):
        if hasattr(value, attr):
            attributes[attr] = _json_safe(getattr(value, attr))
    if attributes:
        return {
            "type": type(value).__name__,
            "attributes": attributes,
        }
    return {"type": type(value).__name__}


def _snapshot_identity_payload(snapshot: MarketSnapshot, *, provider_id: str) -> dict[str, object]:
    """Build the deterministic payload used for snapshot-id generation."""
    provenance = {
        key: value
        for key, value in dict(snapshot.provenance).items()
        if key not in {"provider_id", "requested_provider_id", "resolution_kind", "snapshot_id"}
    }
    return {
        "provider_id": provider_id,
        "source": snapshot.source,
        "as_of": snapshot.as_of.isoformat(),
        "default_discount_curve": snapshot.default_discount_curve,
        "default_vol_surface": snapshot.default_vol_surface,
        "default_credit_curve": snapshot.default_credit_curve,
        "default_fixing_history": snapshot.default_fixing_history,
        "default_state_space": snapshot.default_state_space,
        "default_underlier_spot": snapshot.default_underlier_spot,
        "default_local_vol_surface": snapshot.default_local_vol_surface,
        "default_jump_parameters": snapshot.default_jump_parameters,
        "default_model_parameters": snapshot.default_model_parameters,
        "discount_curves": {
            key: _json_safe(value) for key, value in sorted(snapshot.discount_curves.items())
        },
        "forecast_curves": {
            key: _json_safe(value) for key, value in sorted(snapshot.forecast_curves.items())
        },
        "vol_surfaces": {
            key: _json_safe(value) for key, value in sorted(snapshot.vol_surfaces.items())
        },
        "credit_curves": {
            key: _json_safe(value) for key, value in sorted(snapshot.credit_curves.items())
        },
        "fixing_histories": {
            key: _json_safe(value) for key, value in sorted(snapshot.fixing_histories.items())
        },
        "fx_rates": {
            key: _json_safe(value) for key, value in sorted(snapshot.fx_rates.items())
        },
        "state_spaces": {
            key: _json_safe(value) for key, value in sorted(snapshot.state_spaces.items())
        },
        "underlier_spots": {
            key: float(value) for key, value in sorted(snapshot.underlier_spots.items())
        },
        "local_vol_surfaces": {
            key: _json_safe(value) for key, value in sorted(snapshot.local_vol_surfaces.items())
        },
        "jump_parameter_sets": {
            key: _json_safe(value) for key, value in sorted(snapshot.jump_parameter_sets.items())
        },
        "model_parameter_sets": {
            key: _json_safe(value) for key, value in sorted(snapshot.model_parameter_sets.items())
        },
        "metadata": _json_safe(snapshot.metadata),
        "provenance": _json_safe(provenance),
    }


def _snapshot_id(snapshot: MarketSnapshot, *, provider_id: str) -> str:
    """Return the deterministic snapshot id for one governed snapshot."""
    payload = _snapshot_identity_payload(snapshot, provider_id=provider_id)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"snapshot_{digest}"


def _snapshot_with_identity(
    snapshot: MarketSnapshot,
    *,
    provider_id: str,
    requested_provider_id: str,
    resolution_kind: str,
) -> MarketSnapshot:
    """Attach canonical provider and snapshot identity to one snapshot."""
    provenance = dict(snapshot.provenance)
    provenance.setdefault("source", snapshot.source)
    provenance["provider_id"] = provider_id
    if requested_provider_id and requested_provider_id != provider_id:
        provenance["requested_provider_id"] = requested_provider_id
    provenance["resolution_kind"] = resolution_kind
    provenance["snapshot_id"] = str(provenance.get("snapshot_id") or _snapshot_id(snapshot, provider_id=provider_id))
    return replace(snapshot, provenance=provenance)


class ProviderBindingRequiredError(ValueError):
    """Raised when governed resolution is missing an explicit provider binding."""


class UnknownProviderError(KeyError):
    """Raised when a requested provider id is not registered."""


class ProviderResolutionError(RuntimeError):
    """Raised when a bound provider cannot resolve a governed snapshot."""


class MockDataNotAllowedError(ProviderResolutionError):
    """Raised when governed policy forbids mock provider usage."""


@dataclass(frozen=True)
class ProviderRecord:
    """Canonical provider record for governed runtime bindings."""

    provider_id: str
    kind: str
    display_name: str = ""
    status: str = "available"
    capabilities: tuple[str, ...] = ()
    connection_mode: str = "embedded"
    config_summary: Mapping[str, object] = field(default_factory=dict)
    is_mock: bool = False
    supports_snapshots: bool = True
    supports_streaming: bool = False
    source: str = ""

    def __post_init__(self):
        """Normalize identifiers and freeze config summary."""
        provider_id = _normalize_token(self.provider_id)
        if not provider_id:
            raise ValueError("provider_id is required")
        kind = _normalize_token(self.kind)
        if not kind:
            raise ValueError("kind is required")
        source = _normalize_token(self.source)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", _normalize_token(self.status) or "available")
        object.__setattr__(self, "capabilities", _string_tuple(self.capabilities))
        object.__setattr__(self, "connection_mode", _normalize_token(self.connection_mode) or "embedded")
        object.__setattr__(self, "config_summary", _freeze_mapping(self.config_summary))
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "provider_id": self.provider_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "status": self.status,
            "capabilities": list(self.capabilities),
            "connection_mode": self.connection_mode,
            "config_summary": dict(self.config_summary),
            "is_mock": self.is_mock,
            "supports_snapshots": self.supports_snapshots,
            "supports_streaming": self.supports_streaming,
            "source": self.source,
        }


def _default_provider_records() -> tuple[ProviderRecord, ...]:
    """Return the built-in governed market-data providers."""
    return (
        ProviderRecord(
            provider_id="market_data.mock",
            kind="market_data",
            display_name="Embedded Mock Market Data",
            capabilities=(
                "discount_curve",
                "market_snapshot",
                "fixing_history",
                "forecast_curve",
                "vol_surface",
                "credit_curve",
                "fx_rates",
            ),
            connection_mode="embedded",
            is_mock=True,
            supports_snapshots=True,
            source="mock",
        ),
        ProviderRecord(
            provider_id="market_data.treasury_gov",
            kind="market_data",
            display_name="Treasury.gov",
            capabilities=("discount_curve", "market_snapshot"),
            connection_mode="http_pull",
            supports_snapshots=True,
            source="treasury_gov",
        ),
        ProviderRecord(
            provider_id="market_data.fred",
            kind="market_data",
            display_name="FRED Treasury Series",
            capabilities=("discount_curve", "market_snapshot"),
            connection_mode="http_pull",
            supports_snapshots=True,
            source="fred",
        ),
        ProviderRecord(
            provider_id="market_data.file_import",
            kind="market_data",
            display_name="Explicit File Import",
            capabilities=(
                "discount_curve",
                "forecast_curve",
                "vol_surface",
                "credit_curve",
                "fx_rates",
                "market_snapshot",
                "fixing_history",
                "underlier_spot",
            ),
            connection_mode="local_file",
            supports_snapshots=True,
            source="explicit_input",
        ),
    )


def _default_provider_factories() -> dict[str, Callable[[], object]]:
    """Return provider factories keyed by stable provider id."""
    from trellis.data.fred import FredDataProvider
    from trellis.data.mock import MockDataProvider
    from trellis.data.treasury_gov import TreasuryGovDataProvider

    return {
        "market_data.mock": MockDataProvider,
        "market_data.treasury_gov": TreasuryGovDataProvider,
        "market_data.fred": FredDataProvider,
    }


class ProviderRegistry:
    """Local-first registry for governed provider records."""

    def __init__(
        self,
        *,
        records=(),
        provider_factories: Mapping[str, Callable[[], object]] | None = None,
    ):
        merged_records = {record.provider_id: record for record in _default_provider_records()}
        for record in records or ():
            merged_records[record.provider_id] = record
        merged_factories = _default_provider_factories()
        merged_factories.update(dict(provider_factories or {}))
        self._records = MappingProxyType(merged_records)
        self._provider_factories = MappingProxyType(merged_factories)

    def list_providers(self, *, kind: str | None = None) -> tuple[ProviderRecord, ...]:
        """Return registered providers, optionally filtered by kind."""
        filtered = tuple(self._records.values())
        if kind is not None:
            normalized_kind = _normalize_token(kind)
            filtered = tuple(record for record in filtered if record.kind == normalized_kind)
        return tuple(sorted(filtered, key=lambda record: record.provider_id))

    def get_provider(self, provider_id: str) -> ProviderRecord:
        """Return one provider record by stable provider id."""
        normalized = _normalize_token(provider_id)
        try:
            return self._records[normalized]
        except KeyError as exc:
            raise UnknownProviderError(f"Unknown provider id: {provider_id!r}") from exc

    def _instantiate_provider(self, record: ProviderRecord):
        """Instantiate the provider implementation for one registry record."""
        try:
            factory = self._provider_factories[record.provider_id]
        except KeyError as exc:
            raise ProviderResolutionError(
                f"No provider factory is registered for {record.provider_id}"
            ) from exc
        try:
            return factory()
        except Exception as exc:  # pragma: no cover - exercised via governed failure tests
            raise ProviderResolutionError(
                f"Failed to initialize provider {record.provider_id}: {exc}"
            ) from exc

    def resolve_market_snapshot(
        self,
        *,
        provider_id: str,
        as_of: date | str | None = None,
        fallback_provider_id: str | None = None,
        allow_mock_data: bool = False,
        **resolver_kwargs,
    ) -> MarketSnapshot:
        """Resolve one governed market snapshot from explicit provider bindings."""
        primary_record = self.get_provider(provider_id)
        try:
            return self._resolve_one(
                primary_record,
                requested_provider_id=primary_record.provider_id,
                as_of=as_of,
                allow_mock_data=allow_mock_data,
                resolution_kind="primary",
                **resolver_kwargs,
            )
        except MockDataNotAllowedError:
            raise
        except Exception as primary_exc:
            if not fallback_provider_id:
                raise ProviderResolutionError(
                    f"Governed snapshot resolution failed for {primary_record.provider_id}: {primary_exc}"
                ) from primary_exc

        fallback_record = self.get_provider(fallback_provider_id)
        try:
            return self._resolve_one(
                fallback_record,
                requested_provider_id=primary_record.provider_id,
                as_of=as_of,
                allow_mock_data=allow_mock_data,
                resolution_kind="fallback",
                **resolver_kwargs,
            )
        except MockDataNotAllowedError:
            raise
        except Exception as fallback_exc:
            raise ProviderResolutionError(
                f"Governed snapshot resolution failed for {primary_record.provider_id} and fallback {fallback_record.provider_id}: {fallback_exc}"
            ) from fallback_exc

    def _resolve_one(
        self,
        record: ProviderRecord,
        *,
        requested_provider_id: str,
        as_of: date | str | None = None,
        allow_mock_data: bool = False,
        resolution_kind: str = "primary",
        **resolver_kwargs,
    ) -> MarketSnapshot:
        """Resolve one snapshot through one registered provider record."""
        if record.is_mock and not allow_mock_data:
            raise MockDataNotAllowedError(
                f"Governed execution cannot use mock provider {record.provider_id} without explicit policy allowance"
            )
        provider = self._instantiate_provider(record)
        snapshot = resolve_market_snapshot_from_source(
            as_of=as_of,
            source=record.source or record.provider_id,
            provider=provider,
            **resolver_kwargs,
        )
        return _snapshot_with_identity(
            snapshot,
            provider_id=record.provider_id,
            requested_provider_id=requested_provider_id,
            resolution_kind=resolution_kind,
        )


def resolve_governed_market_snapshot(
    *,
    execution_context: ExecutionContext,
    as_of: date | str | None = None,
    registry: ProviderRegistry | None = None,
    **resolver_kwargs,
) -> MarketSnapshot:
    """Resolve one governed snapshot from explicit market-data provider bindings."""
    binding_set = execution_context.provider_bindings.market_data
    if binding_set.primary is None:
        raise ProviderBindingRequiredError(
            "Governed execution requires an explicit market-data provider binding"
        )
    resolved_registry = registry or ProviderRegistry()
    return resolved_registry.resolve_market_snapshot(
        provider_id=binding_set.primary.provider_id,
        fallback_provider_id=(
            None if binding_set.fallback is None else binding_set.fallback.provider_id
        ),
        as_of=as_of,
        allow_mock_data=execution_context.allow_mock_data,
        **resolver_kwargs,
    )
