"""Non-economic trade and source-document provenance.

Trade-envelope fields are intentionally outside Trellis' semantic contract.
They may identify, report, or reconcile a trade, but they are not pricing-route
or solver-selection inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math
from types import MappingProxyType
from typing import Any, Mapping


def _optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = (value or "").strip()
    return text or None


def _freeze_string_mapping(
    value: Mapping[str, str] | None,
    *,
    field_name: str,
) -> Mapping[str, str]:
    items: dict[str, str] = {}
    for key, item in (value or {}).items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{field_name} values must be non-empty strings")
        normalized_key = key.strip()
        if normalized_key in items:
            raise ValueError(
                f"{field_name} contains duplicate normalized key {normalized_key!r}"
            )
        items[normalized_key] = item.strip()
    return MappingProxyType(dict(sorted(items.items())))


def _freeze_metadata(value: object, *, path: str = "metadata") -> object:
    if value is None or isinstance(value, (str, int, bool, date, datetime)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError(f"{path} keys must be non-empty strings")
            normalized_key = key.strip()
            if normalized_key in frozen:
                raise ValueError(f"{path} contains duplicate key {normalized_key!r}")
            frozen[normalized_key] = _freeze_metadata(
                item,
                path=f"{path}.{normalized_key}",
            )
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only mappings, sequences, scalar values, or dates"
    )


def _summary_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _summary_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_summary_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class TradeParty:
    """One externally identified party carried outside contract economics."""

    party_id: str
    role: str | None = None
    name: str | None = None
    identifiers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        party_id = str(self.party_id or "").strip()
        if not party_id:
            raise ValueError("trade party party_id is required")
        object.__setattr__(self, "party_id", party_id)
        object.__setattr__(
            self,
            "role",
            _optional_text(self.role, field_name="trade party role"),
        )
        object.__setattr__(
            self,
            "name",
            _optional_text(self.name, field_name="trade party name"),
        )
        object.__setattr__(
            self,
            "identifiers",
            _freeze_string_mapping(
                self.identifiers,
                field_name="trade party identifiers",
            ),
        )


@dataclass(frozen=True)
class TradeEnvelope:
    """Immutable operational metadata surrounding one economic contract."""

    source_format: str
    source_view: str | None = None
    source_version: str | None = None
    document_id: str | None = None
    trade_id: str | None = None
    package_id: str | None = None
    trade_date: date | None = None
    lifecycle_state: str | None = None
    parties: tuple[TradeParty, ...] = ()
    identifiers: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_format, str):
            raise TypeError("trade envelope source_format must be a string")
        source_format = self.source_format.strip().lower()
        if not source_format:
            raise ValueError("trade envelope source_format is required")
        object.__setattr__(self, "source_format", source_format)
        for field_name in (
            "source_view",
            "source_version",
            "document_id",
            "trade_id",
            "package_id",
            "lifecycle_state",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(
                    getattr(self, field_name),
                    field_name=f"trade envelope {field_name}",
                ),
            )

        if self.trade_date is not None and (
            not isinstance(self.trade_date, date)
            or isinstance(self.trade_date, datetime)
        ):
            raise TypeError("trade envelope trade_date must be a date")

        parties = tuple(self.parties or ())
        if any(not isinstance(party, TradeParty) for party in parties):
            raise TypeError("trade envelope parties must contain TradeParty values")
        party_ids = [party.party_id for party in parties]
        duplicates = sorted(
            party_id for party_id in set(party_ids) if party_ids.count(party_id) > 1
        )
        if duplicates:
            raise ValueError(
                "duplicate trade party ids: " + ", ".join(duplicates)
            )
        object.__setattr__(
            self,
            "parties",
            tuple(sorted(parties, key=lambda party: (party.party_id, party.role or ""))),
        )
        object.__setattr__(
            self,
            "identifiers",
            _freeze_string_mapping(
                self.identifiers,
                field_name="trade envelope identifiers",
            ),
        )
        frozen_metadata = _freeze_metadata(self.metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise TypeError("trade envelope metadata must be a mapping")
        object.__setattr__(self, "metadata", frozen_metadata)


def trade_envelope_summary(envelope: TradeEnvelope | None) -> dict[str, object] | None:
    """Return a stable serializable provenance projection for diagnostics."""

    if envelope is None:
        return None
    if not isinstance(envelope, TradeEnvelope):
        raise TypeError("envelope must be a TradeEnvelope")
    return {
        "source_format": envelope.source_format,
        "source_view": envelope.source_view,
        "source_version": envelope.source_version,
        "document_id": envelope.document_id,
        "trade_id": envelope.trade_id,
        "package_id": envelope.package_id,
        "trade_date": envelope.trade_date.isoformat() if envelope.trade_date else None,
        "lifecycle_state": envelope.lifecycle_state,
        "parties": [
            {
                "party_id": party.party_id,
                "role": party.role,
                "name": party.name,
                "identifiers": _summary_value(party.identifiers),
            }
            for party in envelope.parties
        ],
        "identifiers": _summary_value(envelope.identifiers),
        "metadata": _summary_value(envelope.metadata),
    }


__all__ = ["TradeEnvelope", "TradeParty", "trade_envelope_summary"]
