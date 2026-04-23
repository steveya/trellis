"""Typed calibrated-object materialization helpers for MarketState bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from trellis.core.market_state import MarketState

CalibratedObjectKind = Literal[
    "model_parameter_set",
    "black_vol_surface",
    "local_vol_surface",
    "credit_curve",
    "correlation_surface",
]

_SUPPORTED_OBJECT_KINDS = frozenset(
    {
        "model_parameter_set",
        "black_vol_surface",
        "local_vol_surface",
        "credit_curve",
        "correlation_surface",
    }
)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping copy."""
    return MappingProxyType(dict(mapping or {}))


def _normalized_string_mapping(mapping: Mapping[str, object] | None) -> dict[str, str]:
    """Return a compact mapping with non-empty string keys and values."""
    result: dict[str, str] = {}
    for key, value in dict(mapping or {}).items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def _to_nested_dict(value: object) -> dict[str, object]:
    """Return a shallow dict copy if ``value`` is mapping-like, else an empty dict."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


@dataclass(frozen=True)
class CalibratedObjectMaterialization:
    """Typed record of how one calibrated object is bound onto ``MarketState``."""

    object_kind: CalibratedObjectKind
    object_name: str
    target_fields: tuple[str, ...]
    source_kind: str
    source_ref: str = ""
    selected_curve_roles: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.object_kind not in _SUPPORTED_OBJECT_KINDS:
            raise ValueError(f"Unsupported calibrated object kind: {self.object_kind}")
        normalized_name = str(self.object_name).strip()
        if not normalized_name:
            raise ValueError("object_name must be non-empty")
        object.__setattr__(self, "object_name", normalized_name)
        target_fields = tuple(str(field_name).strip() for field_name in self.target_fields if str(field_name).strip())
        if not target_fields:
            raise ValueError("target_fields must be non-empty")
        object.__setattr__(self, "target_fields", target_fields)
        normalized_source_kind = str(self.source_kind).strip()
        if not normalized_source_kind:
            raise ValueError("source_kind must be non-empty")
        object.__setattr__(self, "source_kind", normalized_source_kind)
        object.__setattr__(self, "source_ref", str(self.source_ref).strip())
        object.__setattr__(
            self,
            "selected_curve_roles",
            MappingProxyType(_normalized_string_mapping(self.selected_curve_roles)),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly materialization payload."""
        payload: dict[str, object] = {
            "object_kind": self.object_kind,
            "object_name": self.object_name,
            "target_fields": list(self.target_fields),
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "selected_curve_roles": dict(self.selected_curve_roles),
            "metadata": dict(self.metadata),
        }
        return payload


def _record_materialization(
    market_state: MarketState,
    record: CalibratedObjectMaterialization,
) -> MarketState:
    """Attach a calibrated-object materialization record to ``market_state`` provenance."""
    market_provenance = _to_nested_dict(getattr(market_state, "market_provenance", None))
    calibrated_objects = _to_nested_dict(market_provenance.get("calibrated_objects"))
    kind_entries = _to_nested_dict(calibrated_objects.get(record.object_kind))
    kind_entries[record.object_name] = record.to_payload()
    calibrated_objects[record.object_kind] = kind_entries
    selected_objects = _to_nested_dict(market_provenance.get("selected_calibrated_objects"))
    selected_objects[record.object_kind] = record.object_name
    market_provenance["calibrated_objects"] = calibrated_objects
    market_provenance["selected_calibrated_objects"] = selected_objects
    return replace(market_state, market_provenance=market_provenance)


def materialize_model_parameter_set(
    market_state: MarketState,
    *,
    parameter_set_name: str,
    model_parameters: Mapping[str, object],
    source_kind: str,
    source_ref: str = "",
    selected_curve_roles: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MarketState:
    """Materialize a calibrated model-parameter pack onto ``MarketState``."""
    parameter_sets = dict(market_state.model_parameter_sets or {})
    parameter_payload = dict(model_parameters)
    parameter_sets[parameter_set_name] = parameter_payload
    updated_state = replace(
        market_state,
        model_parameters=parameter_payload,
        model_parameter_sets=parameter_sets,
    )
    return _record_materialization(
        updated_state,
        CalibratedObjectMaterialization(
            object_kind="model_parameter_set",
            object_name=parameter_set_name,
            target_fields=("model_parameters", "model_parameter_sets"),
            source_kind=source_kind,
            source_ref=source_ref,
            selected_curve_roles=selected_curve_roles or {},
            metadata=metadata or {},
        ),
    )


def materialize_black_vol_surface(
    market_state: MarketState,
    *,
    surface_name: str,
    vol_surface: object,
    source_kind: str,
    source_ref: str = "",
    selected_curve_roles: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MarketState:
    """Materialize a calibrated Black-vol surface onto ``MarketState``."""
    updated_state = replace(market_state, vol_surface=vol_surface)
    return _record_materialization(
        updated_state,
        CalibratedObjectMaterialization(
            object_kind="black_vol_surface",
            object_name=surface_name,
            target_fields=("vol_surface",),
            source_kind=source_kind,
            source_ref=source_ref,
            selected_curve_roles=selected_curve_roles or {},
            metadata=metadata or {},
        ),
    )


def materialize_local_vol_surface(
    market_state: MarketState,
    *,
    surface_name: str,
    local_vol_surface: object,
    source_kind: str,
    source_ref: str = "",
    selected_curve_roles: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MarketState:
    """Materialize a calibrated local-vol surface onto ``MarketState``."""
    surface_map = dict(market_state.local_vol_surfaces or {})
    surface_map[surface_name] = local_vol_surface
    updated_state = replace(
        market_state,
        local_vol_surface=local_vol_surface,
        local_vol_surfaces=surface_map,
    )
    return _record_materialization(
        updated_state,
        CalibratedObjectMaterialization(
            object_kind="local_vol_surface",
            object_name=surface_name,
            target_fields=("local_vol_surface", "local_vol_surfaces"),
            source_kind=source_kind,
            source_ref=source_ref,
            selected_curve_roles=selected_curve_roles or {},
            metadata=metadata or {},
        ),
    )


def materialize_credit_curve(
    market_state: MarketState,
    *,
    curve_name: str,
    credit_curve: object,
    source_kind: str,
    source_ref: str = "",
    selected_curve_roles: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MarketState:
    """Materialize a calibrated credit curve onto ``MarketState``."""
    updated_state = replace(market_state, credit_curve=credit_curve)
    return _record_materialization(
        updated_state,
        CalibratedObjectMaterialization(
            object_kind="credit_curve",
            object_name=curve_name,
            target_fields=("credit_curve",),
            source_kind=source_kind,
            source_ref=source_ref,
            selected_curve_roles=selected_curve_roles or {},
            metadata=metadata or {},
        ),
    )


def materialize_correlation_surface(
    market_state: MarketState,
    *,
    surface_name: str,
    correlation_surface: object,
    source_kind: str,
    source_ref: str = "",
    selected_curve_roles: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MarketState:
    """Materialize a calibrated correlation surface onto ``MarketState``."""
    surface_map = dict(market_state.correlation_surfaces or {})
    surface_map[surface_name] = correlation_surface
    updated_state = replace(
        market_state,
        correlation_surface=correlation_surface,
        correlation_surfaces=surface_map,
    )
    return _record_materialization(
        updated_state,
        CalibratedObjectMaterialization(
            object_kind="correlation_surface",
            object_name=surface_name,
            target_fields=("correlation_surface", "correlation_surfaces"),
            source_kind=source_kind,
            source_ref=source_ref,
            selected_curve_roles=selected_curve_roles or {},
            metadata=metadata or {},
        ),
    )


def resolve_materialized_object(
    market_state: MarketState,
    *,
    object_kind: CalibratedObjectKind,
    object_name: str | None = None,
) -> dict[str, object] | None:
    """Return one materialized-object record from ``market_state`` provenance."""
    market_provenance = _to_nested_dict(getattr(market_state, "market_provenance", None))
    calibrated_objects = _to_nested_dict(market_provenance.get("calibrated_objects"))
    object_entries = _to_nested_dict(calibrated_objects.get(object_kind))
    selected_objects = _to_nested_dict(market_provenance.get("selected_calibrated_objects"))
    selected_name = object_name
    if selected_name is None:
        selected_name = selected_objects.get(object_kind)
    if not isinstance(selected_name, str) or not selected_name:
        return None
    payload = object_entries.get(selected_name)
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


__all__ = [
    "CalibratedObjectKind",
    "CalibratedObjectMaterialization",
    "materialize_model_parameter_set",
    "materialize_black_vol_surface",
    "materialize_local_vol_surface",
    "materialize_credit_curve",
    "materialize_correlation_surface",
    "resolve_materialized_object",
]
