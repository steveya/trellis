"""Canonical risk-factor identity and sparse risk-vector primitives."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


AxisInput = Mapping[str, object] | Iterable[tuple[str, object]]


def _clean_required(value: object, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_axis_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value).strip()


def _normalize_pairs(values: AxisInput | None, *, field_name: str) -> tuple[tuple[str, str], ...]:
    if values is None:
        return ()
    raw_items = values.items() if isinstance(values, Mapping) else values
    normalized: list[tuple[str, str]] = []
    for key, value in raw_items:
        normalized_key = _clean_required(key, f"{field_name} key")
        normalized.append((normalized_key, _normalize_axis_value(value)))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True)
class RiskFactorId:
    """Stable identity for one differentiable market or model coordinate."""

    object_type: str
    coordinate_type: str
    object_name: str = ""
    currency: str | None = None
    issuer: str | None = None
    axes: AxisInput | None = field(default_factory=tuple)
    provenance_namespace: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_type", _clean_required(self.object_type, "object_type"))
        object.__setattr__(
            self,
            "coordinate_type",
            _clean_required(self.coordinate_type, "coordinate_type"),
        )
        object.__setattr__(self, "object_name", str(self.object_name).strip())
        object.__setattr__(self, "currency", _clean_optional(self.currency))
        object.__setattr__(self, "issuer", _clean_optional(self.issuer))
        object.__setattr__(
            self,
            "provenance_namespace",
            _clean_optional(self.provenance_namespace),
        )
        object.__setattr__(
            self,
            "axes",
            _normalize_pairs(self.axes, field_name="axes"),
        )

    @property
    def key(self) -> str:
        """Return the canonical string key for this factor."""
        parts = [
            f"type={self.object_type}",
            f"object={self.object_name}",
            f"coordinate={self.coordinate_type}",
        ]
        if self.currency is not None:
            parts.append(f"currency={self.currency}")
        if self.issuer is not None:
            parts.append(f"issuer={self.issuer}")
        parts.extend(f"{key}={value}" for key, value in self.axes)
        if self.provenance_namespace is not None:
            parts.append(f"namespace={self.provenance_namespace}")
        return "|".join(parts)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RiskFactorId):
            return NotImplemented
        return self.key < other.key

    def __str__(self) -> str:
        return self.key

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly representation."""
        return {
            "key": self.key,
            "object_type": self.object_type,
            "object_name": self.object_name,
            "coordinate_type": self.coordinate_type,
            "currency": self.currency,
            "issuer": self.issuer,
            "axes": dict(self.axes),
            "provenance_namespace": self.provenance_namespace,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RiskFactorId:
        """Build a factor id from :meth:`to_payload` output."""
        return cls(
            object_type=payload["object_type"],
            coordinate_type=payload["coordinate_type"],
            object_name=str(payload.get("object_name", "")),
            currency=payload.get("currency"),
            issuer=payload.get("issuer"),
            axes=payload.get("axes") or (),
            provenance_namespace=payload.get("provenance_namespace"),
        )


@dataclass(frozen=True)
class RiskFactorCoordinate:
    """Metadata describing where a risk factor lives and how it reports."""

    factor_id: RiskFactorId
    object_path: str = ""
    display_name: str = ""
    unit: str = ""
    transform: str = "identity"
    reporting_buckets: AxisInput | None = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_path", str(self.object_path).strip())
        object.__setattr__(self, "display_name", str(self.display_name).strip())
        object.__setattr__(self, "unit", str(self.unit).strip())
        object.__setattr__(self, "transform", _clean_required(self.transform, "transform"))
        object.__setattr__(
            self,
            "reporting_buckets",
            _normalize_pairs(self.reporting_buckets, field_name="reporting_buckets"),
        )

    def bucket(self, name: str, *, default: str | None = None) -> str | None:
        """Return a reporting bucket value by name."""
        buckets = dict(self.reporting_buckets)
        return buckets.get(str(name), default)

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly representation."""
        return {
            "factor_id": self.factor_id.to_payload(),
            "factor_key": self.factor_id.key,
            "object_path": self.object_path,
            "display_name": self.display_name,
            "unit": self.unit,
            "transform": self.transform,
            "reporting_buckets": dict(self.reporting_buckets),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RiskFactorCoordinate:
        """Build a coordinate from :meth:`to_payload` output."""
        return cls(
            factor_id=RiskFactorId.from_payload(payload["factor_id"]),
            object_path=str(payload.get("object_path", "")),
            display_name=str(payload.get("display_name", "")),
            unit=str(payload.get("unit", "")),
            transform=str(payload.get("transform", "identity")),
            reporting_buckets=payload.get("reporting_buckets") or (),
        )


@dataclass(frozen=True)
class SparseRiskVector(Mapping[RiskFactorId, float]):
    """Sparse sensitivities keyed by canonical risk-factor identity."""

    _items: tuple[tuple[RiskFactorId, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_items", self._normalize_items(self._items))

    @staticmethod
    def _normalize_items(items: Iterable[tuple[RiskFactorId, object]]) -> tuple[tuple[RiskFactorId, float], ...]:
        totals: dict[RiskFactorId, float] = {}
        for factor_id, value in items:
            if not isinstance(factor_id, RiskFactorId):
                raise TypeError("SparseRiskVector keys must be RiskFactorId instances")
            totals[factor_id] = totals.get(factor_id, 0.0) + float(value)
        return tuple(
            (factor_id, value)
            for factor_id, value in sorted(totals.items(), key=lambda item: item[0].key)
            if value != 0.0
        )

    @classmethod
    def from_items(cls, items: Iterable[tuple[RiskFactorId, object]]) -> SparseRiskVector:
        """Build a vector from factor/value pairs, aggregating duplicate factors."""
        return cls(tuple(items))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SparseRiskVector:
        """Build a vector from :meth:`to_payload` output."""
        entries = payload.get("values", ())
        return cls.from_items(
            (
                RiskFactorId.from_payload(entry["factor_id"]),
                entry["sensitivity"],
            )
            for entry in entries
        )

    def __iter__(self) -> Iterator[RiskFactorId]:
        return (factor_id for factor_id, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, factor_id: RiskFactorId) -> float:
        for candidate, value in self._items:
            if candidate == factor_id:
                return value
        raise KeyError(factor_id)

    def items(self) -> tuple[tuple[RiskFactorId, float], ...]:  # type: ignore[override]
        """Return factor/value pairs in deterministic order."""
        return self._items

    def __add__(self, other: SparseRiskVector) -> SparseRiskVector:
        if not isinstance(other, SparseRiskVector):
            return NotImplemented
        return SparseRiskVector.from_items((*self._items, *other._items))

    def scale(self, scalar: float) -> SparseRiskVector:
        """Return a vector scaled by *scalar*."""
        return SparseRiskVector.from_items(
            (factor_id, value * float(scalar))
            for factor_id, value in self._items
        )

    def filter(self, selected_factors: Iterable[RiskFactorId]) -> SparseRiskVector:
        """Return a vector containing only selected factors."""
        selected = set(selected_factors)
        return SparseRiskVector.from_items(
            (factor_id, value)
            for factor_id, value in self._items
            if factor_id in selected
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly representation."""
        return {
            "values": [
                {
                    "factor_id": factor_id.to_payload(),
                    "factor_key": factor_id.key,
                    "sensitivity": float(value),
                }
                for factor_id, value in self._items
            ],
        }

    def bucket_totals(
        self,
        coordinates: Iterable[RiskFactorCoordinate] | Mapping[RiskFactorId, RiskFactorCoordinate],
        bucket_name: str,
        *,
        default_bucket: str = "unbucketed",
    ) -> dict[str, float]:
        """Aggregate sensitivities by a named reporting bucket."""
        if isinstance(coordinates, Mapping):
            coordinate_map = dict(coordinates)
        else:
            coordinate_map = {coordinate.factor_id: coordinate for coordinate in coordinates}
        totals: dict[str, float] = {}
        for factor_id, value in self._items:
            coordinate = coordinate_map.get(factor_id)
            bucket = default_bucket
            if coordinate is not None:
                bucket = coordinate.bucket(bucket_name, default=default_bucket) or default_bucket
            totals[bucket] = totals.get(bucket, 0.0) + float(value)
        return dict(sorted(totals.items()))


__all__ = [
    "RiskFactorCoordinate",
    "RiskFactorId",
    "SparseRiskVector",
]
