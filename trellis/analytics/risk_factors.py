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
    support_status: str = "supported"
    reporting_buckets: AxisInput | None = field(default_factory=tuple)
    metadata: AxisInput | None = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_path", str(self.object_path).strip())
        object.__setattr__(self, "display_name", str(self.display_name).strip())
        object.__setattr__(self, "unit", str(self.unit).strip())
        object.__setattr__(self, "transform", _clean_required(self.transform, "transform"))
        object.__setattr__(
            self,
            "support_status",
            _clean_required(self.support_status, "support_status"),
        )
        object.__setattr__(
            self,
            "reporting_buckets",
            _normalize_pairs(self.reporting_buckets, field_name="reporting_buckets"),
        )
        object.__setattr__(
            self,
            "metadata",
            _normalize_pairs(self.metadata, field_name="metadata"),
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
            "support_status": self.support_status,
            "reporting_buckets": dict(self.reporting_buckets),
            "metadata": dict(self.metadata),
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
            support_status=str(payload.get("support_status", "supported")),
            reporting_buckets=payload.get("reporting_buckets") or (),
            metadata=payload.get("metadata") or (),
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


class UnsupportedRiskFactorObject(ValueError):
    """Raised when registry discovery is asked to inspect an unsupported object."""

    def __init__(self, market_object: object, *, reason: str = "unsupported_market_object"):
        self.object_type = type(market_object).__name__
        self.reason = str(reason)
        super().__init__(
            f"unsupported risk-factor object {self.object_type!r}: {self.reason}"
        )

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-friendly diagnostic payload."""
        return {
            "object_type": self.object_type,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RiskFactorRegistry:
    """Deterministic registry of discovered risk-factor coordinates."""

    coordinates: tuple[RiskFactorCoordinate, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinates",
            self._normalize_coordinates(self.coordinates),
        )

    @staticmethod
    def _normalize_coordinates(
        coordinates: Iterable[RiskFactorCoordinate],
    ) -> tuple[RiskFactorCoordinate, ...]:
        coordinate_map: dict[RiskFactorId, RiskFactorCoordinate] = {}
        for coordinate in coordinates:
            if not isinstance(coordinate, RiskFactorCoordinate):
                raise TypeError("RiskFactorRegistry coordinates must be RiskFactorCoordinate instances")
            existing = coordinate_map.get(coordinate.factor_id)
            if existing is not None and existing != coordinate:
                raise ValueError(f"conflicting coordinate for factor {coordinate.factor_id.key!r}")
            coordinate_map[coordinate.factor_id] = coordinate
        return tuple(
            coordinate
            for _, coordinate in sorted(
                coordinate_map.items(),
                key=lambda item: item[0].key,
            )
        )

    def with_coordinates(
        self,
        coordinates: Iterable[RiskFactorCoordinate],
    ) -> RiskFactorRegistry:
        """Return a registry containing the existing and supplied coordinates."""
        return RiskFactorRegistry((*self.coordinates, *tuple(coordinates)))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly registry payload."""
        return {
            "coordinates": [
                coordinate.to_payload()
                for coordinate in self.coordinates
            ]
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RiskFactorRegistry:
        """Build a registry from :meth:`to_payload` output."""
        return cls(
            tuple(
                RiskFactorCoordinate.from_payload(entry)
                for entry in payload.get("coordinates", ())
            )
        )

    def discover_market_object(
        self,
        market_object: object,
        *,
        object_name: str,
        currency: str | None = None,
        issuer: str | None = None,
        object_path: str = "",
        provenance_namespace: str | None = None,
    ) -> tuple[RiskFactorCoordinate, ...]:
        """Dispatch discovery for a supported market object or fail closed."""
        from trellis.curves.credit_curve import CreditCurve
        from trellis.curves.yield_curve import YieldCurve
        from trellis.models.vol_surface import FlatVol, GridVolSurface

        if isinstance(market_object, YieldCurve):
            return self.discover_yield_curve(
                market_object,
                object_name=object_name,
                currency=currency,
                object_path=object_path,
                provenance_namespace=provenance_namespace,
            )
        if isinstance(market_object, CreditCurve):
            return self.discover_credit_curve(
                market_object,
                object_name=object_name,
                currency=currency,
                issuer=issuer,
                object_path=object_path,
                provenance_namespace=provenance_namespace,
            )
        if isinstance(market_object, FlatVol):
            return self.discover_flat_vol_surface(
                market_object,
                object_name=object_name,
                currency=currency,
                object_path=object_path,
                provenance_namespace=provenance_namespace,
            )
        if isinstance(market_object, GridVolSurface):
            return self.discover_grid_vol_surface(
                market_object,
                object_name=object_name,
                currency=currency,
                object_path=object_path,
                provenance_namespace=provenance_namespace,
            )
        raise UnsupportedRiskFactorObject(market_object)

    def discover_yield_curve(
        self,
        curve: object,
        *,
        object_name: str,
        currency: str | None = None,
        object_path: str = "",
        provenance_namespace: str | None = None,
    ) -> tuple[RiskFactorCoordinate, ...]:
        """Return zero-rate node coordinates for a supported yield curve."""
        tenors = getattr(curve, "tenors", None)
        rates = getattr(curve, "rates", None)
        if tenors is None or rates is None:
            raise UnsupportedRiskFactorObject(curve, reason="yield_curve_nodes_unavailable")
        resolved_path = object_path or f"curves.{object_name}"
        return tuple(
            RiskFactorCoordinate(
                factor_id=RiskFactorId(
                    object_type="curve",
                    object_name=object_name,
                    coordinate_type="zero_rate",
                    currency=currency,
                    axes={"tenor_years": float(tenor)},
                    provenance_namespace=provenance_namespace,
                ),
                object_path=f"{resolved_path}.rates[{index}]",
                display_name=f"{object_name} {format(float(tenor), '.12g')}Y zero rate",
                unit="rate",
                transform="identity",
                support_status="supported",
                reporting_buckets={
                    "risk_class": "rates",
                    "currency": currency or "",
                    "tenor": f"{format(float(tenor), '.12g')}Y",
                    "object_name": object_name,
                },
            )
            for index, tenor in enumerate(tenors)
        )

    def with_yield_curve(self, curve: object, **kwargs: object) -> RiskFactorRegistry:
        """Return a registry extended with yield-curve node coordinates."""
        return self.with_coordinates(self.discover_yield_curve(curve, **kwargs))

    def discover_credit_curve(
        self,
        curve: object,
        *,
        object_name: str,
        issuer: str | None = None,
        currency: str | None = None,
        object_path: str = "",
        provenance_namespace: str | None = None,
    ) -> tuple[RiskFactorCoordinate, ...]:
        """Return discovery-only hazard-rate coordinates for a credit curve."""
        tenors = getattr(curve, "tenors", None)
        hazard_rates = getattr(curve, "hazard_rates", None)
        if tenors is None or hazard_rates is None:
            raise UnsupportedRiskFactorObject(curve, reason="credit_curve_nodes_unavailable")
        resolved_path = object_path or f"credit_curves.{object_name}"
        return tuple(
            RiskFactorCoordinate(
                factor_id=RiskFactorId(
                    object_type="credit_curve",
                    object_name=object_name,
                    coordinate_type="hazard_rate",
                    currency=currency,
                    issuer=issuer,
                    axes={"tenor_years": float(tenor)},
                    provenance_namespace=provenance_namespace,
                ),
                object_path=f"{resolved_path}.hazard_rates[{index}]",
                display_name=f"{object_name} {format(float(tenor), '.12g')}Y hazard rate",
                unit="hazard_rate",
                transform="identity",
                support_status="discovery_only",
                reporting_buckets={
                    "risk_class": "credit",
                    "currency": currency or "",
                    "issuer": issuer or "",
                    "tenor": f"{format(float(tenor), '.12g')}Y",
                    "object_name": object_name,
                },
            )
            for index, tenor in enumerate(tenors)
        )

    def discover_flat_vol_surface(
        self,
        surface: object,
        *,
        object_name: str,
        currency: str | None = None,
        object_path: str = "",
        provenance_namespace: str | None = None,
    ) -> tuple[RiskFactorCoordinate, ...]:
        """Return a discovery-only scalar flat-vol coordinate."""
        if not hasattr(surface, "vol"):
            raise UnsupportedRiskFactorObject(surface, reason="flat_vol_unavailable")
        return (
            RiskFactorCoordinate(
                factor_id=RiskFactorId(
                    object_type="vol_surface",
                    object_name=object_name,
                    coordinate_type="flat_vol",
                    currency=currency,
                    provenance_namespace=provenance_namespace,
                ),
                object_path=object_path or f"vol_surfaces.{object_name}.vol",
                display_name=f"{object_name} flat volatility",
                unit="volatility",
                transform="identity",
                support_status="discovery_only",
                reporting_buckets={
                    "risk_class": "volatility",
                    "currency": currency or "",
                    "object_name": object_name,
                },
            ),
        )

    def discover_grid_vol_surface(
        self,
        surface: object,
        *,
        object_name: str,
        currency: str | None = None,
        object_path: str = "",
        provenance_namespace: str | None = None,
    ) -> tuple[RiskFactorCoordinate, ...]:
        """Return discovery-only Black-vol node coordinates for a grid surface."""
        expiries = getattr(surface, "expiries", None)
        strikes = getattr(surface, "strikes", None)
        vols = getattr(surface, "vols", None)
        if expiries is None or strikes is None or vols is None:
            raise UnsupportedRiskFactorObject(surface, reason="grid_vol_nodes_unavailable")
        resolved_path = object_path or f"vol_surfaces.{object_name}"
        coordinates: list[RiskFactorCoordinate] = []
        for expiry_index, expiry in enumerate(expiries):
            for strike_index, strike in enumerate(strikes):
                expiry_label = format(float(expiry), ".12g")
                strike_label = format(float(strike), ".12g")
                coordinates.append(
                    RiskFactorCoordinate(
                        factor_id=RiskFactorId(
                            object_type="vol_surface",
                            object_name=object_name,
                            coordinate_type="black_vol",
                            currency=currency,
                            axes={
                                "expiry_years": float(expiry),
                                "strike": float(strike),
                            },
                            provenance_namespace=provenance_namespace,
                        ),
                        object_path=f"{resolved_path}.vols[{expiry_index}][{strike_index}]",
                        display_name=f"{object_name} {expiry_label}Y {strike_label} vol",
                        unit="volatility",
                        transform="identity",
                        support_status="discovery_only",
                        reporting_buckets={
                            "risk_class": "volatility",
                            "currency": currency or "",
                            "expiry": f"{expiry_label}Y",
                            "strike": strike_label,
                            "object_name": object_name,
                        },
                    )
                )
        return tuple(coordinates)

    def discover_scalar_model_parameters(
        self,
        parameters: Mapping[str, object],
        *,
        parameter_set_name: str,
        model_family: str | None = None,
        provenance_namespace: str | None = None,
    ) -> tuple[RiskFactorCoordinate, ...]:
        """Return discovery-only scalar model-parameter coordinates."""
        coordinates: list[RiskFactorCoordinate] = []
        for parameter_name in sorted(str(name) for name in parameters):
            coordinates.append(
                RiskFactorCoordinate(
                    factor_id=RiskFactorId(
                        object_type="model_parameter",
                        object_name=parameter_set_name,
                        coordinate_type="model_parameter",
                        axes={"parameter": parameter_name},
                        provenance_namespace=provenance_namespace,
                    ),
                    object_path=f"model_parameter_sets.{parameter_set_name}.{parameter_name}",
                    display_name=f"{parameter_set_name} {parameter_name}",
                    unit="model_parameter",
                    transform="identity",
                    support_status="discovery_only",
                    reporting_buckets={
                        "risk_class": "model",
                        "model_family": model_family or "",
                        "parameter_set": parameter_set_name,
                        "parameter": parameter_name,
                    },
                    metadata={
                        "model_family": model_family or "",
                    },
                )
            )
        return tuple(coordinates)


__all__ = [
    "RiskFactorCoordinate",
    "RiskFactorId",
    "RiskFactorRegistry",
    "SparseRiskVector",
    "UnsupportedRiskFactorObject",
]
