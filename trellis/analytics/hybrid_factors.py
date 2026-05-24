"""Hybrid factor graph contracts for bounded hybrid AD workflows."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from trellis.analytics.risk_factors import RiskFactorCoordinate, RiskFactorId


_DIFFERENTIABILITY_CLASSES = frozenset(
    {
        "smooth",
        "piecewise",
        "discontinuous",
        "projected",
        "held_fixed",
        "unsupported",
    }
)
_DERIVATIVE_METHODS = frozenset(
    {
        "ad",
        "vjp",
        "hvp",
        "jvp",
        "bump",
        "custom_adjoint",
        "smoothed",
        "finite_difference_fallback",
        "held_fixed",
        "unsupported",
    }
)
_SUPPORT_STATUSES = frozenset({"supported", "held_fixed", "unsupported", "discovery_only"})
_GRID_VOL_STATE_CONTROL_LANE_FAMILIES = frozenset(
    {"path_summary", "early_exercise_control"}
)
_GRID_VOL_STATE_CONTROL_UNSUPPORTED_REASONS = frozenset(
    {
        "missing_grid_vol_surface",
        "unsupported_grid_vol_interpolation",
        "unsupported_selected_grid_vol_factors",
        "unsupported_discontinuous_event_monitor",
        "early_exercise_boundary_kink",
    }
)


def _clean_required(value: object, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _validate_member(value: str, allowed: frozenset[str], field_name: str) -> str:
    normalized = _clean_required(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return normalized


def _normalize_coordinates(
    coordinates: Iterable[RiskFactorCoordinate],
) -> tuple[RiskFactorCoordinate, ...]:
    coordinate_map: dict[RiskFactorId, RiskFactorCoordinate] = {}
    for coordinate in coordinates:
        if not isinstance(coordinate, RiskFactorCoordinate):
            raise TypeError("coordinates must contain RiskFactorCoordinate instances")
        existing = coordinate_map.get(coordinate.factor_id)
        if existing is not None and existing != coordinate:
            raise ValueError(f"conflicting coordinate for factor {coordinate.factor_id.key!r}")
        coordinate_map[coordinate.factor_id] = coordinate
    return tuple(
        coordinate
        for _, coordinate in sorted(coordinate_map.items(), key=lambda item: item[0].key)
    )


def _normalize_grid_vol_coordinates(
    coordinates: Iterable[RiskFactorCoordinate],
    *,
    object_name: str,
) -> tuple[RiskFactorCoordinate, ...]:
    normalized = _normalize_coordinates(coordinates)
    if not normalized:
        raise ValueError("grid-vol state/control policy requires at least one coordinate")
    for coordinate in normalized:
        factor_id = coordinate.factor_id
        if (
            factor_id.object_type != "vol_surface"
            or factor_id.coordinate_type != "black_vol"
            or factor_id.object_name != object_name
        ):
            raise ValueError(
                "grid-vol state/control policy requires grid-vol black-vol coordinates"
            )
        axes = dict(factor_id.axes)
        if "expiry_years" not in axes or "strike" not in axes:
            raise ValueError(
                "grid-vol state/control policy requires expiry_years and strike axes"
            )
    return normalized


def _normalize_node_ids(node_ids: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({_clean_required(node_id, "node id") for node_id in node_ids}))


def _as_float(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _validate_correlation(value: object, field_name: str = "correlation") -> float:
    correlation = _as_float(value, field_name)
    if not -1.0 < correlation < 1.0:
        raise ValueError(f"{field_name} must be strictly inside (-1, 1)")
    return correlation


def _normalize_factor_labels(factor_labels: Iterable[object]) -> tuple[str, ...]:
    labels = tuple(_clean_required(label, "factor label") for label in factor_labels)
    if len(labels) < 2:
        raise ValueError("factor_labels must contain at least two labels")
    if len(set(labels)) != len(labels):
        raise ValueError("factor_labels must be unique")
    return labels


def _normalize_grid_vol_lane_family(value: object) -> str:
    return _validate_member(
        str(value).strip(),
        _GRID_VOL_STATE_CONTROL_LANE_FAMILIES,
        "lane_family",
    )


def _normalize_grid_vol_unsupported_reason(value: object) -> str:
    return _validate_member(
        str(value).strip(),
        _GRID_VOL_STATE_CONTROL_UNSUPPORTED_REASONS,
        "reason",
    )


def _grid_vol_differentiability_class(reason: str) -> str:
    if reason == "unsupported_discontinuous_event_monitor":
        return "discontinuous"
    if reason == "early_exercise_boundary_kink":
        return "projected"
    return "unsupported"


def _normalize_correlation_matrix(
    matrix: Iterable[Iterable[object]],
    *,
    dimension: int,
    tolerance: float,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(
        tuple(_as_float(value, "correlation_matrix entry") for value in row)
        for row in matrix
    )
    if len(rows) != dimension or any(len(row) != dimension for row in rows):
        raise ValueError("correlation_matrix must be a square matrix matching factor_labels")
    for index, row in enumerate(rows):
        if abs(row[index] - 1.0) > tolerance:
            raise ValueError("correlation_matrix must have unit diagonal")
    for row in rows:
        for value in row:
            if value < -1.0 - tolerance or value > 1.0 + tolerance:
                raise ValueError("correlation_matrix entries must be inside [-1, 1]")
    for row_index in range(dimension):
        for column_index in range(row_index + 1, dimension):
            if abs(rows[row_index][column_index] - rows[column_index][row_index]) > tolerance:
                raise ValueError("correlation_matrix must be symmetric")
    return rows


def _min_symmetric_eigenvalue(matrix: tuple[tuple[float, ...], ...]) -> float:
    from trellis.core.differentiable import get_numpy

    np = get_numpy()
    return float(np.min(np.linalg.eigvalsh(np.asarray(matrix, dtype=float))))


def _correlation_matrix_coordinates(
    *,
    object_name: str,
    factor_labels: tuple[str, ...],
) -> tuple[RiskFactorCoordinate, ...]:
    coordinates: list[RiskFactorCoordinate] = []
    dimension = len(factor_labels)
    for row_index, row_label in enumerate(factor_labels):
        for column_index in range(row_index + 1, dimension):
            column_label = factor_labels[column_index]
            factor_id = RiskFactorId(
                object_type="correlation_matrix",
                object_name=object_name,
                coordinate_type="correlation",
                axes={
                    "row": row_label,
                    "row_index": row_index,
                    "column": column_label,
                    "column_index": column_index,
                },
                provenance_namespace="hybrid_ad",
            )
            coordinates.append(
                RiskFactorCoordinate(
                    factor_id=factor_id,
                    object_path=(
                        f"correlation_matrix.{object_name}[{row_index},{column_index}]"
                    ),
                    display_name=f"{row_label}/{column_label} correlation",
                    unit="correlation",
                    transform="identity",
                    support_status="unsupported",
                    reporting_buckets={
                        "risk_class": "hybrid",
                        "object_name": object_name,
                        "factor_a": row_label,
                        "factor_b": column_label,
                    },
                    metadata={
                        "matrix_dimension": dimension,
                        "row_index": row_index,
                        "column_index": column_index,
                    },
                )
            )
    return tuple(coordinates)


def _restore_correlation_matrix_coordinate_values(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Restore tuple types in correlation_matrix_psd_policy coordinate_values."""
    result = dict(raw)
    if "factor_labels" in result:
        labels = result["factor_labels"]
        if not isinstance(labels, tuple):
            result["factor_labels"] = tuple(labels) if isinstance(labels, (list, tuple)) else (labels,)
    if "correlation_matrix" in result:
        matrix = result["correlation_matrix"]
        if isinstance(matrix, (list, tuple)):
            result["correlation_matrix"] = tuple(
                tuple(row) if isinstance(row, (list, tuple)) else (row,)
                for row in matrix
            )
    return result


def _restore_correlation_matrix_constraints(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Restore tuple types in correlation_matrix_psd_policy constraints."""
    result = dict(raw)
    bounds = result.get("bounds")
    if bounds is not None and not isinstance(bounds, tuple):
        if isinstance(bounds, (list, tuple)):
            result["bounds"] = tuple(bounds)
    return result


def _restore_grid_vol_coordinate_values(raw: Mapping[str, object]) -> dict[str, object]:
    """Restore tuple types in grid_vol_state_control_policy coordinate_values."""
    result = dict(raw)
    node_keys = result.get("active_node_keys")
    if node_keys is not None and not isinstance(node_keys, tuple):
        if isinstance(node_keys, (list, tuple)):
            result["active_node_keys"] = tuple(node_keys)
        else:
            result["active_node_keys"] = (node_keys,)
    return result


@dataclass(frozen=True)
class MarketObjectCoordinateChart:
    """Coordinate chart owned by one market or model object in a hybrid graph."""

    chart_id: str
    object_type: str
    object_name: str
    coordinates: tuple[RiskFactorCoordinate, ...] = field(default_factory=tuple)
    chart_type: str = "identity"
    coordinate_space: str = "constrained"
    coordinate_values: Mapping[str, object] = field(default_factory=dict)
    constraints: Mapping[str, object] = field(default_factory=dict)
    differentiability_class: str = "smooth"
    support_status: str = "supported"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chart_id", _clean_required(self.chart_id, "chart_id"))
        object.__setattr__(self, "object_type", _clean_required(self.object_type, "object_type"))
        object.__setattr__(self, "object_name", str(self.object_name).strip())
        object.__setattr__(self, "coordinates", _normalize_coordinates(self.coordinates))
        object.__setattr__(self, "chart_type", _clean_required(self.chart_type, "chart_type"))
        object.__setattr__(
            self,
            "coordinate_space",
            _clean_required(self.coordinate_space, "coordinate_space"),
        )
        object.__setattr__(self, "coordinate_values", _freeze_mapping(self.coordinate_values))
        object.__setattr__(self, "constraints", _freeze_mapping(self.constraints))
        object.__setattr__(
            self,
            "differentiability_class",
            _validate_member(
                self.differentiability_class,
                _DIFFERENTIABILITY_CLASSES,
                "differentiability_class",
            ),
        )
        object.__setattr__(
            self,
            "support_status",
            _validate_member(self.support_status, _SUPPORT_STATUSES, "support_status"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def identity(
        cls,
        *,
        chart_id: str,
        object_type: str,
        object_name: str,
        coordinates: Iterable[RiskFactorCoordinate] = (),
        coordinate_values: Mapping[str, object] | None = None,
        differentiability_class: str = "smooth",
        support_status: str = "supported",
        metadata: Mapping[str, object] | None = None,
    ) -> MarketObjectCoordinateChart:
        """Build an identity chart for already-constrained coordinates."""
        return cls(
            chart_id=chart_id,
            object_type=object_type,
            object_name=object_name,
            coordinates=tuple(coordinates),
            chart_type="identity",
            coordinate_space="constrained",
            coordinate_values=coordinate_values or {},
            constraints={},
            differentiability_class=differentiability_class,
            support_status=support_status,
            metadata=metadata or {},
        )

    @classmethod
    def scalar_correlation(
        cls,
        *,
        coordinate: RiskFactorCoordinate,
        correlation: float,
        chart_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MarketObjectCoordinateChart:
        """Build the bounded scalar-correlation chart ``rho = tanh(x)``."""
        rho = _validate_correlation(correlation)
        x_value = math.atanh(rho)
        factor_id = coordinate.factor_id
        return cls(
            chart_id=chart_id or f"chart:{factor_id.key}",
            object_type=factor_id.object_type,
            object_name=factor_id.object_name,
            coordinates=(coordinate,),
            chart_type="tanh_scalar_correlation",
            coordinate_space="unconstrained",
            coordinate_values={
                "rho": rho,
                "x": x_value,
                "d_rho_d_x": 1.0 - rho * rho,
            },
            constraints={
                "lower": -1.0,
                "upper": 1.0,
                "parameterization": "rho=tanh(x)",
            },
            differentiability_class="smooth",
            support_status=coordinate.support_status,
            metadata={
                "coordinate_type": "correlation",
                "chart_family": "scalar_correlation",
                **dict(metadata or {}),
            },
        )

    @classmethod
    def correlation_matrix_policy(
        cls,
        *,
        object_name: str,
        factor_labels: Iterable[object],
        correlation_matrix: Iterable[Iterable[object]],
        chart_id: str | None = None,
        tolerance: float = 1.0e-10,
        metadata: Mapping[str, object] | None = None,
    ) -> MarketObjectCoordinateChart:
        """Build the checked fail-closed policy chart for a correlation matrix."""
        clean_object_name = _clean_required(object_name, "object_name")
        tol = _as_float(tolerance, "tolerance")
        if tol < 0.0:
            raise ValueError("tolerance must be non-negative")
        labels = _normalize_factor_labels(factor_labels)
        matrix = _normalize_correlation_matrix(
            correlation_matrix,
            dimension=len(labels),
            tolerance=tol,
        )
        min_eigenvalue = _min_symmetric_eigenvalue(matrix)
        if min_eigenvalue < -tol:
            raise ValueError("correlation_matrix must be positive semidefinite")
        coordinates = _correlation_matrix_coordinates(
            object_name=clean_object_name,
            factor_labels=labels,
        )
        off_diagonal_values = {
            f"{labels[row_index]}|{labels[column_index]}": matrix[row_index][column_index]
            for row_index in range(len(labels))
            for column_index in range(row_index + 1, len(labels))
        }
        return cls(
            chart_id=chart_id or f"chart:correlation_matrix:{clean_object_name}",
            object_type="correlation_matrix",
            object_name=clean_object_name,
            coordinates=coordinates,
            chart_type="correlation_matrix_psd_policy",
            coordinate_space="matrix",
            coordinate_values={
                "dimension": len(labels),
                "factor_labels": labels,
                "correlation_matrix": matrix,
                "off_diagonal_values": off_diagonal_values,
            },
            constraints={
                "diagonal": 1.0,
                "symmetric": True,
                "bounds": (-1.0, 1.0),
                "psd": True,
                "tolerance": tol,
                "parameterization": "checked_correlation_matrix_entries",
                "projection_policy": "unsupported_no_smoothing_or_projection",
            },
            differentiability_class="smooth",
            support_status="unsupported",
            metadata={
                "coordinate_type": "correlation",
                "chart_family": "correlation_matrix",
                "coordinate_count": len(coordinates),
                "min_eigenvalue": min_eigenvalue,
                **dict(metadata or {}),
            },
        )

    @classmethod
    def grid_vol_state_control_policy(
        cls,
        *,
        object_name: str,
        lane_family: str,
        coordinates: Iterable[RiskFactorCoordinate],
        chart_id: str | None = None,
        interpolation_basis: str = "bilinear_black_vol",
        locality_policy: str = "full_grid_node_vector",
        selected_factor_policy: str = "filter_known_fail_closed_unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> MarketObjectCoordinateChart:
        """Build the grid-vol node policy chart for state/control hybrid lanes."""
        clean_object_name = _clean_required(object_name, "object_name")
        lane = _normalize_grid_vol_lane_family(lane_family)
        interpolation = _clean_required(interpolation_basis, "interpolation_basis")
        locality = _clean_required(locality_policy, "locality_policy")
        selected_policy = _clean_required(
            selected_factor_policy,
            "selected_factor_policy",
        )
        normalized_coordinates = _normalize_grid_vol_coordinates(
            coordinates,
            object_name=clean_object_name,
        )
        active_node_keys = tuple(
            coordinate.factor_id.key for coordinate in normalized_coordinates
        )
        return cls(
            chart_id=chart_id or f"chart:grid_vol_state_control:{clean_object_name}:{lane}",
            object_type="vol_surface",
            object_name=clean_object_name,
            coordinates=normalized_coordinates,
            chart_type="grid_vol_state_control_policy",
            coordinate_space="grid_nodes",
            coordinate_values={
                "parameterization": "grid_node_vols",
                "active_node_keys": active_node_keys,
                "active_node_count": len(active_node_keys),
                "lane_family": lane,
            },
            constraints={
                "interpolation_basis": interpolation,
                "locality_policy": locality,
                "selected_factor_policy": selected_policy,
                "unsupported_selected_factor_reason": (
                    "unsupported_selected_grid_vol_factors"
                ),
                "event_monitor_policy": "unsupported_fail_closed",
                "boundary_kink_policy": "unsupported_fail_closed",
            },
            differentiability_class="piecewise",
            support_status="discovery_only",
            metadata={
                "coordinate_type": "black_vol",
                "chart_family": "grid_vol_state_control",
                "lane_family": lane,
                "market_parameterization": "grid_vol",
                "parameterization": "grid_node_vols",
                "active_node_count": len(active_node_keys),
                **dict(metadata or {}),
            },
        )

    @property
    def coordinate_keys(self) -> tuple[str, ...]:
        """Return coordinate factor keys in deterministic order."""
        return tuple(coordinate.factor_id.key for coordinate in self.coordinates)

    @property
    def constrained_value(self) -> float:
        """Return the current constrained scalar value for scalar charts."""
        if self.chart_type == "tanh_scalar_correlation":
            return _validate_correlation(self.coordinate_values["rho"])
        if "value" in self.coordinate_values:
            return _as_float(self.coordinate_values["value"], "coordinate_values.value")
        raise ValueError(f"chart {self.chart_id!r} does not carry one constrained scalar")

    @property
    def unconstrained_value(self) -> float:
        """Return the current unconstrained scalar value for scalar charts."""
        if self.chart_type == "tanh_scalar_correlation":
            return _as_float(self.coordinate_values["x"], "coordinate_values.x")
        return self.constrained_value

    @property
    def derivative_constrained_wrt_unconstrained(self) -> float:
        """Return ``d constrained / d unconstrained`` at the chart value."""
        if self.chart_type == "tanh_scalar_correlation":
            return _as_float(
                self.coordinate_values["d_rho_d_x"],
                "coordinate_values.d_rho_d_x",
            )
        return 1.0

    def constrained_from_unconstrained(self, value: float) -> float:
        """Map an unconstrained scalar into the constrained coordinate space."""
        x_value = _as_float(value, "unconstrained value")
        if self.chart_type == "tanh_scalar_correlation":
            return math.tanh(x_value)
        return x_value

    def unconstrained_from_constrained(self, value: float) -> float:
        """Map a constrained scalar into the chart's unconstrained space."""
        constrained = _as_float(value, "constrained value")
        if self.chart_type == "tanh_scalar_correlation":
            return math.atanh(_validate_correlation(constrained, "constrained value"))
        return constrained

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly chart payload."""
        return {
            "chart_id": self.chart_id,
            "object_type": self.object_type,
            "object_name": self.object_name,
            "chart_type": self.chart_type,
            "coordinate_space": self.coordinate_space,
            "coordinate_values": dict(self.coordinate_values),
            "constraints": dict(self.constraints),
            "differentiability_class": self.differentiability_class,
            "support_status": self.support_status,
            "metadata": dict(self.metadata),
            "coordinates": [coordinate.to_payload() for coordinate in self.coordinates],
            "coordinate_keys": list(self.coordinate_keys),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MarketObjectCoordinateChart":
        """Rebuild a chart from :meth:`to_payload` output."""
        chart_type = str(payload.get("chart_type", "identity"))
        coordinate_values: Mapping[str, object] = payload.get("coordinate_values") or {}
        constraints: Mapping[str, object] = payload.get("constraints") or {}
        if chart_type == "correlation_matrix_psd_policy":
            coordinate_values = _restore_correlation_matrix_coordinate_values(coordinate_values)
            constraints = _restore_correlation_matrix_constraints(constraints)
        elif chart_type == "grid_vol_state_control_policy":
            coordinate_values = _restore_grid_vol_coordinate_values(coordinate_values)
        return cls(
            chart_id=str(payload["chart_id"]),
            object_type=str(payload["object_type"]),
            object_name=str(payload.get("object_name", "")),
            coordinates=tuple(
                RiskFactorCoordinate.from_payload(entry)
                for entry in payload.get("coordinates", ())
            ),
            chart_type=chart_type,
            coordinate_space=str(payload.get("coordinate_space", "constrained")),
            coordinate_values=coordinate_values,
            constraints=constraints,
            differentiability_class=str(payload.get("differentiability_class", "smooth")),
            support_status=str(payload.get("support_status", "supported")),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class HybridUnsupportedDependency:
    """Typed record for a hybrid graph dependency outside derivative support."""

    dependency_id: str
    node_type: str
    object_name: str
    reason: str
    support_status: str = "unsupported"
    differentiability_class: str = "unsupported"
    derivative_method: str = "unsupported"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dependency_id",
            _clean_required(self.dependency_id, "dependency_id"),
        )
        object.__setattr__(self, "node_type", _clean_required(self.node_type, "node_type"))
        object.__setattr__(self, "object_name", str(self.object_name).strip())
        object.__setattr__(self, "reason", _clean_required(self.reason, "reason"))
        object.__setattr__(
            self,
            "support_status",
            _validate_member(self.support_status, _SUPPORT_STATUSES, "support_status"),
        )
        object.__setattr__(
            self,
            "differentiability_class",
            _validate_member(
                self.differentiability_class,
                _DIFFERENTIABILITY_CLASSES,
                "differentiability_class",
            ),
        )
        object.__setattr__(
            self,
            "derivative_method",
            _validate_member(self.derivative_method, _DERIVATIVE_METHODS, "derivative_method"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def grid_vol_state_control_policy(
        cls,
        *,
        object_name: str,
        lane_family: str,
        reason: str,
        dependency_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> HybridUnsupportedDependency:
        """Build a typed unsupported dependency for grid-vol state/control policy."""
        clean_object_name = _clean_required(object_name, "object_name")
        lane = _normalize_grid_vol_lane_family(lane_family)
        normalized_reason = _normalize_grid_vol_unsupported_reason(reason)
        return cls(
            dependency_id=(
                dependency_id
                or (
                    "unsupported:grid_vol_state_control:"
                    f"{clean_object_name}:{normalized_reason}"
                )
            ),
            node_type="vol_surface",
            object_name=clean_object_name,
            reason=normalized_reason,
            support_status="unsupported",
            differentiability_class=_grid_vol_differentiability_class(
                normalized_reason
            ),
            derivative_method="unsupported",
            metadata={
                "chart_family": "grid_vol_state_control",
                "lane_family": lane,
                "market_parameterization": "grid_vol",
                "parameterization": "grid_node_vols",
                **dict(metadata or {}),
            },
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly unsupported-dependency payload."""
        return {
            "dependency_id": self.dependency_id,
            "node_type": self.node_type,
            "object_name": self.object_name,
            "reason": self.reason,
            "support_status": self.support_status,
            "differentiability_class": self.differentiability_class,
            "derivative_method": self.derivative_method,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HybridUnsupportedDependency:
        """Rebuild an unsupported dependency from :meth:`to_payload` output."""
        return cls(
            dependency_id=str(payload["dependency_id"]),
            node_type=str(payload["node_type"]),
            object_name=str(payload.get("object_name", "")),
            reason=str(payload["reason"]),
            support_status=str(payload.get("support_status", "unsupported")),
            differentiability_class=str(payload.get("differentiability_class", "unsupported")),
            derivative_method=str(payload.get("derivative_method", "unsupported")),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class HybridDependencyNode:
    """One dependency node in a typed hybrid factor graph."""

    node_id: str
    node_type: str
    object_name: str
    coordinate_chart: MarketObjectCoordinateChart | None = None
    coordinates: tuple[RiskFactorCoordinate, ...] = field(default_factory=tuple)
    upstream_node_ids: tuple[str, ...] = field(default_factory=tuple)
    differentiability_class: str = "smooth"
    derivative_method: str = "unsupported"
    support_status: str = "supported"
    provenance: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_required(self.node_id, "node_id"))
        object.__setattr__(self, "node_type", _clean_required(self.node_type, "node_type"))
        object.__setattr__(self, "object_name", str(self.object_name).strip())
        if self.coordinate_chart is not None and not isinstance(
            self.coordinate_chart,
            MarketObjectCoordinateChart,
        ):
            raise TypeError("coordinate_chart must be a MarketObjectCoordinateChart")
        coordinates = self.coordinates
        if self.coordinate_chart is not None and not coordinates:
            coordinates = self.coordinate_chart.coordinates
        object.__setattr__(self, "coordinates", _normalize_coordinates(coordinates))
        object.__setattr__(self, "upstream_node_ids", _normalize_node_ids(self.upstream_node_ids))
        object.__setattr__(
            self,
            "differentiability_class",
            _validate_member(
                self.differentiability_class,
                _DIFFERENTIABILITY_CLASSES,
                "differentiability_class",
            ),
        )
        object.__setattr__(
            self,
            "derivative_method",
            _validate_member(self.derivative_method, _DERIVATIVE_METHODS, "derivative_method"),
        )
        object.__setattr__(
            self,
            "support_status",
            _validate_member(self.support_status, _SUPPORT_STATUSES, "support_status"),
        )
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def coordinate_keys(self) -> tuple[str, ...]:
        """Return coordinate factor keys owned by this node."""
        return tuple(coordinate.factor_id.key for coordinate in self.coordinates)

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly node payload."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "object_name": self.object_name,
            "coordinate_chart": (
                self.coordinate_chart.to_payload()
                if self.coordinate_chart is not None
                else None
            ),
            "coordinates": [coordinate.to_payload() for coordinate in self.coordinates],
            "coordinate_keys": list(self.coordinate_keys),
            "upstream_node_ids": list(self.upstream_node_ids),
            "differentiability_class": self.differentiability_class,
            "derivative_method": self.derivative_method,
            "support_status": self.support_status,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HybridDependencyNode:
        """Rebuild a node from :meth:`to_payload` output."""
        chart_payload = payload.get("coordinate_chart")
        return cls(
            node_id=str(payload["node_id"]),
            node_type=str(payload["node_type"]),
            object_name=str(payload.get("object_name", "")),
            coordinate_chart=(
                MarketObjectCoordinateChart.from_payload(chart_payload)
                if isinstance(chart_payload, Mapping)
                else None
            ),
            coordinates=tuple(
                RiskFactorCoordinate.from_payload(entry)
                for entry in payload.get("coordinates", ())
            ),
            upstream_node_ids=tuple(payload.get("upstream_node_ids", ())),
            differentiability_class=str(payload.get("differentiability_class", "smooth")),
            derivative_method=str(payload.get("derivative_method", "unsupported")),
            support_status=str(payload.get("support_status", "supported")),
            provenance=payload.get("provenance") or {},
            metadata=payload.get("metadata") or {},
        )


def _normalize_nodes(nodes: Iterable[HybridDependencyNode]) -> tuple[HybridDependencyNode, ...]:
    node_map: dict[str, HybridDependencyNode] = {}
    for node in nodes:
        if not isinstance(node, HybridDependencyNode):
            raise TypeError("nodes must contain HybridDependencyNode instances")
        existing = node_map.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting hybrid dependency node {node.node_id!r}")
        node_map[node.node_id] = node
    return tuple(node for _, node in sorted(node_map.items()))


@dataclass(frozen=True)
class HybridFactorGraph:
    """Typed dependency graph for bounded hybrid derivative requests."""

    graph_id: str
    nodes: tuple[HybridDependencyNode, ...] = field(default_factory=tuple)
    unsupported_dependencies: tuple[HybridUnsupportedDependency, ...] = field(
        default_factory=tuple
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _clean_required(self.graph_id, "graph_id"))
        object.__setattr__(self, "nodes", _normalize_nodes(self.nodes))
        unsupported = tuple(self.unsupported_dependencies)
        for dependency in unsupported:
            if not isinstance(dependency, HybridUnsupportedDependency):
                raise TypeError(
                    "unsupported_dependencies must contain HybridUnsupportedDependency instances"
                )
        object.__setattr__(
            self,
            "unsupported_dependencies",
            tuple(sorted(unsupported, key=lambda item: item.dependency_id)),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return graph node ids in deterministic order."""
        return tuple(node.node_id for node in self.nodes)

    @property
    def coordinate_charts(self) -> tuple[MarketObjectCoordinateChart, ...]:
        """Return the coordinate charts carried by graph nodes."""
        return tuple(
            node.coordinate_chart
            for node in self.nodes
            if node.coordinate_chart is not None
        )

    @property
    def coordinates(self) -> tuple[RiskFactorCoordinate, ...]:
        """Return all node-owned coordinates in deterministic factor-key order."""
        return _normalize_coordinates(
            coordinate
            for node in self.nodes
            for coordinate in node.coordinates
        )

    @property
    def coordinate_keys(self) -> tuple[str, ...]:
        """Return all coordinate factor keys in deterministic order."""
        return tuple(coordinate.factor_id.key for coordinate in self.coordinates)

    @property
    def unsupported_reasons(self) -> tuple[str, ...]:
        """Return unsupported-dependency reasons in deterministic order."""
        return tuple(
            sorted({dependency.reason for dependency in self.unsupported_dependencies})
        )

    def node_by_id(self, node_id: str) -> HybridDependencyNode:
        """Return one graph node by id."""
        normalized = _clean_required(node_id, "node_id")
        for node in self.nodes:
            if node.node_id == normalized:
                return node
        raise KeyError(normalized)

    def validate(self) -> HybridFactorGraph:
        """Validate graph-internal dependency references and return ``self``."""
        known = set(self.node_ids)
        for node in self.nodes:
            for upstream in node.upstream_node_ids:
                if upstream not in known:
                    raise KeyError(upstream)
        return self

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly graph payload."""
        return {
            "graph_id": self.graph_id,
            "nodes": [node.to_payload() for node in self.nodes],
            "node_ids": list(self.node_ids),
            "coordinates": [coordinate.to_payload() for coordinate in self.coordinates],
            "coordinate_keys": list(self.coordinate_keys),
            "unsupported_dependencies": [
                dependency.to_payload()
                for dependency in self.unsupported_dependencies
            ],
            "unsupported_reasons": list(self.unsupported_reasons),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HybridFactorGraph:
        """Rebuild a graph from :meth:`to_payload` output."""
        return cls(
            graph_id=str(payload["graph_id"]),
            nodes=tuple(
                HybridDependencyNode.from_payload(entry)
                for entry in payload.get("nodes", ())
            ),
            unsupported_dependencies=tuple(
                HybridUnsupportedDependency.from_payload(entry)
                for entry in payload.get("unsupported_dependencies", ())
            ),
            metadata=payload.get("metadata") or {},
        )


__all__ = [
    "HybridDependencyNode",
    "HybridFactorGraph",
    "HybridUnsupportedDependency",
    "MarketObjectCoordinateChart",
]
