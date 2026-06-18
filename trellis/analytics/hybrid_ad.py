"""Bounded hybrid AD helpers over typed hybrid factor graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from types import MappingProxyType
from typing import Any

from trellis.agent.dynamic_contract_ir import DynamicContractIR
from trellis.analytics.derivative_methods import derivative_method_payload
from trellis.analytics.hybrid_ad_admission import (
    HybridADLaneAdmission,
    admit_hybrid_ad_lane,
)
from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    SparseRiskVector,
)
from trellis.models.vol_surface import _bracket_and_weight


_DERIVATIVE_METHODS = frozenset({"vjp", "jvp", "hvp"})
_COORDINATE_SPACES = frozenset({"constrained", "unconstrained"})
_UNSUPPORTED_SELECTED_POLICIES = frozenset({"empty_vector", "fail_closed"})
_CORRELATION_STRUCTURE_TYPES = frozenset({"correlation_matrix", "correlation_surface"})
_QUANTO_SCALAR_ROLES = frozenset(
    {
        "underlier_spot",
        "fx_spot",
        "domestic_curve",
        "foreign_curve",
        "underlier_vol",
        "fx_vol",
        "correlation",
    }
)


def _clean_member(value: object, allowed: frozenset[str], field_name: str) -> str:
    normalized = str(value).strip()
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return normalized


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _freeze_surface_axes(
    surface_axes: Mapping[str, object] | None,
) -> Mapping[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for key, raw_values in sorted((surface_axes or {}).items()):
        axis_name = str(key).strip()
        if not axis_name:
            raise ValueError("surface_axes keys must be non-empty")
        if isinstance(raw_values, str):
            values = (raw_values,)
        else:
            try:
                values = tuple(raw_values)  # type: ignore[arg-type]
            except TypeError:
                values = (raw_values,)
        normalized[axis_name] = tuple(str(value).strip() for value in values)
    return MappingProxyType(normalized)


def _freeze_optional_matrix(
    matrix: Iterable[Iterable[object]] | None,
) -> tuple[tuple[object, ...], ...] | None:
    if matrix is None:
        return None
    return tuple(tuple(row) for row in matrix)


def _freeze_diagnostics(
    diagnostics: Iterable[Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], ...]:
    return tuple(MappingProxyType(dict(diagnostic)) for diagnostic in (diagnostics or ()))


def _sorted_unique_factors(factors: Iterable[RiskFactorId]) -> tuple[RiskFactorId, ...]:
    factor_map: dict[RiskFactorId, RiskFactorId] = {}
    for factor in factors:
        if not isinstance(factor, RiskFactorId):
            raise TypeError("selected_factors must contain RiskFactorId instances")
        factor_map[factor] = factor
    return tuple(factor for _, factor in sorted(factor_map.items(), key=lambda item: item[0].key))


def _normalize_sparse_vector(value: object, field_name: str) -> SparseRiskVector:
    if isinstance(value, SparseRiskVector):
        return value
    if value is None:
        return SparseRiskVector()
    items = value.items() if isinstance(value, Mapping) else value
    try:
        return SparseRiskVector.from_items(items)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must contain RiskFactorId keyed numeric entries") from exc


def _normalize_semantic_admission(value: object) -> HybridADLaneAdmission | None:
    if value is None:
        return None
    if isinstance(value, HybridADLaneAdmission):
        return value
    if isinstance(value, Mapping):
        return HybridADLaneAdmission.from_payload(value)
    raise TypeError(
        "semantic_admission must be a HybridADLaneAdmission, payload mapping, or None"
    )


@dataclass(frozen=True)
class HybridDerivativeRequest:
    """Request policy for bounded graph-backed hybrid derivative helpers."""

    derivative_method: str = "vjp"
    coordinate_space: str = "constrained"
    selected_factors: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    unsupported_selected_factor_policy: str = "empty_vector"
    hvp_direction: SparseRiskVector = field(default_factory=SparseRiskVector)
    semantic_admission: HybridADLaneAdmission | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derivative_method",
            _clean_member(self.derivative_method, _DERIVATIVE_METHODS, "derivative_method"),
        )
        object.__setattr__(
            self,
            "coordinate_space",
            _clean_member(self.coordinate_space, _COORDINATE_SPACES, "coordinate_space"),
        )
        object.__setattr__(
            self,
            "selected_factors",
            _sorted_unique_factors(self.selected_factors),
        )
        object.__setattr__(
            self,
            "unsupported_selected_factor_policy",
            _clean_member(
                self.unsupported_selected_factor_policy,
                _UNSUPPORTED_SELECTED_POLICIES,
                "unsupported_selected_factor_policy",
            ),
        )
        object.__setattr__(
            self,
            "hvp_direction",
            _normalize_sparse_vector(self.hvp_direction, "hvp_direction"),
        )
        object.__setattr__(
            self,
            "semantic_admission",
            _normalize_semantic_admission(self.semantic_admission),
        )

    @property
    def selects_all_factors(self) -> bool:
        """Return whether all available factors should be returned."""
        return len(self.selected_factors) == 0

    def filter_vector(self, vector: SparseRiskVector) -> SparseRiskVector:
        """Filter a sparse vector according to selected factors."""
        if self.selects_all_factors:
            return vector
        return vector.filter(self.selected_factors)

    def missing_selected_factors(
        self,
        available: Iterable[RiskFactorId],
    ) -> tuple[RiskFactorId, ...]:
        """Return selected factors that were not available in the result."""
        if self.selects_all_factors:
            return ()
        available_set = set(available)
        return tuple(factor for factor in self.selected_factors if factor not in available_set)


@dataclass(frozen=True)
class HybridCorrelationStructureRequest:
    """Unsupported correlation matrix/surface derivative request record."""

    object_name: str
    structure_type: str = "correlation_matrix"
    factors: tuple[str, ...] = field(default_factory=tuple)
    requested_derivative_method: str = "vjp"
    coordinate_space: str = "unconstrained"
    correlation_matrix: tuple[tuple[object, ...], ...] | None = None
    surface_axes: Mapping[str, object] = field(default_factory=dict)
    chart_tolerance: float = 1.0e-10
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_name = str(self.object_name).strip()
        if not object_name:
            raise ValueError("object_name must be non-empty")
        object.__setattr__(self, "object_name", object_name)
        object.__setattr__(
            self,
            "structure_type",
            _clean_member(
                self.structure_type,
                _CORRELATION_STRUCTURE_TYPES,
                "structure_type",
            ),
        )
        factors = tuple(str(factor).strip() for factor in self.factors if str(factor).strip())
        object.__setattr__(self, "factors", factors)
        object.__setattr__(
            self,
            "requested_derivative_method",
            _clean_member(
                self.requested_derivative_method,
                _DERIVATIVE_METHODS,
                "requested_derivative_method",
            ),
        )
        object.__setattr__(
            self,
            "coordinate_space",
            _clean_member(self.coordinate_space, _COORDINATE_SPACES, "coordinate_space"),
        )
        object.__setattr__(
            self,
            "correlation_matrix",
            _freeze_optional_matrix(self.correlation_matrix),
        )
        object.__setattr__(self, "surface_axes", _freeze_surface_axes(self.surface_axes))
        chart_tolerance = float(self.chart_tolerance)
        if chart_tolerance < 0.0:
            raise ValueError("chart_tolerance must be non-negative")
        object.__setattr__(self, "chart_tolerance", chart_tolerance)
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @property
    def unsupported_reason(self) -> str:
        """Return the policy reason for this unsupported structure."""
        return f"{self.structure_type}_chart_not_implemented"


@dataclass(frozen=True)
class HybridMatrixCoordinateContext:
    """Executable context for a checked correlation-matrix coordinate chart."""

    chart: MarketObjectCoordinateChart
    min_eigenvalue_floor: float = 1.0e-6
    active_factor_id: RiskFactorId | Mapping[str, object] | None = None
    support_status: str = "supported"
    diagnostics: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.chart, MarketObjectCoordinateChart):
            raise TypeError("chart must be a MarketObjectCoordinateChart")
        if self.chart.chart_type != "correlation_matrix_psd_policy":
            raise ValueError("chart must be a correlation_matrix_psd_policy chart")
        floor = float(self.min_eigenvalue_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("min_eigenvalue_floor must be a finite non-negative float")
        object.__setattr__(self, "min_eigenvalue_floor", floor)
        if self.chart.support_status != "supported":
            raise ValueError("matrix-coordinate context requires a supported chart")
        status = str(self.support_status).strip()
        if status != "supported":
            raise ValueError("matrix-coordinate context support_status must be supported")
        object.__setattr__(self, "support_status", status)
        active = self.active_factor_id
        if isinstance(active, Mapping):
            active = RiskFactorId.from_payload(active)
        if active is not None and active not in self.factor_ids:
            raise ValueError("active_factor_id must be one of the chart coordinates")
        object.__setattr__(self, "active_factor_id", active)
        object.__setattr__(self, "diagnostics", _freeze_diagnostics(self.diagnostics))

    @property
    def factor_labels(self) -> tuple[str, ...]:
        """Return matrix factor labels in chart order."""
        return tuple(str(label) for label in self.chart.coordinate_values["factor_labels"])

    @property
    def correlation_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return the checked constrained correlation matrix."""
        return tuple(
            tuple(float(value) for value in row)
            for row in self.chart.coordinate_values["correlation_matrix"]
        )

    @property
    def min_eigenvalue(self) -> float:
        """Return the checked minimum symmetric eigenvalue."""
        return float(self.chart.metadata["min_eigenvalue"])

    @property
    def coordinate_count(self) -> int:
        """Return the number of off-diagonal matrix coordinates."""
        return len(self.chart.coordinates)

    @property
    def factor_ids(self) -> tuple[RiskFactorId, ...]:
        """Return supported off-diagonal coordinate factor ids."""
        return tuple(coordinate.factor_id for coordinate in self.chart.coordinates)

    def coordinate_index_for_pair(self, factor_a: object, factor_b: object) -> int:
        """Return the deterministic coordinate index for an unordered factor pair."""
        target = frozenset((str(factor_a).strip(), str(factor_b).strip()))
        if len(target) != 2:
            raise ValueError("matrix coordinate pair must contain two distinct factors")
        for index, coordinate in enumerate(self.chart.coordinates):
            axes = dict(coordinate.factor_id.axes)
            if frozenset((axes.get("row", ""), axes.get("column", ""))) == target:
                return index
        raise KeyError(f"matrix coordinate pair {sorted(target)!r} is not chart-owned")

    def coordinate_for_pair(self, factor_a: object, factor_b: object) -> RiskFactorCoordinate:
        """Return the chart coordinate for an unordered factor pair."""
        return self.chart.coordinates[self.coordinate_index_for_pair(factor_a, factor_b)]

    @property
    def active_coordinate_index(self) -> int | None:
        """Return the active coordinate index when one was requested."""
        if self.active_factor_id is None:
            return None
        index_by_factor = {
            coordinate.factor_id: index
            for index, coordinate in enumerate(self.chart.coordinates)
        }
        return index_by_factor[self.active_factor_id]

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly context payload."""
        return {
            "chart": self.chart.to_payload(),
            "min_eigenvalue_floor": self.min_eigenvalue_floor,
            "active_factor_id": (
                self.active_factor_id.to_payload()
                if self.active_factor_id is not None
                else None
            ),
            "support_status": self.support_status,
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "HybridMatrixCoordinateContext":
        """Rebuild a context from :meth:`to_payload` output."""
        return cls(
            chart=MarketObjectCoordinateChart.from_payload(payload["chart"]),
            min_eigenvalue_floor=float(payload.get("min_eigenvalue_floor", 1.0e-6)),
            active_factor_id=payload.get("active_factor_id"),
            support_status=str(payload.get("support_status", "supported")),
            diagnostics=tuple(payload.get("diagnostics") or ()),
        )


@dataclass(frozen=True)
class HybridDerivativeResult:
    """Result for one bounded graph-backed hybrid derivative request."""

    value: float | None
    risk_vector: SparseRiskVector
    graph: HybridFactorGraph
    support_status: str
    method_metadata: Mapping[str, object] = field(default_factory=dict)
    unsupported_dependencies: tuple[HybridUnsupportedDependency, ...] = field(
        default_factory=tuple
    )
    diagnostics: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.risk_vector, SparseRiskVector):
            raise TypeError("risk_vector must be a SparseRiskVector")
        if not isinstance(self.graph, HybridFactorGraph):
            raise TypeError("graph must be a HybridFactorGraph")
        object.__setattr__(self, "support_status", str(self.support_status))
        object.__setattr__(self, "method_metadata", _freeze_mapping(self.method_metadata))
        unsupported = tuple(self.unsupported_dependencies)
        for dependency in unsupported:
            if not isinstance(dependency, HybridUnsupportedDependency):
                raise TypeError(
                    "unsupported_dependencies must contain HybridUnsupportedDependency instances"
                )
        object.__setattr__(self, "unsupported_dependencies", unsupported)
        object.__setattr__(self, "diagnostics", _freeze_diagnostics(self.diagnostics))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly result payload."""
        return {
            "value": self.value,
            "risk_vector": self.risk_vector.to_payload(),
            "graph": self.graph.to_payload(),
            "support_status": self.support_status,
            "method_metadata": dict(self.method_metadata),
            "unsupported_dependencies": [
                dependency.to_payload()
                for dependency in self.unsupported_dependencies
            ],
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "HybridDerivativeResult":
        """Rebuild a result from :meth:`to_payload` output."""
        return cls(
            value=(
                None
                if payload.get("value") is None
                else float(payload["value"])  # type: ignore[arg-type]
            ),
            risk_vector=SparseRiskVector.from_payload(payload.get("risk_vector") or {}),
            graph=HybridFactorGraph.from_payload(payload["graph"]),
            support_status=str(payload.get("support_status", "unsupported")),
            method_metadata=payload.get("method_metadata") or {},
            unsupported_dependencies=tuple(
                HybridUnsupportedDependency.from_payload(entry)
                for entry in payload.get("unsupported_dependencies", ())
            ),
            diagnostics=tuple(payload.get("diagnostics") or ()),
        )


@dataclass(frozen=True)
class _GraphScalarEntry:
    node_id: str
    role: str
    factor_id: RiskFactorId
    base_value: float


@dataclass(frozen=True)
class _GraphScalarExtraction:
    entries: tuple[_GraphScalarEntry, ...]
    diagnostics: tuple[dict[str, object], ...] = ()


def _correlation_nodes(graph: HybridFactorGraph) -> tuple[HybridDependencyNode, ...]:
    return tuple(
        node
        for node in graph.nodes
        if any(
            coordinate.factor_id.coordinate_type == "correlation"
            for coordinate in node.coordinates
        )
    )


def _correlation_chart(graph: HybridFactorGraph) -> MarketObjectCoordinateChart | None:
    for node in _correlation_nodes(graph):
        if node.coordinate_chart is not None:
            return node.coordinate_chart
    return None


def _correlation_factor(graph: HybridFactorGraph) -> RiskFactorId | None:
    for coordinate in graph.coordinates:
        if coordinate.factor_id.coordinate_type == "correlation":
            return coordinate.factor_id
    return None


def _fallback_graph(spec: object, resolved_inputs: object) -> HybridFactorGraph:
    corr = float(getattr(resolved_inputs, "corr"))
    object_name = str(getattr(spec, "quanto_correlation_key", None) or "quanto_correlation")
    factor_a = str(getattr(spec, "underlier_currency", "underlier"))
    factor_b = str(getattr(spec, "fx_pair", "fx"))
    currency = str(getattr(spec, "underlier_currency", ""))
    from trellis.analytics.risk_factors import RiskFactorRegistry

    coordinate = RiskFactorRegistry().discover_scalar_correlation(
        corr,
        object_name=object_name,
        factor_a=factor_a,
        factor_b=factor_b,
        currency=currency,
        provenance_namespace="hybrid_ad",
        support_status="supported",
    )[0]
    chart = MarketObjectCoordinateChart.scalar_correlation(
        coordinate=coordinate,
        correlation=corr,
        chart_id=f"chart:node:correlation:{object_name}",
    )
    node = HybridDependencyNode(
        node_id=f"node:correlation:{object_name}",
        node_type="correlation",
        object_name=object_name,
        coordinate_chart=chart,
        derivative_method="vjp",
        support_status="supported",
        metadata={"resolved_inputs": ("correlation",)},
    )
    return HybridFactorGraph(
        graph_id=f"quanto:{factor_a}:{getattr(spec, 'domestic_currency', '')}:{factor_b}",
        nodes=(node,),
        metadata={
            "route_family": "bounded_quanto",
            "graph_source": "fallback_scalar_correlation",
        },
    )


def _unsupported_result(
    *,
    value: float | None,
    graph: HybridFactorGraph,
    request: HybridDerivativeRequest,
    code: str,
    message: str,
    method_id: str = "hybrid_scalar_vjp",
    fallback_reason: dict[str, object] | None = None,
    diagnostic_extra: Mapping[str, object] | None = None,
    backend_operator: str | None = None,
    suppress_backend_operator: bool = False,
    method_metadata_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    diagnostic = {
        "code": code,
        "severity": "warning",
        "message": message,
        **dict(diagnostic_extra or {}),
    }
    metadata = derivative_method_payload(
        method_id,
        method_support="unsupported",
        backend_operator=(
            None
            if suppress_backend_operator
            else backend_operator or request.derivative_method
        ),
        coordinate_space=request.coordinate_space,
        hybrid_factor_graph_id=graph.graph_id,
        fallback_reason=fallback_reason or diagnostic,
    )
    metadata.update(dict(method_metadata_extra or {}))
    return HybridDerivativeResult(
        value=value,
        risk_vector=SparseRiskVector(),
        graph=graph,
        support_status="unsupported",
        method_metadata=metadata,
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=(diagnostic,),
    )


def _jvp_backend_support_payload() -> tuple[dict[str, object], tuple[str, ...]]:
    from trellis.core.differentiable import get_backend_capabilities

    capabilities = get_backend_capabilities()
    return capabilities.operator_support("jvp"), capabilities.notes


def _unsupported_jvp_result(
    *,
    value: float | None,
    graph: HybridFactorGraph,
    request: HybridDerivativeRequest,
    message: str,
    diagnostic_extra: Mapping[str, object] | None = None,
    method_metadata_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    support, notes = _jvp_backend_support_payload()
    note_payload = list(notes)
    diagnostic = {
        "backend_id": support["backend_id"],
        "unsupported_operator": "jvp",
        "requested_backend_operator": "jvp",
        "backend_support": support,
        "backend_notes": note_payload,
        **dict(diagnostic_extra or {}),
    }
    metadata_extra = {
        "requested_derivative_method": request.derivative_method,
        "requested_backend_operator": "jvp",
        "backend_support": support,
        "backend_notes": note_payload,
        **dict(method_metadata_extra or {}),
    }
    return _unsupported_result(
        value=value,
        graph=graph,
        request=request,
        code="hybrid_jvp_backend_unsupported",
        message=message,
        method_id="unsupported_hybrid_jvp",
        diagnostic_extra=diagnostic,
        fallback_reason={
            "code": "hybrid_jvp_backend_unsupported",
            **diagnostic,
        },
        suppress_backend_operator=True,
        method_metadata_extra=metadata_extra,
    )


def _path_summary_market_context(
    market_context: object,
    *,
    vol_surface_name: str,
    currency: str | None,
) -> object:
    return _vanilla_equity_vol_market_context(
        market_context,
        vol_surface_name=vol_surface_name,
        currency=currency,
    )


def _vanilla_equity_vol_market_context(
    market_context: object,
    *,
    vol_surface_name: str,
    currency: str | None,
) -> object:
    from trellis.analytics.portfolio_aad import VanillaEquityOptionVolAADMarketContext

    if isinstance(market_context, VanillaEquityOptionVolAADMarketContext):
        if market_context.provenance_namespace == "hybrid_ad":
            return market_context
        return replace(market_context, provenance_namespace="hybrid_ad")
    return VanillaEquityOptionVolAADMarketContext(
        market_state=market_context,
        vol_surface_name=vol_surface_name,
        currency=currency,
        provenance_namespace="hybrid_ad",
    )


def _early_exercise_graph(
    context: object,
    *,
    derivative_method: str,
    position_name: str,
    exercise_style: str,
) -> HybridFactorGraph:
    from trellis.analytics.portfolio_aad import VanillaEquityOptionVolAADMarketContext
    from trellis.models.vol_surface import FlatVol, GridVolSurface

    if not isinstance(context, VanillaEquityOptionVolAADMarketContext):
        raise TypeError(
            "early-exercise helper requires VanillaEquityOptionVolAADMarketContext"
        )
    vol_surface = getattr(context.market_state, "vol_surface", None)
    if isinstance(vol_surface, GridVolSurface):
        coordinates = tuple(
            replace(coordinate, support_status="discovery_only")
            for coordinate in context.coordinates()
        )
        chart = MarketObjectCoordinateChart.grid_vol_state_control_policy(
            object_name=context.vol_surface_name,
            lane_family="early_exercise_control",
            coordinates=coordinates,
            metadata={
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "grid_vol_hard_exercise_projection_pending"
                ),
            },
        )
        dependency = HybridUnsupportedDependency.grid_vol_state_control_policy(
            object_name=context.vol_surface_name,
            lane_family="early_exercise_control",
            reason="unsupported_grid_vol_interpolation",
            metadata={
                "position_name": str(position_name),
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "grid_vol_hard_exercise_projection_pending"
                ),
                "policy_reason": "early_exercise_grid_vol_vjp_pending",
            },
        )
        node = HybridDependencyNode(
            node_id=f"node:early_exercise_grid_vol:{context.vol_surface_name}",
            node_type="vol_surface",
            object_name=context.vol_surface_name,
            coordinate_chart=chart,
            derivative_method="unsupported",
            differentiability_class=chart.differentiability_class,
            support_status=chart.support_status,
            metadata={
                "resolved_inputs": ("underlier_vol",),
                "position_name": str(position_name),
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "grid_vol_hard_exercise_projection_pending"
                ),
                "grid_vol_support_status": "planned",
            },
        )
        graph = HybridFactorGraph(
            graph_id=f"hybrid:early_exercise:{context.vol_surface_name}:{position_name}",
            nodes=(node,),
            unsupported_dependencies=(dependency,),
            metadata={
                "route_family": "bounded_vanilla_early_exercise",
                "graph_source": "vanilla_early_exercise_grid_vol_control_policy",
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "grid_vol_hard_exercise_projection_pending"
                ),
                "vol_surface_name": context.vol_surface_name,
                "position_name": str(position_name),
                "market_parameterization": "grid_vol",
                "grid_vol_support_status": "planned",
                "grid_vol_coordinate_policy": chart.to_payload(),
            },
        )
        graph.validate()
        return graph
    if not isinstance(vol_surface, FlatVol):
        raise ValueError("early-exercise hybrid AD requires FlatVol")
    coordinates = context.coordinates()
    coordinate_values = {"vol": float(vol_surface.vol)}
    chart = MarketObjectCoordinateChart.identity(
        chart_id=f"chart:early_exercise_vol:{context.vol_surface_name}",
        object_type="vol_surface",
        object_name=context.vol_surface_name,
        coordinates=coordinates,
        coordinate_values=coordinate_values,
        metadata={
            "coordinate_type": "flat_vol",
            "chart_family": "flat_vol",
            "exercise_style": exercise_style,
            "early_exercise_policy": "hard_exercise_projection_smooth_interior",
        },
    )
    node = HybridDependencyNode(
        node_id=f"node:early_exercise_vol:{context.vol_surface_name}",
        node_type="vol_surface",
        object_name=context.vol_surface_name,
        coordinate_chart=chart,
        derivative_method=derivative_method,
        support_status="supported",
        metadata={
            "resolved_inputs": ("underlier_vol",),
            "position_name": str(position_name),
            "exercise_style": exercise_style,
            "early_exercise_policy": "hard_exercise_projection_smooth_interior",
        },
    )
    graph = HybridFactorGraph(
        graph_id=f"hybrid:early_exercise:{context.vol_surface_name}:{position_name}",
        nodes=(node,),
        metadata={
            "route_family": "bounded_vanilla_early_exercise",
            "graph_source": "vanilla_early_exercise_smooth_interior",
            "exercise_style": exercise_style,
            "early_exercise_policy": "hard_exercise_projection_smooth_interior",
            "vol_surface_name": context.vol_surface_name,
            "position_name": str(position_name),
        },
    )
    graph.validate()
    return graph


def _path_summary_graph(
    context: object,
    *,
    derivative_method: str,
    position_name: str,
) -> HybridFactorGraph:
    from trellis.analytics.portfolio_aad import VanillaEquityOptionVolAADMarketContext
    from trellis.models.vol_surface import FlatVol, GridVolSurface

    if not isinstance(context, VanillaEquityOptionVolAADMarketContext):
        raise TypeError("path-summary helper requires VanillaEquityOptionVolAADMarketContext")
    coordinates = context.coordinates()
    vol_surface = getattr(context.market_state, "vol_surface", None)
    if isinstance(vol_surface, GridVolSurface):
        discovery_coordinates = tuple(
            replace(coordinate, support_status="discovery_only")
            for coordinate in coordinates
        )
        chart = MarketObjectCoordinateChart.grid_vol_state_control_policy(
            object_name=context.vol_surface_name,
            lane_family="path_summary",
            coordinates=discovery_coordinates,
            metadata={
                "path_summary_type": "arithmetic_mean",
                "path_derivative_policy": (
                    "lognormal_moment_matching_smooth_path_summary"
                ),
            },
        )
        dependency = HybridUnsupportedDependency.grid_vol_state_control_policy(
            object_name=context.vol_surface_name,
            lane_family="path_summary",
            reason="unsupported_grid_vol_interpolation",
            metadata={
                "position_name": str(position_name),
                "path_summary_type": "arithmetic_mean",
                "path_derivative_policy": (
                    "lognormal_moment_matching_smooth_path_summary"
                ),
                "policy_reason": "path_summary_grid_vol_vjp_pending",
            },
        )
        node = HybridDependencyNode(
            node_id=f"node:path_summary_grid_vol:{context.vol_surface_name}",
            node_type="vol_surface",
            object_name=context.vol_surface_name,
            coordinate_chart=chart,
            derivative_method="unsupported",
            differentiability_class=chart.differentiability_class,
            support_status=chart.support_status,
            metadata={
                "resolved_inputs": ("underlier_vol",),
                "position_name": str(position_name),
                "path_summary_type": "arithmetic_mean",
                "path_derivative_policy": (
                    "lognormal_moment_matching_smooth_path_summary"
                ),
                "grid_vol_support_status": "planned",
            },
        )
        graph = HybridFactorGraph(
            graph_id=f"hybrid:path_summary:{context.vol_surface_name}:{position_name}",
            nodes=(node,),
            unsupported_dependencies=(dependency,),
            metadata={
                "route_family": "bounded_arithmetic_asian",
                "graph_source": "arithmetic_asian_grid_vol_path_summary_policy",
                "path_summary_type": "arithmetic_mean",
                "path_derivative_policy": (
                    "lognormal_moment_matching_smooth_path_summary"
                ),
                "vol_surface_name": context.vol_surface_name,
                "position_name": str(position_name),
                "market_parameterization": "grid_vol",
                "grid_vol_support_status": "planned",
                "grid_vol_coordinate_policy": chart.to_payload(),
            },
        )
        graph.validate()
        return graph
    coordinate_values: dict[str, object] = {}
    if isinstance(vol_surface, FlatVol):
        coordinate_values["vol"] = float(vol_surface.vol)
    chart = MarketObjectCoordinateChart.identity(
        chart_id=f"chart:path_summary_vol:{context.vol_surface_name}",
        object_type="vol_surface",
        object_name=context.vol_surface_name,
        coordinates=coordinates,
        coordinate_values=coordinate_values,
        metadata={
            "coordinate_type": "flat_vol",
            "chart_family": "flat_vol",
            "path_summary_type": "arithmetic_mean",
        },
    )
    node = HybridDependencyNode(
        node_id=f"node:path_summary_vol:{context.vol_surface_name}",
        node_type="vol_surface",
        object_name=context.vol_surface_name,
        coordinate_chart=chart,
        derivative_method=derivative_method,
        support_status="supported",
        metadata={
            "resolved_inputs": ("underlier_vol",),
            "position_name": str(position_name),
            "path_summary_type": "arithmetic_mean",
            "path_derivative_policy": "lognormal_moment_matching_smooth_path_summary",
        },
    )
    graph = HybridFactorGraph(
        graph_id=f"hybrid:path_summary:{context.vol_surface_name}:{position_name}",
        nodes=(node,),
        metadata={
            "route_family": "bounded_arithmetic_asian",
            "graph_source": "arithmetic_asian_path_summary",
            "path_summary_type": "arithmetic_mean",
            "path_derivative_policy": "lognormal_moment_matching_smooth_path_summary",
            "vol_surface_name": context.vol_surface_name,
            "position_name": str(position_name),
        },
    )
    graph.validate()
    return graph


def _unsupported_early_exercise_graph(
    *,
    reason: str,
    position_name: str,
    vol_surface_name: str,
    exercise_style: str,
) -> HybridFactorGraph:
    dependency = HybridUnsupportedDependency(
        dependency_id=f"node:early_exercise:{position_name}",
        node_type="early_exercise_control",
        object_name=str(position_name),
        reason=reason,
        metadata={
            "vol_surface_name": vol_surface_name,
            "exercise_style": exercise_style,
            "early_exercise_policy": "hard_exercise_projection_smooth_interior",
        },
    )
    return HybridFactorGraph(
        graph_id=f"hybrid:early_exercise:{vol_surface_name}:{position_name}",
        unsupported_dependencies=(dependency,),
        metadata={
            "route_family": "bounded_vanilla_early_exercise",
            "graph_source": "vanilla_early_exercise_smooth_interior",
            "exercise_style": exercise_style,
            "vol_surface_name": vol_surface_name,
        },
    )


def _unsupported_path_summary_graph(
    *,
    reason: str,
    position_name: str,
    vol_surface_name: str,
) -> HybridFactorGraph:
    dependency = HybridUnsupportedDependency(
        dependency_id=f"node:path_summary:{position_name}",
        node_type="path_summary",
        object_name=str(position_name),
        reason=reason,
        metadata={
            "vol_surface_name": vol_surface_name,
            "path_summary_type": "arithmetic_mean",
        },
    )
    return HybridFactorGraph(
        graph_id=f"hybrid:path_summary:{vol_surface_name}:{position_name}",
        unsupported_dependencies=(dependency,),
        metadata={
            "route_family": "bounded_arithmetic_asian",
            "graph_source": "arithmetic_asian_path_summary",
            "path_summary_type": "arithmetic_mean",
            "vol_surface_name": vol_surface_name,
        },
    )


def _grid_vol_state_control_chart(
    graph: HybridFactorGraph,
) -> MarketObjectCoordinateChart | None:
    for chart in graph.coordinate_charts:
        if chart.chart_type == "grid_vol_state_control_policy":
            return chart
    return None


def _grid_vol_state_control_policy_metadata(
    graph: HybridFactorGraph,
) -> dict[str, object]:
    chart = _grid_vol_state_control_chart(graph)
    if chart is None:
        return {}
    return {
        "grid_vol_coordinate_policy": chart.to_payload(),
        "grid_vol_unsupported_dependency_reasons": list(graph.unsupported_reasons),
    }


def _grid_vol_path_summary_policy_metadata(
    graph: HybridFactorGraph,
) -> dict[str, object]:
    return _grid_vol_state_control_policy_metadata(graph)


def _grid_vol_selected_factor_diagnostics(
    request: HybridDerivativeRequest,
    graph: HybridFactorGraph,
) -> tuple[dict[str, object], ...]:
    if request.selects_all_factors:
        return ()
    available_factors = tuple(coordinate.factor_id for coordinate in graph.coordinates)
    missing_factors = request.missing_selected_factors(available_factors)
    if missing_factors:
        return (
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
                "unsupported_selected_factor_policy": (
                    request.unsupported_selected_factor_policy
                ),
            },
        )
    selected_factor_keys = [factor.key for factor in request.selected_factors]
    if selected_factor_keys:
        return (
            {
                "code": "unsupported_selected_grid_vol_factors",
                "severity": "warning",
                "selected_factor_keys": selected_factor_keys,
                "unsupported_selected_factor_policy": (
                    request.unsupported_selected_factor_policy
                ),
            },
        )
    return ()


def _unsupported_selected_factor_diagnostics(
    request: HybridDerivativeRequest,
    graph: HybridFactorGraph,
    *,
    code: str,
) -> tuple[dict[str, object], ...]:
    if request.selects_all_factors:
        return ()
    available_factors = tuple(coordinate.factor_id for coordinate in graph.coordinates)
    missing_factors = request.missing_selected_factors(available_factors)
    if missing_factors:
        return (
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
                "unsupported_selected_factor_policy": (
                    request.unsupported_selected_factor_policy
                ),
            },
        )
    selected_factor_keys = [factor.key for factor in request.selected_factors]
    if selected_factor_keys:
        return (
            {
                "code": code,
                "severity": "warning",
                "selected_factor_keys": selected_factor_keys,
                "unsupported_selected_factor_policy": (
                    request.unsupported_selected_factor_policy
                ),
            },
        )
    return ()


def _unsupported_grid_vol_jvp_result(
    *,
    value: float | None,
    graph: HybridFactorGraph,
    request: HybridDerivativeRequest,
    message: str,
    diagnostic_extra: Mapping[str, object] | None = None,
    method_metadata_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    policy_metadata = _grid_vol_state_control_policy_metadata(graph)
    unsupported_reasons = list(graph.unsupported_reasons)
    return _unsupported_jvp_result(
        value=value,
        graph=graph,
        request=request,
        message=message,
        diagnostic_extra={
            "market_parameterization": "grid_vol",
            "unsupported_dependency_reasons": unsupported_reasons,
            **dict(diagnostic_extra or {}),
        },
        method_metadata_extra={
            **policy_metadata,
            **dict(method_metadata_extra or {}),
        },
    )


def _unsupported_grid_vol_path_summary_result(
    *,
    value: float | None,
    graph: HybridFactorGraph,
    request: HybridDerivativeRequest,
    code: str,
    message: str,
    diagnostic_extra: Mapping[str, object] | None = None,
    fallback_reason: dict[str, object] | None = None,
    method_metadata_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    policy_metadata = _grid_vol_path_summary_policy_metadata(graph)
    unsupported_reasons = list(graph.unsupported_reasons)
    fallback_payload = {
        "code": code,
        "unsupported_dependency_reasons": unsupported_reasons,
        **dict(fallback_reason or {}),
    }
    result = _unsupported_result(
        value=value,
        graph=graph,
        request=request,
        code=code,
        message=message,
        method_id="hybrid_path_summary_vjp",
        diagnostic_extra={
            "path_summary_type": "arithmetic_mean",
            "market_parameterization": "grid_vol",
            "unsupported_dependency_reasons": unsupported_reasons,
            **dict(diagnostic_extra or {}),
        },
        fallback_reason=fallback_payload,
        method_metadata_extra={
            **policy_metadata,
            **dict(method_metadata_extra or {}),
        },
    )
    selected_diagnostics = _grid_vol_selected_factor_diagnostics(request, graph)
    if selected_diagnostics:
        return replace(result, diagnostics=result.diagnostics + selected_diagnostics)
    return result


def _unsupported_grid_vol_early_exercise_result(
    *,
    value: float | None,
    graph: HybridFactorGraph,
    request: HybridDerivativeRequest,
    code: str,
    message: str,
    exercise_style: str,
    diagnostic_extra: Mapping[str, object] | None = None,
    fallback_reason: dict[str, object] | None = None,
    method_metadata_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    policy_metadata = _grid_vol_state_control_policy_metadata(graph)
    unsupported_reasons = list(graph.unsupported_reasons)
    fallback_payload = {
        "code": code,
        "unsupported_dependency_reasons": unsupported_reasons,
        **dict(fallback_reason or {}),
    }
    result = _unsupported_result(
        value=value,
        graph=graph,
        request=request,
        code=code,
        message=message,
        method_id="hybrid_early_exercise_vjp",
        diagnostic_extra={
            "exercise_style": exercise_style,
            "early_exercise_policy": "grid_vol_hard_exercise_projection_pending",
            "market_parameterization": "grid_vol",
            "unsupported_dependency_reasons": unsupported_reasons,
            **dict(diagnostic_extra or {}),
        },
        fallback_reason=fallback_payload,
        method_metadata_extra={
            **policy_metadata,
            **dict(method_metadata_extra or {}),
        },
    )
    selected_diagnostics = _grid_vol_selected_factor_diagnostics(request, graph)
    if selected_diagnostics:
        return replace(result, diagnostics=result.diagnostics + selected_diagnostics)
    return result


def _node_role(node: HybridDependencyNode) -> str | None:
    roles = tuple(str(role) for role in node.metadata.get("resolved_inputs", ()))
    for role in roles:
        if role in _QUANTO_SCALAR_ROLES:
            return role
    return None


def _axis_float(factor_id: RiskFactorId, axis_name: str) -> float:
    return float(_axis_key(factor_id, axis_name))


def _axis_key(factor_id: RiskFactorId, axis_name: str) -> str:
    axes = dict(factor_id.axes)
    if axis_name not in axes:
        raise KeyError(axis_name)
    return str(axes[axis_name])


def _float_key(value: object) -> str:
    return format(float(value), ".12g")


def _chart_float_tuple(chart: MarketObjectCoordinateChart, key: str) -> tuple[float, ...]:
    return tuple(float(value) for value in chart.coordinate_values[key])


def _chart_float_matrix(
    chart: MarketObjectCoordinateChart,
    key: str,
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in chart.coordinate_values[key])


def _linear_interp_from_values(value: float, grid: tuple[float, ...], node_values: tuple[object, ...]):
    lower, upper, weight = _bracket_and_weight(value, grid)
    return (1.0 - weight) * node_values[lower] + weight * node_values[upper]


def _discount_from_curve_chart(
    chart: MarketObjectCoordinateChart,
    node_values: tuple[object, ...],
    np: object,
):
    tenors = _chart_float_tuple(chart, "tenors")
    expiry = float(chart.coordinate_values["time_to_expiry"])
    rate_by_tenor = {
        _axis_key(coordinate.factor_id, "tenor_years"): node_values[index]
        for index, coordinate in enumerate(chart.coordinates)
    }
    ordered_rates = tuple(rate_by_tenor[_float_key(tenor)] for tenor in tenors)
    rate = _linear_interp_from_values(expiry, tenors, ordered_rates)
    return np.exp(-rate * expiry)


def _vol_from_surface_chart(
    chart: MarketObjectCoordinateChart,
    node_values: tuple[object, ...],
):
    surface_type = str(chart.coordinate_values.get("surface_type", ""))
    if surface_type == "FlatVol":
        return node_values[0]

    expiries = _chart_float_tuple(chart, "expiries")
    strikes = _chart_float_tuple(chart, "strikes")
    expiry = float(chart.coordinate_values["query_expiry"])
    strike = float(chart.coordinate_values["query_strike"])
    expiry_lower, expiry_upper, expiry_weight = _bracket_and_weight(expiry, expiries)
    strike_lower, strike_upper, strike_weight = _bracket_and_weight(strike, strikes)
    value_by_node = {
        (
            _axis_key(coordinate.factor_id, "expiry_years"),
            _axis_key(coordinate.factor_id, "strike"),
        ): node_values[index]
        for index, coordinate in enumerate(chart.coordinates)
    }

    def at(expiry_index: int, strike_index: int):
        return value_by_node[
            (
                _float_key(expiries[expiry_index]),
                _float_key(strikes[strike_index]),
            )
        ]

    lower = (1.0 - strike_weight) * at(expiry_lower, strike_lower) + strike_weight * at(
        expiry_lower,
        strike_upper,
    )
    upper = (1.0 - strike_weight) * at(expiry_upper, strike_lower) + strike_weight * at(
        expiry_upper,
        strike_upper,
    )
    return (1.0 - expiry_weight) * lower + expiry_weight * upper


def _entries_for_node(
    node: HybridDependencyNode,
    *,
    coordinate_space: str,
) -> tuple[_GraphScalarEntry, ...]:
    role = _node_role(node)
    chart = node.coordinate_chart
    if role is None or chart is None or node.support_status != "supported":
        return ()
    values = dict(chart.coordinate_values)
    if role in {"underlier_spot", "fx_spot"}:
        if len(chart.coordinates) != 1:
            return ()
        return (
            _GraphScalarEntry(
                node_id=node.node_id,
                role=role,
                factor_id=chart.coordinates[0].factor_id,
                base_value=float(values["spot"]),
            ),
        )
    if role in {"domestic_curve", "foreign_curve"}:
        tenors = _chart_float_tuple(chart, "tenors")
        rates = _chart_float_tuple(chart, "rates")
        rate_by_tenor = {_float_key(tenor): rate for tenor, rate in zip(tenors, rates)}
        return tuple(
            _GraphScalarEntry(
                node_id=node.node_id,
                role=role,
                factor_id=coordinate.factor_id,
                base_value=rate_by_tenor[_axis_key(coordinate.factor_id, "tenor_years")],
            )
            for coordinate in chart.coordinates
            if coordinate.factor_id.coordinate_type == "zero_rate"
        )
    if role in {"underlier_vol", "fx_vol"}:
        if str(values.get("surface_type")) == "FlatVol":
            return (
                _GraphScalarEntry(
                    node_id=node.node_id,
                    role=role,
                    factor_id=chart.coordinates[0].factor_id,
                    base_value=float(values["flat_vol"]),
                ),
            )
        expiries = _chart_float_tuple(chart, "expiries")
        strikes = _chart_float_tuple(chart, "strikes")
        vols = _chart_float_matrix(chart, "vols")
        vol_by_node = {
            (_float_key(expiry), _float_key(strike)): vols[expiry_index][strike_index]
            for expiry_index, expiry in enumerate(expiries)
            for strike_index, strike in enumerate(strikes)
        }
        return tuple(
            _GraphScalarEntry(
                node_id=node.node_id,
                role=role,
                factor_id=coordinate.factor_id,
                base_value=vol_by_node[
                    (
                        _axis_key(coordinate.factor_id, "expiry_years"),
                        _axis_key(coordinate.factor_id, "strike"),
                    )
                ],
            )
            for coordinate in chart.coordinates
            if coordinate.factor_id.coordinate_type == "black_vol"
        )
    if role == "correlation":
        if len(chart.coordinates) != 1:
            return ()
        base_value = (
            chart.unconstrained_value
            if coordinate_space == "unconstrained"
            else chart.constrained_value
        )
        return (
            _GraphScalarEntry(
                node_id=node.node_id,
                role=role,
                factor_id=chart.coordinates[0].factor_id,
                base_value=float(base_value),
            ),
        )
    return ()


def _scalar_chart_diagnostic(
    node: HybridDependencyNode,
    role: str,
    reason: str,
    exc: Exception | None = None,
) -> dict[str, object]:
    diagnostic: dict[str, object] = {
        "code": "scalar_chart_context_unavailable",
        "severity": "warning",
        "node_id": node.node_id,
        "node_type": node.node_type,
        "object_name": node.object_name,
        "resolved_input": role,
        "reason": reason,
    }
    if exc is not None:
        diagnostic["error_type"] = type(exc).__name__
        diagnostic["error"] = str(exc)
    return diagnostic


def _extract_entries_for_node(
    node: HybridDependencyNode,
    *,
    coordinate_space: str,
) -> _GraphScalarExtraction:
    role = _node_role(node)
    if role is None or node.coordinate_chart is None or node.support_status != "supported":
        return _GraphScalarExtraction(())
    try:
        entries = _entries_for_node(node, coordinate_space=coordinate_space)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _GraphScalarExtraction(
            (),
            (_scalar_chart_diagnostic(node, role, "executable_chart_context_invalid", exc),),
        )
    if not entries:
        return _GraphScalarExtraction(
            (),
            (_scalar_chart_diagnostic(node, role, "executable_chart_coordinates_unavailable"),),
        )
    return _GraphScalarExtraction(entries)


def _quanto_scalar_entries(
    graph: HybridFactorGraph,
    *,
    coordinate_space: str,
) -> _GraphScalarExtraction:
    entries: list[_GraphScalarEntry] = []
    diagnostics: list[dict[str, object]] = []
    for node in graph.nodes:
        extraction = _extract_entries_for_node(node, coordinate_space=coordinate_space)
        entries.extend(extraction.entries)
        diagnostics.extend(extraction.diagnostics)
    return _GraphScalarExtraction(tuple(entries), tuple(diagnostics))


def _values_by_node(
    entries: tuple[_GraphScalarEntry, ...],
    theta,
) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = {}
    for index, entry in enumerate(entries):
        grouped.setdefault(entry.node_id, []).append(theta[index])
    return {node_id: tuple(values) for node_id, values in grouped.items()}


def _hvp_direction_for_entries(
    request: HybridDerivativeRequest,
    entries: tuple[_GraphScalarEntry, ...],
) -> tuple[tuple[float, ...], dict[str, object] | None]:
    if len(request.hvp_direction) == 0:
        return (), {
            "code": "hvp_direction_required",
            "message": "Hybrid scalar-coordinate HVP requires a non-empty direction vector.",
        }
    entry_index_by_factor = {
        entry.factor_id: index
        for index, entry in enumerate(entries)
    }
    missing_factors = tuple(
        factor
        for factor in request.hvp_direction
        if factor not in entry_index_by_factor
    )
    if missing_factors:
        return (), {
            "code": "hvp_direction_factors_unavailable",
            "message": "HVP direction contains factors that are not graph-owned scalar coordinates.",
            "missing_factor_keys": [factor.key for factor in missing_factors],
        }
    values = [0.0] * len(entries)
    for factor, value in request.hvp_direction.items():
        values[entry_index_by_factor[factor]] = float(value)
    return tuple(values), None


def _replace_inputs_from_graph(
    resolved_inputs: object,
    graph: HybridFactorGraph,
    entries: tuple[_GraphScalarEntry, ...],
    theta,
    *,
    coordinate_space: str,
    np: object,
):
    grouped_values = _values_by_node(entries, theta)
    updates: dict[str, object] = {}
    for node in graph.nodes:
        role = _node_role(node)
        chart = node.coordinate_chart
        node_values = grouped_values.get(node.node_id)
        if role is None or chart is None or node_values is None:
            continue
        if role == "underlier_spot":
            updates["spot"] = node_values[0]
        elif role == "fx_spot":
            updates["fx_spot"] = node_values[0]
        elif role == "domestic_curve":
            updates["domestic_df"] = _discount_from_curve_chart(chart, node_values, np)
        elif role == "foreign_curve":
            updates["foreign_df"] = _discount_from_curve_chart(chart, node_values, np)
        elif role == "underlier_vol":
            updates["sigma_underlier"] = _vol_from_surface_chart(chart, node_values)
        elif role == "fx_vol":
            updates["sigma_fx"] = _vol_from_surface_chart(chart, node_values)
        elif role == "correlation":
            updates["corr"] = (
                np.tanh(node_values[0])
                if coordinate_space == "unconstrained"
                else node_values[0]
            )
    return replace(resolved_inputs, **updates)


def _correlation_matrix_validation_code(message: str) -> str:
    normalized = message.lower()
    if "square" in normalized:
        return "invalid_correlation_matrix_shape"
    if "label" in normalized or "unique" in normalized:
        return "invalid_correlation_matrix_labels"
    if "unit diagonal" in normalized:
        return "invalid_correlation_matrix_unit_diagonal"
    if "inside [-1, 1]" in normalized:
        return "invalid_correlation_matrix_bounds"
    if "symmetric" in normalized:
        return "invalid_correlation_matrix_symmetry"
    if "positive semidefinite" in normalized:
        return "invalid_correlation_matrix_psd"
    return "invalid_correlation_matrix_chart"


def _surface_axes_payload(request: HybridCorrelationStructureRequest) -> dict[str, object]:
    surface_axes = {key: tuple(values) for key, values in request.surface_axes.items()}
    axis_names = tuple(sorted(surface_axes))
    return {
        "surface_axes": surface_axes,
        "surface_axis_names": axis_names,
        "surface_axis_count": len(axis_names),
    }


def _correlation_structure_policy_payload(
    request: HybridCorrelationStructureRequest,
) -> tuple[str, str, dict[str, object], MarketObjectCoordinateChart | None]:
    if request.structure_type == "correlation_surface":
        extra = {
            "chart_policy_status": "surface_chart_not_implemented",
            **_surface_axes_payload(request),
        }
        return (
            request.unsupported_reason,
            (
                "Hybrid correlation surface derivatives require a checked "
                "surface chart and are fail-closed until one exists."
            ),
            extra,
            None,
        )
    if request.correlation_matrix is None:
        return (
            request.unsupported_reason,
            (
                "Hybrid correlation matrix derivatives require a checked "
                "coordinate chart and are fail-closed until one exists."
            ),
            {"chart_policy_status": "matrix_payload_missing"},
            None,
        )
    try:
        chart = MarketObjectCoordinateChart.correlation_matrix_policy(
            object_name=request.object_name,
            factor_labels=request.factors,
            correlation_matrix=request.correlation_matrix,
            tolerance=request.chart_tolerance,
        )
    except ValueError as exc:
        validation_message = str(exc)
        code = _correlation_matrix_validation_code(validation_message)
        return (
            code,
            f"Hybrid correlation matrix request failed chart validation: {validation_message}.",
            {
                "chart_policy_status": "invalid_fail_closed",
                "validation_code": code,
                "validation_message": validation_message,
            },
            None,
        )
    extra = {
        "chart_policy_status": "validated_fail_closed",
        "chart_id": chart.chart_id,
        "chart_type": chart.chart_type,
        "matrix_dimension": chart.coordinate_values["dimension"],
        "coordinate_count": chart.metadata["coordinate_count"],
        "coordinate_keys": chart.coordinate_keys,
        "factor_labels": chart.coordinate_values["factor_labels"],
        "min_eigenvalue": chart.metadata["min_eigenvalue"],
        "projection_policy": chart.constraints["projection_policy"],
    }
    return (
        "correlation_matrix_derivative_not_implemented",
        (
            "Hybrid correlation matrix derivatives remain fail-closed because "
            "the checked matrix chart policy is not yet an executable AD lane."
        ),
        extra,
        chart,
    )


def _clean_min_eigenvalue_floor(value: object) -> float:
    floor = float(value)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("min_eigenvalue_floor must be a finite non-negative float")
    return floor


def _executable_correlation_matrix_chart(
    chart: MarketObjectCoordinateChart,
    *,
    min_eigenvalue_floor: float,
) -> MarketObjectCoordinateChart:
    supported_coordinates = tuple(
        replace(coordinate, support_status="supported")
        for coordinate in chart.coordinates
    )
    return replace(
        chart,
        coordinates=supported_coordinates,
        support_status="supported",
        metadata={
            **dict(chart.metadata),
            "chart_policy_status": "validated_executable",
            "min_eigenvalue_floor": min_eigenvalue_floor,
            "projection_policy": chart.constraints["projection_policy"],
        },
    )


def build_correlation_matrix_coordinate_context(
    request: HybridCorrelationStructureRequest,
    *,
    active_factor_pair: tuple[object, object] | None = None,
    min_eigenvalue_floor: float = 1.0e-6,
) -> HybridMatrixCoordinateContext:
    """Build an executable context for a well-conditioned matrix chart.

    The context uses the checked matrix entries directly. It does not smooth,
    project, or repair invalid matrices, and it rejects matrices too close to
    the PSD boundary for stable coordinate derivatives.
    """
    if not isinstance(request, HybridCorrelationStructureRequest):
        raise TypeError("request must be a HybridCorrelationStructureRequest")
    floor = _clean_min_eigenvalue_floor(min_eigenvalue_floor)
    code, message, _, chart = _correlation_structure_policy_payload(request)
    if chart is None:
        raise ValueError(f"{code}: {message}")
    min_eigenvalue = float(chart.metadata["min_eigenvalue"])
    if min_eigenvalue < floor:
        raise ValueError(
            "correlation_matrix_near_psd_boundary: "
            f"minimum eigenvalue {min_eigenvalue:.12g} is below the executable "
            f"context floor {floor:.12g}."
        )

    supported_chart = _executable_correlation_matrix_chart(
        chart,
        min_eigenvalue_floor=floor,
    )
    context = HybridMatrixCoordinateContext(
        chart=supported_chart,
        min_eigenvalue_floor=floor,
        diagnostics=(
            {
                "code": "correlation_matrix_context_executable",
                "severity": "info",
                "message": (
                    "Correlation matrix chart is well-conditioned enough for "
                    "direct off-diagonal coordinate execution."
                ),
                "min_eigenvalue": min_eigenvalue,
                "min_eigenvalue_floor": floor,
                "projection_policy": supported_chart.constraints["projection_policy"],
            },
        ),
    )
    if active_factor_pair is None:
        return context
    try:
        active_coordinate = context.coordinate_for_pair(*active_factor_pair)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "active_factor_pair_unavailable: "
            f"{tuple(str(factor) for factor in active_factor_pair)!r} "
            "is not an off-diagonal coordinate in the matrix chart."
        ) from exc
    return replace(context, active_factor_id=active_coordinate.factor_id)


def _quanto_matrix_active_factor_pair(spec: object) -> tuple[str, str]:
    return (
        str(getattr(spec, "underlier_currency", "underlier")),
        str(getattr(spec, "fx_pair", "fx")),
    )


def _matrix_coordinate_values(
    context: HybridMatrixCoordinateContext,
) -> tuple[float, ...]:
    matrix = context.correlation_matrix
    values: list[float] = []
    for coordinate in context.chart.coordinates:
        axes = dict(coordinate.factor_id.axes)
        row_index = int(axes["row_index"])
        column_index = int(axes["column_index"])
        values.append(float(matrix[row_index][column_index]))
    return tuple(values)


def _matrix_entries_for_context(
    request: HybridCorrelationStructureRequest,
    context: HybridMatrixCoordinateContext,
) -> tuple[_GraphScalarEntry, ...]:
    node_id = f"node:{request.structure_type}:{request.object_name}"
    values = _matrix_coordinate_values(context)
    return tuple(
        _GraphScalarEntry(
            node_id=node_id,
            role="correlation_matrix",
            factor_id=coordinate.factor_id,
            base_value=values[index],
        )
        for index, coordinate in enumerate(context.chart.coordinates)
    )


def _matrix_context_graph(
    request: HybridCorrelationStructureRequest,
    context: HybridMatrixCoordinateContext,
    *,
    derivative_method: str,
) -> HybridFactorGraph:
    active_factor_key = (
        context.active_factor_id.key
        if context.active_factor_id is not None
        else None
    )
    node = HybridDependencyNode(
        node_id=f"node:{request.structure_type}:{request.object_name}",
        node_type=request.structure_type,
        object_name=request.object_name,
        coordinate_chart=context.chart,
        differentiability_class=context.chart.differentiability_class,
        derivative_method=derivative_method,
        support_status=context.support_status,
        metadata={
            "policy": "executable_no_projection",
            "chart_policy_status": context.chart.metadata["chart_policy_status"],
            "resolved_inputs": ("correlation_matrix", "correlation"),
            "active_factor_key": active_factor_key,
            "active_coordinate_index": context.active_coordinate_index,
        },
    )
    graph = HybridFactorGraph(
        graph_id=f"hybrid:{request.structure_type}:{request.object_name}",
        nodes=(node,),
        metadata={
            "structure_type": request.structure_type,
            "object_name": request.object_name,
            "factors": request.factors,
            "chart_policy_status": context.chart.metadata["chart_policy_status"],
            "matrix_dimension": context.chart.coordinate_values["dimension"],
            "matrix_coordinate_count": context.coordinate_count,
            "active_factor_key": active_factor_key,
            "min_eigenvalue": context.min_eigenvalue,
            "min_eigenvalue_floor": context.min_eigenvalue_floor,
            "projection_policy": context.chart.constraints["projection_policy"],
        },
    )
    graph.validate()
    return graph


def _matrix_context_error_code_and_message(exc: ValueError) -> tuple[str, str]:
    raw = str(exc)
    code, separator, message = raw.partition(":")
    if separator:
        return code.strip(), message.strip()
    return "correlation_matrix_context_unavailable", raw


def _unsupported_correlation_matrix_result(
    *,
    value: float | None,
    correlation_request: HybridCorrelationStructureRequest,
    derivative_request: HybridDerivativeRequest,
    code: str,
    message: str,
    method_id: str = "hybrid_matrix_vector_vjp",
    backend_operator: str | None = None,
    diagnostic_extra: Mapping[str, object] | None = None,
    method_metadata_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    dependency = HybridUnsupportedDependency(
        dependency_id=f"node:{correlation_request.structure_type}:{correlation_request.object_name}",
        node_type=correlation_request.structure_type,
        object_name=correlation_request.object_name,
        reason=code,
        metadata={
            "factors": correlation_request.factors,
            "requested_derivative_method": derivative_request.derivative_method,
            "coordinate_space": "matrix",
            "policy": "fail_closed_no_projection",
            **dict(correlation_request.provenance),
            **dict(diagnostic_extra or {}),
        },
    )
    graph = HybridFactorGraph(
        graph_id=f"hybrid:{correlation_request.structure_type}:{correlation_request.object_name}",
        unsupported_dependencies=(dependency,),
        metadata={
            "structure_type": correlation_request.structure_type,
            "object_name": correlation_request.object_name,
            "factors": correlation_request.factors,
            "chart_policy_status": "matrix_context_unavailable",
            **dict(diagnostic_extra or {}),
        },
    )
    return _unsupported_result(
        value=value,
        graph=graph,
        request=derivative_request,
        code=code,
        message=message,
        method_id=method_id,
        diagnostic_extra={
            "structure_type": correlation_request.structure_type,
            "object_name": correlation_request.object_name,
            "factors": correlation_request.factors,
            "coordinate_space": "matrix",
            **dict(diagnostic_extra or {}),
        },
        fallback_reason={
            "code": code,
            "structure_type": correlation_request.structure_type,
            "object_name": correlation_request.object_name,
            "coordinate_space": "matrix",
            **dict(diagnostic_extra or {}),
        },
        backend_operator=backend_operator,
        method_metadata_extra=method_metadata_extra,
    )


def differentiate_arithmetic_asian_path_summary(
    instrument: object,
    market_context: object,
    request: HybridDerivativeRequest | None = None,
    *,
    position_name: str = "path_summary",
    vol_surface_name: str = "default_vol_surface",
    currency: str | None = None,
) -> HybridDerivativeResult:
    """Return VJP risk for one bounded arithmetic-average smooth path summary."""
    from trellis.analytics.portfolio_aad import (
        ArithmeticAsianOptionVolAADAdapter,
        PortfolioAADRequest,
    )
    from trellis.core.differentiable import get_backend_capabilities

    resolved_request = request or HybridDerivativeRequest()
    method_id = "hybrid_path_summary_vjp"
    adapter = ArithmeticAsianOptionVolAADAdapter()
    context = _path_summary_market_context(
        market_context,
        vol_surface_name=vol_surface_name,
        currency=currency,
    )
    portfolio_request = PortfolioAADRequest()
    decision = adapter.support_decision(
        position_name,
        instrument,
        context,
        portfolio_request,
    )
    admission_metadata = _semantic_admission_metadata(
        resolved_request.semantic_admission
    )

    graph: HybridFactorGraph
    try:
        graph = _path_summary_graph(
            context,
            derivative_method="vjp",
            position_name=position_name,
        )
    except Exception:
        graph = _unsupported_path_summary_graph(
            reason=decision.reason,
            position_name=position_name,
            vol_surface_name=vol_surface_name,
        )
    grid_vol_policy_metadata = _grid_vol_path_summary_policy_metadata(graph)

    value: float | None = None
    if decision.supported:
        value = float(adapter.value(instrument, context, portfolio_request))

    if resolved_request.semantic_admission is not None:
        admission = resolved_request.semantic_admission
        if grid_vol_policy_metadata and not admission.supported:
            state_metadata = _semantic_state_policy_metadata(admission)
            admission_metadata = _semantic_admission_metadata(admission)
            if admission.reason == "hybrid_jvp_backend_unsupported":
                admission_result = _unsupported_grid_vol_jvp_result(
                    value=value,
                    graph=graph,
                    request=resolved_request,
                    message=(
                        "Semantic hybrid AD admission rejected grid-vol "
                        "path-summary JVP at the backend boundary."
                    ),
                    diagnostic_extra={
                        "semantic_admission_status": admission.support_status,
                        "semantic_admission_lane_id": admission.lane_id,
                        "path_summary_type": "arithmetic_mean",
                        **state_metadata,
                    },
                    method_metadata_extra={
                        **grid_vol_policy_metadata,
                        **admission_metadata,
                    },
                )
            else:
                admission_result = _unsupported_grid_vol_path_summary_result(
                    value=value,
                    graph=graph,
                    request=resolved_request,
                    code=admission.reason,
                    message=(
                        "Semantic hybrid AD admission did not allow grid-vol "
                        f"path-summary execution: {admission.reason}."
                    ),
                    diagnostic_extra={
                        "semantic_admission_status": admission.support_status,
                        "semantic_admission_lane_id": admission.lane_id,
                        **state_metadata,
                    },
                    fallback_reason={
                        "code": admission.reason,
                        "semantic_admission_status": admission.support_status,
                        "semantic_admission_reason": admission.reason,
                        "semantic_admission_lane_id": admission.lane_id,
                        **state_metadata,
                    },
                    method_metadata_extra={
                        **grid_vol_policy_metadata,
                        **admission_metadata,
                    },
                )
        else:
            admission_result = _unsupported_semantic_admission_result(
                value=value,
                graph=graph,
                request=resolved_request,
                admission=admission,
                method_id=method_id,
                allowed_lane_prefixes=("arithmetic_asian_path_summary_",),
            )
        if admission_result is not None:
            return admission_result

    if resolved_request.derivative_method != "vjp":
        if grid_vol_policy_metadata:
            if resolved_request.derivative_method == "jvp":
                return _unsupported_grid_vol_jvp_result(
                    value=value,
                    graph=graph,
                    request=resolved_request,
                    message=(
                        "Grid-vol path-summary hybrid AD currently supports "
                        "no executable JVP lane; JVP remains fail-closed at "
                        "the backend boundary."
                    ),
                    diagnostic_extra={
                        "requested_derivative_method": (
                            resolved_request.derivative_method
                        ),
                        "path_summary_type": "arithmetic_mean",
                    },
                    method_metadata_extra=admission_metadata,
                )
            return _unsupported_grid_vol_path_summary_result(
                value=value,
                graph=graph,
                request=resolved_request,
                code=(
                    "path_summary_grid_vol_"
                    f"{resolved_request.derivative_method}_pending"
                ),
                message=(
                    "Grid-vol path-summary hybrid AD currently has no "
                    "executable HVP lane under the current coordinate policy."
                ),
                diagnostic_extra={
                    "requested_derivative_method": resolved_request.derivative_method,
                    "path_summary_type": "arithmetic_mean",
                },
                fallback_reason={
                    "requested_derivative_method": resolved_request.derivative_method,
                    "path_summary_type": "arithmetic_mean",
                },
                method_metadata_extra=admission_metadata,
            )
        if resolved_request.derivative_method == "jvp":
            return _unsupported_jvp_result(
                value=value,
                graph=graph,
                request=resolved_request,
                message=(
                    "Smooth path-summary hybrid AD currently supports only "
                    "the bounded arithmetic-Asian VJP lane; JVP remains "
                    "fail-closed at the backend boundary."
                ),
                diagnostic_extra={
                    "requested_derivative_method": resolved_request.derivative_method,
                    "path_summary_type": "arithmetic_mean",
                },
                method_metadata_extra=admission_metadata,
            )
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="path_summary_hvp_pending",
            message=(
                "Smooth path-summary hybrid AD currently supports only the "
                "bounded arithmetic-Asian VJP lane."
            ),
            method_id=method_id,
            diagnostic_extra={
                "requested_derivative_method": resolved_request.derivative_method,
                "path_summary_type": "arithmetic_mean",
            },
            fallback_reason={
                "code": "path_summary_hvp_pending",
                "requested_derivative_method": resolved_request.derivative_method,
                "path_summary_type": "arithmetic_mean",
            },
            method_metadata_extra=admission_metadata,
        )

    if not decision.supported:
        if grid_vol_policy_metadata:
            return _unsupported_grid_vol_path_summary_result(
                value=value,
                graph=graph,
                request=resolved_request,
                code=decision.reason,
                message=(
                    "Arithmetic-Asian path-summary VJP is unavailable for "
                    "this grid-vol market context under the current coordinate "
                    f"policy: {decision.reason}."
                ),
                diagnostic_extra={
                    "position_name": str(position_name),
                    "support_decision": decision.to_payload(),
                },
                fallback_reason={
                    "code": decision.reason,
                    "position_name": str(position_name),
                    "path_summary_type": "arithmetic_mean",
                    "support_decision": decision.to_payload(),
                },
                method_metadata_extra={
                    **grid_vol_policy_metadata,
                    **admission_metadata,
                },
            )
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code=decision.reason,
            message=(
                "Arithmetic-Asian path-summary VJP is unavailable for this "
                f"instrument or market context: {decision.reason}."
            ),
            method_id=method_id,
            diagnostic_extra={
                "position_name": str(position_name),
                "path_summary_type": "arithmetic_mean",
                "support_decision": decision.to_payload(),
            },
            fallback_reason={
                "code": decision.reason,
                "position_name": str(position_name),
                "path_summary_type": "arithmetic_mean",
            },
            method_metadata_extra=admission_metadata,
        )

    capabilities = get_backend_capabilities()
    full_vector = adapter.vjp(
        instrument,
        context,
        portfolio_request,
        weight=1.0,
    )
    selected_vector = resolved_request.filter_vector(full_vector)
    missing_factors = resolved_request.missing_selected_factors(full_vector)
    selected_diagnostics: list[dict[str, object]] = []
    support_status = "supported"
    if missing_factors:
        selected_diagnostics.append(
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
                "unsupported_selected_factor_policy": (
                    resolved_request.unsupported_selected_factor_policy
                ),
            }
        )
        support_status = "unsupported" if len(selected_vector) == 0 else "partial"
        if resolved_request.unsupported_selected_factor_policy == "fail_closed":
            selected_vector = SparseRiskVector()
            support_status = "unsupported"

    coordinates = graph.coordinates
    metadata = derivative_method_payload(
        method_id,
        method_support=support_status,
        backend_id=capabilities.backend_id,
        backend_operator="vjp",
        coordinate_space="constrained",
        chart_type="identity",
        hybrid_factor_graph_id=graph.graph_id,
        path_summary_type="arithmetic_mean",
        path_derivative_policy="lognormal_moment_matching_smooth_path_summary",
        factor_count=len(full_vector),
        sparse_nonzero_factor_count=len(full_vector),
        selected_factor_count=len(selected_vector),
        node_count=len(graph.nodes),
        unsupported_dependency_count=len(graph.unsupported_dependencies),
        risk_factor_coordinates=[
            coordinate.to_payload()
            for coordinate in coordinates
            if coordinate.factor_id in set(full_vector)
        ],
    )
    if value is not None and math.isfinite(value):
        metadata["base_value"] = value
    metadata.update(admission_metadata)

    diagnostics = tuple(selected_diagnostics) + tuple(
        dict(diagnostic) for diagnostic in decision.diagnostics
    )
    return HybridDerivativeResult(
        value=value,
        risk_vector=selected_vector,
        graph=graph,
        support_status=support_status,
        method_metadata=metadata,
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=diagnostics,
    )


def differentiate_vanilla_early_exercise(
    instrument: object,
    market_context: object,
    request: HybridDerivativeRequest | None = None,
    *,
    position_name: str = "early_exercise",
    vol_surface_name: str = "default_vol_surface",
    currency: str | None = None,
) -> HybridDerivativeResult:
    """Return VJP risk for one bounded vanilla early-exercise smooth-interior lane."""
    from trellis.analytics.portfolio_aad import (
        AADSupportDecision,
        PortfolioAADRequest,
        VanillaEquityOptionVolAADAdapter,
    )
    from trellis.core.differentiable import get_backend_capabilities

    resolved_request = request or HybridDerivativeRequest()
    method_id = "hybrid_early_exercise_vjp"
    adapter = VanillaEquityOptionVolAADAdapter()
    context = _vanilla_equity_vol_market_context(
        market_context,
        vol_surface_name=vol_surface_name,
        currency=currency,
    )
    portfolio_request = PortfolioAADRequest()
    decision = adapter.support_decision(
        position_name,
        instrument,
        context,
        portfolio_request,
    )
    exercise_style = str(getattr(instrument, "exercise_style", "european")).strip().lower()
    if exercise_style not in {"american", "bermudan"}:
        decision = AADSupportDecision(
            False,
            "unsupported_early_exercise_style",
            diagnostics=(
                {
                    "position_name": str(position_name),
                    "exercise_style": exercise_style,
                },
            ),
        )
    admission_metadata = _semantic_admission_metadata(
        resolved_request.semantic_admission
    )

    graph: HybridFactorGraph
    try:
        graph = _early_exercise_graph(
            context,
            derivative_method="vjp",
            position_name=position_name,
            exercise_style=exercise_style,
        )
    except Exception:
        graph = _unsupported_early_exercise_graph(
            reason=decision.reason,
            position_name=position_name,
            vol_surface_name=vol_surface_name,
            exercise_style=exercise_style,
        )
    grid_vol_policy_metadata = _grid_vol_state_control_policy_metadata(graph)

    value: float | None = None
    if decision.supported:
        value = float(adapter.value(instrument, context, portfolio_request))

    if resolved_request.semantic_admission is not None:
        admission = resolved_request.semantic_admission
        if grid_vol_policy_metadata and not admission.supported:
            state_metadata = _semantic_state_policy_metadata(admission)
            admission_metadata = _semantic_admission_metadata(admission)
            if admission.reason == "hybrid_jvp_backend_unsupported":
                admission_result = _unsupported_grid_vol_jvp_result(
                    value=value,
                    graph=graph,
                    request=resolved_request,
                    message=(
                        "Semantic hybrid AD admission rejected grid-vol "
                        "early-exercise JVP at the backend boundary."
                    ),
                    diagnostic_extra={
                        "semantic_admission_status": admission.support_status,
                        "semantic_admission_lane_id": admission.lane_id,
                        "exercise_style": exercise_style,
                        "early_exercise_policy": (
                            "grid_vol_hard_exercise_projection_pending"
                        ),
                        **state_metadata,
                    },
                    method_metadata_extra={
                        **grid_vol_policy_metadata,
                        **admission_metadata,
                    },
                )
            else:
                admission_result = _unsupported_grid_vol_early_exercise_result(
                    value=value,
                    graph=graph,
                    request=resolved_request,
                    code=admission.reason,
                    message=(
                        "Semantic hybrid AD admission did not allow grid-vol "
                        f"early-exercise execution: {admission.reason}."
                    ),
                    exercise_style=exercise_style,
                    diagnostic_extra={
                        "semantic_admission_status": admission.support_status,
                        "semantic_admission_lane_id": admission.lane_id,
                        **state_metadata,
                    },
                    fallback_reason={
                        "code": admission.reason,
                        "semantic_admission_status": admission.support_status,
                        "semantic_admission_reason": admission.reason,
                        "semantic_admission_lane_id": admission.lane_id,
                        **state_metadata,
                    },
                    method_metadata_extra={
                        **grid_vol_policy_metadata,
                        **admission_metadata,
                    },
                )
        else:
            admission_result = _unsupported_semantic_admission_result(
                value=value,
                graph=graph,
                request=resolved_request,
                admission=admission,
                method_id=method_id,
                allowed_lane_prefixes=("early_exercise_smooth_interior_",),
            )
        if admission_result is not None:
            return admission_result

    if resolved_request.derivative_method != "vjp":
        if grid_vol_policy_metadata:
            if resolved_request.derivative_method == "jvp":
                return _unsupported_grid_vol_jvp_result(
                    value=value,
                    graph=graph,
                    request=resolved_request,
                    message=(
                        "Grid-vol early-exercise hybrid AD currently supports "
                        "no executable JVP lane; JVP remains fail-closed at "
                        "the backend boundary."
                    ),
                    diagnostic_extra={
                        "requested_derivative_method": (
                            resolved_request.derivative_method
                        ),
                        "exercise_style": exercise_style,
                        "early_exercise_policy": (
                            "grid_vol_hard_exercise_projection_pending"
                        ),
                    },
                    method_metadata_extra=admission_metadata,
                )
            return _unsupported_grid_vol_early_exercise_result(
                value=value,
                graph=graph,
                request=resolved_request,
                code=(
                    "early_exercise_grid_vol_"
                    f"{resolved_request.derivative_method}_pending"
                ),
                message=(
                    "Grid-vol early-exercise hybrid AD currently has no "
                    "executable HVP lane under the current coordinate policy."
                ),
                exercise_style=exercise_style,
                diagnostic_extra={
                    "requested_derivative_method": resolved_request.derivative_method,
                    "exercise_style": exercise_style,
                },
                fallback_reason={
                    "requested_derivative_method": resolved_request.derivative_method,
                    "exercise_style": exercise_style,
                },
                method_metadata_extra=admission_metadata,
            )
        if resolved_request.derivative_method == "jvp":
            return _unsupported_jvp_result(
                value=value,
                graph=graph,
                request=resolved_request,
                message=(
                    "Early-exercise hybrid AD currently supports only the "
                    "bounded vanilla flat-vol VJP lane; JVP remains "
                    "fail-closed at the backend boundary."
                ),
                diagnostic_extra={
                    "requested_derivative_method": resolved_request.derivative_method,
                    "exercise_style": exercise_style,
                    "early_exercise_policy": (
                        "hard_exercise_projection_smooth_interior"
                    ),
                },
                method_metadata_extra=admission_metadata,
            )
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="early_exercise_hvp_pending",
            message=(
                "Early-exercise hybrid AD currently supports only the bounded "
                "vanilla flat-vol VJP lane."
            ),
            method_id=method_id,
            diagnostic_extra={
                "requested_derivative_method": resolved_request.derivative_method,
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "hard_exercise_projection_smooth_interior"
                ),
            },
            fallback_reason={
                "code": "early_exercise_hvp_pending",
                "requested_derivative_method": resolved_request.derivative_method,
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "hard_exercise_projection_smooth_interior"
                ),
            },
            method_metadata_extra=admission_metadata,
        )

    if not decision.supported:
        if grid_vol_policy_metadata:
            return _unsupported_grid_vol_early_exercise_result(
                value=value,
                graph=graph,
                request=resolved_request,
                code=decision.reason,
                message=(
                    "Vanilla early-exercise VJP is unavailable for this "
                    "grid-vol market context under the current coordinate "
                    f"policy: {decision.reason}."
                ),
                exercise_style=exercise_style,
                diagnostic_extra={
                    "position_name": str(position_name),
                    "support_decision": decision.to_payload(),
                },
                fallback_reason={
                    "code": decision.reason,
                    "position_name": str(position_name),
                    "exercise_style": exercise_style,
                    "early_exercise_policy": (
                        "grid_vol_hard_exercise_projection_pending"
                    ),
                    "support_decision": decision.to_payload(),
                },
                method_metadata_extra={
                    **grid_vol_policy_metadata,
                    **admission_metadata,
                },
            )
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code=decision.reason,
            message=(
                "Vanilla early-exercise VJP is unavailable for this instrument "
                f"or market context: {decision.reason}."
            ),
            method_id=method_id,
            diagnostic_extra={
                "position_name": str(position_name),
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "hard_exercise_projection_smooth_interior"
                ),
                "support_decision": decision.to_payload(),
            },
            fallback_reason={
                "code": decision.reason,
                "position_name": str(position_name),
                "exercise_style": exercise_style,
                "early_exercise_policy": (
                    "hard_exercise_projection_smooth_interior"
                ),
            },
            method_metadata_extra=admission_metadata,
        )

    capabilities = get_backend_capabilities()
    full_vector = adapter.vjp(
        instrument,
        context,
        portfolio_request,
        weight=1.0,
    )
    selected_vector = resolved_request.filter_vector(full_vector)
    missing_factors = resolved_request.missing_selected_factors(full_vector)
    selected_diagnostics: list[dict[str, object]] = []
    support_status = "supported"
    if missing_factors:
        selected_diagnostics.append(
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
                "unsupported_selected_factor_policy": (
                    resolved_request.unsupported_selected_factor_policy
                ),
            }
        )
        support_status = "unsupported" if len(selected_vector) == 0 else "partial"
        if resolved_request.unsupported_selected_factor_policy == "fail_closed":
            selected_vector = SparseRiskVector()
            support_status = "unsupported"

    coordinates = graph.coordinates
    metadata = derivative_method_payload(
        method_id,
        method_support=support_status,
        backend_id=capabilities.backend_id,
        backend_operator="vjp",
        coordinate_space="constrained",
        chart_type="identity",
        hybrid_factor_graph_id=graph.graph_id,
        exercise_style=exercise_style,
        early_exercise_policy="hard_exercise_projection_smooth_interior",
        factor_count=len(full_vector),
        sparse_nonzero_factor_count=len(full_vector),
        selected_factor_count=len(selected_vector),
        node_count=len(graph.nodes),
        unsupported_dependency_count=len(graph.unsupported_dependencies),
        risk_factor_coordinates=[
            coordinate.to_payload()
            for coordinate in coordinates
            if coordinate.factor_id in set(full_vector)
        ],
    )
    if value is not None and math.isfinite(value):
        metadata["base_value"] = value
    metadata.update(admission_metadata)

    diagnostics = tuple(selected_diagnostics) + tuple(
        dict(diagnostic) for diagnostic in decision.diagnostics
    )
    return HybridDerivativeResult(
        value=value,
        risk_vector=selected_vector,
        graph=graph,
        support_status=support_status,
        method_metadata=metadata,
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=diagnostics,
    )


def differentiate_quanto_correlation_matrix(
    spec: object,
    resolved_inputs: object,
    correlation_request: HybridCorrelationStructureRequest,
    request: HybridDerivativeRequest | None = None,
    *,
    active_factor_pair: tuple[object, object] | None = None,
    min_eigenvalue_floor: float = 1.0e-6,
) -> HybridDerivativeResult:
    """Return VJP or HVP risk for a checked quanto correlation-matrix coordinate."""
    from trellis.core.differentiable import (
        get_backend_capabilities,
        get_numpy,
        hessian_vector_product,
        vjp,
    )
    from trellis.models.analytical.quanto import price_quanto_option_raw

    if not isinstance(correlation_request, HybridCorrelationStructureRequest):
        raise TypeError("correlation_request must be a HybridCorrelationStructureRequest")
    resolved_request = request or HybridDerivativeRequest()
    method_id = (
        "hybrid_matrix_vector_hvp"
        if resolved_request.derivative_method == "hvp"
        else "hybrid_matrix_vector_vjp"
    )
    active_pair = active_factor_pair or _quanto_matrix_active_factor_pair(spec)
    base_value = float(price_quanto_option_raw(spec, resolved_inputs))
    admission_metadata = _semantic_admission_metadata(resolved_request.semantic_admission)
    try:
        context = build_correlation_matrix_coordinate_context(
            correlation_request,
            active_factor_pair=active_pair,
            min_eigenvalue_floor=min_eigenvalue_floor,
        )
    except ValueError as exc:
        code, message = _matrix_context_error_code_and_message(exc)
        return _unsupported_correlation_matrix_result(
            value=base_value,
            correlation_request=correlation_request,
            derivative_request=resolved_request,
            code=code,
            message=message,
            diagnostic_extra={"active_factor_pair": tuple(str(factor) for factor in active_pair)},
            method_metadata_extra=admission_metadata,
        )

    graph = _matrix_context_graph(
        correlation_request,
        context,
        derivative_method=resolved_request.derivative_method,
    )
    capabilities = get_backend_capabilities()
    if resolved_request.semantic_admission is not None:
        admission_result = _unsupported_semantic_admission_result(
            value=base_value,
            graph=graph,
            request=resolved_request,
            admission=resolved_request.semantic_admission,
            method_id=method_id,
            allowed_lane_prefixes=("quanto_matrix_graph_",),
        )
        if admission_result is not None:
            return admission_result
    if resolved_request.derivative_method == "jvp":
        return _unsupported_jvp_result(
            value=base_value,
            graph=graph,
            request=resolved_request,
            message=(
                "Hybrid matrix-coordinate JVP remains fail-closed because the "
                "active backend does not provide checked JVP coverage."
            ),
            diagnostic_extra={"active_factor_key": context.active_factor_id.key},
            method_metadata_extra=admission_metadata,
        )
    if resolved_request.derivative_method not in {"vjp", "hvp"}:
        return _unsupported_result(
            value=base_value,
            graph=graph,
            request=resolved_request,
            code="hybrid_derivative_method_unsupported",
            message="Only VJP and HVP are supported by the bounded matrix-coordinate lane.",
            method_id=method_id,
            backend_operator=resolved_request.derivative_method,
            diagnostic_extra={"active_factor_key": context.active_factor_id.key},
            method_metadata_extra=admission_metadata,
        )
    if not is_dataclass(resolved_inputs):
        return _unsupported_result(
            value=base_value,
            graph=graph,
            request=resolved_request,
            code="resolved_inputs_dataclass_required",
            message="Matrix-coordinate quanto VJP requires dataclass resolved inputs.",
            method_id=method_id,
            diagnostic_extra={"active_factor_key": context.active_factor_id.key},
            method_metadata_extra=admission_metadata,
        )
    if context.active_coordinate_index is None or context.active_factor_id is None:
        return _unsupported_result(
            value=base_value,
            graph=graph,
            request=resolved_request,
            code="active_matrix_coordinate_required",
            message="Matrix-coordinate quanto VJP requires one active correlation pair.",
            method_id=method_id,
            method_metadata_extra=admission_metadata,
        )

    np = get_numpy()
    entries = _matrix_entries_for_context(correlation_request, context)
    base_vector = np.asarray(tuple(entry.base_value for entry in entries))
    active_index = context.active_coordinate_index

    def value_from_matrix_entries(theta):
        traced_inputs = replace(resolved_inputs, corr=theta[active_index])
        return price_quanto_option_raw(spec, traced_inputs)

    metadata_extra: dict[str, object] = {}
    if resolved_request.derivative_method == "hvp":
        if not capabilities.supports("hessian_vector_product"):
            diagnostic_extra = {
                "backend_id": capabilities.backend_id,
                "unsupported_operator": "hessian_vector_product",
                "backend_notes": capabilities.notes,
                "active_factor_key": context.active_factor_id.key,
            }
            return _unsupported_result(
                value=base_value,
                graph=graph,
                request=resolved_request,
                code="hybrid_hvp_backend_unsupported",
                message=(
                    "Hybrid matrix-coordinate HVP is fail-closed because the "
                    "active backend does not provide checked scalar-objective HVP support."
                ),
                method_id=method_id,
                diagnostic_extra=diagnostic_extra,
                fallback_reason={
                    "code": "hybrid_hvp_backend_unsupported",
                    **diagnostic_extra,
                },
                backend_operator="hessian_vector_product",
                method_metadata_extra=admission_metadata,
            )
        direction_values, direction_diagnostic = _hvp_direction_for_entries(
            resolved_request,
            entries,
        )
        if direction_diagnostic is not None:
            return _unsupported_result(
                value=base_value,
                graph=graph,
                request=resolved_request,
                code=str(direction_diagnostic["code"]),
                message=str(direction_diagnostic["message"]),
                method_id=method_id,
                diagnostic_extra={
                    key: val
                    for key, val in direction_diagnostic.items()
                    if key not in {"code", "message"}
                },
                fallback_reason=direction_diagnostic,
                backend_operator="hessian_vector_product",
                method_metadata_extra=admission_metadata,
            )
        direction_vector = np.asarray(direction_values)
        traced_value = value_from_matrix_entries(base_vector)
        sensitivities = np.reshape(
            np.asarray(
                hessian_vector_product(
                    value_from_matrix_entries,
                    base_vector,
                    direction_vector,
                )
            ),
            (-1,),
        )
        metadata_extra = {
            "hvp_direction_factor_count": len(resolved_request.hvp_direction),
            "hvp_direction_coordinate_count": sum(
                1 for value in direction_values if value != 0.0
            ),
            "hvp_direction_norm": float(np.sqrt(np.sum(direction_vector * direction_vector))),
        }
        backend_operator = "hessian_vector_product"
    else:
        traced_value, pullback = vjp(value_from_matrix_entries, base_vector)
        sensitivities = np.reshape(np.asarray(pullback(np.asarray(1.0))), (-1,))
        backend_operator = "vjp"
    full_vector = SparseRiskVector.from_items(
        (entry.factor_id, float(sensitivities[index]))
        for index, entry in enumerate(entries)
    )
    selected_vector = resolved_request.filter_vector(full_vector)
    missing_factors = resolved_request.missing_selected_factors(context.factor_ids)
    diagnostics: list[dict[str, object]] = []
    support_status = "supported"
    if missing_factors:
        diagnostics.append(
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
                "unsupported_selected_factor_policy": (
                    resolved_request.unsupported_selected_factor_policy
                ),
            }
        )
        support_status = "unsupported" if len(selected_vector) == 0 else "partial"
        if resolved_request.unsupported_selected_factor_policy == "fail_closed":
            selected_vector = SparseRiskVector()
            support_status = "unsupported"

    value = float(traced_value)
    metadata = derivative_method_payload(
        method_id,
        method_support=support_status,
        backend_id=capabilities.backend_id,
        backend_operator=backend_operator,
        coordinate_space=context.chart.coordinate_space,
        chart_type=context.chart.chart_type,
        hybrid_factor_graph_id=graph.graph_id,
        matrix_dimension=context.chart.coordinate_values["dimension"],
        matrix_coordinate_count=context.coordinate_count,
        factor_count=context.coordinate_count,
        sparse_nonzero_factor_count=len(full_vector),
        active_factor_key=context.active_factor_id.key,
        active_coordinate_index=active_index,
        min_eigenvalue=context.min_eigenvalue,
        min_eigenvalue_floor=context.min_eigenvalue_floor,
        projection_policy=context.chart.constraints["projection_policy"],
        node_count=len(graph.nodes),
        unsupported_dependency_count=0,
        **metadata_extra,
    )
    if math.isfinite(value):
        metadata["base_value"] = value
    metadata.update(admission_metadata)

    return HybridDerivativeResult(
        value=value,
        risk_vector=selected_vector,
        graph=graph,
        support_status=support_status,
        method_metadata=metadata,
        diagnostics=tuple(diagnostics),
    )


def fail_closed_correlation_structure_derivative(
    request: HybridCorrelationStructureRequest,
) -> HybridDerivativeResult:
    """Return a fail-closed result for unsupported matrix/surface correlation AD."""
    if not isinstance(request, HybridCorrelationStructureRequest):
        raise TypeError("request must be a HybridCorrelationStructureRequest")
    code, message, policy_extra, chart = _correlation_structure_policy_payload(request)
    node = None
    if chart is not None:
        node = HybridDependencyNode(
            node_id=f"node:{request.structure_type}:{request.object_name}",
            node_type=request.structure_type,
            object_name=request.object_name,
            coordinate_chart=chart,
            differentiability_class=chart.differentiability_class,
            derivative_method="unsupported",
            support_status=chart.support_status,
            metadata={
                "policy": "fail_closed_no_projection",
                "chart_policy_status": policy_extra["chart_policy_status"],
            },
        )
    dependency = HybridUnsupportedDependency(
        dependency_id=f"node:{request.structure_type}:{request.object_name}",
        node_type=request.structure_type,
        object_name=request.object_name,
        reason=code,
        metadata={
            "factors": request.factors,
            "requested_derivative_method": request.requested_derivative_method,
            "coordinate_space": request.coordinate_space,
            "policy": "fail_closed_no_projection",
            **policy_extra,
            **dict(request.provenance),
        },
    )
    graph = HybridFactorGraph(
        graph_id=f"hybrid:{request.structure_type}:{request.object_name}",
        nodes=(node,) if node is not None else (),
        unsupported_dependencies=(dependency,),
        metadata={
            "structure_type": request.structure_type,
            "object_name": request.object_name,
            "factors": request.factors,
            **policy_extra,
        },
    )
    diagnostic = {
        "code": code,
        "severity": "warning",
        "message": message,
        "structure_type": request.structure_type,
        "object_name": request.object_name,
        "factors": request.factors,
        "psd_chart_required": request.structure_type == "correlation_matrix",
        "projection_policy": "unsupported_no_smoothing_or_projection",
        **policy_extra,
    }
    metadata = derivative_method_payload(
        "unsupported_hybrid_structure",
        method_support="unsupported",
        backend_operator=request.requested_derivative_method,
        coordinate_space=request.coordinate_space,
        hybrid_factor_graph_id=graph.graph_id,
        correlation_structure_type=request.structure_type,
        fallback_reason=diagnostic,
    )
    return HybridDerivativeResult(
        value=None,
        risk_vector=SparseRiskVector(),
        graph=graph,
        support_status="unsupported",
        method_metadata=metadata,
        unsupported_dependencies=(dependency,),
        diagnostics=(diagnostic,),
    )


def _dynamic_state_policy_payload(
    admission: HybridADLaneAdmission,
) -> dict[str, object]:
    state_policy = admission.metadata.get("state_policy")
    return dict(state_policy) if isinstance(state_policy, Mapping) else {}


def _graph_support_status_from_admission(admission: HybridADLaneAdmission) -> str:
    if admission.support_status == "supported":
        return "supported"
    if admission.support_status == "planned":
        return "discovery_only"
    return "unsupported"


def _is_dynamic_state_admission(admission: HybridADLaneAdmission) -> bool:
    return (
        admission.semantic_contract_type == "DynamicContractIR"
        and admission.contract_shape == "dynamic_hybrid_state"
    )


def _dynamic_state_factor_coordinate(
    *,
    position_name: str,
    semantic_family: str,
    base_track: str,
    admission: HybridADLaneAdmission,
    graph_support_status: str,
) -> RiskFactorCoordinate:
    factor = RiskFactorId(
        object_type="dynamic_contract",
        object_name=position_name,
        coordinate_type="dynamic_state_policy",
        axes={
            "semantic_family": semantic_family or admission.product_family,
            "base_track": base_track or "unspecified",
            "lane_id": admission.lane_id,
        },
        provenance_namespace="hybrid_ad",
    )
    return RiskFactorCoordinate(
        factor_id=factor,
        object_path=f"dynamic_contract:{position_name}",
        display_name=f"{position_name} dynamic state policy",
        unit="policy",
        support_status=graph_support_status,
        reporting_buckets={
            "risk_class": "hybrid",
            "state_kind": "dynamic_state",
        },
        metadata={
            "semantic_contract_type": admission.semantic_contract_type,
            "contract_shape": admission.contract_shape,
            "lane_id": admission.lane_id,
            "support_status": admission.support_status,
        },
    )


def _dynamic_state_graph(
    *,
    semantic_contract: DynamicContractIR,
    admission: HybridADLaneAdmission,
    position_name: str,
    market_parameterization: str,
) -> HybridFactorGraph:
    semantic_family = semantic_contract.semantic_family or admission.product_family
    base_track = semantic_contract.base_track or ""
    state_policy = _dynamic_state_policy_payload(admission)
    graph_support_status = _graph_support_status_from_admission(admission)
    coordinate = _dynamic_state_factor_coordinate(
        position_name=position_name,
        semantic_family=semantic_family,
        base_track=base_track,
        admission=admission,
        graph_support_status=graph_support_status,
    )
    chart = MarketObjectCoordinateChart(
        chart_id=f"chart:dynamic_state:{position_name}",
        object_type="dynamic_contract",
        object_name=position_name,
        coordinates=(coordinate,),
        chart_type="dynamic_state_policy",
        coordinate_space="state_policy",
        coordinate_values={
            "state_kind": state_policy.get("state_kind", "dynamic_state"),
            "state_variable_roles": tuple(
                state_policy.get("state_variable_roles", ())
            ),
            "event_policy": state_policy.get("event_policy", "not_applicable"),
            "control_policy": state_policy.get("control_policy", "not_applicable"),
            "fail_closed": bool(state_policy.get("fail_closed", True)),
        },
        differentiability_class=str(
            state_policy.get("differentiability_class", "piecewise")
        ),
        support_status=graph_support_status,
        metadata={
            "chart_family": "dynamic_state_policy",
            "state_kind": state_policy.get("state_kind", "dynamic_state"),
            "semantic_family": semantic_family,
            "base_track": base_track,
            "lane_id": admission.lane_id,
            "admission_reason": admission.reason,
            "market_parameterization": market_parameterization,
        },
    )
    node = HybridDependencyNode(
        node_id=f"node:dynamic_state:{position_name}",
        node_type="dynamic_contract",
        object_name=position_name,
        coordinate_chart=chart,
        differentiability_class=chart.differentiability_class,
        derivative_method="unsupported",
        support_status=chart.support_status,
        provenance={
            "semantic_contract_type": admission.semantic_contract_type,
            "semantic_family": semantic_family,
            "base_track": base_track,
        },
        metadata={
            "policy": "fail_closed_dynamic_state",
            "lane_id": admission.lane_id,
            "state_kind": state_policy.get("state_kind", "dynamic_state"),
            "event_policy": state_policy.get("event_policy", "not_applicable"),
            "control_policy": state_policy.get("control_policy", "not_applicable"),
        },
    )
    dependency = HybridUnsupportedDependency(
        dependency_id=f"unsupported:dynamic_state:{position_name}:{admission.reason}",
        node_type="dynamic_contract",
        object_name=position_name,
        reason=admission.reason,
        differentiability_class=chart.differentiability_class,
        derivative_method="unsupported",
        metadata={
            "policy": "fail_closed_dynamic_state",
            "semantic_contract_type": admission.semantic_contract_type,
            "semantic_family": semantic_family,
            "product_family": admission.product_family,
            "contract_shape": admission.contract_shape,
            "lane_id": admission.lane_id,
            "state_policy": state_policy,
            "market_parameterization": market_parameterization,
        },
    )
    return HybridFactorGraph(
        graph_id=f"hybrid:dynamic_state:{position_name}",
        nodes=(node,),
        unsupported_dependencies=(dependency,),
        metadata={
            "route_family": "bounded_dynamic_state",
            "graph_source": "dynamic_contract_ir_admission",
            "semantic_contract_type": admission.semantic_contract_type,
            "semantic_family": semantic_family,
            "product_family": admission.product_family,
            "contract_shape": admission.contract_shape,
            "base_track": base_track,
            "market_parameterization": market_parameterization,
            "state_policy": state_policy,
        },
    )


def _dynamic_state_unavailable_graph(
    *,
    semantic_contract: DynamicContractIR,
    admission: HybridADLaneAdmission,
    position_name: str,
    market_parameterization: str,
) -> HybridFactorGraph:
    semantic_family = semantic_contract.semantic_family or admission.product_family
    actual_admission = admission.metadata.get("actual_semantic_admission")
    actual_contract_shape = admission.contract_shape
    if isinstance(actual_admission, Mapping):
        actual_contract_shape = str(
            actual_admission.get("contract_shape", actual_contract_shape)
        )
    dependency = HybridUnsupportedDependency(
        dependency_id=f"unsupported:dynamic_state_unavailable:{position_name}",
        node_type="dynamic_contract",
        object_name=position_name,
        reason=admission.reason,
        differentiability_class="unsupported",
        derivative_method="unsupported",
        metadata={
            "policy": "fail_closed_dynamic_state_wrong_lane",
            "semantic_contract_type": admission.semantic_contract_type,
            "semantic_family": semantic_family,
            "product_family": admission.product_family,
            "contract_shape": admission.contract_shape,
            "actual_contract_shape": actual_contract_shape,
            "expected_contract_shape": "dynamic_hybrid_state",
            "lane_id": admission.lane_id,
            "market_parameterization": market_parameterization,
        },
    )
    return HybridFactorGraph(
        graph_id=f"hybrid:dynamic_state_unavailable:{position_name}",
        unsupported_dependencies=(dependency,),
        metadata={
            "route_family": "bounded_dynamic_state",
            "graph_source": "dynamic_contract_ir_admission",
            "semantic_contract_type": admission.semantic_contract_type,
            "semantic_family": semantic_family,
            "product_family": admission.product_family,
            "contract_shape": admission.contract_shape,
            "actual_contract_shape": actual_contract_shape,
            "expected_contract_shape": "dynamic_hybrid_state",
            "base_track": semantic_contract.base_track or "",
            "market_parameterization": market_parameterization,
        },
    )


def _resolve_dynamic_state_admission(
    semantic_contract: DynamicContractIR,
    request: HybridDerivativeRequest,
    *,
    product_family: str | None,
    market_parameterization: str,
) -> HybridADLaneAdmission:
    base_admission = admit_hybrid_ad_lane(
        semantic_contract,
        derivative_method=request.derivative_method,
        market_parameterization=market_parameterization,
        product_family=product_family or semantic_contract.semantic_family or None,
    )
    supplied = request.semantic_admission
    if supplied is None:
        if _is_dynamic_state_admission(base_admission):
            return base_admission
        return replace(
            base_admission,
            support_status="unsupported",
            reason="semantic_admission_lane_unavailable",
            metadata={
                **dict(base_admission.metadata),
                "actual_semantic_admission": base_admission.to_payload(),
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "semantic_admission_lane_unavailable",
                    "severity": "warning",
                    "actual_lane_id": base_admission.lane_id,
                    "actual_contract_shape": base_admission.contract_shape,
                    "expected_contract_shape": "dynamic_hybrid_state",
                },
            ),
        )
    if _is_dynamic_state_admission(supplied):
        return supplied
    if _is_dynamic_state_admission(base_admission):
        return replace(
            base_admission,
            support_status="unsupported",
            reason="semantic_admission_lane_unavailable",
            metadata={
                **dict(base_admission.metadata),
                "supplied_semantic_admission": supplied.to_payload(),
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "semantic_admission_lane_unavailable",
                    "severity": "warning",
                    "supplied_lane_id": supplied.lane_id,
                    "supplied_contract_shape": supplied.contract_shape,
                    "expected_contract_shape": "dynamic_hybrid_state",
                },
            ),
        )
    return replace(
        base_admission,
        support_status="unsupported",
        reason="semantic_admission_lane_unavailable",
        metadata={
            **dict(base_admission.metadata),
            "supplied_semantic_admission": supplied.to_payload(),
            "actual_semantic_admission": base_admission.to_payload(),
            "fail_closed": True,
        },
        diagnostics=(
            {
                "code": "semantic_admission_lane_unavailable",
                "severity": "warning",
                "supplied_lane_id": supplied.lane_id,
                "supplied_contract_shape": supplied.contract_shape,
                "actual_lane_id": base_admission.lane_id,
                "actual_contract_shape": base_admission.contract_shape,
                "expected_contract_shape": "dynamic_hybrid_state",
            },
        ),
    )


def fail_closed_dynamic_state_derivative(
    semantic_contract: DynamicContractIR,
    *,
    request: HybridDerivativeRequest | None = None,
    value: float | None = None,
    position_name: str = "dynamic_contract",
    product_family: str | None = None,
    market_parameterization: str = "flat_vol",
) -> HybridDerivativeResult:
    """Return a typed fail-closed result for unsupported dynamic-state hybrid AD."""
    if not isinstance(semantic_contract, DynamicContractIR):
        raise TypeError("semantic_contract must be a DynamicContractIR")
    position = str(position_name or "dynamic_contract").strip() or "dynamic_contract"
    resolved_request = request or HybridDerivativeRequest()
    admission = _resolve_dynamic_state_admission(
        semantic_contract,
        resolved_request,
        product_family=product_family,
        market_parameterization=market_parameterization,
    )
    resolved_request = replace(resolved_request, semantic_admission=admission)
    if _is_dynamic_state_admission(admission):
        graph = _dynamic_state_graph(
            semantic_contract=semantic_contract,
            admission=admission,
            position_name=position,
            market_parameterization=market_parameterization,
        )
    else:
        graph = _dynamic_state_unavailable_graph(
            semantic_contract=semantic_contract,
            admission=admission,
            position_name=position,
            market_parameterization=market_parameterization,
        )
    admission_metadata = _semantic_admission_metadata(admission)
    selected_diagnostics = _unsupported_selected_factor_diagnostics(
        resolved_request,
        graph,
        code="unsupported_selected_dynamic_state_factors",
    )
    state_policy = graph.metadata.get("state_policy")
    if not isinstance(state_policy, Mapping):
        state_policy = admission.metadata.get("state_policy")
    diagnostic_extra_state = (
        {"state_kind": state_policy.get("state_kind", "dynamic_state")}
        if isinstance(state_policy, Mapping)
        else {}
    )
    admission_diagnostic_extra = dict(admission.diagnostics[0]) if admission.diagnostics else {}
    admission_diagnostic_extra.pop("code", None)
    admission_diagnostic_extra.pop("severity", None)
    diagnostic_extra = {
        "semantic_admission_status": admission.support_status,
        "semantic_admission_reason": admission.reason,
        "semantic_contract_type": admission.semantic_contract_type,
        "product_family": admission.product_family,
        "contract_shape": admission.contract_shape,
        "lane_id": admission.lane_id,
        **admission_diagnostic_extra,
        **diagnostic_extra_state,
    }
    if resolved_request.derivative_method == "jvp":
        result = _unsupported_jvp_result(
            value=value,
            graph=graph,
            request=resolved_request,
            message=(
                "DynamicContractIR dynamic-state JVP is fail-closed because the "
                "active backend has no checked forward-mode primitive lane."
            ),
            diagnostic_extra=diagnostic_extra,
            method_metadata_extra=admission_metadata,
        )
    else:
        result = _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code=admission.reason,
            message=(
                "DynamicContractIR dynamic-state derivatives are fail-closed "
                "until an executable state/control replay lane is admitted."
            ),
            method_id="unsupported_hybrid_structure",
            fallback_reason={
                "code": admission.reason,
                **diagnostic_extra,
            },
            diagnostic_extra=diagnostic_extra,
            method_metadata_extra=admission_metadata,
        )
    if selected_diagnostics:
        return replace(
            result,
            diagnostics=tuple(result.diagnostics) + selected_diagnostics,
        )
    return result


def _semantic_admission_metadata(
    admission: HybridADLaneAdmission | None,
) -> dict[str, object]:
    if admission is None:
        return {}
    return {
        "semantic_admission": admission.to_payload(),
        **_semantic_state_policy_metadata(admission),
    }


def _semantic_state_policy_metadata(
    admission: HybridADLaneAdmission,
) -> dict[str, object]:
    state_policy = admission.metadata.get("state_policy")
    if not isinstance(state_policy, Mapping):
        return {}
    payload = dict(state_policy)
    return {
        "semantic_state_policy": payload,
        "semantic_state_kind": str(payload.get("state_kind", "")),
        "semantic_state_policy_support_status": str(payload.get("support_status", "")),
        "semantic_state_differentiability_class": str(
            payload.get("differentiability_class", "")
        ),
        "semantic_state_event_policy": str(payload.get("event_policy", "")),
        "semantic_state_control_policy": str(payload.get("control_policy", "")),
        "semantic_state_fail_closed": bool(payload.get("fail_closed", True)),
    }


def _unsupported_semantic_admission_result(
    *,
    value: float | None,
    graph: HybridFactorGraph,
    request: HybridDerivativeRequest,
    admission: HybridADLaneAdmission,
    method_id: str,
    allowed_lane_prefixes: tuple[str, ...] = ("quanto_scalar_graph_",),
) -> HybridDerivativeResult | None:
    admission_metadata = _semantic_admission_metadata(admission)
    state_metadata = _semantic_state_policy_metadata(admission)
    if not admission.supported:
        fallback_reason = {
            "code": admission.reason,
            "semantic_admission_status": admission.support_status,
            "semantic_admission_reason": admission.reason,
            "semantic_admission_lane_id": admission.lane_id,
            **state_metadata,
        }
        return _unsupported_result(
            value=value,
            graph=graph,
            request=request,
            code=admission.reason,
            message=(
                "Semantic hybrid AD admission did not allow runtime execution: "
                f"{admission.reason}."
            ),
            method_id=method_id,
            diagnostic_extra={
                "semantic_admission_status": admission.support_status,
                "semantic_admission_lane_id": admission.lane_id,
                **state_metadata,
            },
            fallback_reason=fallback_reason,
            method_metadata_extra=admission_metadata,
        )
    if request.derivative_method not in admission.derivative_methods:
        fallback_reason = {
            "code": "semantic_admission_derivative_method_unavailable",
            "semantic_admission_status": admission.support_status,
            "semantic_admission_reason": admission.reason,
            "semantic_admission_lane_id": admission.lane_id,
            "requested_derivative_method": request.derivative_method,
            "admitted_derivative_methods": list(admission.derivative_methods),
            **state_metadata,
        }
        return _unsupported_result(
            value=value,
            graph=graph,
            request=request,
            code="semantic_admission_derivative_method_unavailable",
            message=(
                "Semantic hybrid AD admission does not include the requested "
                "derivative method."
            ),
            method_id=method_id,
            diagnostic_extra={
                "semantic_admission_lane_id": admission.lane_id,
                "requested_derivative_method": request.derivative_method,
                "admitted_derivative_methods": list(admission.derivative_methods),
                **state_metadata,
            },
            fallback_reason=fallback_reason,
            method_metadata_extra=admission_metadata,
        )
    if not any(admission.lane_id.startswith(prefix) for prefix in allowed_lane_prefixes):
        fallback_reason = {
            "code": "semantic_admission_lane_unavailable",
            "semantic_admission_status": admission.support_status,
            "semantic_admission_reason": admission.reason,
            "semantic_admission_lane_id": admission.lane_id,
            **state_metadata,
        }
        return _unsupported_result(
            value=value,
            graph=graph,
            request=request,
            code="semantic_admission_lane_unavailable",
            message=(
                "Semantic hybrid AD admission refers to a lane that this "
                "runtime helper does not execute."
            ),
            method_id=method_id,
            diagnostic_extra={
                "semantic_admission_lane_id": admission.lane_id,
                **state_metadata,
            },
            fallback_reason=fallback_reason,
            method_metadata_extra=admission_metadata,
        )
    return None


def differentiate_quanto_scalar_inputs(
    spec: object,
    resolved_inputs: object,
    request: HybridDerivativeRequest | None = None,
) -> HybridDerivativeResult:
    """Return VJP or HVP sensitivities to supported graph-owned quanto scalar inputs."""
    from trellis.core.differentiable import (
        get_backend_capabilities,
        get_numpy,
        hessian_vector_product,
        vjp,
    )
    from trellis.models.analytical.quanto import price_quanto_option_raw

    resolved_request = request or HybridDerivativeRequest()
    method_id = (
        "hybrid_scalar_vector_hvp"
        if resolved_request.derivative_method == "hvp"
        else "hybrid_scalar_vector_vjp"
    )
    graph = getattr(resolved_inputs, "hybrid_factor_graph", None)
    if graph is None:
        graph = _fallback_graph(spec, resolved_inputs)
    if not isinstance(graph, HybridFactorGraph):
        raise TypeError("resolved_inputs.hybrid_factor_graph must be a HybridFactorGraph")
    graph.validate()
    value = float(price_quanto_option_raw(spec, resolved_inputs))
    capabilities = get_backend_capabilities()
    admission_metadata = _semantic_admission_metadata(resolved_request.semantic_admission)
    if resolved_request.semantic_admission is not None:
        admission_result = _unsupported_semantic_admission_result(
            value=value,
            graph=graph,
            request=resolved_request,
            admission=resolved_request.semantic_admission,
            method_id=method_id,
        )
        if admission_result is not None:
            return admission_result

    if resolved_request.derivative_method == "jvp":
        return _unsupported_jvp_result(
            value=value,
            graph=graph,
            request=resolved_request,
            message=(
                "Hybrid JVP remains fail-closed because the active backend "
                "does not provide checked JVP coverage for pricing primitives."
            ),
            method_metadata_extra=admission_metadata,
        )
    if resolved_request.derivative_method not in {"vjp", "hvp"}:
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="hybrid_derivative_method_unsupported",
            message="Only VJP and HVP are supported by the bounded scalar quanto input lane.",
            method_id=method_id,
            method_metadata_extra=admission_metadata,
        )
    if not is_dataclass(resolved_inputs):
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="resolved_inputs_dataclass_required",
            message="Scalar quanto input VJP/HVP requires dataclass resolved inputs.",
            method_id=method_id,
            method_metadata_extra=admission_metadata,
        )

    extraction = _quanto_scalar_entries(
        graph,
        coordinate_space=resolved_request.coordinate_space,
    )
    entries = extraction.entries
    if not entries:
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="graph_scalar_coordinates_unavailable",
            message="Hybrid graph does not contain supported scalar quanto coordinates.",
            method_id=method_id,
            diagnostic_extra={
                "scalar_chart_diagnostics": list(extraction.diagnostics),
            },
            method_metadata_extra=admission_metadata,
        )

    np = get_numpy()
    base_vector = np.asarray(tuple(entry.base_value for entry in entries))

    def value_from_vector(theta):
        traced_inputs = _replace_inputs_from_graph(
            resolved_inputs,
            graph,
            entries,
            theta,
            coordinate_space=resolved_request.coordinate_space,
            np=np,
        )
        return price_quanto_option_raw(spec, traced_inputs)

    metadata_extra: dict[str, object] = {}
    if resolved_request.derivative_method == "hvp":
        if not capabilities.supports("hessian_vector_product"):
            diagnostic_extra = {
                "backend_id": capabilities.backend_id,
                "unsupported_operator": "hessian_vector_product",
                "backend_notes": capabilities.notes,
            }
            return _unsupported_result(
                value=value,
                graph=graph,
                request=resolved_request,
                code="hybrid_hvp_backend_unsupported",
                message=(
                    "Hybrid HVP is fail-closed because the active backend "
                    "does not provide checked scalar-objective HVP support."
                ),
                method_id=method_id,
                diagnostic_extra=diagnostic_extra,
                fallback_reason={
                    "code": "hybrid_hvp_backend_unsupported",
                    **diagnostic_extra,
                },
                backend_operator="hessian_vector_product",
                method_metadata_extra=admission_metadata,
            )
        direction_values, direction_diagnostic = _hvp_direction_for_entries(
            resolved_request,
            entries,
        )
        if direction_diagnostic is not None:
            return _unsupported_result(
                value=value,
                graph=graph,
                request=resolved_request,
                code=str(direction_diagnostic["code"]),
                message=str(direction_diagnostic["message"]),
                method_id=method_id,
                diagnostic_extra={
                    key: val
                    for key, val in direction_diagnostic.items()
                    if key not in {"code", "message"}
                },
                fallback_reason=direction_diagnostic,
                backend_operator="hessian_vector_product",
                method_metadata_extra=admission_metadata,
            )
        direction_vector = np.asarray(direction_values)
        sensitivities = np.reshape(
            np.asarray(
                hessian_vector_product(
                    value_from_vector,
                    base_vector,
                    direction_vector,
                )
            ),
            (-1,),
        )
        metadata_extra = {
            "hvp_direction_factor_count": len(resolved_request.hvp_direction),
            "hvp_direction_coordinate_count": sum(
                1 for v in direction_values if v != 0.0
            ),
            "hvp_direction_norm": float(np.sqrt(np.sum(direction_vector * direction_vector))),
        }
        backend_operator = "hessian_vector_product"
    else:
        _value, pullback = vjp(value_from_vector, base_vector)
        sensitivities = np.reshape(np.asarray(pullback(np.asarray(1.0))), (-1,))
        backend_operator = "vjp"

    full_vector = SparseRiskVector.from_items(
        (entry.factor_id, float(sensitivities[index]))
        for index, entry in enumerate(entries)
    )
    selected_vector = resolved_request.filter_vector(full_vector)
    available_factors = tuple(entry.factor_id for entry in entries)
    missing_factors = resolved_request.missing_selected_factors(available_factors)
    diagnostics: list[dict[str, object]] = list(extraction.diagnostics)
    support_status = (
        "partial"
        if graph.unsupported_dependencies or extraction.diagnostics
        else "supported"
    )
    if graph.unsupported_dependencies:
        diagnostics.append(
            {
                "code": "unsupported_graph_dependencies",
                "severity": "warning",
                "unsupported_dependency_ids": [
                    dependency.dependency_id for dependency in graph.unsupported_dependencies
                ],
                "unsupported_dependency_reasons": list(graph.unsupported_reasons),
            }
        )
    if missing_factors:
        diagnostics.append(
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
                "unsupported_selected_factor_policy": (
                    resolved_request.unsupported_selected_factor_policy
                ),
            }
        )
        support_status = "unsupported" if len(selected_vector) == 0 else "partial"
        if resolved_request.unsupported_selected_factor_policy == "fail_closed":
            selected_vector = SparseRiskVector()
            support_status = "unsupported"

    metadata = derivative_method_payload(
        method_id,
        method_support=support_status,
        backend_id=capabilities.backend_id,
        backend_operator=backend_operator,
        coordinate_space=resolved_request.coordinate_space,
        hybrid_factor_graph_id=graph.graph_id,
        graph_scalar_coordinate_count=len(entries),
        factor_count=len(full_vector),
        node_count=len(graph.nodes),
        unsupported_scalar_node_count=len(extraction.diagnostics),
        unsupported_dependency_count=len(graph.unsupported_dependencies),
        **metadata_extra,
    )
    if math.isfinite(value):
        metadata["base_value"] = value
    metadata.update(admission_metadata)

    return HybridDerivativeResult(
        value=value,
        risk_vector=selected_vector,
        graph=graph,
        support_status=support_status,
        method_metadata=metadata,
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=tuple(diagnostics),
    )


def differentiate_quanto_scalar_correlation(
    spec: object,
    resolved_inputs: object,
    request: HybridDerivativeRequest | None = None,
) -> HybridDerivativeResult:
    """Return scalar quanto sensitivity to the graph-owned correlation coordinate."""
    from trellis.core.differentiable import get_backend_capabilities, get_numpy, vjp
    from trellis.models.analytical.quanto import price_quanto_option_raw

    resolved_request = request or HybridDerivativeRequest()
    graph = getattr(resolved_inputs, "hybrid_factor_graph", None)
    if graph is None:
        graph = _fallback_graph(spec, resolved_inputs)
    if not isinstance(graph, HybridFactorGraph):
        raise TypeError("resolved_inputs.hybrid_factor_graph must be a HybridFactorGraph")
    graph.validate()
    value = float(price_quanto_option_raw(spec, resolved_inputs))

    capabilities = get_backend_capabilities()
    if resolved_request.derivative_method == "jvp":
        return _unsupported_jvp_result(
            value=value,
            graph=graph,
            request=resolved_request,
            message=(
                "Hybrid JVP remains fail-closed because the active backend "
                "does not provide checked JVP coverage for pricing primitives."
            ),
        )
    if resolved_request.derivative_method != "vjp":
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="hybrid_derivative_method_unsupported",
            message="Only VJP is supported by the bounded scalar quanto derivative lane.",
        )
    if not is_dataclass(resolved_inputs):
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="resolved_inputs_dataclass_required",
            message="Scalar quanto hybrid VJP requires dataclass resolved inputs.",
        )

    factor = _correlation_factor(graph)
    chart = _correlation_chart(graph)
    if factor is None or chart is None:
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="correlation_coordinate_unavailable",
            message="Hybrid graph does not contain one scalar correlation coordinate.",
        )

    base_corr = float(getattr(resolved_inputs, "corr"))
    if not -0.999 < base_corr < 0.999:
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="correlation_bounds_required",
            message="Scalar quanto hybrid VJP requires correlation strictly inside bounds.",
        )

    np = get_numpy()
    if resolved_request.coordinate_space == "constrained":

        def value_from_corr(corr):
            traced_inputs = replace(resolved_inputs, corr=corr)
            return price_quanto_option_raw(spec, traced_inputs)

        _value, pullback = vjp(value_from_corr, base_corr)
        sensitivity = float(np.reshape(np.asarray(pullback(np.asarray(1.0))), (-1,))[0])
        chart_derivative = chart.derivative_constrained_wrt_unconstrained
    else:
        x_value = chart.unconstrained_from_constrained(base_corr)

        def value_from_x(x_coord):
            traced_inputs = replace(resolved_inputs, corr=np.tanh(x_coord))
            return price_quanto_option_raw(spec, traced_inputs)

        _value, pullback = vjp(value_from_x, x_value)
        sensitivity = float(np.reshape(np.asarray(pullback(np.asarray(1.0))), (-1,))[0])
        chart_derivative = chart.derivative_constrained_wrt_unconstrained

    full_vector = SparseRiskVector.from_items(((factor, sensitivity),))
    selected_vector = resolved_request.filter_vector(full_vector)
    missing_factors = resolved_request.missing_selected_factors(full_vector)
    diagnostics: list[dict[str, object]] = []
    support_status = "supported"
    if missing_factors:
        diagnostics.append(
            {
                "code": "selected_factors_unavailable",
                "severity": "warning",
                "missing_factor_keys": [factor.key for factor in missing_factors],
            }
        )
        support_status = "unsupported" if len(selected_vector) == 0 else "partial"
        if resolved_request.unsupported_selected_factor_policy == "fail_closed":
            selected_vector = SparseRiskVector()
            support_status = "unsupported"

    metadata = derivative_method_payload(
        "hybrid_scalar_vjp",
        method_support=support_status,
        backend_id=capabilities.backend_id,
        backend_operator="vjp",
        coordinate_space=resolved_request.coordinate_space,
        chart_type=chart.chart_type,
        chart_derivative=float(chart_derivative),
        hybrid_factor_graph_id=graph.graph_id,
        correlation_factor_key=factor.key,
        unsupported_dependency_count=len(graph.unsupported_dependencies),
    )
    if math.isfinite(value):
        metadata["base_value"] = value

    return HybridDerivativeResult(
        value=value,
        risk_vector=selected_vector,
        graph=graph,
        support_status=support_status,
        method_metadata=metadata,
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "HybridCorrelationStructureRequest",
    "HybridDerivativeRequest",
    "HybridDerivativeResult",
    "HybridMatrixCoordinateContext",
    "build_correlation_matrix_coordinate_context",
    "differentiate_arithmetic_asian_path_summary",
    "differentiate_vanilla_early_exercise",
    "differentiate_quanto_correlation_matrix",
    "differentiate_quanto_scalar_inputs",
    "differentiate_quanto_scalar_correlation",
    "fail_closed_correlation_structure_derivative",
    "fail_closed_dynamic_state_derivative",
]
