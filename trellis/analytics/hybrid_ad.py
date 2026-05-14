"""Bounded hybrid AD helpers over typed hybrid factor graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from types import MappingProxyType
from typing import Any

from trellis.analytics.derivative_methods import derivative_method_payload
from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.risk_factors import RiskFactorId, SparseRiskVector


_DERIVATIVE_METHODS = frozenset({"vjp", "jvp", "hvp"})
_COORDINATE_SPACES = frozenset({"constrained", "unconstrained"})
_UNSUPPORTED_SELECTED_POLICIES = frozenset({"empty_vector", "fail_closed"})
_CORRELATION_STRUCTURE_TYPES = frozenset({"correlation_matrix", "correlation_surface"})


def _clean_member(value: object, allowed: frozenset[str], field_name: str) -> str:
    normalized = str(value).strip()
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return normalized


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


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


@dataclass(frozen=True)
class HybridDerivativeRequest:
    """Request policy for bounded graph-backed hybrid derivative helpers."""

    derivative_method: str = "vjp"
    coordinate_space: str = "constrained"
    selected_factors: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    unsupported_selected_factor_policy: str = "empty_vector"

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
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @property
    def unsupported_reason(self) -> str:
        """Return the policy reason for this unsupported structure."""
        return f"{self.structure_type}_chart_not_implemented"


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
    fallback_reason: dict[str, object] | None = None,
    diagnostic_extra: Mapping[str, object] | None = None,
) -> HybridDerivativeResult:
    diagnostic = {
        "code": code,
        "severity": "warning",
        "message": message,
        **dict(diagnostic_extra or {}),
    }
    metadata = derivative_method_payload(
        "hybrid_scalar_vjp",
        method_support="unsupported",
        backend_operator=request.derivative_method,
        coordinate_space=request.coordinate_space,
        hybrid_factor_graph_id=graph.graph_id,
        fallback_reason=fallback_reason or diagnostic,
    )
    return HybridDerivativeResult(
        value=value,
        risk_vector=SparseRiskVector(),
        graph=graph,
        support_status="unsupported",
        method_metadata=metadata,
        unsupported_dependencies=graph.unsupported_dependencies,
        diagnostics=(diagnostic,),
    )


def fail_closed_correlation_structure_derivative(
    request: HybridCorrelationStructureRequest,
) -> HybridDerivativeResult:
    """Return a fail-closed result for unsupported matrix/surface correlation AD."""
    if not isinstance(request, HybridCorrelationStructureRequest):
        raise TypeError("request must be a HybridCorrelationStructureRequest")
    dependency = HybridUnsupportedDependency(
        dependency_id=f"node:{request.structure_type}:{request.object_name}",
        node_type=request.structure_type,
        object_name=request.object_name,
        reason=request.unsupported_reason,
        metadata={
            "factors": request.factors,
            "requested_derivative_method": request.requested_derivative_method,
            "coordinate_space": request.coordinate_space,
            "policy": "fail_closed_no_projection",
            **dict(request.provenance),
        },
    )
    graph = HybridFactorGraph(
        graph_id=f"hybrid:{request.structure_type}:{request.object_name}",
        unsupported_dependencies=(dependency,),
        metadata={
            "structure_type": request.structure_type,
            "object_name": request.object_name,
            "factors": request.factors,
        },
    )
    diagnostic = {
        "code": request.unsupported_reason,
        "severity": "warning",
        "message": (
            "Hybrid correlation matrix/surface derivatives require a checked "
            "coordinate chart and are fail-closed until one exists."
        ),
        "structure_type": request.structure_type,
        "object_name": request.object_name,
        "factors": request.factors,
        "psd_chart_required": request.structure_type == "correlation_matrix",
        "projection_policy": "unsupported_no_smoothing_or_projection",
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
        diagnostic_extra = {
            "backend_id": capabilities.backend_id,
            "unsupported_operator": "jvp",
            "backend_notes": capabilities.notes,
        }
        return _unsupported_result(
            value=value,
            graph=graph,
            request=resolved_request,
            code="hybrid_jvp_backend_unsupported",
            message=(
                "Hybrid JVP remains fail-closed because the active backend "
                "does not provide checked JVP coverage for pricing primitives."
            ),
            diagnostic_extra=diagnostic_extra,
            fallback_reason={
                "code": "hybrid_jvp_backend_unsupported",
                **diagnostic_extra,
            },
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
    "differentiate_quanto_scalar_correlation",
    "fail_closed_correlation_structure_derivative",
]
