"""Shared market-resolution helpers for single-name quanto routes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Protocol

from trellis.analytics.hybrid_factors import (
    HybridDependencyNode,
    HybridFactorGraph,
    HybridUnsupportedDependency,
    MarketObjectCoordinateChart,
)
from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    RiskFactorRegistry,
)
from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState

np = get_numpy()


class QuantoSpecLike(Protocol):
    """Minimal spec surface required by the shared quanto resolvers."""

    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str
    domestic_currency: str
    day_count: object
    quanto_correlation_key: str | None


@dataclass(frozen=True)
class ResolvedQuantoInputs:
    """Normalized market inputs consumed by quanto pricing routes."""

    spot: float
    fx_spot: float
    valuation_date: date
    T: float
    domestic_df: float
    foreign_df: float
    sigma_underlier: float
    sigma_fx: float
    corr: float
    provenance: dict[str, object] = field(default_factory=dict)
    hybrid_factor_graph: HybridFactorGraph | None = None

    _ALIASES = {
        "underlier_spot": "spot",
        "time_to_expiry": "T",
        "domestic_discount_factor": "domestic_df",
        "foreign_discount_factor": "foreign_df",
        "underlier_vol": "sigma_underlier",
        "fx_vol": "sigma_fx",
        "correlation": "corr",
        "quanto_correlation": "corr",
        "provenance": "provenance",
        "valuation_date": "valuation_date",
        "as_of_date": "valuation_date",
        "settlement_date": "valuation_date",
    }

    @property
    def underlier_spot(self) -> float:
        return self.spot

    @property
    def time_to_expiry(self) -> float:
        return self.T

    @property
    def domestic_discount_factor(self) -> float:
        return self.domestic_df

    @property
    def foreign_discount_factor(self) -> float:
        return self.foreign_df

    @property
    def underlier_vol(self) -> float:
        return self.sigma_underlier

    @property
    def correlation(self) -> float:
        return self.corr

    @property
    def quanto_correlation(self) -> float:
        return self.corr

    @property
    def as_of_date(self) -> date:
        return self.valuation_date

    @property
    def settlement_date(self) -> date:
        return self.valuation_date

    def __getitem__(self, key: str) -> Any:
        field_name = self._ALIASES.get(key, key)
        if not hasattr(self, field_name):
            raise KeyError(key)
        return getattr(self, field_name)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _market_input_source_kind(market_state: MarketState) -> str:
    """Return the snapshot/runtime source kind backing the market state."""
    provenance = dict(getattr(market_state, "market_provenance", None) or {})
    return str(provenance.get("source_kind") or "runtime_state")


def _market_input_source_family(market_state: MarketState) -> str:
    """Map snapshot/runtime source kinds to broad input provenance families."""
    normalized = _market_input_source_kind(market_state).strip().lower().replace("-", "_")
    if normalized in {"explicit_input", "user_supplied_snapshot"}:
        return "user_supplied"
    if normalized in {"synthetic_snapshot", "synthetic"}:
        return "synthetic"
    if normalized in {"estimated_snapshot", "estimated", "empirical"}:
        return "estimated"
    if normalized == "mixed":
        return "mixed"
    if normalized in {"provider_snapshot", "direct_quote"}:
        return "observed"
    return "observed"


def _build_quanto_input_provenance(
    market_state: MarketState,
    *,
    source_family: str,
    source_kind: str,
    source_key: str | None = None,
    source_estimator: str | None = None,
    source_seed: int | None = None,
    source_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a compact, traceable provenance payload for one resolved input."""
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    parameters = dict(source_parameters or {})
    if market_provenance.get("source") is not None:
        parameters.setdefault("market_source", market_provenance["source"])
    if market_provenance.get("source_ref") is not None:
        parameters.setdefault("market_source_ref", market_provenance["source_ref"])
    parameters.setdefault("upstream_source_family", _market_input_source_family(market_state))
    parameters.setdefault("upstream_source_kind", _market_input_source_kind(market_state))

    payload: dict[str, object] = {
        "source_family": source_family,
        "source_kind": source_kind,
        "source_parameters": parameters,
    }
    if source_key is not None:
        payload["source_key"] = source_key
    if source_estimator is not None:
        payload["source_estimator"] = source_estimator
    if source_seed is not None:
        payload["source_seed"] = int(source_seed)
    return payload


def _provenance_for(resolved: ResolvedQuantoInputs, role: str) -> dict[str, object]:
    return dict((resolved.provenance or {}).get(role) or {})


def _source_key_for(
    resolved: ResolvedQuantoInputs,
    role: str,
    fallback: str,
) -> str:
    return str(_provenance_for(resolved, role).get("source_key") or fallback)


def _identity_coordinate_node(
    *,
    node_id: str,
    node_type: str,
    object_name: str,
    coordinate: RiskFactorCoordinate,
    resolved_input: str,
    coordinate_values: dict[str, object],
    provenance: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> HybridDependencyNode:
    chart = MarketObjectCoordinateChart.identity(
        chart_id=f"chart:{node_id}",
        object_type=coordinate.factor_id.object_type,
        object_name=coordinate.factor_id.object_name,
        coordinates=(coordinate,),
        coordinate_values=coordinate_values,
        metadata=metadata or {},
    )
    return HybridDependencyNode(
        node_id=node_id,
        node_type=node_type,
        object_name=object_name,
        coordinate_chart=chart,
        derivative_method="vjp",
        support_status="supported",
        provenance=provenance,
        metadata={
            "resolved_inputs": (resolved_input,),
            **dict(metadata or {}),
        },
    )


def _held_fixed_node(
    *,
    node_id: str,
    node_type: str,
    object_name: str,
    resolved_input: str,
    reason: str,
    provenance: dict[str, object] | None = None,
) -> HybridDependencyNode:
    return HybridDependencyNode(
        node_id=node_id,
        node_type=node_type,
        object_name=object_name,
        differentiability_class="held_fixed",
        derivative_method="held_fixed",
        support_status="held_fixed",
        provenance=provenance or {},
        metadata={
            "resolved_inputs": (resolved_input,),
            "reason": reason,
        },
    )


def _unsupported_dependency(
    *,
    dependency_id: str,
    node_type: str,
    object_name: str,
    reason: str,
    metadata: dict[str, object] | None = None,
) -> HybridUnsupportedDependency:
    return HybridUnsupportedDependency(
        dependency_id=dependency_id,
        node_type=node_type,
        object_name=object_name,
        reason=reason,
        metadata=metadata or {},
    )


def _float_tuple(values: object) -> tuple[float, ...]:
    """Return a deterministic tuple of floats from a scalar array-like object."""
    return tuple(float(value) for value in values)


def _float_matrix(values: object) -> tuple[tuple[float, ...], ...]:
    """Return a deterministic tuple-of-tuples float matrix."""
    return tuple(tuple(float(value) for value in row) for row in values)


def _curve_chart_values(
    curve: object,
    *,
    time_to_expiry: float,
    discount_factor: float,
) -> dict[str, object]:
    """Build executable scalar context for a supported yield-curve chart."""
    return {
        "coordinate_family": "curve_zero_rate_nodes",
        "time_to_expiry": float(time_to_expiry),
        "discount_factor": float(discount_factor),
        "tenors": _float_tuple(getattr(curve, "tenors")),
        "rates": _float_tuple(getattr(curve, "rates")),
    }


def _vol_surface_chart_values(surface: object) -> dict[str, object]:
    """Build executable scalar context for supported volatility charts."""
    from trellis.models.vol_surface import FlatVol, GridVolSurface

    if isinstance(surface, FlatVol):
        return {
            "surface_type": "FlatVol",
            "flat_vol": float(surface.vol),
        }
    if isinstance(surface, GridVolSurface):
        return {
            "surface_type": "GridVolSurface",
            "expiries": _float_tuple(surface.expiries),
            "strikes": _float_tuple(surface.strikes),
            "vols": _float_matrix(surface.vols),
        }
    return {"surface_type": type(surface).__name__}


def _spot_coordinate(
    spec: QuantoSpecLike,
    resolved: ResolvedQuantoInputs,
) -> RiskFactorCoordinate:
    source_key = _source_key_for(resolved, "underlier_spot", spec.underlier_currency)
    object_path = (
        f"market_state.underlier_spots[{source_key}]"
        if source_key != "spot"
        else "market_state.spot"
    )
    return RiskFactorCoordinate(
        factor_id=RiskFactorId(
            object_type="spot",
            object_name=source_key,
            coordinate_type="spot",
            currency=spec.underlier_currency,
            provenance_namespace="hybrid_ad",
        ),
        object_path=object_path,
        display_name=f"{source_key} spot",
        unit="price",
        transform="identity",
        support_status="supported",
        reporting_buckets={
            "risk_class": "spot",
            "currency": spec.underlier_currency,
            "object_name": source_key,
        },
    )


def _fx_spot_coordinate(
    spec: QuantoSpecLike,
    resolved: ResolvedQuantoInputs,
) -> RiskFactorCoordinate:
    source_key = _source_key_for(resolved, "fx_spot", spec.fx_pair)
    return RiskFactorCoordinate(
        factor_id=RiskFactorId(
            object_type="fx_rate",
            object_name=source_key,
            coordinate_type="spot",
            currency=spec.domestic_currency,
            axes={
                "foreign_currency": spec.underlier_currency,
                "domestic_currency": spec.domestic_currency,
            },
            provenance_namespace="hybrid_ad",
        ),
        object_path=f"market_state.fx_rates[{source_key}].spot",
        display_name=f"{source_key} FX spot",
        unit="fx_rate",
        transform="identity",
        support_status="supported",
        reporting_buckets={
            "risk_class": "fx",
            "currency": spec.domestic_currency,
            "object_name": source_key,
            "foreign_currency": spec.underlier_currency,
        },
    )


def _curve_node(
    *,
    role: str,
    curve: object | None,
    object_name: str,
    currency: str,
    object_path: str,
    provenance: dict[str, object],
    time_to_expiry: float,
    discount_factor: float,
    unsupported_dependencies: list[HybridUnsupportedDependency],
) -> HybridDependencyNode:
    node_id = f"node:curve:{role}:{object_name}"
    if curve is None:
        reason = f"{role}_unresolved"
        unsupported_dependencies.append(
            _unsupported_dependency(
                dependency_id=node_id,
                node_type="curve",
                object_name=object_name,
                reason=reason,
                metadata={"resolved_input": role},
            )
        )
        return _held_fixed_node(
            node_id=node_id,
            node_type="curve",
            object_name=object_name,
            resolved_input=role,
            reason=reason,
            provenance=provenance,
        )

    try:
        coordinates = RiskFactorRegistry().discover_yield_curve(
            curve,
            object_name=object_name,
            currency=currency,
            object_path=object_path,
            provenance_namespace="hybrid_ad",
        )
    except Exception as exc:
        reason = f"{role}_nodes_unavailable"
        unsupported_dependencies.append(
            _unsupported_dependency(
                dependency_id=node_id,
                node_type="curve",
                object_name=object_name,
                reason=reason,
                metadata={
                    "resolved_input": role,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        )
        return _held_fixed_node(
            node_id=node_id,
            node_type="curve",
            object_name=object_name,
            resolved_input=role,
            reason=reason,
            provenance=provenance,
        )

    chart = MarketObjectCoordinateChart.identity(
        chart_id=f"chart:{node_id}",
        object_type="curve",
        object_name=object_name,
        coordinates=coordinates,
        coordinate_values=_curve_chart_values(
            curve,
            time_to_expiry=time_to_expiry,
            discount_factor=discount_factor,
        ),
        metadata={"resolved_input": role},
    )
    return HybridDependencyNode(
        node_id=node_id,
        node_type="curve",
        object_name=object_name,
        coordinate_chart=chart,
        derivative_method="vjp",
        support_status="supported",
        provenance=provenance,
        metadata={"resolved_inputs": (role,)},
    )


def _vol_surface_coordinates(
    surface: object,
    *,
    object_name: str,
    currency: str,
    object_path: str,
) -> tuple[RiskFactorCoordinate, ...]:
    from trellis.models.vol_surface import FlatVol, GridVolSurface

    registry = RiskFactorRegistry()
    if isinstance(surface, FlatVol):
        return registry.discover_flat_vol_surface(
            surface,
            object_name=object_name,
            currency=currency,
            object_path=object_path,
            provenance_namespace="hybrid_ad",
            support_status="supported",
        )
    if isinstance(surface, GridVolSurface):
        return registry.discover_grid_vol_surface(
            surface,
            object_name=object_name,
            currency=currency,
            object_path=object_path,
            provenance_namespace="hybrid_ad",
            support_status="supported",
        )
    raise TypeError(f"unsupported vol surface type {type(surface).__name__!r}")


def _vol_node(
    *,
    role: str,
    surface: object | None,
    object_name: str,
    currency: str,
    object_path: str,
    coordinate_values: dict[str, object],
    provenance: dict[str, object],
    unsupported_dependencies: list[HybridUnsupportedDependency],
) -> HybridDependencyNode:
    node_id = f"node:vol_surface:{role}:{object_name}"
    if surface is None:
        reason = f"{role}_surface_unresolved"
        unsupported_dependencies.append(
            _unsupported_dependency(
                dependency_id=node_id,
                node_type="vol_surface",
                object_name=object_name,
                reason=reason,
                metadata={"resolved_input": role},
            )
        )
        return _held_fixed_node(
            node_id=node_id,
            node_type="vol_surface",
            object_name=object_name,
            resolved_input=role,
            reason=reason,
            provenance=provenance,
        )

    try:
        coordinates = _vol_surface_coordinates(
            surface,
            object_name=object_name,
            currency=currency,
            object_path=object_path,
        )
    except Exception as exc:
        reason = f"{role}_surface_coordinates_unavailable"
        unsupported_dependencies.append(
            _unsupported_dependency(
                dependency_id=node_id,
                node_type="vol_surface",
                object_name=object_name,
                reason=reason,
                metadata={
                    "resolved_input": role,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        )
        return _held_fixed_node(
            node_id=node_id,
            node_type="vol_surface",
            object_name=object_name,
            resolved_input=role,
            reason=reason,
            provenance=provenance,
        )

    chart = MarketObjectCoordinateChart.identity(
        chart_id=f"chart:{node_id}",
        object_type="vol_surface",
        object_name=object_name,
        coordinates=coordinates,
        coordinate_values={
            "coordinate_family": "vol_surface_nodes",
            "query_expiry": float(coordinate_values["expiry"]),
            "query_strike": float(coordinate_values["strike"]),
            "resolved_vol": float(coordinate_values["vol"]),
            **coordinate_values,
            **_vol_surface_chart_values(surface),
        },
        metadata={"resolved_input": role},
    )
    return HybridDependencyNode(
        node_id=node_id,
        node_type="vol_surface",
        object_name=object_name,
        coordinate_chart=chart,
        derivative_method="vjp",
        support_status="supported",
        provenance=provenance,
        metadata={"resolved_inputs": (role,)},
    )


def _correlation_node(
    spec: QuantoSpecLike,
    resolved: ResolvedQuantoInputs,
) -> HybridDependencyNode:
    provenance = _provenance_for(resolved, "correlation")
    object_name = str(
        provenance.get("source_key")
        or spec.quanto_correlation_key
        or "quanto_correlation"
    )
    coordinate = RiskFactorRegistry().discover_scalar_correlation(
        resolved.corr,
        object_name=object_name,
        factor_a=spec.underlier_currency,
        factor_b=spec.fx_pair,
        currency=spec.underlier_currency,
        object_path=f"market_state.model_parameters[{object_name}].value",
        provenance_namespace="hybrid_ad",
        support_status="supported",
    )[0]
    chart = MarketObjectCoordinateChart.scalar_correlation(
        coordinate=coordinate,
        correlation=float(resolved.corr),
        chart_id=f"chart:node:correlation:{object_name}",
        metadata={"source_kind": provenance.get("source_kind", "")},
    )
    return HybridDependencyNode(
        node_id=f"node:correlation:{object_name}",
        node_type="correlation",
        object_name=object_name,
        coordinate_chart=chart,
        derivative_method="vjp",
        support_status="supported",
        provenance=provenance,
        metadata={"resolved_inputs": ("correlation",)},
    )


def build_quanto_hybrid_factor_graph(
    market_state: MarketState,
    spec: QuantoSpecLike,
    resolved_inputs: ResolvedQuantoInputs,
) -> HybridFactorGraph:
    """Build a typed hybrid factor graph for one resolved quanto route."""
    selected_curve_names = dict(getattr(market_state, "selected_curve_names", None) or {})
    unsupported_dependencies: list[HybridUnsupportedDependency] = []
    underlier_spot_key = _source_key_for(
        resolved_inputs,
        "underlier_spot",
        spec.underlier_currency,
    )
    nodes: list[HybridDependencyNode] = [
        _identity_coordinate_node(
            node_id=f"node:spot:{underlier_spot_key}",
            node_type="spot",
            object_name=underlier_spot_key,
            coordinate=_spot_coordinate(spec, resolved_inputs),
            resolved_input="underlier_spot",
            coordinate_values={"spot": float(resolved_inputs.spot)},
            provenance=_provenance_for(resolved_inputs, "underlier_spot"),
        ),
        _identity_coordinate_node(
            node_id=f"node:fx_spot:{spec.fx_pair}",
            node_type="fx_rate",
            object_name=spec.fx_pair,
            coordinate=_fx_spot_coordinate(spec, resolved_inputs),
            resolved_input="fx_spot",
            coordinate_values={"spot": float(resolved_inputs.fx_spot)},
            provenance=_provenance_for(resolved_inputs, "fx_spot"),
        ),
    ]

    domestic_name = str(
        selected_curve_names.get("discount_curve")
        or _source_key_for(resolved_inputs, "domestic_curve", f"{spec.domestic_currency}-DISC")
    )
    nodes.append(
        _curve_node(
            role="domestic_curve",
            curve=market_state.discount,
            object_name=domestic_name,
            currency=spec.domestic_currency,
            object_path="market_state.discount",
            provenance=_provenance_for(resolved_inputs, "domestic_curve"),
            time_to_expiry=float(resolved_inputs.T),
            discount_factor=float(resolved_inputs.domestic_df),
            unsupported_dependencies=unsupported_dependencies,
        )
    )

    try:
        foreign_curve, foreign_provenance = _resolve_quanto_foreign_curve_details(
            market_state,
            spec,
        )
    except Exception as exc:
        foreign_curve = None
        foreign_provenance = {
            "source_family": "unsupported",
            "source_kind": "missing_foreign_curve",
            "source_parameters": {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
    foreign_name = str(
        foreign_provenance.get("source_key")
        or selected_curve_names.get("forecast_curve")
        or f"{spec.underlier_currency}-DISC"
    )
    nodes.append(
        _curve_node(
            role="foreign_curve",
            curve=foreign_curve,
            object_name=foreign_name,
            currency=spec.underlier_currency,
            object_path=f"market_state.forecast_curves[{foreign_name}]",
            provenance=foreign_provenance,
            time_to_expiry=float(resolved_inputs.T),
            discount_factor=float(resolved_inputs.foreign_df),
            unsupported_dependencies=unsupported_dependencies,
        )
    )

    vol_surface_name = str(
        selected_curve_names.get("vol_surface")
        or _source_key_for(resolved_inputs, "underlier_vol", "default_vol_surface")
    )
    vol_surface = getattr(market_state, "vol_surface", None)
    nodes.append(
        _vol_node(
            role="underlier_vol",
            surface=vol_surface,
            object_name=vol_surface_name,
            currency=spec.underlier_currency,
            object_path="market_state.vol_surface",
            coordinate_values={
                "expiry": float(resolved_inputs.T),
                "strike": float(spec.strike),
                "vol": float(resolved_inputs.sigma_underlier),
            },
            provenance=_provenance_for(resolved_inputs, "underlier_vol"),
            unsupported_dependencies=unsupported_dependencies,
        )
    )
    nodes.append(
        _vol_node(
            role="fx_vol",
            surface=vol_surface,
            object_name=vol_surface_name,
            currency=spec.domestic_currency,
            object_path="market_state.vol_surface",
            coordinate_values={
                "expiry": float(resolved_inputs.T),
                "strike": float(resolved_inputs.fx_spot),
                "vol": float(resolved_inputs.sigma_fx),
            },
            provenance=_provenance_for(resolved_inputs, "fx_vol"),
            unsupported_dependencies=unsupported_dependencies,
        )
    )
    nodes.append(_correlation_node(spec, resolved_inputs))
    return HybridFactorGraph(
        graph_id=f"quanto:{spec.underlier_currency}:{spec.domestic_currency}:{spec.fx_pair}",
        nodes=tuple(nodes),
        unsupported_dependencies=tuple(unsupported_dependencies),
        metadata={
            "route_family": "bounded_quanto",
            "fx_pair": spec.fx_pair,
            "underlier_currency": spec.underlier_currency,
            "domestic_currency": spec.domestic_currency,
            "valuation_date": resolved_inputs.valuation_date.isoformat(),
            "time_to_expiry": float(resolved_inputs.T),
        },
    )


def _attach_quanto_hybrid_factor_graph(
    market_state: MarketState,
    spec: QuantoSpecLike,
    resolved_inputs: ResolvedQuantoInputs,
    *,
    include_hybrid_factor_graph: bool,
) -> ResolvedQuantoInputs:
    if not include_hybrid_factor_graph:
        return resolved_inputs
    return replace(
        resolved_inputs,
        hybrid_factor_graph=build_quanto_hybrid_factor_graph(
            market_state,
            spec,
            resolved_inputs,
        ),
    )


def resolve_quanto_underlier_spot(
    market_state: MarketState,
    spec: QuantoSpecLike,
) -> float:
    """Resolve the named underlier spot from the market state."""
    spot, _ = _resolve_quanto_underlier_spot_details(market_state, spec)
    return spot


def _resolve_quanto_underlier_spot_details(
    market_state: MarketState,
    spec: QuantoSpecLike,
) -> tuple[float, dict[str, object]]:
    """Resolve the underlier spot plus traceable binding provenance."""
    base_family = _market_input_source_family(market_state)
    if market_state.underlier_spots:
        for key in (
            spec.underlier_currency,
            spec.fx_pair,
            spec.underlier_currency.upper(),
            spec.underlier_currency.lower(),
        ):
            if key in market_state.underlier_spots:
                return market_state.underlier_spots[key], _build_quanto_input_provenance(
                    market_state,
                    source_family=base_family,
                    source_kind="underlier_spot",
                    source_key=key,
                    source_parameters={
                        "binding_kind": "named_underlier_spot",
                        "underlier_currency": spec.underlier_currency,
                    },
                )
    if market_state.spot is not None:
        return market_state.spot, _build_quanto_input_provenance(
            market_state,
            source_family="derived",
            source_kind="default_spot_bridge",
            source_key="spot",
            source_parameters={
                "binding_kind": "generic_spot_fallback",
                "underlier_currency": spec.underlier_currency,
            },
        )
    raise ValueError(
        f"Quanto pricing requires spot or underlier_spots for {spec.underlier_currency!r}"
    )


def resolve_quanto_foreign_curve(
    market_state: MarketState,
    spec: QuantoSpecLike,
):
    """Resolve the foreign discount curve used for carry and quanto adjustment."""
    curve, _ = _resolve_quanto_foreign_curve_details(market_state, spec)
    return curve


def _resolve_quanto_foreign_curve_details(
    market_state: MarketState,
    spec: QuantoSpecLike,
):
    """Resolve the foreign discount curve plus explicit binding provenance."""
    forecast_curves = market_state.forecast_curves or {}
    base_family = _market_input_source_family(market_state)
    selected_curve_names = dict(getattr(market_state, "selected_curve_names", None) or {})
    for key in (
        f"{spec.underlier_currency}-DISC",
        f"{spec.underlier_currency}_DISC",
        spec.underlier_currency,
        spec.underlier_currency.upper(),
    ):
        if key in forecast_curves:
            return forecast_curves[key], _build_quanto_input_provenance(
                market_state,
                source_family=base_family,
                source_kind="forecast_curve",
                source_key=key,
                source_parameters={
                    "binding_kind": "canonical_foreign_curve",
                    "underlier_currency": spec.underlier_currency,
                    "selected_curve_name": selected_curve_names.get("forecast_curve"),
                },
            )

    policy_payload, policy_key = _resolve_quanto_foreign_curve_policy(market_state)
    if policy_payload is not None:
        curve, provenance = _apply_quanto_foreign_curve_policy(
            market_state,
            spec,
            forecast_curves,
            policy_payload=policy_payload,
            policy_key=policy_key,
        )
        if curve is not None:
            return curve, provenance
    raise ValueError(
        "Quanto pricing requires a foreign carry/discount curve bound to the "
        f"underlier currency {spec.underlier_currency!r}. Provide "
        f"`market_state.forecast_curves[{spec.underlier_currency!r}]` or "
        f"`market_state.forecast_curves[{spec.underlier_currency + '-DISC'!r}]`, "
        "or set an explicit `quanto_foreign_curve_policy` bridge."
    )


def _resolve_quanto_foreign_curve_policy(
    market_state: MarketState,
) -> tuple[dict[str, object] | None, str | None]:
    """Return an explicit foreign-curve bridge policy, when one is configured."""
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    params = dict(getattr(market_state, "model_parameters", None) or {})
    for scope_name, mapping in (
        ("market_provenance", market_provenance),
        ("model_parameters", params),
    ):
        for key in (
            "quanto_foreign_curve_policy",
            "quanto_foreign_curve_bridge",
            "foreign_curve_policy",
        ):
            if key not in mapping:
                continue
            descriptor = mapping[key]
            if isinstance(descriptor, str):
                normalized = descriptor.strip()
                if normalized.lower().replace("-", "_").replace(" ", "_") in {
                    "selected",
                    "selected_curve",
                    "selected_forecast_curve",
                    "discount_curve",
                    "domestic_discount_curve",
                    "reuse_domestic_discount",
                    "domestic_discount_bridge",
                }:
                    return {"kind": normalized}, f"{scope_name}.{key}"
                return {"kind": "explicit_curve_name", "curve_name": normalized}, f"{scope_name}.{key}"
            if not isinstance(descriptor, dict):
                raise ValueError(
                    f"{scope_name}.{key} must be a mapping or string for quanto pricing"
                )
            return dict(descriptor), f"{scope_name}.{key}"
    return None, None


def _apply_quanto_foreign_curve_policy(
    market_state: MarketState,
    spec: QuantoSpecLike,
    forecast_curves: dict[str, object],
    *,
    policy_payload: dict[str, object],
    policy_key: str | None,
):
    """Apply an explicit bridge policy for the quanto foreign curve."""
    kind = str(
        policy_payload.get("kind")
        or policy_payload.get("policy")
        or policy_payload.get("source_kind")
        or ""
    ).strip()
    normalized = kind.lower().replace("-", "_").replace(" ", "_")
    selected_curve_names = dict(getattr(market_state, "selected_curve_names", None) or {})
    curve_name = str(
        policy_payload.get("curve_name")
        or policy_payload.get("name")
        or policy_payload.get("selected_curve_name")
        or ""
    ).strip() or None

    expected_currency = policy_payload.get("underlier_currency")
    if expected_currency is not None and str(expected_currency) != str(spec.underlier_currency):
        raise ValueError(
            f"{policy_key} targets underlier_currency={expected_currency!r}, "
            f"but quanto pricing requested {spec.underlier_currency!r}"
        )

    base_family = _market_input_source_family(market_state)
    if normalized in {"", "explicit_curve_name", "curve_name", "forecast_curve"}:
        if curve_name is None:
            raise ValueError(f"{policy_key} requires `curve_name` for quanto pricing")
        if curve_name not in forecast_curves:
            raise ValueError(f"{policy_key} references unknown forecast curve {curve_name!r}")
        return forecast_curves[curve_name], _build_quanto_input_provenance(
            market_state,
            source_family="derived",
            source_kind="explicit_forecast_curve_bridge",
            source_key=curve_name,
            source_parameters={
                "binding_kind": "explicit_curve_name",
                "policy_key": policy_key,
                "underlier_currency": spec.underlier_currency,
                "curve_source_family": base_family,
                "curve_source_kind": _market_input_source_kind(market_state),
            },
        )

    if normalized in {"selected", "selected_curve", "selected_forecast_curve"}:
        selected_name = curve_name or selected_curve_names.get("forecast_curve")
        if selected_name is None:
            raise ValueError(f"{policy_key} requires a selected forecast curve name")
        if selected_name not in forecast_curves:
            raise ValueError(
                f"{policy_key} selected forecast curve {selected_name!r} is not available"
            )
        return forecast_curves[selected_name], _build_quanto_input_provenance(
            market_state,
            source_family="derived",
            source_kind="selected_forecast_curve_bridge",
            source_key=selected_name,
            source_parameters={
                "binding_kind": "selected_forecast_curve",
                "policy_key": policy_key,
                "underlier_currency": spec.underlier_currency,
                "curve_source_family": base_family,
                "curve_source_kind": _market_input_source_kind(market_state),
            },
        )

    if normalized in {
        "discount_curve",
        "domestic_discount_curve",
        "reuse_domestic_discount",
        "domestic_discount_bridge",
    }:
        if market_state.discount is None:
            raise ValueError(f"{policy_key} cannot reuse market_state.discount because it is missing")
        return market_state.discount, _build_quanto_input_provenance(
            market_state,
            source_family="derived",
            source_kind="domestic_discount_bridge",
            source_key=selected_curve_names.get("discount_curve"),
            source_parameters={
                "binding_kind": "domestic_discount_bridge",
                "policy_key": policy_key,
                "underlier_currency": spec.underlier_currency,
                "curve_source_family": base_family,
                "curve_source_kind": _market_input_source_kind(market_state),
            },
        )

    raise ValueError(f"Unsupported quanto_foreign_curve_policy kind {kind!r}")


def resolve_quanto_correlation(
    market_state: MarketState,
    spec: QuantoSpecLike,
) -> float:
    """Resolve the underlier/FX correlation input from model parameters."""
    corr, _ = _resolve_quanto_correlation_details(market_state, spec)
    return corr


def _resolve_quanto_correlation_details(
    market_state: MarketState,
    spec: QuantoSpecLike,
) -> tuple[float, dict[str, object]]:
    """Resolve the quanto correlation value plus traceable provenance."""
    params = market_state.model_parameters or {}
    candidate_keys = [
        spec.quanto_correlation_key,
        "quanto_correlation",
        f"{spec.underlier_currency}_{spec.domestic_currency}_correlation",
        f"{spec.underlier_currency}{spec.domestic_currency}_correlation",
        "underlier_fx_correlation",
        "rho",
    ]
    for key in candidate_keys:
        if key and key in params:
            value = params[key]
            if isinstance(value, dict):
                kind = str(value.get("kind") or value.get("source_kind") or "explicit")
                corr_value = value.get("value", value.get("rho"))
                if corr_value is None:
                    raise ValueError(
                        f"Quanto correlation descriptor `{key}` requires a scalar `value`."
                    )
                source_parameters = dict(value.get("parameters") or {})
                if value.get("source_ref") is not None:
                    source_parameters.setdefault("source_ref", value["source_ref"])
                if value.get("seed") is not None:
                    source_parameters.setdefault("seed", int(value["seed"]))
                if value.get("sample_size") is not None:
                    source_parameters.setdefault("sample_size", int(value["sample_size"]))
                if value.get("estimator") is not None:
                    source_parameters.setdefault("estimator", value["estimator"])
                return float(corr_value), {
                    "source_family": _normalize_quanto_source_family(kind),
                    "source_kind": _normalize_quanto_source_kind(kind),
                    "source_key": key,
                    "source_estimator": value.get("estimator"),
                    "source_seed": value.get("seed"),
                    "source_parameters": source_parameters,
                }
            return float(value), {
                "source_family": "explicit",
                "source_kind": "explicit_scalar",
                "source_key": key,
                "source_estimator": "explicit_input",
                "source_parameters": {"value": float(value)},
            }
    source_source = params.get("correlation_source")
    if source_source is not None:
        if isinstance(source_source, str):
            source_source = {"kind": source_source}
        if not isinstance(source_source, dict):
            raise ValueError("correlation_source must be a mapping or string")
        kind = str(source_source.get("kind") or source_source.get("source_kind") or "").strip()
        corr_value = source_source.get("value", source_source.get("rho"))
        if corr_value is None:
            raise ValueError("correlation_source requires a scalar value for quanto pricing")
        source_parameters = dict(source_source.get("parameters") or {})
        if source_source.get("source_ref") is not None:
            source_parameters.setdefault("source_ref", source_source["source_ref"])
        if source_source.get("seed") is not None:
            source_parameters.setdefault("seed", int(source_source["seed"]))
        if source_source.get("sample_size") is not None:
            source_parameters.setdefault("sample_size", int(source_source["sample_size"]))
        if source_source.get("estimator") is not None:
            source_parameters.setdefault("estimator", source_source["estimator"])
        return float(corr_value), {
            "source_family": _normalize_quanto_source_family(kind),
            "source_kind": _normalize_quanto_source_kind(kind),
            "source_key": source_source.get("source_key", "correlation_source"),
            "source_estimator": source_source.get("estimator"),
            "source_seed": source_source.get("seed"),
            "source_parameters": source_parameters,
        }
    raise ValueError(
        "Quanto pricing requires an underlier/FX correlation in "
        "`market_state.model_parameters`."
    )


def resolve_quanto_inputs(
    market_state: MarketState,
    spec: QuantoSpecLike,
    *,
    include_hybrid_factor_graph: bool = False,
) -> ResolvedQuantoInputs:
    """Resolve the deterministic market inputs needed by quanto routes."""
    if market_state.discount is None:
        raise ValueError("market_state.discount is required for quanto pricing")
    if market_state.vol_surface is None:
        raise ValueError("market_state.vol_surface is required for quanto pricing")

    fx_quote = (market_state.fx_rates or {}).get(spec.fx_pair)
    if fx_quote is None:
        raise ValueError(
            f"Quanto pricing requires market_state.fx_rates[{spec.fx_pair!r}]"
        )

    spot, spot_provenance = _resolve_quanto_underlier_spot_details(market_state, spec)
    fx_spot = fx_quote.spot
    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    if T <= 0.0:
        market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
        resolved = ResolvedQuantoInputs(
            spot=spot,
            fx_spot=fx_spot,
            valuation_date=market_state.settlement,
            T=0.0,
            domestic_df=1.0,
            foreign_df=1.0,
            sigma_underlier=0.0,
            sigma_fx=0.0,
            corr=0.0,
            provenance={
                "selected_curve_names": dict(market_state.selected_curve_names or {}),
                "market_provenance": market_provenance,
                "underlier_spot": spot_provenance,
                "fx_spot": _build_quanto_input_provenance(
                    market_state,
                    source_family=_market_input_source_family(market_state),
                    source_kind="fx_rate",
                    source_key=spec.fx_pair,
                    source_parameters={
                        "binding_kind": "named_fx_rate",
                        "domestic_currency": spec.domestic_currency,
                        "underlier_currency": spec.underlier_currency,
                    },
                ),
                "correlation": {"source_family": "identity", "source_kind": "identity_default"},
            },
        )
        return _attach_quanto_hybrid_factor_graph(
            market_state,
            spec,
            resolved,
            include_hybrid_factor_graph=include_hybrid_factor_graph,
        )

    foreign_curve, foreign_curve_provenance = _resolve_quanto_foreign_curve_details(
        market_state,
        spec,
    )
    domestic_df = market_state.discount.discount(T)
    foreign_df = foreign_curve.discount(T)
    sigma_underlier = market_state.vol_surface.black_vol(T, spec.strike)
    sigma_fx = market_state.vol_surface.black_vol(T, fx_spot)
    corr, correlation_provenance = _resolve_quanto_correlation_details(market_state, spec)
    corr = np.clip(corr, -0.999, 0.999)
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    surface_source_family = _market_input_source_family(market_state)
    surface_source_kind = _market_input_source_kind(market_state)
    resolved = ResolvedQuantoInputs(
        spot=spot,
        fx_spot=fx_spot,
        valuation_date=market_state.settlement,
        T=T,
        domestic_df=domestic_df,
        foreign_df=foreign_df,
        sigma_underlier=sigma_underlier,
        sigma_fx=sigma_fx,
        corr=corr,
        provenance={
            "selected_curve_names": dict(market_state.selected_curve_names or {}),
            "market_provenance": market_provenance,
            "underlier_spot": spot_provenance,
            "fx_spot": _build_quanto_input_provenance(
                market_state,
                source_family=surface_source_family,
                source_kind="fx_rate",
                source_key=spec.fx_pair,
                source_parameters={
                    "binding_kind": "named_fx_rate",
                    "domestic_currency": spec.domestic_currency,
                    "underlier_currency": spec.underlier_currency,
                },
            ),
            "domestic_curve": _build_quanto_input_provenance(
                market_state,
                source_family=surface_source_family,
                source_kind="discount_curve",
                source_key=(market_state.selected_curve_names or {}).get("discount_curve"),
                source_parameters={
                    "binding_kind": "domestic_discount_curve",
                    "time_to_expiry": float(T),
                },
            ),
            "foreign_curve": foreign_curve_provenance,
            "underlier_vol": _build_quanto_input_provenance(
                market_state,
                source_family="derived",
                source_kind="surface_lookup",
                source_key="vol_surface",
                source_parameters={
                    "binding_kind": "underlier_vol_lookup",
                    "time_to_expiry": float(T),
                    "lookup_strike": float(spec.strike),
                    "surface_source_family": surface_source_family,
                    "surface_source_kind": surface_source_kind,
                },
            ),
            "fx_vol": _build_quanto_input_provenance(
                market_state,
                source_family="derived",
                source_kind="surface_lookup",
                source_key="vol_surface",
                source_parameters={
                    "binding_kind": "fx_vol_lookup",
                    "time_to_expiry": float(T),
                    "lookup_strike": float(fx_spot),
                    "surface_source_family": surface_source_family,
                    "surface_source_kind": surface_source_kind,
                },
            ),
            "correlation": correlation_provenance,
        },
    )
    return _attach_quanto_hybrid_factor_graph(
        market_state,
        spec,
        resolved,
        include_hybrid_factor_graph=include_hybrid_factor_graph,
    )


def _normalize_quanto_source_family(kind: str | None) -> str:
    normalized = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "",
        "explicit",
        "explicit_scalar",
        "explicit_matrix",
        "calibrated",
        "quoted",
        "bootstrapped",
    }:
        return "explicit"
    if normalized in {"estimated", "empirical", "empirical_matrix", "empirical_observations"}:
        return "empirical"
    if normalized in {"implied", "implied_matrix", "implied_scalar"}:
        return "implied"
    if normalized in {"synthetic", "synthetic_matrix", "synthetic_scalar", "sampled"}:
        return "synthetic"
    if normalized in {"identity", "identity_default"}:
        return "identity"
    return normalized


def _normalize_quanto_source_kind(kind: str | None) -> str:
    normalized = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "explicit", "calibrated", "quoted", "bootstrapped"}:
        return "explicit_scalar"
    if normalized in {"explicit_scalar", "explicit_matrix"}:
        return normalized
    if normalized in {"estimated", "empirical"}:
        return "empirical_scalar"
    if normalized in {"implied"}:
        return "implied_scalar"
    if normalized in {"synthetic", "sampled"}:
        return "synthetic_scalar"
    return normalized
