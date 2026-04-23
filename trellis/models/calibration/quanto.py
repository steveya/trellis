"""Bounded rates + equity/FX quanto-correlation calibration workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from math import isfinite
from types import MappingProxyType

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type
from trellis.models.calibration.dependency_graph import (
    CalibrationDependencyGraph,
    CalibrationDependencyNode,
)
from trellis.models.calibration.materialization import materialize_model_parameter_set
from trellis.models.calibration.solve_request import (
    ObjectiveBundle,
    SolveBounds,
    SolveProvenance,
    SolveReplayArtifact,
    SolveRequest,
    SolveResult,
    build_solve_provenance,
    build_solve_replay_artifact,
    execute_solve_request,
)
from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state
from trellis.models.resolution.quanto import resolve_quanto_inputs

_SUPPORT_BOUNDARY = "bounded_quanto_correlation"
_SOURCE_REF = "calibrate_quanto_correlation_workflow"
_DEFAULT_CORRELATION_BOUNDS = (-0.999, 0.999)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow mapping copy."""
    return MappingProxyType(dict(mapping or {}))


def _finite_float(value: float, *, field_name: str) -> float:
    """Return one finite float value or raise ``ValueError``."""
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _positive_float(value: float, *, field_name: str) -> float:
    """Return one finite positive float value or raise ``ValueError``."""
    normalized = _finite_float(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return normalized


def _non_negative_float(value: float, *, field_name: str) -> float:
    """Return one finite non-negative float value or raise ``ValueError``."""
    normalized = _finite_float(value, field_name=field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


def _normalized_token(value: object, *, field_name: str) -> str:
    """Return a stripped string token or raise ``ValueError``."""
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{field_name} must be non-empty")
    return token


def _validate_correlation_bounds(bounds: Sequence[float]) -> tuple[float, float]:
    """Return valid finite correlation bounds."""
    if len(bounds) != 2:
        raise ValueError("correlation_bounds must contain lower and upper values")
    lower = _finite_float(bounds[0], field_name="correlation_bounds.lower")
    upper = _finite_float(bounds[1], field_name="correlation_bounds.upper")
    if lower < -0.999 or upper > 0.999 or lower >= upper:
        raise ValueError("correlation_bounds must satisfy -0.999 <= lower < upper <= 0.999")
    return lower, upper


def _clamp_correlation(value: float, *, bounds: tuple[float, float]) -> float:
    """Return ``value`` clipped inside the configured correlation bounds."""
    lower, upper = bounds
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class QuantoCorrelationCalibrationQuote:
    """One quanto option price quote used to infer underlier/FX correlation."""

    market_price: float
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    label: str = ""
    weight: float = 1.0
    quanto_correlation_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "market_price",
            _non_negative_float(self.market_price, field_name="market_price"),
        )
        object.__setattr__(
            self,
            "notional",
            _positive_float(self.notional, field_name="notional"),
        )
        object.__setattr__(
            self,
            "strike",
            _positive_float(self.strike, field_name="strike"),
        )
        object.__setattr__(self, "fx_pair", _normalized_token(self.fx_pair, field_name="fx_pair"))
        object.__setattr__(
            self,
            "underlier_currency",
            _normalized_token(self.underlier_currency, field_name="underlier_currency"),
        )
        object.__setattr__(
            self,
            "domestic_currency",
            _normalized_token(self.domestic_currency, field_name="domestic_currency"),
        )
        object.__setattr__(self, "option_type", normalized_option_type(self.option_type))
        object.__setattr__(self, "label", str(self.label).strip())
        object.__setattr__(self, "weight", _positive_float(self.weight, field_name="weight"))
        if self.quanto_correlation_key is not None:
            key = _normalized_token(
                self.quanto_correlation_key,
                field_name="quanto_correlation_key",
            )
            object.__setattr__(self, "quanto_correlation_key", key)

    def resolved_label(self, index: int) -> str:
        """Return a stable quote label."""
        if self.label:
            return self.label
        expiry = self.expiry_date.isoformat().replace("-", "")
        strike = str(float(self.strike)).replace(".", "_")
        return f"quanto_{self.option_type}_{expiry}_{strike}_{index}"

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly quote payload."""
        return {
            "market_price": float(self.market_price),
            "notional": float(self.notional),
            "strike": float(self.strike),
            "expiry_date": self.expiry_date.isoformat(),
            "fx_pair": self.fx_pair,
            "underlier_currency": self.underlier_currency,
            "domestic_currency": self.domestic_currency,
            "option_type": self.option_type,
            "day_count": str(self.day_count),
            "label": self.label,
            "weight": float(self.weight),
            "quanto_correlation_key": self.quanto_correlation_key,
        }


@dataclass(frozen=True)
class QuantoCorrelationQuoteResidual:
    """One model-vs-market quote residual after correlation calibration."""

    label: str
    market_price: float
    model_price: float
    price_residual: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "market_price", float(self.market_price))
        object.__setattr__(self, "model_price", float(self.model_price))
        object.__setattr__(self, "price_residual", float(self.price_residual))
        object.__setattr__(self, "weight", float(self.weight))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly residual payload."""
        return {
            "label": self.label,
            "market_price": float(self.market_price),
            "model_price": float(self.model_price),
            "price_residual": float(self.price_residual),
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class QuantoCorrelationCalibrationDiagnostics:
    """Diagnostics for the bounded quanto-correlation calibration workflow."""

    quote_residuals: tuple[QuantoCorrelationQuoteResidual, ...]
    max_abs_price_residual: float
    l2_norm: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        residuals = tuple(self.quote_residuals)
        object.__setattr__(self, "quote_residuals", residuals)
        object.__setattr__(
            self,
            "max_abs_price_residual",
            float(self.max_abs_price_residual),
        )
        object.__setattr__(self, "l2_norm", float(self.l2_norm))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly diagnostics payload."""
        return {
            "quote_residuals": [residual.to_payload() for residual in self.quote_residuals],
            "max_abs_price_residual": float(self.max_abs_price_residual),
            "l2_norm": float(self.l2_norm),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class QuantoCorrelationCalibrationResult:
    """Structured result for bounded rates + equity/FX quanto-correlation calibration."""

    quotes: tuple[QuantoCorrelationCalibrationQuote, ...]
    correlation: float
    parameter_set_name: str
    diagnostics: QuantoCorrelationCalibrationDiagnostics
    solve_result: SolveResult
    solve_provenance: SolveProvenance
    solve_replay_artifact: SolveReplayArtifact
    dependency_graph: CalibrationDependencyGraph
    provenance: Mapping[str, object] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quotes", tuple(self.quotes))
        object.__setattr__(self, "correlation", float(self.correlation))
        object.__setattr__(
            self,
            "parameter_set_name",
            _normalized_token(self.parameter_set_name, field_name="parameter_set_name"),
        )
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with the calibrated quanto correlation."""
        descriptor = {
            "kind": "calibrated",
            "value": float(self.correlation),
            "source_ref": _SOURCE_REF,
            "parameters": {
                "parameter_set_name": self.parameter_set_name,
                "support_boundary": _SUPPORT_BOUNDARY,
                "quote_count": len(self.quotes),
                "max_abs_price_residual": float(self.diagnostics.max_abs_price_residual),
            },
        }
        parameter_payload = dict(market_state.model_parameters or {})
        parameter_payload["quanto_correlation"] = descriptor
        selected_names = dict(market_state.selected_curve_names or {})
        selected_curve_roles = {
            "discount_curve": str(selected_names.get("discount_curve") or ""),
            "forecast_curve": str(selected_names.get("forecast_curve") or ""),
            "model_parameter_set": self.parameter_set_name,
        }
        return materialize_model_parameter_set(
            market_state,
            parameter_set_name=self.parameter_set_name,
            model_parameters=parameter_payload,
            source_kind="calibrated_model_parameter_set",
            source_ref=_SOURCE_REF,
            selected_curve_roles=selected_curve_roles,
            metadata={
                "instrument_family": "hybrid_quanto",
                "instrument_kind": "quanto_correlation",
                "support_boundary": _SUPPORT_BOUNDARY,
                "quote_count": len(self.quotes),
                "quote_labels": [quote.resolved_label(index) for index, quote in enumerate(self.quotes)],
                "max_abs_price_residual": float(self.diagnostics.max_abs_price_residual),
                "diagnostics": self.diagnostics.to_payload(),
                "dependency_graph": _dependency_graph_payload(self.dependency_graph),
                "solve_provenance": self.solve_provenance.to_payload(),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly calibration payload."""
        return {
            "quotes": [quote.to_payload() for quote in self.quotes],
            "correlation": float(self.correlation),
            "parameter_set_name": self.parameter_set_name,
            "diagnostics": self.diagnostics.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solve_provenance": self.solve_provenance.to_payload(),
            "solve_replay_artifact": self.solve_replay_artifact.to_payload(),
            "dependency_graph": _dependency_graph_payload(self.dependency_graph),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
            "assumptions": list(self.assumptions),
        }


def calibrate_quanto_correlation_workflow(
    quotes: Sequence[QuantoCorrelationCalibrationQuote],
    market_state: MarketState,
    *,
    parameter_set_name: str = "quanto_correlation",
    initial_correlation: float = 0.0,
    correlation_bounds: Sequence[float] = _DEFAULT_CORRELATION_BOUNDS,
    solver_hint: str = "trf",
    options: Mapping[str, object] | None = None,
) -> QuantoCorrelationCalibrationResult:
    """Calibrate one bounded underlier/FX correlation for quanto pricing.

    This is the first hybrid calibration slice: it composes existing market
    inputs and materialized single-asset records, but only solves the scalar
    quanto-correlation bridge.
    """
    normalized_quotes = tuple(quotes)
    if not normalized_quotes:
        raise ValueError("quanto correlation calibration requires at least one quote")
    parameter_set_name = _normalized_token(parameter_set_name, field_name="parameter_set_name")
    bounds = _validate_correlation_bounds(correlation_bounds)
    initial = _clamp_correlation(
        _finite_float(initial_correlation, field_name="initial_correlation"),
        bounds=bounds,
    )
    trial_state = _state_with_quanto_correlation(market_state, initial)
    _preflight_required_inputs(normalized_quotes, trial_state)

    labels = tuple(quote.resolved_label(index) for index, quote in enumerate(normalized_quotes))
    market_prices = tuple(float(quote.market_price) for quote in normalized_quotes)
    weights = tuple(float(quote.weight) for quote in normalized_quotes)
    solve_options = {
        "ftol": 1.0e-12,
        "xtol": 1.0e-12,
        "gtol": 1.0e-12,
        "maxiter": 1000,
    }
    solve_options.update(dict(options or {}))

    def model_prices(parameters: object) -> tuple[float, ...]:
        rho = _parameter_value(parameters)
        priced_state = _state_with_quanto_correlation(market_state, rho)
        return tuple(_price_quote(priced_state, quote) for quote in normalized_quotes)

    objective = ObjectiveBundle(
        objective_kind="least_squares",
        labels=labels,
        target_values=market_prices,
        weights=weights,
        vector_objective_fn=model_prices,
    )
    request = SolveRequest(
        request_id=f"{parameter_set_name}_quanto_correlation",
        problem_kind="least_squares",
        parameter_names=("quanto_correlation",),
        initial_guess=(initial,),
        objective=objective,
        bounds=SolveBounds(lower=(bounds[0],), upper=(bounds[1],)),
        solver_hint=solver_hint,
        metadata={
            "source_ref": _SOURCE_REF,
            "support_boundary": _SUPPORT_BOUNDARY,
            "quote_count": len(normalized_quotes),
        },
        options=solve_options,
    )
    solve_result = execute_solve_request(request)
    if not solve_result.success:
        message = str(solve_result.metadata.get("message") or "solver reported failure")
        raise ValueError(f"Quanto correlation calibration failed: {message}")
    correlation = float(solve_result.solution[0])
    solved_state = _state_with_quanto_correlation(market_state, correlation)
    residuals = tuple(
        QuantoCorrelationQuoteResidual(
            label=label,
            market_price=quote.market_price,
            model_price=model_price,
            price_residual=model_price - quote.market_price,
            weight=quote.weight,
        )
        for label, quote, model_price in zip(labels, normalized_quotes, model_prices((correlation,)))
    )
    max_abs_residual = max((abs(residual.price_residual) for residual in residuals), default=0.0)
    l2_norm = sum(residual.price_residual ** 2 for residual in residuals) ** 0.5
    diagnostics = QuantoCorrelationCalibrationDiagnostics(
        quote_residuals=residuals,
        max_abs_price_residual=float(max_abs_residual),
        l2_norm=float(l2_norm),
        warnings=(),
    )
    solve_provenance = build_solve_provenance(request, solve_result)
    solve_replay_artifact = build_solve_replay_artifact(request, solve_result)
    dependency_graph = _build_dependency_graph(market_state, parameter_set_name=parameter_set_name)
    dependency_graph_payload = _dependency_graph_payload(dependency_graph)
    upstream_materializations = _upstream_materializations(market_state)
    provenance = {
        "source_kind": "calibrated_model_parameter_set",
        "source_ref": _SOURCE_REF,
        "support_boundary": _SUPPORT_BOUNDARY,
        "dependency_graph": dependency_graph_payload,
        "dependency_order": _dependency_order(dependency_graph),
        "upstream_materializations": upstream_materializations,
        "calibration_target": {
            "instrument_family": "quanto_option",
            "parameter_name": "quanto_correlation",
            "quote_values": [quote.to_payload() for quote in normalized_quotes],
        },
        "solver": {
            "request": request.to_payload(),
            "result": solve_result.to_payload(),
            "provenance": solve_provenance.to_payload(),
            "replay_artifact": solve_replay_artifact.to_payload(),
        },
        "diagnostics": diagnostics.to_payload(),
    }
    summary = {
        "parameter_set_name": parameter_set_name,
        "support_boundary": _SUPPORT_BOUNDARY,
        "quote_count": len(normalized_quotes),
        "correlation": float(correlation),
        "correlation_bounds": [float(bounds[0]), float(bounds[1])],
        "max_abs_price_residual": float(max_abs_residual),
        "upstream_materialization_kinds": sorted(upstream_materializations),
    }
    assumptions = (
        "single-underlier quanto option prices share one scalar underlier/FX correlation",
        "domestic and foreign curves, spots, and volatility inputs are already bound on MarketState",
    )
    # Resolve once after solving so provenance catches late pricing failures before materialization.
    _preflight_required_inputs(normalized_quotes, solved_state)
    return QuantoCorrelationCalibrationResult(
        quotes=normalized_quotes,
        correlation=correlation,
        parameter_set_name=parameter_set_name,
        diagnostics=diagnostics,
        solve_result=solve_result,
        solve_provenance=solve_provenance,
        solve_replay_artifact=solve_replay_artifact,
        dependency_graph=dependency_graph,
        provenance=provenance,
        summary=summary,
        assumptions=assumptions,
    )


def _parameter_value(parameters: object) -> float:
    """Return the scalar correlation from a solver parameter container."""
    if isinstance(parameters, (int, float)):
        return float(parameters)
    try:
        return float(parameters[0])  # type: ignore[index]
    except (TypeError, IndexError) as exc:
        raise ValueError("quanto correlation objective requires one scalar parameter") from exc


def _state_with_quanto_correlation(market_state: MarketState, correlation: float) -> MarketState:
    """Return ``market_state`` with a trial quanto correlation injected."""
    params = dict(market_state.model_parameters or {})
    params["quanto_correlation"] = float(correlation)
    return replace(market_state, model_parameters=params)


def _price_quote(
    market_state: MarketState,
    quote: QuantoCorrelationCalibrationQuote,
) -> float:
    """Price one quote through the existing quanto analytical runtime helper."""
    return float(price_quanto_option_analytical_from_market_state(market_state, quote))


def _preflight_required_inputs(
    quotes: Sequence[QuantoCorrelationCalibrationQuote],
    market_state: MarketState,
) -> None:
    """Validate all non-correlation hybrid inputs through the quanto resolver."""
    for index, quote in enumerate(quotes):
        try:
            resolve_quanto_inputs(market_state, quote)
        except ValueError as exc:
            label = quote.resolved_label(index)
            raise ValueError(
                f"Quanto correlation calibration requires valid hybrid inputs for "
                f"quote {label!r}: {exc}"
            ) from exc


def _build_dependency_graph(
    market_state: MarketState,
    *,
    parameter_set_name: str,
) -> CalibrationDependencyGraph:
    """Build the bounded hybrid dependency graph for this workflow."""
    selected_names = dict(market_state.selected_curve_names or {})
    nodes = (
        CalibrationDependencyNode(
            node_id="domestic_discount_curve",
            object_kind="discount_curve",
            object_name=str(selected_names.get("discount_curve") or "market_state.discount"),
            source_ref="market_state.discount",
            required=True,
            description="Domestic discount curve consumed by quanto pricing.",
        ),
        CalibrationDependencyNode(
            node_id="foreign_curve",
            object_kind="forecast_curve",
            object_name=str(selected_names.get("forecast_curve") or "foreign_currency_curve"),
            source_ref="market_state.forecast_curves",
            required=True,
            description="Foreign carry curve consumed by quanto pricing.",
        ),
        CalibrationDependencyNode(
            node_id="underlier_spot",
            object_kind="underlier_spot",
            object_name="underlier_spot",
            source_ref="market_state.spot",
            required=True,
            description="Underlying spot consumed by quanto pricing.",
        ),
        CalibrationDependencyNode(
            node_id="fx_spot",
            object_kind="fx_rate",
            object_name="fx_pair",
            source_ref="market_state.fx_rates",
            required=True,
            description="FX spot bridge consumed by quanto pricing.",
        ),
        CalibrationDependencyNode(
            node_id="black_vol_surface",
            object_kind="black_vol_surface",
            object_name="vol_surface",
            source_ref="market_state.vol_surface",
            required=True,
            description="Volatility input consumed for underlier and FX lookup.",
        ),
        CalibrationDependencyNode(
            node_id="market_quotes",
            object_kind="market_quotes",
            object_name="quanto_option_prices",
            source_ref="quanto_quotes",
            required=True,
            description="Quanto option prices used as calibration targets.",
        ),
        CalibrationDependencyNode(
            node_id="quanto_correlation",
            object_kind="model_parameter",
            object_name="quanto_correlation",
            source_ref=_SOURCE_REF,
            required=True,
            description="Calibrated scalar underlier/FX correlation.",
            depends_on=(
                "domestic_discount_curve",
                "foreign_curve",
                "underlier_spot",
                "fx_spot",
                "black_vol_surface",
                "market_quotes",
            ),
        ),
        CalibrationDependencyNode(
            node_id="model_parameter_set",
            object_kind="model_parameter_set",
            object_name=parameter_set_name,
            source_ref="materialize_model_parameter_set",
            required=True,
            description="Runtime materialization target for downstream quanto pricing.",
            depends_on=("quanto_correlation",),
        ),
    )
    return CalibrationDependencyGraph(
        workflow_id="quanto_correlation_calibration",
        nodes=nodes,
        edges=(),
    )


def _dependency_order(graph: CalibrationDependencyGraph) -> list[str]:
    """Return dependency order from the graph while tolerating older graph objects."""
    order_attr = getattr(graph, "dependency_order", None)
    if callable(order_attr):
        return list(order_attr())
    if order_attr is not None:
        return list(order_attr)
    topo_attr = getattr(graph, "topological_order", None)
    if callable(topo_attr):
        return list(topo_attr())
    if topo_attr is not None:
        return list(topo_attr)
    return []


def _dependency_graph_payload(graph: CalibrationDependencyGraph) -> dict[str, object]:
    """Return a JSON-friendly graph payload."""
    to_payload = getattr(graph, "to_payload", None)
    if callable(to_payload):
        return dict(to_payload())
    return {
        "workflow_id": getattr(graph, "workflow_id", ""),
        "dependency_order": _dependency_order(graph),
    }


def _upstream_materializations(market_state: MarketState) -> dict[str, dict[str, object]]:
    """Return materialized upstream records already available on the market state."""
    records: dict[str, dict[str, object]] = {}
    for kind in (
        "black_vol_surface",
        "local_vol_surface",
        "credit_curve",
        "correlation_surface",
        "model_parameter_set",
    ):
        record = market_state.materialized_calibrated_object(object_kind=kind)
        if record is not None:
            records[kind] = record
    return records


__all__ = [
    "QuantoCorrelationCalibrationDiagnostics",
    "QuantoCorrelationCalibrationQuote",
    "QuantoCorrelationCalibrationResult",
    "QuantoCorrelationQuoteResidual",
    "calibrate_quanto_correlation_workflow",
]
