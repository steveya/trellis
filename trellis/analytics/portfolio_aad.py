"""Typed portfolio-AAD request and result contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    RiskFactorRegistry,
    SparseRiskVector,
)


def _sorted_unique_factors(factors: Iterable[RiskFactorId]) -> tuple[RiskFactorId, ...]:
    factor_map: dict[RiskFactorId, RiskFactorId] = {}
    for factor in factors:
        if not isinstance(factor, RiskFactorId):
            raise TypeError("selected factors must be RiskFactorId instances")
        factor_map[factor] = factor
    return tuple(factor for _, factor in sorted(factor_map.items(), key=lambda item: item[0].key))


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _copy_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(diagnostic) for diagnostic in (diagnostics or ()))


def _axis_key(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value


@dataclass(frozen=True)
class AADSupportDecision:
    """Adapter support decision for one position and request."""

    supported: bool
    reason: str
    factor_dependencies: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported", bool(self.supported))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "factor_dependencies",
            _sorted_unique_factors(self.factor_dependencies),
        )
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def support_status(self) -> str:
        """Return a compact support status string."""
        return "supported" if self.supported else "unsupported"

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly support-decision payload."""
        return {
            "supported": self.supported,
            "support_status": self.support_status,
            "reason": self.reason,
            "factor_dependencies": [
                factor.to_payload()
                for factor in self.factor_dependencies
            ],
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AADSupportDecision:
        """Build a support decision from :meth:`to_payload` output."""
        return cls(
            supported=bool(payload["supported"]),
            reason=str(payload["reason"]),
            factor_dependencies=tuple(
                RiskFactorId.from_payload(entry)
                for entry in payload.get("factor_dependencies", ())
            ),
            diagnostics=payload.get("diagnostics") or (),
        )


@runtime_checkable
class TradeAADAdapter(Protocol):
    """Structural protocol for product-family portfolio-AAD adapters."""

    def support_decision(
        self,
        position_name: str,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> AADSupportDecision:
        """Return whether this adapter supports a position under *request*."""
        ...

    def factor_dependencies(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> tuple[RiskFactorId, ...]:
        """Return the factors needed to differentiate the supplied position."""
        ...

    def value(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> float:
        """Return the scalar value represented by this adapter."""
        ...

    def vjp(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
        weight: float = 1.0,
    ) -> SparseRiskVector:
        """Return a sparse VJP risk vector for the supplied position."""
        ...


@dataclass(frozen=True)
class DefaultUnsupportedAADPolicy:
    """Default fail-closed policy for positions outside portfolio-AAD support."""

    include_value_when_priced: bool = True

    def record(
        self,
        *,
        position_name: str,
        instrument: object,
        reason: str,
        request: PortfolioAADRequest,
        priced_value_available: bool = False,
    ) -> UnsupportedAADPosition:
        """Return the typed unsupported-position record for a failed adapter match."""
        return UnsupportedAADPosition(
            position_name=position_name,
            instrument_type=type(instrument).__name__,
            reason=reason,
            requested_factors=request.selected_factors,
            included_in_value=bool(self.include_value_when_priced and priced_value_available),
            included_in_risk=False,
            fallback_method=None,
        )


@dataclass(frozen=True)
class BondCurveAADMarketContext:
    """Market context for the bounded shared-curve bond AAD lane."""

    curve: object
    settlement: date
    curve_name: str = "shared_curve"
    currency: str | None = None
    provenance_namespace: str | None = "portfolio_aad"
    object_path: str = ""

    def coordinates(self) -> tuple[RiskFactorCoordinate, ...]:
        """Return canonical curve coordinates for this context."""
        return RiskFactorRegistry().discover_yield_curve(
            self.curve,
            object_name=self.curve_name,
            currency=self.currency,
            object_path=self.object_path,
            provenance_namespace=self.provenance_namespace,
        )


@dataclass(frozen=True)
class BondCurveAADAdapter:
    """Concrete adapter for fixed-rate bond risk over one shared yield curve."""

    def support_decision(
        self,
        position_name: str,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> AADSupportDecision:
        """Return whether this adapter supports a bond under *market_context*."""
        from trellis.instruments.bond import Bond

        if not isinstance(instrument, Bond):
            return AADSupportDecision(False, "unsupported_instrument_type")
        if instrument.maturity_date is None:
            return AADSupportDecision(False, "bond_maturity_date_required")
        try:
            dependencies = self.factor_dependencies(instrument, market_context, request)
        except Exception as exc:
            return AADSupportDecision(
                False,
                "yield_curve_nodes_unavailable",
                diagnostics=(
                    {
                        "position_name": str(position_name),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        return AADSupportDecision(
            True,
            "supported_bond_curve_aad",
            factor_dependencies=dependencies,
        )

    def factor_dependencies(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> tuple[RiskFactorId, ...]:
        """Return shared-curve zero-rate factors for a supported bond."""
        context = self._context(market_context)
        return tuple(coordinate.factor_id for coordinate in context.coordinates())

    def value(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> float:
        """Return the bond value for this adapter's market context."""
        context = self._context(market_context)
        return float(instrument.price(context.curve, context.settlement))

    def vjp(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
        weight: float = 1.0,
    ) -> SparseRiskVector:
        """Return the sparse VJP vector for one bond position."""
        from trellis.core.differentiable import get_numpy, vjp

        context = self._context(market_context)
        curve = context.curve
        coordinates = context.coordinates()
        tenors = getattr(curve, "tenors", None)
        rates = getattr(curve, "rates", None)
        if tenors is None or rates is None:
            raise ValueError("bond curve AAD requires curve tenors and rates")
        np = get_numpy()
        curve_cls = type(curve)
        tenors_arr = np.asarray(tenors, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)

        def value_from_rates(rates_vec):
            traced_curve = curve_cls(tenors_arr, rates_vec)
            return instrument.price(traced_curve, context.settlement)

        _value, pullback = vjp(value_from_rates, rates_arr)
        gradient = np.asarray(pullback(float(weight)), dtype=float)
        vector = SparseRiskVector.from_items(
            (coordinate.factor_id, sensitivity)
            for coordinate, sensitivity in zip(coordinates, gradient)
        )
        return request.filter_vector(vector)

    @staticmethod
    def _context(market_context: object) -> BondCurveAADMarketContext:
        if not isinstance(market_context, BondCurveAADMarketContext):
            raise TypeError("BondCurveAADAdapter requires BondCurveAADMarketContext")
        return market_context


@dataclass(frozen=True)
class PortfolioAADRequest:
    """Request policy for factorized portfolio AAD."""

    selected_factors: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    unsupported_position_policy: str = "exclude_from_risk"
    include_unsupported_value: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_factors",
            _sorted_unique_factors(self.selected_factors),
        )
        object.__setattr__(
            self,
            "unsupported_position_policy",
            str(self.unsupported_position_policy).strip() or "exclude_from_risk",
        )
        object.__setattr__(
            self,
            "include_unsupported_value",
            bool(self.include_unsupported_value),
        )

    @property
    def selects_all_factors(self) -> bool:
        """Return whether the request leaves the full factor set selected."""
        return not self.selected_factors

    def filter_vector(self, vector: SparseRiskVector) -> SparseRiskVector:
        """Apply this request's selected-factor policy to a sparse vector."""
        if self.selects_all_factors:
            return vector
        return vector.filter(self.selected_factors)

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly request payload."""
        return {
            "selected_factors": "all"
            if self.selects_all_factors
            else [factor.to_payload() for factor in self.selected_factors],
            "unsupported_position_policy": self.unsupported_position_policy,
            "include_unsupported_value": self.include_unsupported_value,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PortfolioAADRequest:
        """Build a request from :meth:`to_payload` output."""
        selected_payload = payload.get("selected_factors", "all")
        if selected_payload == "all":
            selected_factors: tuple[RiskFactorId, ...] = ()
        else:
            selected_factors = tuple(
                RiskFactorId.from_payload(entry)
                for entry in selected_payload
            )
        return cls(
            selected_factors=selected_factors,
            unsupported_position_policy=str(
                payload.get("unsupported_position_policy", "exclude_from_risk")
            ),
            include_unsupported_value=bool(payload.get("include_unsupported_value", True)),
        )


@dataclass(frozen=True)
class UnsupportedAADPosition:
    """Typed record for a position excluded from portfolio-AAD risk."""

    position_name: str
    instrument_type: str
    reason: str
    requested_factors: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    included_in_value: bool = False
    included_in_risk: bool = False
    fallback_method: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_name", str(self.position_name))
        object.__setattr__(self, "instrument_type", str(self.instrument_type))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "requested_factors",
            _sorted_unique_factors(self.requested_factors),
        )
        object.__setattr__(self, "included_in_value", bool(self.included_in_value))
        object.__setattr__(self, "included_in_risk", bool(self.included_in_risk))
        if self.fallback_method is not None:
            object.__setattr__(self, "fallback_method", str(self.fallback_method))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly unsupported-position payload."""
        return {
            "position_name": self.position_name,
            "instrument_type": self.instrument_type,
            "reason": self.reason,
            "requested_factors": [
                factor.to_payload()
                for factor in self.requested_factors
            ],
            "included_in_value": self.included_in_value,
            "included_in_risk": self.included_in_risk,
            "fallback_method": self.fallback_method,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> UnsupportedAADPosition:
        """Build an unsupported-position record from :meth:`to_payload` output."""
        return cls(
            position_name=str(payload["position_name"]),
            instrument_type=str(payload["instrument_type"]),
            reason=str(payload["reason"]),
            requested_factors=tuple(
                RiskFactorId.from_payload(entry)
                for entry in payload.get("requested_factors", ())
            ),
            included_in_value=bool(payload.get("included_in_value", False)),
            included_in_risk=bool(payload.get("included_in_risk", False)),
            fallback_method=(
                None
                if payload.get("fallback_method") is None
                else str(payload.get("fallback_method"))
            ),
        )


@dataclass(frozen=True)
class PortfolioAADResult:
    """Factorized portfolio-AAD result with metadata and diagnostics."""

    portfolio_value: float | None = None
    risk_vector: SparseRiskVector = field(default_factory=SparseRiskVector)
    coordinates: tuple[RiskFactorCoordinate, ...] = field(default_factory=tuple)
    unsupported_positions: tuple[UnsupportedAADPosition, ...] = field(default_factory=tuple)
    method_metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self) -> None:
        if self.portfolio_value is not None:
            object.__setattr__(self, "portfolio_value", float(self.portfolio_value))
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "unsupported_positions", tuple(self.unsupported_positions))
        object.__setattr__(self, "method_metadata", _copy_metadata(self.method_metadata))
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def support_status(self) -> str:
        """Return the normalized derivative support status."""
        return str(
            self.method_metadata.get(
                "derivative_method_support",
                self.method_metadata.get("support_status", "unsupported"),
            )
        )

    def apply_request(self, request: PortfolioAADRequest) -> PortfolioAADResult:
        """Return this result filtered according to *request*."""
        filtered_vector = request.filter_vector(self.risk_vector)
        selected = set(filtered_vector)
        filtered_coordinates = tuple(
            coordinate
            for coordinate in self.coordinates
            if coordinate.factor_id in selected
        )
        return PortfolioAADResult(
            portfolio_value=self.portfolio_value,
            risk_vector=filtered_vector,
            coordinates=filtered_coordinates,
            unsupported_positions=self.unsupported_positions,
            method_metadata=self.method_metadata,
            diagnostics=self.diagnostics,
        )

    def missing_selected_factors(self, request: PortfolioAADRequest) -> tuple[RiskFactorId, ...]:
        """Return explicitly requested factors absent from this result."""
        if request.selects_all_factors:
            return ()
        available_factors = set(self.risk_vector)
        available_factors.update(coordinate.factor_id for coordinate in self.coordinates)
        return tuple(
            factor
            for factor in request.selected_factors
            if factor not in available_factors
        )

    def values_by_axis(self, axis_name: str) -> dict[str | float, float]:
        """Expose sparse values keyed by one factor-axis value for legacy views."""
        values: dict[str | float, float] = {}
        for factor_id, sensitivity in self.risk_vector.items():
            axes = dict(factor_id.axes)
            if axis_name not in axes:
                continue
            values[_axis_key(axes[axis_name])] = float(sensitivity)
        return dict(sorted(values.items(), key=lambda item: str(item[0])))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly result payload."""
        return {
            "portfolio_value": self.portfolio_value,
            "risk_vector": self.risk_vector.to_payload(),
            "coordinates": [
                coordinate.to_payload()
                for coordinate in self.coordinates
            ],
            "unsupported_positions": [
                position.to_payload()
                for position in self.unsupported_positions
            ],
            "metadata": dict(self.method_metadata),
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
            "support_status": self.support_status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PortfolioAADResult:
        """Build a result from :meth:`to_payload` output."""
        return cls(
            portfolio_value=payload.get("portfolio_value"),
            risk_vector=SparseRiskVector.from_payload(payload.get("risk_vector") or {}),
            coordinates=tuple(
                RiskFactorCoordinate.from_payload(entry)
                for entry in payload.get("coordinates", ())
            ),
            unsupported_positions=tuple(
                UnsupportedAADPosition.from_payload(entry)
                for entry in payload.get("unsupported_positions", ())
            ),
            method_metadata=payload.get("metadata") or {},
            diagnostics=payload.get("diagnostics") or (),
        )


__all__ = [
    "AADSupportDecision",
    "BondCurveAADAdapter",
    "BondCurveAADMarketContext",
    "DefaultUnsupportedAADPolicy",
    "PortfolioAADRequest",
    "PortfolioAADResult",
    "TradeAADAdapter",
    "UnsupportedAADPosition",
]
