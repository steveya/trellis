"""Bounded multi-product Hybrid AD fixture/result contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from trellis.analytics.hybrid_ad import HybridDerivativeResult
from trellis.analytics.hybrid_ad_admission import HybridADLaneAdmission
from trellis.analytics.risk_factors import RiskFactorId, SparseRiskVector


_DERIVATIVE_METHODS = frozenset({"vjp", "hvp", "jvp"})
_UNSUPPORTED_LANE_POLICIES = frozenset({"collect_supported", "fail_closed"})


def _clean_required(value: object, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_member(value: object, allowed: frozenset[str], field_name: str) -> str:
    cleaned = _clean_required(value, field_name)
    if cleaned not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return cleaned


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _normalize_factor(value: RiskFactorId | Mapping[str, object]) -> RiskFactorId:
    if isinstance(value, RiskFactorId):
        return value
    if isinstance(value, Mapping):
        return RiskFactorId.from_payload(value)
    raise TypeError("selected_factors must contain RiskFactorId instances or payloads")


def _normalize_factors(
    factors: Iterable[RiskFactorId | Mapping[str, object]],
) -> tuple[RiskFactorId, ...]:
    by_factor: dict[RiskFactorId, RiskFactorId] = {}
    for factor in factors:
        normalized = _normalize_factor(factor)
        by_factor[normalized] = normalized
    return tuple(
        factor for _, factor in sorted(by_factor.items(), key=lambda item: item[0].key)
    )


def _normalize_admission(
    value: HybridADLaneAdmission | Mapping[str, object] | None,
) -> HybridADLaneAdmission | None:
    if value is None:
        return None
    if isinstance(value, HybridADLaneAdmission):
        return value
    if isinstance(value, Mapping):
        return HybridADLaneAdmission.from_payload(value)
    raise TypeError("semantic_admission must be a HybridADLaneAdmission or payload")


def _normalize_derivative_result(
    value: HybridDerivativeResult | Mapping[str, object],
) -> HybridDerivativeResult:
    if isinstance(value, HybridDerivativeResult):
        return value
    if isinstance(value, Mapping):
        return HybridDerivativeResult.from_payload(value)
    raise TypeError("derivative_result must be a HybridDerivativeResult or payload")


def _normalize_request(
    value: "HybridADMultiProductRequest" | Mapping[str, object],
) -> "HybridADMultiProductRequest":
    if isinstance(value, HybridADMultiProductRequest):
        return value
    if isinstance(value, Mapping):
        return HybridADMultiProductRequest.from_payload(value)
    raise TypeError("request must be a HybridADMultiProductRequest or payload")


@dataclass(frozen=True)
class HybridADMultiProductRequest:
    """Request metadata for a bounded multi-product Hybrid AD fixture."""

    request_id: str = "hybrid_multi_product"
    derivative_method: str = "vjp"
    unsupported_lane_policy: str = "collect_supported"
    selected_factors: tuple[RiskFactorId | Mapping[str, object], ...] = field(
        default_factory=tuple
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _clean_required(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "derivative_method",
            _clean_member(self.derivative_method, _DERIVATIVE_METHODS, "derivative_method"),
        )
        object.__setattr__(
            self,
            "unsupported_lane_policy",
            _clean_member(
                self.unsupported_lane_policy,
                _UNSUPPORTED_LANE_POLICIES,
                "unsupported_lane_policy",
            ),
        )
        object.__setattr__(
            self,
            "selected_factors",
            _normalize_factors(self.selected_factors),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly request payload."""
        return {
            "request_id": self.request_id,
            "derivative_method": self.derivative_method,
            "unsupported_lane_policy": self.unsupported_lane_policy,
            "selected_factors": [
                {
                    **factor.to_payload(),
                    "factor_key": factor.key,
                }
                for factor in self.selected_factors
            ],
            "selected_factor_keys": [factor.key for factor in self.selected_factors],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "HybridADMultiProductRequest":
        """Rebuild a request from :meth:`to_payload` output."""
        return cls(
            request_id=str(payload.get("request_id", "hybrid_multi_product")),
            derivative_method=str(payload.get("derivative_method", "vjp")),
            unsupported_lane_policy=str(
                payload.get("unsupported_lane_policy", "collect_supported")
            ),
            selected_factors=tuple(payload.get("selected_factors") or ()),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class HybridADMultiProductLaneResult:
    """One lane result inside a bounded multi-product Hybrid AD fixture."""

    lane_id: str
    position_name: str
    product_family: str
    requested_derivative_method: str
    derivative_result: HybridDerivativeResult | Mapping[str, object]
    quantity: float = 1.0
    semantic_admission: HybridADLaneAdmission | Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", _clean_required(self.lane_id, "lane_id"))
        object.__setattr__(
            self,
            "position_name",
            _clean_required(self.position_name, "position_name"),
        )
        object.__setattr__(
            self,
            "product_family",
            _clean_required(self.product_family, "product_family"),
        )
        object.__setattr__(
            self,
            "requested_derivative_method",
            _clean_member(
                self.requested_derivative_method,
                _DERIVATIVE_METHODS,
                "requested_derivative_method",
            ),
        )
        quantity = float(self.quantity)
        if not math.isfinite(quantity):
            raise ValueError("quantity must be finite")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(
            self,
            "derivative_result",
            _normalize_derivative_result(self.derivative_result),
        )
        object.__setattr__(
            self,
            "semantic_admission",
            _normalize_admission(self.semantic_admission),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def support_status(self) -> str:
        """Return the wrapped lane support status."""
        return self.derivative_result.support_status

    @property
    def value_contribution(self) -> float | None:
        """Return quantity-scaled value contribution when the lane has value."""
        if self.derivative_result.value is None:
            return None
        return self.quantity * self.derivative_result.value

    @property
    def risk_vector_contribution(self) -> SparseRiskVector:
        """Return quantity-scaled sparse risk contribution for this lane."""
        return self.derivative_result.risk_vector.scale(self.quantity)

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        """Return diagnostic codes carried by the lane result."""
        return tuple(
            str(diagnostic["code"])
            for diagnostic in self.derivative_result.diagnostics
            if "code" in diagnostic
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly lane payload."""
        return {
            "lane_id": self.lane_id,
            "position_name": self.position_name,
            "product_family": self.product_family,
            "requested_derivative_method": self.requested_derivative_method,
            "quantity": self.quantity,
            "support_status": self.support_status,
            "value_contribution": self.value_contribution,
            "risk_vector_contribution": self.risk_vector_contribution.to_payload(),
            "semantic_admission": (
                self.semantic_admission.to_payload()
                if self.semantic_admission is not None
                else None
            ),
            "derivative_result": self.derivative_result.to_payload(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "HybridADMultiProductLaneResult":
        """Rebuild a lane result from :meth:`to_payload` output."""
        return cls(
            lane_id=str(payload["lane_id"]),
            position_name=str(payload["position_name"]),
            product_family=str(payload["product_family"]),
            requested_derivative_method=str(payload.get("requested_derivative_method", "vjp")),
            quantity=float(payload.get("quantity", 1.0)),
            semantic_admission=payload.get("semantic_admission"),
            derivative_result=payload["derivative_result"],
            metadata=payload.get("metadata") or {},
        )


def _normalize_lane_results(
    lane_results: Iterable[HybridADMultiProductLaneResult | Mapping[str, object]],
) -> tuple[HybridADMultiProductLaneResult, ...]:
    lanes_by_id: dict[str, HybridADMultiProductLaneResult] = {}
    for lane in lane_results:
        normalized = (
            lane
            if isinstance(lane, HybridADMultiProductLaneResult)
            else HybridADMultiProductLaneResult.from_payload(lane)
        )
        existing = lanes_by_id.get(normalized.lane_id)
        if existing is not None and existing != normalized:
            raise ValueError(f"conflicting multi-product lane {normalized.lane_id!r}")
        lanes_by_id[normalized.lane_id] = normalized
    return tuple(
        lane for _, lane in sorted(lanes_by_id.items(), key=lambda item: item[0])
    )


@dataclass(frozen=True)
class HybridADMultiProductResult:
    """Typed result envelope for a bounded multi-product Hybrid AD fixture."""

    request: HybridADMultiProductRequest | Mapping[str, object]
    lane_results: tuple[HybridADMultiProductLaneResult | Mapping[str, object], ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", _normalize_request(self.request))
        object.__setattr__(
            self,
            "lane_results",
            _normalize_lane_results(self.lane_results),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def lane_ids(self) -> tuple[str, ...]:
        """Return lane ids in deterministic order."""
        return tuple(lane.lane_id for lane in self.lane_results)

    @property
    def supported_lane_count(self) -> int:
        """Return the number of lanes with supported or partial derivative output."""
        return sum(
            1
            for lane in self.lane_results
            if lane.support_status in {"supported", "partial"}
        )

    @property
    def unsupported_lane_count(self) -> int:
        """Return the number of fail-closed unsupported lanes."""
        return sum(1 for lane in self.lane_results if lane.support_status == "unsupported")

    @property
    def support_status(self) -> str:
        """Return aggregate support status implied by the lane set and policy."""
        if not self.lane_results:
            return "unsupported"
        if (
            self.unsupported_lane_count
            and self.request.unsupported_lane_policy == "fail_closed"
        ):
            return "unsupported"
        if self.supported_lane_count == len(self.lane_results):
            if any(lane.support_status == "partial" for lane in self.lane_results):
                return "partial"
            return "supported"
        if self.supported_lane_count:
            return "partial"
        return "unsupported"

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        """Return unique diagnostic codes carried by all lane results."""
        return tuple(
            sorted(
                {
                    code
                    for lane in self.lane_results
                    for code in lane.diagnostic_codes
                }
            )
        )

    @property
    def risk_vector(self) -> SparseRiskVector:
        """Return aggregate sparse risk over supported lane contributions."""
        if (
            self.unsupported_lane_count
            and self.request.unsupported_lane_policy == "fail_closed"
        ):
            return SparseRiskVector()
        return SparseRiskVector.from_items(
            item
            for lane in self.lane_results
            if lane.support_status in {"supported", "partial"}
            for item in lane.risk_vector_contribution.items()
        )

    @property
    def risk_factor_keys(self) -> tuple[str, ...]:
        """Return aggregate risk factor keys in deterministic order."""
        return tuple(factor.key for factor in self.risk_vector)

    @property
    def value_contribution(self) -> float | None:
        """Return the sum of lane value contributions, ignoring absent values."""
        if (
            self.unsupported_lane_count
            and self.request.unsupported_lane_policy == "fail_closed"
        ):
            return None
        values = tuple(
            value
            for value in (lane.value_contribution for lane in self.lane_results)
            if value is not None
        )
        if not values:
            return None
        return float(sum(values))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly multi-product result payload."""
        return {
            "request": self.request.to_payload(),
            "support_status": self.support_status,
            "lane_results": [lane.to_payload() for lane in self.lane_results],
            "lane_ids": list(self.lane_ids),
            "supported_lane_count": self.supported_lane_count,
            "unsupported_lane_count": self.unsupported_lane_count,
            "diagnostic_codes": list(self.diagnostic_codes),
            "risk_vector": self.risk_vector.to_payload(),
            "risk_factor_keys": list(self.risk_factor_keys),
            "value_contribution": self.value_contribution,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "HybridADMultiProductResult":
        """Rebuild a multi-product result from :meth:`to_payload` output."""
        return cls(
            request=payload["request"],
            lane_results=tuple(payload.get("lane_results") or ()),
            metadata=payload.get("metadata") or {},
        )


def aggregate_hybrid_ad_lane_results(
    request: HybridADMultiProductRequest | Mapping[str, object],
    lane_results: Iterable[HybridADMultiProductLaneResult | Mapping[str, object]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> HybridADMultiProductResult:
    """Build a bounded aggregate result from lane-local Hybrid AD results."""
    return HybridADMultiProductResult(
        request=request,
        lane_results=tuple(lane_results),
        metadata=metadata or {},
    )


__all__ = [
    "HybridADMultiProductLaneResult",
    "HybridADMultiProductRequest",
    "HybridADMultiProductResult",
    "aggregate_hybrid_ad_lane_results",
]
