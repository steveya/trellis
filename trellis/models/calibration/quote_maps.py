"""Bounded quote-map surface for calibration-target transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Callable, Mapping

_SUPPORTED_QUOTE_FAMILIES = (
    "price",
    "implied_vol",
    "par_rate",
    "spread",
    "upfront",
    "hazard",
)

_SUPPORTED_IMPLIED_VOL_CONVENTIONS = (
    "black",
    "normal",
)


def _normalize_quote_family_and_convention(
    quote_family: str | None,
    convention: str | None = None,
) -> tuple[str, str]:
    """Return one validated quote-family / convention pair."""
    family = _normalize_token(quote_family)
    normalized_convention = _normalize_token(convention)
    if family not in _SUPPORTED_QUOTE_FAMILIES:
        raise ValueError(
            f"unsupported quote_family `{family}`; expected one of {sorted(_SUPPORTED_QUOTE_FAMILIES)}"
        )
    if family == "implied_vol" and normalized_convention not in _SUPPORTED_IMPLIED_VOL_CONVENTIONS:
        raise ValueError(
            "implied_vol quote maps require convention `black` or `normal`"
        )
    if family != "implied_vol":
        normalized_convention = ""
    return family, normalized_convention


def _normalize_token(value: str | None, *, fallback: str = "") -> str:
    """Return a stable lowercase token string."""
    token = str(value or "").strip().lower().replace(" ", "_")
    return token or fallback


@dataclass(frozen=True)
class QuoteAxisSpec:
    """Typed semantics for one quote axis."""

    axis_name: str
    axis_kind: str = ""
    value_type: str = ""
    unit: str = ""
    role: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axis_name = _normalize_token(self.axis_name)
        if not axis_name:
            raise ValueError("quote axes require a non-empty axis_name")
        object.__setattr__(self, "axis_name", axis_name)
        object.__setattr__(self, "axis_kind", _normalize_token(self.axis_kind))
        object.__setattr__(self, "value_type", _normalize_token(self.value_type))
        object.__setattr__(self, "unit", _normalize_token(self.unit))
        object.__setattr__(self, "role", _normalize_token(self.role))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        payload = {"axis_name": self.axis_name}
        if self.axis_kind:
            payload["axis_kind"] = self.axis_kind
        if self.value_type:
            payload["value_type"] = self.value_type
        if self.unit:
            payload["unit"] = self.unit
        if self.role:
            payload["role"] = self.role
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class QuoteUnitSpec:
    """Typed semantics for quote units and scaling."""

    unit_name: str
    value_domain: str = ""
    scaling: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_name = _normalize_token(self.unit_name)
        if not unit_name:
            raise ValueError("quote units require a non-empty unit_name")
        object.__setattr__(self, "unit_name", unit_name)
        object.__setattr__(self, "value_domain", _normalize_token(self.value_domain))
        object.__setattr__(self, "scaling", _normalize_token(self.scaling))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        payload = {"unit_name": self.unit_name}
        if self.value_domain:
            payload["value_domain"] = self.value_domain
        if self.scaling:
            payload["scaling"] = self.scaling
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class QuoteSettlementSpec:
    """Typed settlement and numeraire semantics for one quote family."""

    numeraire: str = ""
    settlement_style: str = ""
    discount_curve_role: str = ""
    forecast_curve_role: str = ""
    rate_index: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeraire", _normalize_token(self.numeraire))
        object.__setattr__(self, "settlement_style", _normalize_token(self.settlement_style))
        object.__setattr__(self, "discount_curve_role", _normalize_token(self.discount_curve_role))
        object.__setattr__(self, "forecast_curve_role", _normalize_token(self.forecast_curve_role))
        object.__setattr__(self, "rate_index", _normalize_token(self.rate_index))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        payload: dict[str, object] = {}
        if self.numeraire:
            payload["numeraire"] = self.numeraire
        if self.settlement_style:
            payload["settlement_style"] = self.settlement_style
        if self.discount_curve_role:
            payload["discount_curve_role"] = self.discount_curve_role
        if self.forecast_curve_role:
            payload["forecast_curve_role"] = self.forecast_curve_role
        if self.rate_index:
            payload["rate_index"] = self.rate_index
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class QuoteSemanticsSpec:
    """Typed quote semantics shared by calibration, comparison, and runtime binding."""

    quote_family: str
    convention: str = ""
    quote_subject: str = ""
    axes: tuple[QuoteAxisSpec, ...] = ()
    unit: QuoteUnitSpec | None = None
    settlement: QuoteSettlementSpec | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family, convention = _normalize_quote_family_and_convention(
            self.quote_family,
            self.convention,
        )
        axes = tuple(self.axes)
        object.__setattr__(self, "quote_family", family)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "quote_subject", _normalize_token(self.quote_subject))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        payload = {
            "quote_family": self.quote_family,
            "convention": self.convention,
        }
        if self.quote_subject:
            payload["quote_subject"] = self.quote_subject
        if self.axes:
            payload["quote_axes"] = [axis.to_payload() for axis in self.axes]
        if self.unit is not None:
            payload["quote_unit"] = self.unit.unit_name
            payload["quote_unit_spec"] = self.unit.to_payload()
        if self.settlement is not None:
            payload["quote_settlement"] = self.settlement.to_payload()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class QuoteMapSpec:
    """Serializable quote-map descriptor for one calibration quote convention."""

    quote_family: str
    convention: str = ""
    semantics: QuoteSemanticsSpec | None = None
    assumptions: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family, convention = _normalize_quote_family_and_convention(
            self.quote_family,
            self.convention,
        )
        semantics = self.semantics or QuoteSemanticsSpec(
            quote_family=family,
            convention=convention,
        )
        if semantics.quote_family != family or semantics.convention != convention:
            raise ValueError(
                "quote-map semantics must agree with the top-level quote_family and convention"
            )
        assumptions = tuple(str(assumption) for assumption in self.assumptions)
        metadata = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "quote_family", family)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "semantics", semantics)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "metadata", metadata)

    @property
    def quote_subject(self) -> str:
        """Return the normalized quote subject carried by the structured semantics."""
        return self.semantics.quote_subject

    @property
    def quote_axes(self) -> tuple[QuoteAxisSpec, ...]:
        """Return the normalized structured quote axes."""
        return self.semantics.axes

    @property
    def quote_unit(self) -> str:
        """Return the normalized quote-unit token when available."""
        unit = self.semantics.unit
        return unit.unit_name if unit is not None else ""

    @property
    def quote_settlement(self) -> QuoteSettlementSpec | None:
        """Return the structured quote-settlement semantics when available."""
        return self.semantics.settlement

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        payload = {
            "quote_family": self.quote_family,
            "convention": self.convention,
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }
        semantics_payload = self.semantics.to_payload()
        payload["quote_semantics"] = semantics_payload
        if self.quote_subject:
            payload["quote_subject"] = self.quote_subject
        if self.quote_axes:
            payload["quote_axes"] = [axis.to_payload() for axis in self.quote_axes]
        if self.quote_unit:
            payload["quote_unit"] = self.quote_unit
        if self.quote_settlement is not None:
            payload["quote_settlement"] = self.quote_settlement.to_payload()
        return payload


@dataclass(frozen=True)
class QuoteTransformResult:
    """One directional quote-transform result with explicit failure reporting."""

    value: float | None
    warnings: tuple[str, ...] = ()
    failure: str | None = None

    def __post_init__(self) -> None:
        value = self.value
        if value is not None:
            value = float(value)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        if self.failure is not None:
            object.__setattr__(self, "failure", str(self.failure))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "value": self.value,
            "warnings": list(self.warnings),
            "failure": self.failure,
        }


@dataclass(frozen=True)
class CalibrationQuoteMap:
    """Two-sided quote map for calibration target assembly and diagnostics."""

    spec: QuoteMapSpec
    quote_to_price_fn: Callable[[float], float]
    price_to_quote_fn: Callable[[float], float] | None = None
    source_ref: str = ""
    assumptions: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def _apply(
        self,
        fn: Callable[[float], float],
        value: float,
        *,
        direction: str,
    ) -> QuoteTransformResult:
        try:
            transformed = float(fn(float(value)))
        except Exception as exc:  # pragma: no cover - exercised by quote-map tests
            label = self.source_ref or "quote_map"
            return QuoteTransformResult(
                value=None,
                failure=f"{label} {direction} failed: {exc}",
            )
        if not isfinite(transformed):
            label = self.source_ref or "quote_map"
            return QuoteTransformResult(
                value=None,
                failure=f"{label} {direction} produced a non-finite value",
            )
        return QuoteTransformResult(value=transformed)

    def target_price(self, quote: float) -> QuoteTransformResult:
        """Transform one market quote onto target price space."""
        return self._apply(self.quote_to_price_fn, quote, direction="quote_to_price")

    def model_quote(self, price: float) -> QuoteTransformResult:
        """Transform one model price into quote space for residual reporting."""
        if self.price_to_quote_fn is None:
            return QuoteTransformResult(
                value=None,
                failure=(
                    f"{self.source_ref or 'quote_map'} price_to_quote is unavailable "
                    f"for quote_family `{self.spec.quote_family}`"
                ),
            )
        return self._apply(self.price_to_quote_fn, price, direction="price_to_quote")

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        payload = self.spec.to_payload()
        payload["assumptions"] = list(self.spec.assumptions + self.assumptions)
        payload["source_ref"] = self.source_ref
        payload["has_inverse_transform"] = self.price_to_quote_fn is not None
        payload.update(dict(self.spec.metadata))
        payload.update(dict(self.metadata))
        return payload


def build_identity_quote_map(
    spec: QuoteMapSpec,
    *,
    source_ref: str = "",
    assumptions: tuple[str, ...] = (),
    metadata: Mapping[str, object] | None = None,
) -> CalibrationQuoteMap:
    """Build an identity quote map (price-to-price or quote-space passthrough)."""
    return CalibrationQuoteMap(
        spec=spec,
        quote_to_price_fn=lambda value: float(value),
        price_to_quote_fn=lambda value: float(value),
        source_ref=source_ref,
        assumptions=assumptions,
        metadata=metadata or {},
    )


def build_implied_vol_quote_map(
    *,
    convention: str,
    quote_to_price_fn: Callable[[float], float],
    price_to_quote_fn: Callable[[float], float] | None,
    semantics: QuoteSemanticsSpec | None = None,
    source_ref: str = "",
    assumptions: tuple[str, ...] = (),
    metadata: Mapping[str, object] | None = None,
) -> CalibrationQuoteMap:
    """Build a two-sided implied-vol quote map."""
    return CalibrationQuoteMap(
        spec=QuoteMapSpec(
            quote_family="implied_vol",
            convention=convention,
            semantics=semantics,
        ),
        quote_to_price_fn=quote_to_price_fn,
        price_to_quote_fn=price_to_quote_fn,
        source_ref=source_ref,
        assumptions=assumptions,
        metadata=metadata or {},
    )


def supported_quote_map_surface() -> tuple[QuoteMapSpec, ...]:
    """Return the bounded shipped quote-map vocabulary."""
    return (
        QuoteMapSpec(quote_family="price"),
        QuoteMapSpec(quote_family="implied_vol", convention="black"),
        QuoteMapSpec(quote_family="implied_vol", convention="normal"),
        QuoteMapSpec(quote_family="par_rate"),
        QuoteMapSpec(quote_family="spread"),
        QuoteMapSpec(quote_family="upfront"),
        QuoteMapSpec(quote_family="hazard"),
    )


__all__ = [
    "CalibrationQuoteMap",
    "QuoteAxisSpec",
    "QuoteMapSpec",
    "QuoteSemanticsSpec",
    "QuoteSettlementSpec",
    "QuoteTransformResult",
    "QuoteUnitSpec",
    "build_identity_quote_map",
    "build_implied_vol_quote_map",
    "supported_quote_map_surface",
]
