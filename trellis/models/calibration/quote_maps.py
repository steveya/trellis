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
    "hazard",
)

_SUPPORTED_IMPLIED_VOL_CONVENTIONS = (
    "black",
    "normal",
)


def _normalize_token(value: str | None, *, fallback: str = "") -> str:
    """Return a stable lowercase token string."""
    token = str(value or "").strip().lower().replace(" ", "_")
    return token or fallback


@dataclass(frozen=True)
class QuoteMapSpec:
    """Serializable quote-map descriptor for one calibration quote convention."""

    quote_family: str
    convention: str = ""
    assumptions: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = _normalize_token(self.quote_family)
        convention = _normalize_token(self.convention)
        if family not in _SUPPORTED_QUOTE_FAMILIES:
            raise ValueError(
                f"unsupported quote_family `{family}`; expected one of {sorted(_SUPPORTED_QUOTE_FAMILIES)}"
            )
        if family == "implied_vol" and convention not in _SUPPORTED_IMPLIED_VOL_CONVENTIONS:
            raise ValueError(
                "implied_vol quote maps require convention `black` or `normal`"
            )
        if family != "implied_vol":
            convention = ""
        assumptions = tuple(str(assumption) for assumption in self.assumptions)
        metadata = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "quote_family", family)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "metadata", metadata)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "quote_family": self.quote_family,
            "convention": self.convention,
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }


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
        payload = {
            "quote_family": self.spec.quote_family,
            "convention": self.spec.convention,
            "assumptions": list(self.spec.assumptions + self.assumptions),
            "source_ref": self.source_ref,
            "has_inverse_transform": self.price_to_quote_fn is not None,
        }
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
    source_ref: str = "",
    assumptions: tuple[str, ...] = (),
    metadata: Mapping[str, object] | None = None,
) -> CalibrationQuoteMap:
    """Build a two-sided implied-vol quote map."""
    return CalibrationQuoteMap(
        spec=QuoteMapSpec(quote_family="implied_vol", convention=convention),
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
        QuoteMapSpec(quote_family="hazard"),
    )


__all__ = [
    "CalibrationQuoteMap",
    "QuoteMapSpec",
    "QuoteTransformResult",
    "build_identity_quote_map",
    "build_implied_vol_quote_map",
    "supported_quote_map_surface",
]
