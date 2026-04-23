"""Bounded homogeneous basket-credit tranche-implied correlation calibration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from types import MappingProxyType
from typing import Literal

from scipy.optimize import brentq

from trellis.core.date_utils import add_months
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.calibration.materialization import materialize_correlation_surface
from trellis.models.calibration.quote_maps import (
    QuoteAxisSpec,
    QuoteMapSpec,
    QuoteSemanticsSpec,
    QuoteSettlementSpec,
    QuoteUnitSpec,
    build_identity_quote_map,
)
from trellis.models.credit_basket_copula import price_credit_basket_tranche_result

QuoteFamily = Literal["price", "spread"]
QuoteStyle = Literal["present_value", "expected_loss_fraction", "fair_spread_bp"]
RootFailurePolicy = Literal["raise", "warn"]

_SUPPORT_BOUNDARY = "homogeneous_representative_curve"
_DEFAULT_CORRELATION_BOUNDS = (1.0e-8, 0.999999)


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


def _token(value: object, *, fallback: str = "") -> str:
    """Return a stable lowercase token."""
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return token or fallback


def _normalize_quote_family_and_style(
    quote_family: object,
    quote_style: object,
) -> tuple[QuoteFamily, QuoteStyle]:
    """Normalize tranche quote aliases onto the bounded quote-map vocabulary."""
    family = _token(quote_family, fallback="price")
    style = _token(quote_style)

    if family in {"fair_spread", "tranche_spread", "running_spread", "fair_spread_bp"}:
        family = "spread"
        style = "fair_spread_bp"
    elif family in {"tranche_price", "pv", "present_value"}:
        family = "price"
        style = "present_value"

    if style in {"fair_spread", "tranche_spread", "running_spread", "spread"}:
        style = "fair_spread_bp"
        family = "spread"
    elif style in {"expected_loss", "loss_fraction", "etl"}:
        style = "expected_loss_fraction"
        family = "price"
    elif style in {"pv", "price", "tranche_price"}:
        style = "present_value"
        family = "price"

    if not style:
        style = "fair_spread_bp" if family == "spread" else "present_value"

    if family not in {"price", "spread"}:
        raise ValueError("quote_family must normalize to 'price' or 'spread'")
    if style not in {"present_value", "expected_loss_fraction", "fair_spread_bp"}:
        raise ValueError(
            "quote_style must be 'present_value', 'expected_loss_fraction', or 'fair_spread_bp'"
        )
    if family == "spread" and style != "fair_spread_bp":
        raise ValueError("spread quote_family requires quote_style 'fair_spread_bp'")
    if family == "price" and style == "fair_spread_bp":
        raise ValueError("fair_spread_bp quote_style requires quote_family 'spread'")
    return family, style


def _validate_tranche_bounds(attachment: float, detachment: float) -> tuple[float, float]:
    """Return normalized tranche bounds or raise ``ValueError``."""
    normalized_attachment = _finite_float(attachment, field_name="attachment")
    normalized_detachment = _finite_float(detachment, field_name="detachment")
    if not 0.0 <= normalized_attachment < normalized_detachment <= 1.0:
        raise ValueError("tranche attachment/detachment must satisfy 0 <= attachment < detachment <= 1")
    return normalized_attachment, normalized_detachment


def _maturity_date_from_years(settlement: date, maturity_years: float) -> date:
    """Return a deterministic maturity date for the bounded tenor fixture path."""
    months = max(int(round(float(maturity_years) * 12.0)), 1)
    return add_months(settlement, months)


@dataclass(frozen=True)
class BasketCreditTrancheQuote:
    """One normalized basket-credit tranche quote used for implied-correlation fit."""

    maturity_years: float
    attachment: float
    detachment: float
    quote_value: float
    quote_family: str = "price"
    quote_style: str = ""
    label: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        maturity_years = _positive_float(self.maturity_years, field_name="maturity_years")
        attachment, detachment = _validate_tranche_bounds(self.attachment, self.detachment)
        quote_value = _non_negative_float(self.quote_value, field_name="quote_value")
        quote_family, quote_style = _normalize_quote_family_and_style(
            self.quote_family,
            self.quote_style,
        )
        weight = _positive_float(self.weight, field_name="weight")
        object.__setattr__(self, "maturity_years", maturity_years)
        object.__setattr__(self, "attachment", attachment)
        object.__setattr__(self, "detachment", detachment)
        object.__setattr__(self, "quote_value", quote_value)
        object.__setattr__(self, "quote_family", quote_family)
        object.__setattr__(self, "quote_style", quote_style)
        object.__setattr__(self, "label", str(self.label).strip())
        object.__setattr__(self, "weight", weight)

    @property
    def tranche_width(self) -> float:
        """Return detachment minus attachment."""
        return float(self.detachment - self.attachment)

    def resolved_label(self, index: int) -> str:
        """Return a stable quote label."""
        if self.label:
            return self.label
        maturity = str(float(self.maturity_years)).replace(".", "_")
        attachment = str(float(self.attachment)).replace(".", "_")
        detachment = str(float(self.detachment)).replace(".", "_")
        return f"{self.quote_style}_{maturity}y_{attachment}_{detachment}_{index}"

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly quote payload."""
        return {
            "maturity_years": float(self.maturity_years),
            "attachment": float(self.attachment),
            "detachment": float(self.detachment),
            "quote_value": float(self.quote_value),
            "quote_family": self.quote_family,
            "quote_style": self.quote_style,
            "label": self.label,
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class BasketCreditCorrelationPoint:
    """One fitted tranche-implied correlation node."""

    maturity_years: float
    attachment: float
    detachment: float
    correlation: float
    quote_label: str
    quote_family: QuoteFamily
    quote_style: QuoteStyle
    market_quote_value: float
    model_quote_value: float
    quote_residual: float

    def __post_init__(self) -> None:
        maturity_years = _positive_float(self.maturity_years, field_name="maturity_years")
        attachment, detachment = _validate_tranche_bounds(self.attachment, self.detachment)
        correlation = _finite_float(self.correlation, field_name="correlation")
        if correlation < 0.0 or correlation >= 1.0:
            raise ValueError("correlation must satisfy 0 <= correlation < 1")
        object.__setattr__(self, "maturity_years", maturity_years)
        object.__setattr__(self, "attachment", attachment)
        object.__setattr__(self, "detachment", detachment)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "quote_label", str(self.quote_label))
        object.__setattr__(self, "market_quote_value", float(self.market_quote_value))
        object.__setattr__(self, "model_quote_value", float(self.model_quote_value))
        object.__setattr__(self, "quote_residual", float(self.quote_residual))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly point payload."""
        return {
            "maturity_years": float(self.maturity_years),
            "attachment": float(self.attachment),
            "detachment": float(self.detachment),
            "correlation": float(self.correlation),
            "quote_label": self.quote_label,
            "quote_family": self.quote_family,
            "quote_style": self.quote_style,
            "market_quote_value": float(self.market_quote_value),
            "model_quote_value": float(self.model_quote_value),
            "quote_residual": float(self.quote_residual),
        }


@dataclass(frozen=True)
class BasketCreditCorrelationSurface:
    """Bounded exact-node tranche-implied correlation surface."""

    points: tuple[BasketCreditCorrelationPoint, ...]
    surface_name: str = "basket_credit_tranche_correlation"
    copula_family: str = "gaussian"
    n_names: int = 0
    recovery: float = 0.4
    notional: float = 1.0
    support_boundary: str = _SUPPORT_BOUNDARY
    representative_credit_curve: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        surface_name = str(self.surface_name).strip()
        if not surface_name:
            raise ValueError("surface_name must be non-empty")
        n_names = int(self.n_names)
        if n_names < 2:
            raise ValueError("n_names must be at least 2")
        recovery = _finite_float(self.recovery, field_name="recovery")
        if recovery <= 0.0 or recovery >= 1.0:
            raise ValueError("recovery must be strictly between 0 and 1")
        notional = _positive_float(self.notional, field_name="notional")
        points = tuple(
            sorted(
                self.points,
                key=lambda point: (
                    float(point.maturity_years),
                    float(point.attachment),
                    float(point.detachment),
                ),
            )
        )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "surface_name", surface_name)
        object.__setattr__(self, "copula_family", _normalize_copula_family(self.copula_family))
        object.__setattr__(self, "n_names", n_names)
        object.__setattr__(self, "recovery", recovery)
        object.__setattr__(self, "notional", notional)
        object.__setattr__(self, "support_boundary", str(self.support_boundary))
        object.__setattr__(self, "representative_credit_curve", _freeze_mapping(self.representative_credit_curve))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def correlation_for(
        self,
        maturity_years: float,
        attachment: float,
        detachment: float,
        *,
        default: float | None = None,
        tolerance: float = 1.0e-8,
    ) -> float:
        """Return an exact-node correlation for one tranche/maturity axis."""
        maturity = float(maturity_years)
        normalized_attachment, normalized_detachment = _validate_tranche_bounds(
            attachment,
            detachment,
        )
        for point in self.points:
            if (
                abs(float(point.maturity_years) - maturity) <= tolerance
                and abs(float(point.attachment) - normalized_attachment) <= tolerance
                and abs(float(point.detachment) - normalized_detachment) <= tolerance
            ):
                return float(point.correlation)
        if default is not None:
            return float(default)
        raise ValueError(
            "correlation surface has no node for "
            f"maturity={maturity}, attachment={normalized_attachment}, "
            f"detachment={normalized_detachment}"
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly surface payload."""
        return {
            "surface_name": self.surface_name,
            "copula_family": self.copula_family,
            "n_names": int(self.n_names),
            "recovery": float(self.recovery),
            "notional": float(self.notional),
            "support_boundary": self.support_boundary,
            "representative_credit_curve": dict(self.representative_credit_curve),
            "points": [point.to_payload() for point in self.points],
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BasketCreditQuoteResidual:
    """One quote residual after fitting an implied correlation."""

    label: str
    market_quote_value: float
    model_quote_value: float
    residual: float
    quote_family: QuoteFamily
    quote_style: QuoteStyle

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly residual payload."""
        return {
            "label": self.label,
            "market_quote_value": float(self.market_quote_value),
            "model_quote_value": float(self.model_quote_value),
            "residual": float(self.residual),
            "quote_family": self.quote_family,
            "quote_style": self.quote_style,
        }


@dataclass(frozen=True)
class BasketCreditRootFailure:
    """Root-bracketing failure for one tranche quote."""

    label: str
    reason: str
    maturity_years: float
    attachment: float
    detachment: float
    quote_value: float
    quote_family: QuoteFamily
    quote_style: QuoteStyle
    lower_correlation: float
    upper_correlation: float
    min_model_quote: float
    max_model_quote: float

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly root-failure payload."""
        return {
            "label": self.label,
            "reason": self.reason,
            "maturity_years": float(self.maturity_years),
            "attachment": float(self.attachment),
            "detachment": float(self.detachment),
            "quote_value": float(self.quote_value),
            "quote_family": self.quote_family,
            "quote_style": self.quote_style,
            "lower_correlation": float(self.lower_correlation),
            "upper_correlation": float(self.upper_correlation),
            "min_model_quote": float(self.min_model_quote),
            "max_model_quote": float(self.max_model_quote),
        }


@dataclass(frozen=True)
class BasketCreditCalibrationDiagnostics:
    """Governance diagnostics for homogeneous basket-credit correlation calibration."""

    quote_residuals: tuple[BasketCreditQuoteResidual, ...] = ()
    root_failures: tuple[BasketCreditRootFailure, ...] = ()
    monotonicity_warnings: tuple[str, ...] = ()
    smoothness_warnings: tuple[str, ...] = ()
    tranche_arbitrage_warnings: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    max_abs_quote_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_residuals", tuple(self.quote_residuals))
        object.__setattr__(self, "root_failures", tuple(self.root_failures))
        object.__setattr__(
            self,
            "monotonicity_warnings",
            tuple(str(warning) for warning in self.monotonicity_warnings),
        )
        object.__setattr__(
            self,
            "smoothness_warnings",
            tuple(str(warning) for warning in self.smoothness_warnings),
        )
        object.__setattr__(
            self,
            "tranche_arbitrage_warnings",
            tuple(str(warning) for warning in self.tranche_arbitrage_warnings),
        )
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "max_abs_quote_residual", float(self.max_abs_quote_residual))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly diagnostics payload."""
        return {
            "quote_residuals": [residual.to_payload() for residual in self.quote_residuals],
            "root_failures": [failure.to_payload() for failure in self.root_failures],
            "monotonicity_warnings": list(self.monotonicity_warnings),
            "smoothness_warnings": list(self.smoothness_warnings),
            "tranche_arbitrage_warnings": list(self.tranche_arbitrage_warnings),
            "warnings": list(self.warnings),
            "max_abs_quote_residual": float(self.max_abs_quote_residual),
        }


@dataclass(frozen=True)
class BasketCreditCorrelationCalibrationResult:
    """Structured result for the bounded basket-credit correlation workflow."""

    quotes: tuple[BasketCreditTrancheQuote, ...]
    surface: BasketCreditCorrelationSurface
    diagnostics: BasketCreditCalibrationDiagnostics
    provenance: Mapping[str, object] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quotes", tuple(self.quotes))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with this calibrated correlation surface."""
        if self.diagnostics.root_failures:
            raise ValueError("cannot materialize basket-credit correlation surface with root failures")
        selected_names = dict(market_state.selected_curve_names or {})
        representative_credit_curve = dict(self.surface.representative_credit_curve)
        selected_curve_roles = {
            "discount_curve": str(selected_names.get("discount_curve") or ""),
            "credit_curve": str(representative_credit_curve.get("object_name") or ""),
            "correlation_surface": self.surface.surface_name,
        }
        return materialize_correlation_surface(
            market_state,
            surface_name=self.surface.surface_name,
            correlation_surface=self.surface,
            source_kind="calibrated_correlation_surface",
            source_ref="calibrate_homogeneous_basket_tranche_correlation_workflow",
            selected_curve_roles=selected_curve_roles,
            metadata={
                "instrument_family": "basket_credit",
                "instrument_kind": "homogeneous_tranche_implied_correlation",
                "support_boundary": self.surface.support_boundary,
                "copula_family": self.surface.copula_family,
                "n_names": int(self.surface.n_names),
                "recovery": float(self.surface.recovery),
                "representative_credit_curve": representative_credit_curve,
                "quote_styles": sorted({quote.quote_style for quote in self.quotes}),
                "max_abs_quote_residual": float(self.diagnostics.max_abs_quote_residual),
                "diagnostics": self.diagnostics.to_payload(),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly calibration payload."""
        return {
            "quotes": [quote.to_payload() for quote in self.quotes],
            "surface": self.surface.to_payload(),
            "diagnostics": self.diagnostics.to_payload(),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class _BasketTrancheSpec:
    """Internal spec consumed by existing basket-credit pricing helpers."""

    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    correlation: float
    recovery: float
    day_count: DayCountConvention = DayCountConvention.ACT_360


@dataclass(frozen=True)
class _RootSolveOutcome:
    """Internal implied-correlation solve outcome."""

    correlation: float | None
    model_quote_value: float | None
    residual: float | None
    failure: BasketCreditRootFailure | None
    warnings: tuple[str, ...] = ()


def _normalize_copula_family(value: object) -> str:
    """Return the bounded copula-family token supported by this workflow."""
    family = _token(value, fallback="gaussian")
    aliases = {
        "factor": "gaussian",
        "factor_gaussian": "gaussian",
        "gaussian_copula": "gaussian",
    }
    family = aliases.get(family, family)
    if family != "gaussian":
        raise ValueError("basket-credit correlation calibration currently supports gaussian copula only")
    return family


def _normalize_quotes(
    quotes: Sequence[BasketCreditTrancheQuote],
) -> tuple[BasketCreditTrancheQuote, ...]:
    """Return sorted tranche quotes and reject duplicate axes."""
    normalized = tuple(quotes)
    if not normalized:
        raise ValueError("at least one basket-credit tranche quote is required")
    sorted_quotes = tuple(
        sorted(
            normalized,
            key=lambda quote: (
                float(quote.maturity_years),
                float(quote.attachment),
                float(quote.detachment),
                quote.quote_family,
                quote.quote_style,
            ),
        )
    )
    seen: set[tuple[float, float, float]] = set()
    for quote in sorted_quotes:
        key = (
            round(float(quote.maturity_years), 12),
            round(float(quote.attachment), 12),
            round(float(quote.detachment), 12),
        )
        if key in seen:
            raise ValueError("basket-credit tranche quote axes must be unique")
        seen.add(key)
    return sorted_quotes


def _representative_credit_curve_payload(market_state: MarketState) -> dict[str, object]:
    """Return the provenance link to the single-name representative credit curve."""
    materialized = market_state.materialized_calibrated_object(object_kind="credit_curve")
    if materialized is not None:
        return {
            "object_name": materialized.get("object_name", "market_state.credit_curve"),
            "object_kind": "credit_curve",
            "source_kind": materialized.get("source_kind", ""),
            "source_ref": materialized.get("source_ref", ""),
            "selected_curve_roles": dict(materialized.get("selected_curve_roles") or {}),
            "metadata": dict(materialized.get("metadata") or {}),
        }
    return {
        "object_name": market_state.selected_curve_name("credit_curve") or "market_state.credit_curve",
        "object_kind": "credit_curve",
        "source_kind": "market_state",
        "source_ref": "",
        "selected_curve_roles": dict(market_state.selected_curve_names or {}),
        "metadata": {},
        "warning": "credit curve was supplied on MarketState without calibrated-object provenance",
    }


def _quote_map_payload(
    quote: BasketCreditTrancheQuote,
    *,
    assumptions: tuple[str, ...],
) -> dict[str, object]:
    """Return quote-map provenance using the shipped bounded quote vocabulary."""
    if quote.quote_style == "fair_spread_bp":
        unit = QuoteUnitSpec(
            unit_name="basis_points",
            value_domain="credit_spread",
            scaling="absolute",
        )
    elif quote.quote_style == "expected_loss_fraction":
        unit = QuoteUnitSpec(
            unit_name="loss_fraction",
            value_domain="portfolio_loss",
            scaling="absolute",
        )
    else:
        unit = QuoteUnitSpec(
            unit_name="present_value",
            value_domain="currency_amount",
            scaling="absolute",
        )
    quote_map = build_identity_quote_map(
        QuoteMapSpec(
            quote_family=quote.quote_family,
            semantics=QuoteSemanticsSpec(
                quote_family=quote.quote_family,
                quote_subject="basket_credit_tranche",
                axes=(
                    QuoteAxisSpec("maturity", axis_kind="tenor", unit="years"),
                    QuoteAxisSpec("attachment", axis_kind="tranche_bound", unit="loss_fraction"),
                    QuoteAxisSpec("detachment", axis_kind="tranche_bound", unit="loss_fraction"),
                ),
                unit=unit,
                settlement=QuoteSettlementSpec(
                    numeraire="discount_curve",
                    settlement_style="tranche_loss",
                    discount_curve_role="discount_curve",
                    metadata={"credit_curve_role": "credit_curve"},
                ),
                metadata={"quote_style": quote.quote_style},
            ),
            assumptions=assumptions,
            metadata={"quote_style": quote.quote_style},
        ),
        source_ref="_basket_credit_tranche_quote_map",
        assumptions=assumptions,
        metadata={
            "normalization_method": "identity_quote_space",
            "support_boundary": _SUPPORT_BOUNDARY,
        },
    )
    return quote_map.to_payload()


def _model_quote_value(
    quote: BasketCreditTrancheQuote,
    market_state: MarketState,
    *,
    n_names: int,
    recovery: float,
    notional: float,
    correlation: float,
    copula_family: str,
) -> float:
    """Return the model quote value for one tranche quote and correlation."""
    spec = _BasketTrancheSpec(
        notional=float(notional),
        n_names=int(n_names),
        attachment=float(quote.attachment),
        detachment=float(quote.detachment),
        end_date=_maturity_date_from_years(market_state.settlement, quote.maturity_years),
        correlation=float(correlation),
        recovery=float(recovery),
    )
    priced = price_credit_basket_tranche_result(
        market_state,
        spec,
        copula_family=copula_family,
    )
    if quote.quote_style == "fair_spread_bp":
        return float(priced.fair_spread_bp)
    if quote.quote_style == "expected_loss_fraction":
        return float(priced.expected_loss_fraction)
    return float(priced.price)


def _root_failure(
    quote: BasketCreditTrancheQuote,
    *,
    label: str,
    reason: str,
    bounds: tuple[float, float],
    sampled_values: Sequence[float],
) -> BasketCreditRootFailure:
    finite_values = [float(value) for value in sampled_values if isfinite(float(value))]
    if not finite_values:
        finite_values = [float("nan")]
    return BasketCreditRootFailure(
        label=label,
        reason=reason,
        maturity_years=float(quote.maturity_years),
        attachment=float(quote.attachment),
        detachment=float(quote.detachment),
        quote_value=float(quote.quote_value),
        quote_family=quote.quote_family,
        quote_style=quote.quote_style,
        lower_correlation=float(bounds[0]),
        upper_correlation=float(bounds[1]),
        min_model_quote=min(finite_values),
        max_model_quote=max(finite_values),
    )


def _solve_implied_correlation(
    quote: BasketCreditTrancheQuote,
    market_state: MarketState,
    *,
    label: str,
    n_names: int,
    recovery: float,
    notional: float,
    copula_family: str,
    correlation_bounds: tuple[float, float],
    root_grid_size: int,
    root_tolerance: float,
    preferred_correlation: float,
) -> _RootSolveOutcome:
    """Solve one quote's tranche-implied correlation with root-scan diagnostics."""
    lower, upper = correlation_bounds
    grid = [
        lower + (upper - lower) * index / float(root_grid_size - 1)
        for index in range(root_grid_size)
    ]
    model_values: list[float] = []
    residual_values: list[float] = []
    warnings: list[str] = []

    def objective(correlation: float) -> float:
        return (
            _model_quote_value(
                quote,
                market_state,
                n_names=n_names,
                recovery=recovery,
                notional=notional,
                correlation=float(correlation),
                copula_family=copula_family,
            )
            - float(quote.quote_value)
        )

    for correlation in grid:
        try:
            model_value = _model_quote_value(
                quote,
                market_state,
                n_names=n_names,
                recovery=recovery,
                notional=notional,
                correlation=float(correlation),
                copula_family=copula_family,
            )
        except Exception:
            model_value = float("nan")
        model_values.append(float(model_value))
        residual_values.append(float(model_value - quote.quote_value) if isfinite(model_value) else float("nan"))

    for correlation, residual in zip(grid, residual_values):
        if isfinite(residual) and abs(residual) <= root_tolerance:
            return _RootSolveOutcome(
                correlation=float(correlation),
                model_quote_value=float(quote.quote_value + residual),
                residual=float(residual),
                failure=None,
                warnings=tuple(warnings),
            )

    intervals: list[tuple[float, float]] = []
    for left_index in range(len(grid) - 1):
        left_residual = residual_values[left_index]
        right_residual = residual_values[left_index + 1]
        if not (isfinite(left_residual) and isfinite(right_residual)):
            continue
        if left_residual == 0.0 or right_residual == 0.0 or left_residual * right_residual < 0.0:
            intervals.append((float(grid[left_index]), float(grid[left_index + 1])))

    if intervals:
        roots: list[tuple[float, float]] = []
        for left, right in intervals:
            try:
                root = float(brentq(objective, left, right, xtol=root_tolerance, rtol=1.0e-12))
            except Exception as exc:
                warnings.append(f"{label}: brent root solve failed on [{left:.6f}, {right:.6f}]: {exc}")
                continue
            roots.append((root, abs(root - float(preferred_correlation))))
        if roots:
            if len(roots) > 1:
                warnings.append(
                    f"{label}: multiple implied-correlation roots were bracketed; "
                    "selected the root closest to the smooth-surface seed."
                )
            root = min(roots, key=lambda item: item[1])[0]
            model_value = _model_quote_value(
                quote,
                market_state,
                n_names=n_names,
                recovery=recovery,
                notional=notional,
                correlation=float(root),
                copula_family=copula_family,
            )
            residual = float(model_value - quote.quote_value)
            return _RootSolveOutcome(
                correlation=float(root),
                model_quote_value=float(model_value),
                residual=residual,
                failure=None,
                warnings=tuple(warnings),
            )

    finite_model_values = [value for value in model_values if isfinite(value)]
    if not finite_model_values:
        reason = "model quote scan produced no finite values"
    elif float(quote.quote_value) < min(finite_model_values) - root_tolerance or (
        float(quote.quote_value) > max(finite_model_values) + root_tolerance
    ):
        reason = "quote is outside model quote range for the configured correlation bounds"
    else:
        reason = "could not bracket implied-correlation root on the scan grid"
    return _RootSolveOutcome(
        correlation=None,
        model_quote_value=None,
        residual=None,
        failure=_root_failure(
            quote,
            label=label,
            reason=reason,
            bounds=correlation_bounds,
            sampled_values=model_values,
        ),
        warnings=tuple(warnings),
    )


def _quote_upper_bound_warning(
    quote: BasketCreditTrancheQuote,
    market_state: MarketState,
    *,
    notional: float,
) -> str | None:
    """Return a simple tranche-bound warning for impossible quote magnitudes."""
    if quote.quote_style == "fair_spread_bp":
        return None
    if quote.quote_style == "expected_loss_fraction":
        upper = quote.tranche_width
        unit_text = "expected loss fraction"
    else:
        discount = float(market_state.discount.discount(float(quote.maturity_years)))
        upper = float(notional) * discount * quote.tranche_width
        unit_text = "present value"
    if float(quote.quote_value) > upper + 1.0e-10:
        return (
            f"{quote.label or 'tranche quote'} {unit_text} exceeds the tranche maximum "
            f"{upper:.12g} for width {quote.tranche_width:.12g}."
        )
    return None


def _normalized_loss_quote(
    quote: BasketCreditTrancheQuote,
    market_state: MarketState,
    *,
    notional: float,
) -> float | None:
    """Return tranche-width-normalized expected loss for arbitrage checks."""
    width = max(float(quote.tranche_width), 1.0e-12)
    if quote.quote_style == "expected_loss_fraction":
        return float(quote.quote_value) / width
    if quote.quote_style == "present_value":
        discount = float(market_state.discount.discount(float(quote.maturity_years)))
        denominator = max(float(notional) * discount * width, 1.0e-12)
        return float(quote.quote_value) / denominator
    return None


def _tranche_arbitrage_warnings(
    quotes: Sequence[BasketCreditTrancheQuote],
    market_state: MarketState,
    *,
    notional: float,
) -> tuple[str, ...]:
    """Return bounded tranche quote sanity warnings."""
    warnings: list[str] = []
    for quote in quotes:
        warning = _quote_upper_bound_warning(quote, market_state, notional=notional)
        if warning is not None:
            warnings.append(warning)

    by_maturity: dict[float, list[BasketCreditTrancheQuote]] = defaultdict(list)
    for quote in quotes:
        by_maturity[round(float(quote.maturity_years), 12)].append(quote)
    for maturity, maturity_quotes in by_maturity.items():
        ordered = sorted(maturity_quotes, key=lambda quote: (float(quote.attachment), float(quote.detachment)))
        previous_quote: BasketCreditTrancheQuote | None = None
        previous_loss: float | None = None
        for quote in ordered:
            normalized_loss = _normalized_loss_quote(quote, market_state, notional=notional)
            if normalized_loss is None:
                continue
            if (
                previous_quote is not None
                and previous_loss is not None
                and float(quote.attachment) >= float(previous_quote.detachment) - 1.0e-12
                and normalized_loss > previous_loss + 1.0e-8
            ):
                warnings.append(
                    "senior tranche quote has larger width-normalized loss than the adjacent "
                    f"junior tranche at maturity {maturity:g}y."
                )
            previous_quote = quote
            previous_loss = normalized_loss
    return tuple(warnings)


def _surface_governance_warnings(
    points: Sequence[BasketCreditCorrelationPoint],
    *,
    smoothness_jump_threshold: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return monotonicity and smoothness warnings for the fitted surface."""
    monotonicity: list[str] = []
    smoothness: list[str] = []
    by_maturity: dict[float, list[BasketCreditCorrelationPoint]] = defaultdict(list)
    by_tranche: dict[tuple[float, float], list[BasketCreditCorrelationPoint]] = defaultdict(list)
    for point in points:
        by_maturity[round(float(point.maturity_years), 12)].append(point)
        by_tranche[(round(float(point.attachment), 12), round(float(point.detachment), 12))].append(point)

    for maturity, maturity_points in by_maturity.items():
        ordered = sorted(maturity_points, key=lambda point: (float(point.detachment), float(point.attachment)))
        for left, right in zip(ordered, ordered[1:]):
            diff = float(right.correlation - left.correlation)
            if diff < -1.0e-8:
                monotonicity.append(
                    "tranche-implied correlation decreases with detachment at "
                    f"maturity {maturity:g}y: {left.correlation:.6f} -> {right.correlation:.6f}."
                )
            if abs(diff) > float(smoothness_jump_threshold):
                smoothness.append(
                    "tranche-implied correlation jump exceeds smoothness threshold at "
                    f"maturity {maturity:g}y: {left.correlation:.6f} -> {right.correlation:.6f}."
                )

    for (attachment, detachment), tranche_points in by_tranche.items():
        ordered = sorted(tranche_points, key=lambda point: float(point.maturity_years))
        for left, right in zip(ordered, ordered[1:]):
            diff = float(right.correlation - left.correlation)
            if diff < -1.0e-8:
                monotonicity.append(
                    "tranche-implied correlation decreases with maturity for "
                    f"[{attachment:g}, {detachment:g}]: {left.correlation:.6f} -> {right.correlation:.6f}."
                )
            if abs(diff) > float(smoothness_jump_threshold):
                smoothness.append(
                    "tranche-implied correlation term jump exceeds smoothness threshold for "
                    f"[{attachment:g}, {detachment:g}]: {left.correlation:.6f} -> {right.correlation:.6f}."
                )
    return tuple(monotonicity), tuple(smoothness)


def calibrate_homogeneous_basket_tranche_correlation_workflow(
    quotes: Sequence[BasketCreditTrancheQuote],
    market_state: MarketState,
    *,
    n_names: int,
    recovery: float = 0.4,
    notional: float = 1.0,
    surface_name: str = "basket_credit_tranche_correlation",
    copula_family: str = "gaussian",
    correlation_bounds: tuple[float, float] = _DEFAULT_CORRELATION_BOUNDS,
    root_grid_size: int = 81,
    root_tolerance: float = 1.0e-10,
    root_failure_policy: RootFailurePolicy = "raise",
    smoothness_jump_threshold: float = 0.25,
) -> BasketCreditCorrelationCalibrationResult:
    """Calibrate homogeneous tranche-implied correlations from basket tranche quotes."""
    if market_state.discount is None:
        raise ValueError("basket-credit correlation calibration requires market_state.discount")
    if market_state.credit_curve is None:
        raise ValueError(
            "basket-credit correlation calibration requires market_state.credit_curve; "
            "apply a calibrated single-name CreditCurve to MarketState first"
        )
    if market_state.settlement is None:
        raise ValueError("basket-credit correlation calibration requires market_state.settlement")
    n_names = int(n_names)
    if n_names < 2:
        raise ValueError("n_names must be at least 2")
    recovery = _finite_float(recovery, field_name="recovery")
    if recovery <= 0.0 or recovery >= 1.0:
        raise ValueError("recovery must be strictly between 0 and 1")
    notional = _positive_float(notional, field_name="notional")
    copula_family = _normalize_copula_family(copula_family)
    lower, upper = (float(correlation_bounds[0]), float(correlation_bounds[1]))
    if lower < 0.0 or lower >= upper or upper >= 1.0:
        raise ValueError("correlation_bounds must satisfy 0 <= lower < upper < 1")
    if int(root_grid_size) < 3:
        raise ValueError("root_grid_size must be at least 3")
    root_grid_size = int(root_grid_size)
    if root_failure_policy not in {"raise", "warn"}:
        raise ValueError("root_failure_policy must be 'raise' or 'warn'")
    smoothness_jump_threshold = _positive_float(
        smoothness_jump_threshold,
        field_name="smoothness_jump_threshold",
    )

    normalized_quotes = _normalize_quotes(quotes)
    representative_curve = _representative_credit_curve_payload(market_state)
    assumptions = (
        "Homogeneous basket-credit calibration uses one representative calibrated CreditCurve "
        "from MarketState for every name.",
        "The workflow does not reconstruct hidden single-name curves or heterogeneous portfolios.",
        "The first slice fits exact-node tranche-implied correlations under a one-factor Gaussian copula.",
    )
    labels = tuple(quote.resolved_label(index) for index, quote in enumerate(normalized_quotes))
    quote_maps = tuple(
        _quote_map_payload(quote, assumptions=assumptions)
        for quote in normalized_quotes
    )
    tranche_warnings = _tranche_arbitrage_warnings(
        normalized_quotes,
        market_state,
        notional=notional,
    )

    points: list[BasketCreditCorrelationPoint] = []
    residuals: list[BasketCreditQuoteResidual] = []
    root_failures: list[BasketCreditRootFailure] = []
    root_warnings: list[str] = []
    preferred_correlation = min(max(0.30, lower), upper)
    for label, quote in zip(labels, normalized_quotes):
        outcome = _solve_implied_correlation(
            quote,
            market_state,
            label=label,
            n_names=n_names,
            recovery=recovery,
            notional=notional,
            copula_family=copula_family,
            correlation_bounds=(lower, upper),
            root_grid_size=root_grid_size,
            root_tolerance=float(root_tolerance),
            preferred_correlation=preferred_correlation,
        )
        root_warnings.extend(outcome.warnings)
        if outcome.failure is not None:
            root_failures.append(outcome.failure)
            continue
        assert outcome.correlation is not None
        assert outcome.model_quote_value is not None
        assert outcome.residual is not None
        preferred_correlation = float(outcome.correlation)
        point = BasketCreditCorrelationPoint(
            maturity_years=quote.maturity_years,
            attachment=quote.attachment,
            detachment=quote.detachment,
            correlation=float(outcome.correlation),
            quote_label=label,
            quote_family=quote.quote_family,
            quote_style=quote.quote_style,
            market_quote_value=float(quote.quote_value),
            model_quote_value=float(outcome.model_quote_value),
            quote_residual=float(outcome.residual),
        )
        points.append(point)
        residuals.append(
            BasketCreditQuoteResidual(
                label=label,
                market_quote_value=float(quote.quote_value),
                model_quote_value=float(outcome.model_quote_value),
                residual=float(outcome.residual),
                quote_family=quote.quote_family,
                quote_style=quote.quote_style,
            )
        )

    if root_failures and root_failure_policy == "raise":
        details = "; ".join(f"{failure.label}: {failure.reason}" for failure in root_failures)
        raise ValueError(f"basket-credit correlation calibration failed: {details}")

    monotonicity_warnings, smoothness_warnings = _surface_governance_warnings(
        points,
        smoothness_jump_threshold=smoothness_jump_threshold,
    )
    warnings = tuple(
        list(root_warnings)
        + [f"{failure.label}: {failure.reason}" for failure in root_failures]
        + list(monotonicity_warnings)
        + list(smoothness_warnings)
        + list(tranche_warnings)
    )
    max_abs_quote_residual = max((abs(residual.residual) for residual in residuals), default=0.0)
    diagnostics = BasketCreditCalibrationDiagnostics(
        quote_residuals=tuple(residuals),
        root_failures=tuple(root_failures),
        monotonicity_warnings=monotonicity_warnings,
        smoothness_warnings=smoothness_warnings,
        tranche_arbitrage_warnings=tranche_warnings,
        warnings=warnings,
        max_abs_quote_residual=float(max_abs_quote_residual),
    )
    provenance = {
        "source_kind": "calibrated_correlation_surface",
        "source_ref": "calibrate_homogeneous_basket_tranche_correlation_workflow",
        "support_boundary": _SUPPORT_BOUNDARY,
        "representative_credit_curve": dict(representative_curve),
        "calibration_target": {
            "quote_maps": list(quote_maps),
            "quote_axes": ["maturity", "attachment", "detachment"],
            "quote_values": [quote.to_payload() for quote in normalized_quotes],
        },
        "solver": {
            "method": "brentq_root_scan",
            "correlation_bounds": [float(lower), float(upper)],
            "root_grid_size": int(root_grid_size),
            "root_tolerance": float(root_tolerance),
        },
        "diagnostics": diagnostics.to_payload(),
    }
    surface = BasketCreditCorrelationSurface(
        points=tuple(points),
        surface_name=surface_name,
        copula_family=copula_family,
        n_names=n_names,
        recovery=recovery,
        notional=notional,
        support_boundary=_SUPPORT_BOUNDARY,
        representative_credit_curve=representative_curve,
        provenance=provenance,
        warnings=warnings,
    )
    summary = {
        "surface_name": surface.surface_name,
        "support_boundary": _SUPPORT_BOUNDARY,
        "copula_family": copula_family,
        "n_names": int(n_names),
        "recovery": float(recovery),
        "notional": float(notional),
        "quote_count": len(normalized_quotes),
        "fitted_point_count": len(points),
        "root_failure_count": len(root_failures),
        "quote_families": [quote.quote_family for quote in normalized_quotes],
        "quote_styles": [quote.quote_style for quote in normalized_quotes],
        "representative_credit_curve": dict(representative_curve),
        "max_abs_quote_residual": float(max_abs_quote_residual),
    }
    return BasketCreditCorrelationCalibrationResult(
        quotes=normalized_quotes,
        surface=surface,
        diagnostics=diagnostics,
        provenance=provenance,
        summary=summary,
        assumptions=assumptions,
    )


__all__ = [
    "BasketCreditCalibrationDiagnostics",
    "BasketCreditCorrelationCalibrationResult",
    "BasketCreditCorrelationPoint",
    "BasketCreditCorrelationSurface",
    "BasketCreditQuoteResidual",
    "BasketCreditRootFailure",
    "BasketCreditTrancheQuote",
    "calibrate_homogeneous_basket_tranche_correlation_workflow",
]
