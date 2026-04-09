"""Shared resolver helpers for bounded short-rate claim families."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Mapping, Protocol

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.quote_maps import (
    QuoteAxisSpec,
    QuoteSemanticsSpec,
    QuoteUnitSpec,
)
from trellis.models.hull_white_parameters import (
    extract_hull_white_parameter_payload,
    resolve_hull_white_mean_reversion,
)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy of one metadata mapping."""
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class FlatShortRateVolSurface:
    """Flat short-rate volatility object with explicit quote semantics."""

    sigma: float
    quote_family: str = "implied_vol"
    quote_subject: str = "discount_bond_option"
    quote_convention: str = "black"
    quote_unit: str = "decimal_volatility"
    source_kind: str = "comparison_regime"
    quote_semantics: QuoteSemanticsSpec | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if float(self.sigma) < 0.0:
            raise ValueError("short-rate comparison sigma must be non-negative")
        semantics = self.quote_semantics or QuoteSemanticsSpec(
            quote_family=str(self.quote_family or "implied_vol"),
            convention=str(self.quote_convention or "black"),
            quote_subject=str(self.quote_subject or "discount_bond_option"),
            axes=(
                QuoteAxisSpec("expiry", axis_kind="time_to_expiry", unit="years"),
                QuoteAxisSpec("bond_maturity", axis_kind="discount_bond_maturity", unit="years"),
                QuoteAxisSpec("strike", axis_kind="bond_price", unit="decimal_price"),
            ),
            unit=QuoteUnitSpec(
                unit_name=str(self.quote_unit or "decimal_volatility"),
                value_domain="volatility",
                scaling="absolute",
            ),
            metadata={"surface_shape": "flat"},
        )
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "quote_family", semantics.quote_family)
        object.__setattr__(self, "quote_subject", semantics.quote_subject)
        object.__setattr__(self, "quote_convention", semantics.convention)
        object.__setattr__(self, "quote_unit", semantics.unit.unit_name if semantics.unit is not None else "")
        object.__setattr__(self, "quote_semantics", semantics)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return the same volatility for every expiry/strike pair."""
        return float(self.sigma)

    def to_payload(self) -> dict[str, object]:
        """Return a stable serializable representation of the flat surface."""
        payload = {
            "surface_family": "short_rate_flat_vol",
            "quote_family": self.quote_family,
            "quote_subject": self.quote_subject,
            "quote_convention": self.quote_convention,
            "quote_unit": self.quote_unit,
            "sigma": float(self.sigma),
            "source_kind": self.source_kind,
            "quote_semantics": self.quote_semantics.to_payload(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ShortRateComparisonRegime:
    """Typed bounded comparison regime for short-rate task flows."""

    regime_name: str = "short_rate_comparison_regime"
    flat_discount_rate: float = 0.05
    flat_sigma: float = 0.01
    hull_white_mean_reversion: float = 0.1
    ho_lee_mean_reversion: float = 0.0
    quote_family: str = "implied_vol"
    quote_subject: str = "discount_bond_option"
    quote_convention: str = "black"
    quote_unit: str = "decimal_volatility"
    source_kind: str = "task_comparison_regime"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if float(self.flat_sigma) < 0.0:
            raise ValueError("flat short-rate sigma must be non-negative")
        object.__setattr__(self, "flat_discount_rate", float(self.flat_discount_rate))
        object.__setattr__(self, "flat_sigma", float(self.flat_sigma))
        object.__setattr__(self, "hull_white_mean_reversion", float(self.hull_white_mean_reversion))
        object.__setattr__(self, "ho_lee_mean_reversion", float(self.ho_lee_mean_reversion))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def from_task_spec(cls, payload: Mapping[str, object]) -> "ShortRateComparisonRegime":
        """Construct one bounded short-rate regime from task/runtime metadata."""
        return cls(
            regime_name=str(payload.get("regime_name") or "short_rate_comparison_regime").strip(),
            flat_discount_rate=float(payload.get("flat_discount_rate", 0.05)),
            flat_sigma=float(payload.get("flat_sigma", 0.01)),
            hull_white_mean_reversion=float(payload.get("hull_white_mean_reversion", 0.1)),
            ho_lee_mean_reversion=float(payload.get("ho_lee_mean_reversion", 0.0)),
            quote_family=(
                "implied_vol"
                if str(payload.get("quote_family") or "implied_vol").strip().lower() == "black"
                else str(payload.get("quote_family") or "implied_vol").strip()
            ),
            quote_subject=str(payload.get("quote_subject") or "discount_bond_option").strip(),
            quote_convention=(
                "black"
                if str(payload.get("quote_convention") or "black").strip().lower() == "flat"
                else str(payload.get("quote_convention") or "black").strip()
            ),
            quote_unit=str(payload.get("quote_unit") or "decimal_volatility").strip(),
            source_kind=str(payload.get("source_kind") or "task_comparison_regime").strip(),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_payload(self) -> dict[str, object]:
        """Return a stable serializable payload for runtime provenance."""
        payload = {
            "regime_family": "short_rate",
            "regime_name": self.regime_name,
            "flat_discount_rate": float(self.flat_discount_rate),
            "flat_sigma": float(self.flat_sigma),
            "hull_white_mean_reversion": float(self.hull_white_mean_reversion),
            "ho_lee_mean_reversion": float(self.ho_lee_mean_reversion),
            "quote_family": self.quote_family,
            "quote_subject": self.quote_subject,
            "quote_convention": self.quote_convention,
            "quote_unit": self.quote_unit,
            "source_kind": self.source_kind,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        payload["vol_surface"] = self.build_vol_surface().to_payload()
        return payload

    def build_vol_surface(self) -> FlatShortRateVolSurface:
        """Return the regime's typed flat short-rate volatility surface."""
        return FlatShortRateVolSurface(
            sigma=self.flat_sigma,
            quote_family=self.quote_family,
            quote_subject=self.quote_subject,
            quote_convention=self.quote_convention,
            quote_unit=self.quote_unit,
            source_kind=self.source_kind,
            metadata={
                "regime_name": self.regime_name,
                **dict(self.metadata),
            },
        )

    def build_discount_curve(self, *, max_tenor: float = 31.0):
        """Return the regime's flat discount curve."""
        return YieldCurve.flat(self.flat_discount_rate, max_tenor=max_tenor)

    def mean_reversion_for_model(self, model: str) -> float:
        """Return the model-specific mean reversion implied by the regime."""
        normalized_model = str(model or "hull_white").strip().lower()
        if normalized_model == "ho_lee":
            return float(self.ho_lee_mean_reversion)
        return float(self.hull_white_mean_reversion)


class DiscountCurveLike(Protocol):
    """Discount interface required by the short-rate claim resolver."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return a zero rate to time ``t``."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the short-rate claim resolver."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class ShortRateClaimMarketStateLike(Protocol):
    """Minimal market-state interface required by the short-rate claim helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None
    market_provenance: Mapping[str, object] | None


class DiscountBondClaimSpecLike(Protocol):
    """Minimal semantic spec surface for discount-bond option style claims."""

    notional: float
    strike: float
    expiry_date: date
    bond_maturity_date: date


@dataclass(frozen=True)
class ResolvedShortRateRegime:
    """Resolved model inputs shared by bounded short-rate claim families."""

    model: str
    initial_rate: float
    sigma: float
    mean_reversion: float
    comparison_regime: ShortRateComparisonRegime | None = None


@dataclass(frozen=True)
class ResolvedDiscountBondClaim:
    """Resolved claim semantics shared by analytical and tree ZCB helpers."""

    notional: float
    option_type: str
    settlement: date
    strike_unit: float
    expiry_time: float
    bond_maturity_time: float
    discount_factor_expiry: float
    discount_factor_bond: float
    regime: ResolvedShortRateRegime


def normalize_discount_bond_strike(strike_quote: float, notional: float) -> float:
    """Normalize strikes to unit-face form when quoted per 100 face."""
    strike = float(strike_quote)
    face = abs(float(notional))
    if abs(strike) > 1.0 and face > 1.0:
        strike /= face
    return strike


def resolve_discount_bond_option_type(spec) -> str:
    """Resolve call/put semantics from modern or legacy bond-option spec fields."""
    option_type = getattr(spec, "option_type", None)
    if option_type is not None:
        normalized = str(option_type).strip().strip("'\"").lower()
        if normalized in {"call", "put"}:
            return normalized
        if normalized == "payer":
            return "put"
        if normalized == "receiver":
            return "call"
    if hasattr(spec, "is_call"):
        return "call" if bool(spec.is_call) else "put"
    if hasattr(spec, "is_payer"):
        return "put" if bool(spec.is_payer) else "call"
    return "call"


def extract_short_rate_comparison_regime(
    market_state: ShortRateClaimMarketStateLike,
) -> ShortRateComparisonRegime | None:
    """Return the first typed short-rate comparison regime on one market state."""
    provenance = dict(getattr(market_state, "market_provenance", None) or {})
    payload = provenance.get("comparison_regime")
    if not isinstance(payload, Mapping):
        return None
    if str(payload.get("regime_family") or "").strip().lower() != "short_rate":
        return None
    return ShortRateComparisonRegime.from_task_spec(payload)


def resolve_short_rate_regime(
    market_state: ShortRateClaimMarketStateLike,
    *,
    model: str = "hull_white",
    maturity: float,
    strike: float,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    default_mean_reversion: float = 0.1,
) -> ResolvedShortRateRegime:
    """Resolve bounded short-rate model inputs from market state and overrides."""
    normalized_model = str(model or "hull_white").strip().lower()
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("short-rate claim pricing requires market_state.discount")

    comparison_regime = extract_short_rate_comparison_regime(market_state)
    lookup_time = max(float(maturity), 1e-6)
    initial_rate = float(discount_curve.zero_rate(lookup_time))

    hull_white_payload = extract_hull_white_parameter_payload(market_state)

    if sigma is not None:
        resolved_sigma = float(sigma)
    elif comparison_regime is not None:
        resolved_sigma = float(comparison_regime.flat_sigma)
    elif hull_white_payload is not None and hull_white_payload.get("sigma") is not None:
        resolved_sigma = float(hull_white_payload["sigma"])
    elif market_state.vol_surface is not None:
        resolved_sigma = float(market_state.vol_surface.black_vol(lookup_time, float(strike)))
    else:
        raise ValueError("short-rate claim pricing requires market_state.vol_surface or an explicit sigma")

    if mean_reversion is not None:
        resolved_mean_reversion = float(mean_reversion)
    elif comparison_regime is not None:
        resolved_mean_reversion = comparison_regime.mean_reversion_for_model(normalized_model)
    elif normalized_model == "ho_lee":
        resolved_mean_reversion = 0.0
    else:
        resolved_mean_reversion = resolve_hull_white_mean_reversion(
            market_state,
            default_mean_reversion=default_mean_reversion,
        )

    return ResolvedShortRateRegime(
        model=normalized_model,
        initial_rate=initial_rate,
        sigma=float(resolved_sigma),
        mean_reversion=float(resolved_mean_reversion),
        comparison_regime=comparison_regime,
    )


def resolve_discount_bond_claim_inputs(
    market_state: ShortRateClaimMarketStateLike,
    spec: DiscountBondClaimSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    default_mean_reversion: float = 0.1,
) -> ResolvedDiscountBondClaim:
    """Resolve one bounded discount-bond option claim under a short-rate regime."""
    settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for short-rate claims")

    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("short-rate claim pricing requires market_state.discount")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    expiry_time = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    bond_maturity_time = float(year_fraction(settlement, spec.bond_maturity_date, day_count))
    if bond_maturity_time <= expiry_time:
        raise ValueError("bond_maturity_date must be after expiry_date for discount-bond claims")

    strike_unit = normalize_discount_bond_strike(spec.strike, spec.notional)
    regime = resolve_short_rate_regime(
        market_state,
        model=model,
        maturity=min(expiry_time, bond_maturity_time) if expiry_time > 0.0 else bond_maturity_time,
        strike=strike_unit,
        mean_reversion=mean_reversion,
        sigma=sigma,
        default_mean_reversion=default_mean_reversion,
    )

    return ResolvedDiscountBondClaim(
        notional=float(spec.notional),
        option_type=resolve_discount_bond_option_type(spec),
        settlement=settlement,
        strike_unit=strike_unit,
        expiry_time=expiry_time,
        bond_maturity_time=bond_maturity_time,
        discount_factor_expiry=float(discount_curve.discount(max(expiry_time, 0.0))),
        discount_factor_bond=float(discount_curve.discount(max(bond_maturity_time, 0.0))),
        regime=regime,
    )


__all__ = [
    "DiscountBondClaimSpecLike",
    "DiscountCurveLike",
    "FlatShortRateVolSurface",
    "ResolvedDiscountBondClaim",
    "ResolvedShortRateRegime",
    "ShortRateClaimMarketStateLike",
    "ShortRateComparisonRegime",
    "VolSurfaceLike",
    "extract_short_rate_comparison_regime",
    "normalize_discount_bond_strike",
    "resolve_discount_bond_claim_inputs",
    "resolve_discount_bond_option_type",
    "resolve_short_rate_regime",
]
