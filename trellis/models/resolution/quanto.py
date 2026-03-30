"""Shared market-resolution helpers for single-name quanto routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

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


def resolve_quanto_underlier_spot(
    market_state: MarketState,
    spec: QuantoSpecLike,
) -> float:
    """Resolve the named underlier spot from the market state."""
    if market_state.underlier_spots:
        for key in (
            spec.underlier_currency,
            spec.fx_pair,
            spec.underlier_currency.upper(),
            spec.underlier_currency.lower(),
        ):
            if key in market_state.underlier_spots:
                return market_state.underlier_spots[key]
    if market_state.spot is not None:
        return market_state.spot
    raise ValueError(
        f"Quanto pricing requires spot or underlier_spots for {spec.underlier_currency!r}"
    )


def resolve_quanto_foreign_curve(
    market_state: MarketState,
    spec: QuantoSpecLike,
):
    """Resolve the foreign discount curve used for carry and quanto adjustment."""
    forecast_curves = market_state.forecast_curves or {}
    for key in (
        f"{spec.underlier_currency}-DISC",
        f"{spec.underlier_currency}_DISC",
        spec.underlier_currency,
        spec.underlier_currency.upper(),
    ):
        if key in forecast_curves:
            return forecast_curves[key]
    if len(forecast_curves) == 1:
        return next(iter(forecast_curves.values()))
    if market_state.discount is not None:
        return market_state.discount
    raise ValueError(
        "Quanto pricing requires a foreign carry/discount curve in "
        "`market_state.forecast_curves` or a fallback discount curve."
    )


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

    spot = resolve_quanto_underlier_spot(market_state, spec)
    fx_spot = fx_quote.spot
    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    if T <= 0.0:
        market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
        return ResolvedQuantoInputs(
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
                "correlation": {"source_family": "identity", "source_kind": "identity_default"},
            },
        )

    domestic_df = market_state.discount.discount(T)
    foreign_df = resolve_quanto_foreign_curve(market_state, spec).discount(T)
    sigma_underlier = market_state.vol_surface.black_vol(T, spec.strike)
    sigma_fx = market_state.vol_surface.black_vol(T, fx_spot)
    corr, correlation_provenance = _resolve_quanto_correlation_details(market_state, spec)
    corr = np.clip(corr, -0.999, 0.999)
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    return ResolvedQuantoInputs(
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
            "correlation": correlation_provenance,
        },
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
