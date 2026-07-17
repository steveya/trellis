"""Product-neutral market binding for one-factor short-rate lattices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from typing import Protocol, SupportsFloat, SupportsIndex, cast

from trellis.models.trees.models import MODEL_REGISTRY


class DiscountCurveLike(Protocol):
    """Discount-curve surface required by short-rate lattice resolution."""

    def zero_rate(self, time: float) -> float:
        """Return the continuously compounded zero rate at ``time``."""
        ...


class VolSurfaceLike(Protocol):
    """Black-volatility surface used as a bounded lattice fallback."""

    def black_vol(self, time: float, strike: float) -> float:
        """Return the Black-style volatility quote at ``time`` and ``strike``."""
        ...


class ShortRateLatticeMarketStateLike(Protocol):
    """Market-state fields consumed by the short-rate lattice resolver."""

    @property
    def discount(self) -> DiscountCurveLike | None:
        """Return the discount curve used for calibration."""
        ...

    @property
    def vol_surface(self) -> VolSurfaceLike | None:
        """Return optional Black-style volatility evidence."""
        ...


@dataclass(frozen=True)
class ResolvedShortRateLatticeInputs:
    """Canonical inputs for a calibrated one-factor short-rate lattice."""

    model_name: str
    volatility_type: str
    horizon: float
    r0: float
    mean_reversion: float
    sigma: float
    n_steps: int


_MODEL_FAMILY_ALIASES = {
    "black_derman_toy": "bdt",
    "blackdermantoy": "bdt",
    "blackkarasinski": "black_karasinski",
    "hullwhite": "hull_white",
    "holee": "ho_lee",
}


def _normalize_model_family(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    for alias, canonical in _MODEL_FAMILY_ALIASES.items():
        if normalized == alias or normalized.startswith(f"{alias}_"):
            return f"{canonical}{normalized[len(alias):]}"
    return normalized


def _model_parameter_payload(
    market_state: ShortRateLatticeMarketStateLike,
    *,
    model_key: str,
) -> Mapping[str, object]:
    """Return calibrated parameters associated with the selected tree model."""
    candidates: list[tuple[str, Mapping[str, object]]] = []
    direct = getattr(market_state, "model_parameters", None)
    if isinstance(direct, Mapping):
        nested = direct.get(model_key)
        if isinstance(nested, Mapping):
            candidates.append((model_key, nested))
        candidates.append(("", direct))

    parameter_sets = getattr(market_state, "model_parameter_sets", None)
    if isinstance(parameter_sets, Mapping):
        candidates.extend(
            (str(name), payload)
            for name, payload in parameter_sets.items()
            if isinstance(payload, Mapping)
        )

    for _, payload in candidates:
        family = _normalize_model_family(
            payload.get("model_family") or payload.get("model_name") or payload.get("family")
        )
        if family == model_key:
            return payload

    for name, payload in candidates:
        normalized_name = _normalize_model_family(name)
        family = _normalize_model_family(
            payload.get("model_family") or payload.get("model_name") or payload.get("family")
        )
        if not family and (
            normalized_name == model_key or normalized_name.startswith(f"{model_key}_")
        ):
            return payload

    for name, payload in candidates:
        family = _normalize_model_family(
            payload.get("model_family") or payload.get("model_name") or payload.get("family")
        )
        if not name and not family and (
            payload.get("sigma") is not None or payload.get("mean_reversion") is not None
        ):
            return payload
    return {}


def _parameter_float(
    payload: Mapping[str, object],
    *names: str,
) -> float | None:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return float(cast(str | SupportsFloat | SupportsIndex, value))
    return None


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        resolved = int(integer_index(cast(SupportsIndex, value)))
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def resolve_short_rate_lattice_inputs(
    market_state: ShortRateLatticeMarketStateLike,
    *,
    horizon: float,
    model: str = "hull_white",
    volatility_time: float | None = None,
    volatility_strike: float | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
    default_mean_reversion: float = 0.1,
    minimum_steps: int = 50,
    maximum_steps: int = 200,
    steps_per_year: float = 50.0,
) -> ResolvedShortRateLatticeInputs:
    """Resolve market and discretization inputs for a short-rate lattice.

    Explicit or calibrated model parameters take precedence. When sigma is not
    available from either source, a Black-style surface quote is retained for
    lognormal models and converted to absolute rate volatility for normal
    models.
    """
    resolved_horizon = float(horizon)
    if not isfinite(resolved_horizon) or resolved_horizon <= 0.0:
        raise ValueError("short-rate lattice horizon must be positive")

    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("short-rate lattice resolution requires market_state.discount")

    model_key = str(model).strip().lower()
    try:
        tree_model = MODEL_REGISTRY[model_key]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported short-rate lattice model {model!r}; expected one of: {supported}"
        ) from exc

    resolved_r0 = float(discount_curve.zero_rate(max(resolved_horizon / 2.0, 1e-6)))
    if not isfinite(resolved_r0):
        raise ValueError("short-rate lattice initial rate must be finite")
    parameter_payload = _model_parameter_payload(market_state, model_key=model_key)
    parameter_sigma = (
        None
        if sigma is not None
        else _parameter_float(
            parameter_payload,
            "sigma",
            "volatility",
            "short_rate_vol",
        )
    )
    fallback_sigma: float | None = None
    if sigma is None and parameter_sigma is None:
        vol_surface = market_state.vol_surface
        if vol_surface is None:
            raise ValueError(
                "short-rate lattice sigma must be provided explicitly, through model parameters, "
                "or through market_state.vol_surface"
            )
        quote_time = (
            resolved_horizon / 2.0
            if volatility_time is None
            else float(volatility_time)
        )
        if not isfinite(quote_time) or quote_time <= 0.0:
            raise ValueError("short-rate lattice volatility_time must be positive")
        quote_strike = (
            max(resolved_r0, 1e-6)
            if volatility_strike is None
            else float(volatility_strike)
        )
        if not isfinite(quote_strike):
            raise ValueError("short-rate lattice volatility_strike must be finite")
        black_vol = float(vol_surface.black_vol(quote_time, quote_strike))
        if not isfinite(black_vol) or black_vol < 0.0:
            raise ValueError(
                "short-rate lattice Black volatility must be finite and non-negative"
            )
        fallback_sigma = (
            black_vol
            if tree_model.vol_type == "lognormal"
            else black_vol * max(abs(resolved_r0), 1e-6)
        )

    parameter_mean_reversion = (
        None
        if mean_reversion is not None
        else _parameter_float(
            parameter_payload,
            "mean_reversion",
            "a",
        )
    )
    resolved_mean_reversion = (
        float(mean_reversion)
        if mean_reversion is not None
        else parameter_mean_reversion
        if parameter_mean_reversion is not None
        else float(default_mean_reversion)
    )
    resolved_sigma = float(sigma) if sigma is not None else parameter_sigma
    if resolved_sigma is None:
        resolved_sigma = fallback_sigma
    if resolved_sigma is None:
        raise ValueError(
            "short-rate lattice sigma must be provided explicitly, through model parameters, "
            "or through market_state.vol_surface"
        )

    if not isfinite(float(resolved_mean_reversion)):
        raise ValueError("short-rate lattice mean reversion must be finite")
    if not isfinite(float(resolved_sigma)) or float(resolved_sigma) < 0.0:
        raise ValueError("short-rate lattice sigma must be finite and non-negative")

    lower_steps = _positive_integer(minimum_steps, name="minimum_steps")
    upper_steps = _positive_integer(maximum_steps, name="maximum_steps")
    if upper_steps < lower_steps:
        raise ValueError("maximum_steps must be at least minimum_steps")
    resolved_steps_per_year = float(steps_per_year)
    if not isfinite(resolved_steps_per_year) or resolved_steps_per_year <= 0.0:
        raise ValueError("steps_per_year must be positive")
    if n_steps is None:
        resolved_steps = min(
            upper_steps,
            max(lower_steps, int(resolved_horizon * resolved_steps_per_year)),
        )
    else:
        resolved_steps = _positive_integer(n_steps, name="n_steps")

    return ResolvedShortRateLatticeInputs(
        model_name=tree_model.name,
        volatility_type=tree_model.vol_type,
        horizon=resolved_horizon,
        r0=resolved_r0,
        mean_reversion=float(resolved_mean_reversion),
        sigma=float(resolved_sigma),
        n_steps=resolved_steps,
    )


__all__ = [
    "DiscountCurveLike",
    "ResolvedShortRateLatticeInputs",
    "ShortRateLatticeMarketStateLike",
    "VolSurfaceLike",
    "resolve_short_rate_lattice_inputs",
]
