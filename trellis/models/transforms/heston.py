"""Heston transform pricing helpers.

The helpers in this module bind explicit Heston model parameters to the
existing transform kernels. They intentionally do not read Black volatility
surfaces; a bumped surface is calibration evidence, not a substitute for a
Heston parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.processes.heston import (
    HestonRuntimeBinding,
    resolve_heston_runtime_binding,
)
from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price


@dataclass(frozen=True)
class ResolvedHestonTransformInputs:
    """Resolved inputs for Heston characteristic-function transform pricing."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    method: str
    fft_alpha: float
    fft_points: int
    fft_eta: float
    cos_points: int
    cos_truncation: float
    runtime_binding: HestonRuntimeBinding
    characteristic_family: str = "heston_log_spot"
    validation_bundle: str = "heston:transform"


@dataclass(frozen=True)
class HestonTransformResult:
    """Structured Heston transform-pricing result."""

    price: float
    method: str
    maturity: float
    model_parameters: dict[str, object]
    runtime_binding: dict[str, object]
    characteristic_family: str
    validation_bundle: str
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


class UnsupportedHestonTransformMethod(NotImplementedError):
    """Raised when a Heston transform target lacks a checked kernel."""

    def __init__(self, method: object) -> None:
        self.method = str(method or "").strip() or "<unspecified>"
        self.repair_packet = heston_transform_capability_packet(self.method)
        super().__init__(self.repair_packet["summary"])


def resolve_heston_transform_inputs(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> ResolvedHestonTransformInputs:
    """Resolve market, contract, and model inputs for Heston transform pricing."""
    settlement = getattr(market_state, "settlement", None) or getattr(
        market_state,
        "as_of",
        None,
    )
    if settlement is None:
        raise ValueError(
            "market_state must provide settlement or as_of for Heston transform pricing"
        )
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = _resolve_maturity(settlement, spec, day_count)
    if getattr(market_state, "discount", None) is None and maturity > 0.0:
        raise ValueError("Heston transform pricing requires market_state.discount")

    rate = (
        0.0
        if maturity <= 0.0
        else float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    )
    dividend_yield = _resolve_dividend_yield(spec)
    runtime_mu = float(mu) if mu is not None else rate - dividend_yield
    spec_parameters = _resolve_spec_heston_parameters(spec)
    runtime_binding = resolve_heston_runtime_binding(
        market_state,
        mu=runtime_mu,
        kappa=spec_parameters.get("kappa"),
        theta=spec_parameters.get("theta"),
        xi=spec_parameters.get("xi"),
        rho=spec_parameters.get("rho"),
        v0=spec_parameters.get("v0"),
        parameter_set_name=parameter_set_name,
    )
    resolved_method = _normalized_heston_transform_method(
        method
        if method is not None
        else getattr(spec, "transform_method", getattr(spec, "method", "fft"))
    )

    spot = _resolve_spot(market_state, spec)
    strike = _resolve_strike(spec, spot=spot)
    notional_value = getattr(spec, "notional", 1.0)
    notional = 1.0 if notional_value is None else float(notional_value)

    return ResolvedHestonTransformInputs(
        notional=notional,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=normalized_option_type(getattr(spec, "option_type", "call")),
        method=resolved_method,
        fft_alpha=float(fft_alpha if fft_alpha is not None else getattr(spec, "fft_alpha", 1.5)),
        fft_points=max(
            int(fft_points if fft_points is not None else getattr(spec, "fft_points", 4096)),
            32,
        ),
        fft_eta=float(fft_eta if fft_eta is not None else getattr(spec, "fft_eta", 0.25)),
        cos_points=max(
            int(cos_points if cos_points is not None else getattr(spec, "cos_points", 256)),
            16,
        ),
        cos_truncation=float(
            cos_truncation
            if cos_truncation is not None
            else getattr(spec, "cos_truncation", 10.0)
        ),
        runtime_binding=runtime_binding,
    )


def price_heston_option_transform_result(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> HestonTransformResult:
    """Return a structured Heston transform price result for a European option."""
    resolved = resolve_heston_transform_inputs(
        market_state,
        spec,
        method=method,
        fft_alpha=fft_alpha,
        fft_points=fft_points,
        fft_eta=fft_eta,
        cos_points=cos_points,
        cos_truncation=cos_truncation,
        mu=mu,
        parameter_set_name=parameter_set_name,
    )

    if resolved.maturity <= 0.0:
        raw_price = terminal_intrinsic(
            resolved.option_type,
            spot=resolved.spot,
            strike=resolved.strike,
        )
    elif resolved.method == "fft":
        call_price = fft_price(
            _heston_log_spot_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            alpha=resolved.fft_alpha,
            N=resolved.fft_points,
            eta=resolved.fft_eta,
        )
        raw_price = (
            call_price
            if resolved.option_type == "call"
            else _put_from_call_parity(call_price, resolved)
        )
    elif resolved.method == "cos":
        raw_price = cos_price(
            _heston_log_ratio_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            N=resolved.cos_points,
            L=resolved.cos_truncation,
            option_type=resolved.option_type,
        )
    else:
        raise UnsupportedHestonTransformMethod(resolved.method)

    runtime_payload = resolved.runtime_binding.to_payload()
    return HestonTransformResult(
        price=float(resolved.notional) * float(raw_price),
        method=resolved.method,
        maturity=resolved.maturity,
        model_parameters=dict(resolved.runtime_binding.model_parameters),
        runtime_binding=runtime_payload,
        characteristic_family=resolved.characteristic_family,
        validation_bundle=resolved.validation_bundle,
        warnings=tuple(resolved.runtime_binding.warnings),
        assumptions=tuple(resolved.runtime_binding.assumptions),
    )


def price_heston_option_transform(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> float:
    """Return a scalar Heston transform price for a European option."""
    return float(
        price_heston_option_transform_result(
            market_state,
            spec,
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
            mu=mu,
            parameter_set_name=parameter_set_name,
        ).price
    )


def heston_transform_capability_packet(method: object = "gauss_laguerre") -> dict[str, Any]:
    """Return a machine-readable repair packet for an unsupported transform target."""
    normalized = str(method or "").strip().lower().replace("-", "_") or "unknown"
    return {
        "packet_type": "missing_heston_gauss_laguerre_transform_kernel",
        "summary": (
            "Heston Gauss-Laguerre transform pricing needs a checked quadrature "
            "kernel before route binding can admit the target."
        ),
        "missing_primitive": "heston_gauss_laguerre_transform_kernel",
        "unsupported_class": "heston_gauss_laguerre_transform",
        "suggested_action": "open_remediation_packet",
        "evidence": [f"method={normalized}"],
    }


def _heston_log_spot_char_fn(resolved: ResolvedHestonTransformInputs):
    process = resolved.runtime_binding.process
    log_spot = raw_np.log(resolved.spot)

    def phi(u):
        return process.characteristic_function(
            u,
            resolved.maturity,
            log_spot=float(log_spot),
        )

    return phi


def _heston_log_ratio_char_fn(resolved: ResolvedHestonTransformInputs):
    process = resolved.runtime_binding.process

    def phi(u):
        return process.characteristic_function(u, resolved.maturity, log_spot=0.0)

    return phi


def _put_from_call_parity(
    call_price: float,
    resolved: ResolvedHestonTransformInputs,
) -> float:
    discounted_strike = resolved.strike * raw_np.exp(-resolved.rate * resolved.maturity)
    discounted_spot = resolved.spot * raw_np.exp(-resolved.dividend_yield * resolved.maturity)
    return float(call_price - discounted_spot + discounted_strike)


def _resolve_dividend_yield(spec) -> float:
    for attr in ("dividend_yield", "continuous_dividend_yield", "q"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _resolve_maturity(settlement: date, spec, day_count) -> float:
    """Resolve maturity from common date aliases or a direct year fraction."""
    for attr in (
        "expiry_date",
        "maturity_date",
        "expiration_date",
        "exercise_date",
        "expiry",
        "maturity",
        "expiry_years",
        "maturity_years",
        "time_to_maturity",
        "tenor_years",
    ):
        value = getattr(spec, attr, None)
        if value is None:
            continue
        if isinstance(value, date):
            return max(float(year_fraction(settlement, value, day_count)), 0.0)
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value.strip())
            except ValueError:
                parsed = None
            if parsed is not None:
                return max(float(year_fraction(settlement, parsed, day_count)), 0.0)
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            continue

    for attr in ("surface_maturities", "maturity_dates", "expiry_dates"):
        values = getattr(spec, attr, None)
        if not values:
            continue
        for value in _iter_alias_values(values):
            if isinstance(value, date):
                return max(float(year_fraction(settlement, value, day_count)), 0.0)
            if isinstance(value, str):
                try:
                    parsed = date.fromisoformat(value.strip())
                except ValueError:
                    parsed = None
                if parsed is not None:
                    return max(float(year_fraction(settlement, parsed, day_count)), 0.0)
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
    if _is_surface_shaped_spec(spec):
        return 1.0
    raise ValueError(
        "spec must provide expiry_date, maturity_date, numeric maturity, or surface_maturities for Heston transform pricing"
    )


def _resolve_spot(market_state, spec) -> float:
    """Resolve the underlier spot from spec aliases or the market state."""
    for attr in ("spot", "underlier_spot", "s0"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    value = getattr(market_state, "spot", None)
    if value is not None:
        return float(value)
    underlier_spots = getattr(market_state, "underlier_spots", None)
    if isinstance(underlier_spots, dict) and len(underlier_spots) == 1:
        return float(next(iter(underlier_spots.values())))
    raise ValueError("spec or market_state must provide spot for Heston transform pricing")


def _resolve_strike(spec, *, spot: float) -> float:
    """Resolve a scalar strike, selecting a representative ATM/grid strike when needed."""
    for attr in ("strike", "strike_price", "k"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)

    for attr in ("surface_strikes", "strike_grid", "strikes"):
        values = getattr(spec, attr, None)
        if not values:
            continue
        numeric = []
        for value in _iter_alias_values(values):
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if numeric:
            return min(numeric, key=lambda strike: abs(strike - spot))

    if any(
        getattr(spec, attr, None)
        for attr in (
            "surface_maturities",
            "maturity_dates",
            "expiry_dates",
            "compute_implied_vol_surface",
            "implied_vol_surface_extraction",
        )
    ):
        return float(spot)

    if _is_surface_shaped_spec(spec):
        return float(spot)

    raise ValueError("spec must provide strike or a numeric strike grid for Heston transform pricing")


def _is_surface_shaped_spec(spec) -> bool:
    """Return whether a task spec represents a surface/smile workflow."""
    spec_type = type(spec)
    haystack = " ".join(
        str(part or "")
        for part in (
            spec_type.__name__,
            getattr(spec_type, "__doc__", ""),
            getattr(spec, "description", ""),
        )
    ).lower()
    return any(token in haystack for token in ("surface", "smile", "implied vol"))


def _iter_alias_values(values) -> tuple[object, ...]:
    """Return scalar or iterable alias payloads as a normalized tuple."""
    if values is None:
        return ()
    if isinstance(values, str):
        pieces = tuple(piece.strip() for piece in values.split(",") if piece.strip())
        return pieces or (values,)
    try:
        iterator = iter(values)
    except TypeError:
        return (values,)
    return tuple(iterator)


def _resolve_spec_heston_parameters(spec) -> dict[str, float]:
    """Return explicit Heston parameters declared on a task spec, if present."""
    aliases = {
        "kappa": ("kappa", "mean_reversion"),
        "theta": ("theta", "long_run_variance", "long_variance"),
        "xi": ("xi", "vol_of_vol", "variance_vol", "sigma"),
        "rho": ("rho", "correlation"),
        "v0": ("v0", "initial_variance", "initial_var", "variance0"),
    }
    resolved: dict[str, float] = {}
    for canonical, names in aliases.items():
        for name in names:
            value = getattr(spec, name, None)
            if value is None:
                continue
            resolved[canonical] = float(value)
            break
    return resolved


def _normalized_heston_transform_method(value: object) -> str:
    method = str(value or "fft").strip().lower().replace("-", "_")
    aliases = {
        "carr_madan": "fft",
        "fft_heston": "fft",
        "heston_fft": "fft",
        "heston_analytical": "fft",
        "analytical_heston": "fft",
        "semi_analytical_heston": "fft",
        "fang_oosterlee": "cos",
        "cos_heston": "cos",
        "heston_cos": "cos",
        "laguerre": "gauss_laguerre",
        "laguerre_heston": "gauss_laguerre",
        "heston_laguerre": "gauss_laguerre",
        "gauss_laguerre_heston": "gauss_laguerre",
        "heston_gauss_laguerre": "gauss_laguerre",
    }
    method = aliases.get(method, method)
    if method in {"fft", "cos"}:
        return method
    if method == "gauss_laguerre":
        raise UnsupportedHestonTransformMethod(method)
    raise ValueError(f"Unsupported Heston transform method {value!r}")


__all__ = [
    "HestonTransformResult",
    "ResolvedHestonTransformInputs",
    "UnsupportedHestonTransformMethod",
    "heston_transform_capability_packet",
    "price_heston_option_transform",
    "price_heston_option_transform_result",
    "resolve_heston_transform_inputs",
]
