"""Bounded affine short-rate zero-coupon bond helpers.

This module owns plain zero-coupon bond pricing under one-factor affine
short-rate models. It is intentionally narrower than the existing
``zcb_option`` helpers: no embedded optionality, no coupons, and no calibration
plant. The task/runtime surface uses it for Vasicek/CIR analytical-vs-tree
comparisons where generated adapters should stay thin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import exp, sqrt
from typing import Protocol

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.resolution.short_rate_claims import (
    DiscountCurveLike,
    ShortRateClaimMarketStateLike,
    extract_short_rate_comparison_regime,
)


class ShortRateBondMarketStateLike(ShortRateClaimMarketStateLike, Protocol):
    """Market-state interface required by short-rate bond helpers."""


class ShortRateBondSpecLike(Protocol):
    """Spec fields consumed by the short-rate zero-coupon bond helpers."""

    notional: float
    maturity_date: date
    day_count: DayCountConvention


@dataclass(frozen=True)
class ResolvedShortRateBondInputs:
    """Resolved affine short-rate inputs for a zero-coupon bond."""

    model: str
    notional: float
    settlement: date
    maturity_date: date
    maturity_time: float
    initial_rate: float
    mean_reversion: float
    long_term_rate: float
    sigma: float
    discount_curve: DiscountCurveLike
    parameter_source: str


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _candidate_parameter_payload(
    market_state: ShortRateBondMarketStateLike,
    model: str,
) -> Mapping[str, object]:
    """Return the most specific model-parameter payload for ``model``."""
    normalized = str(model or "").strip().lower()
    direct = _as_mapping(getattr(market_state, "model_parameters", None))
    if isinstance(direct.get(normalized), Mapping):
        return _as_mapping(direct[normalized])
    family = str(direct.get("model_family") or direct.get("family") or "").strip().lower()
    if family == normalized:
        return direct
    stochastic_vol_keys = {"xi", "rho", "v0", "initial_variance", "variance0"}
    has_stochastic_vol_shape = bool(stochastic_vol_keys.intersection(direct))
    has_short_rate_shape = any(
        key in direct
        for key in (
            "a",
            "b",
            "mean_reversion",
            "r0",
            "initial_rate",
            "short_rate",
            "rate",
            "sigma",
            "volatility",
            "short_rate_vol",
        )
    )
    has_cir_shape = {"kappa", "theta", "sigma"}.issubset(direct)
    if not has_stochastic_vol_shape and (has_short_rate_shape or has_cir_shape):
        return direct

    sets = _as_mapping(getattr(market_state, "model_parameter_sets", None))
    preferred_keys = (
        normalized,
        f"{normalized}_validation",
        f"{normalized}_short_rate",
        "short_rate",
        "short_rate_validation",
    )
    for key in preferred_keys:
        candidate = sets.get(key)
        if isinstance(candidate, Mapping):
            return _as_mapping(candidate)
    for candidate in sets.values():
        payload = _as_mapping(candidate)
        family = str(payload.get("model_family") or payload.get("family") or "").strip().lower()
        if family == normalized:
            return payload
    return {}


def _first_float(mapping: Mapping[str, object], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def resolve_short_rate_bond_inputs(
    market_state: ShortRateBondMarketStateLike,
    spec: ShortRateBondSpecLike,
    *,
    model: str = "vasicek",
    initial_rate: float | None = None,
    mean_reversion: float | None = None,
    long_term_rate: float | None = None,
    sigma: float | None = None,
    allow_benchmark_defaults: bool = False,
) -> ResolvedShortRateBondInputs:
    """Resolve Vasicek/CIR zero-coupon bond inputs from market state and spec."""
    normalized_model = str(model or "vasicek").strip().lower().replace("-", "_")
    if normalized_model not in {"vasicek", "cir"}:
        raise ValueError(f"Unsupported short-rate bond model {model!r}")

    settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for short-rate bond pricing")
    discount_curve = getattr(market_state, "discount", None)
    if discount_curve is None:
        raise ValueError("short-rate bond pricing requires market_state.discount")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity_time = max(float(year_fraction(settlement, spec.maturity_date, day_count)), 0.0)
    lookup_time = max(maturity_time, 1e-6)
    params = _candidate_parameter_payload(market_state, normalized_model)
    comparison_regime = extract_short_rate_comparison_regime(market_state)

    resolved_initial_rate = (
        float(initial_rate)
        if initial_rate is not None
        else _first_float(params, ("r0", "initial_rate", "short_rate", "rate"))
    )
    if resolved_initial_rate is None:
        resolved_initial_rate = float(discount_curve.zero_rate(lookup_time))

    resolved_mean_reversion = (
        float(mean_reversion)
        if mean_reversion is not None
        else _first_float(params, ("a", "kappa", "mean_reversion", "speed"))
    )
    if resolved_mean_reversion is None and comparison_regime is not None:
        resolved_mean_reversion = float(comparison_regime.hull_white_mean_reversion)
    if resolved_mean_reversion is None:
        resolved_mean_reversion = 0.1

    resolved_long_term_rate = (
        float(long_term_rate)
        if long_term_rate is not None
        else _first_float(params, ("b", "theta", "long_term_rate", "long_term_mean", "mean_level"))
    )
    if resolved_long_term_rate is None and comparison_regime is not None:
        resolved_long_term_rate = float(comparison_regime.flat_discount_rate)
    if resolved_long_term_rate is None:
        resolved_long_term_rate = float(resolved_initial_rate)

    resolved_sigma = (
        float(sigma)
        if sigma is not None
        else _first_float(params, ("sigma", "volatility", "short_rate_vol", "flat_sigma"))
    )
    if resolved_sigma is None and comparison_regime is not None:
        resolved_sigma = float(comparison_regime.flat_sigma)
    if resolved_sigma is None and allow_benchmark_defaults:
        resolved_sigma = 0.01
    if resolved_sigma is None:
        raise ValueError("short-rate bond pricing requires model sigma, a short-rate comparison regime, or explicit benchmark defaults")

    source = "explicit_overrides" if any(
        value is not None for value in (initial_rate, mean_reversion, long_term_rate, sigma)
    ) else (
        "model_parameters"
        if params
        else ("comparison_regime" if comparison_regime is not None else "benchmark_defaults")
    )

    return ResolvedShortRateBondInputs(
        model=normalized_model,
        notional=float(getattr(spec, "notional", 1.0)),
        settlement=settlement,
        maturity_date=spec.maturity_date,
        maturity_time=float(maturity_time),
        initial_rate=float(resolved_initial_rate),
        mean_reversion=float(resolved_mean_reversion),
        long_term_rate=float(resolved_long_term_rate),
        sigma=float(resolved_sigma),
        discount_curve=discount_curve,
        parameter_source=source,
    )


def _mean_reversion_B(a: float, maturity: float) -> float:
    if abs(float(a)) < 1e-12:
        return float(maturity)
    return (1.0 - exp(-float(a) * float(maturity))) / float(a)


def _deterministic_short_rate_integral(inputs: ResolvedShortRateBondInputs) -> float:
    B = _mean_reversion_B(inputs.mean_reversion, inputs.maturity_time)
    return inputs.long_term_rate * inputs.maturity_time + (
        inputs.initial_rate - inputs.long_term_rate
    ) * B


def _vasicek_unit_zcb(inputs: ResolvedShortRateBondInputs) -> float:
    T = inputs.maturity_time
    if T <= 0.0:
        return 1.0
    a = inputs.mean_reversion
    sigma = max(inputs.sigma, 0.0)
    if sigma == 0.0 or abs(a) < 1e-12:
        return exp(-_deterministic_short_rate_integral(inputs))
    B = _mean_reversion_B(a, T)
    a2 = a * a
    log_A = (B - T) * (a2 * inputs.long_term_rate - 0.5 * sigma * sigma) / a2
    log_A -= (sigma * sigma * B * B) / (4.0 * a)
    return exp(log_A - B * inputs.initial_rate)


def _cir_unit_zcb(inputs: ResolvedShortRateBondInputs) -> float:
    T = inputs.maturity_time
    if T <= 0.0:
        return 1.0
    a = inputs.mean_reversion
    sigma = max(inputs.sigma, 0.0)
    if sigma == 0.0:
        return exp(-_deterministic_short_rate_integral(inputs))
    gamma = sqrt(max(a * a + 2.0 * sigma * sigma, 1e-24))
    exp_gamma_t = exp(gamma * T)
    denominator = (gamma + a) * (exp_gamma_t - 1.0) + 2.0 * gamma
    B = 2.0 * (exp_gamma_t - 1.0) / denominator
    A_base = 2.0 * gamma * exp((a + gamma) * T / 2.0) / denominator
    power = 2.0 * a * max(inputs.long_term_rate, 0.0) / (sigma * sigma)
    return (A_base ** power) * exp(-B * max(inputs.initial_rate, 0.0))


def price_short_rate_zero_coupon_bond_analytical(
    market_state: ShortRateBondMarketStateLike,
    spec: ShortRateBondSpecLike,
    *,
    model: str = "vasicek",
    initial_rate: float | None = None,
    mean_reversion: float | None = None,
    long_term_rate: float | None = None,
    sigma: float | None = None,
    allow_benchmark_defaults: bool = False,
) -> float:
    """Price a plain ZCB analytically under Vasicek or CIR."""
    inputs = resolve_short_rate_bond_inputs(
        market_state,
        spec,
        model=model,
        initial_rate=initial_rate,
        mean_reversion=mean_reversion,
        long_term_rate=long_term_rate,
        sigma=sigma,
        allow_benchmark_defaults=allow_benchmark_defaults,
    )
    unit_price = _cir_unit_zcb(inputs) if inputs.model == "cir" else _vasicek_unit_zcb(inputs)
    return float(inputs.notional * unit_price)


def price_vasicek_zero_coupon_bond_analytical(
    market_state: ShortRateBondMarketStateLike,
    spec: ShortRateBondSpecLike,
    **kwargs,
) -> float:
    """Price a plain ZCB analytically under the Vasicek model."""
    return price_short_rate_zero_coupon_bond_analytical(
        market_state,
        spec,
        model="vasicek",
        **kwargs,
    )


def price_cir_zero_coupon_bond_analytical(
    market_state: ShortRateBondMarketStateLike,
    spec: ShortRateBondSpecLike,
    **kwargs,
) -> float:
    """Price a plain ZCB analytically under the CIR model."""
    return price_short_rate_zero_coupon_bond_analytical(
        market_state,
        spec,
        model="cir",
        **kwargs,
    )


def _deterministic_mean(inputs: ResolvedShortRateBondInputs, t: float) -> float:
    return inputs.long_term_rate + (
        inputs.initial_rate - inputs.long_term_rate
    ) * exp(-inputs.mean_reversion * t)


def _transition_probabilities(
    *,
    current_rate: float,
    expected_rate: float,
    variance: float,
    next_mid_rate: float,
    dx: float,
) -> tuple[float, float, float]:
    if variance <= 1e-18 or dx <= 1e-12:
        offset = 0.0 if dx <= 1e-12 else (expected_rate - next_mid_rate) / dx
        if offset <= -0.5:
            return (1.0, 0.0, 0.0)
        if offset >= 0.5:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)
    z = (expected_rate - next_mid_rate) / dx
    v = max(float(variance) / (dx * dx), 0.0)
    p_down = 0.5 * (v + z * z - z)
    p_up = 0.5 * (v + z * z + z)
    p_mid = 1.0 - v - z * z
    probs = [max(0.0, p_down), max(0.0, p_mid), max(0.0, p_up)]
    total = sum(probs)
    if total <= 0.0:
        return (0.0, 1.0, 0.0)
    return tuple(float(p / total) for p in probs)  # type: ignore[return-value]


def price_short_rate_zero_coupon_bond_tree(
    market_state: ShortRateBondMarketStateLike,
    spec: ShortRateBondSpecLike,
    *,
    model: str = "vasicek",
    n_steps: int | None = None,
    initial_rate: float | None = None,
    mean_reversion: float | None = None,
    long_term_rate: float | None = None,
    sigma: float | None = None,
    allow_benchmark_defaults: bool = False,
) -> float:
    """Price a plain ZCB with a bounded recombining trinomial short-rate tree."""
    inputs = resolve_short_rate_bond_inputs(
        market_state,
        spec,
        model=model,
        initial_rate=initial_rate,
        mean_reversion=mean_reversion,
        long_term_rate=long_term_rate,
        sigma=sigma,
        allow_benchmark_defaults=allow_benchmark_defaults,
    )
    T = inputs.maturity_time
    if T <= 0.0:
        return float(inputs.notional)
    steps = int(n_steps or min(720, max(120, round(T * 36))))
    steps = max(2, steps)
    dt = T / steps
    dx = max(abs(inputs.sigma) * sqrt(3.0 * dt), 1e-5)
    centers = [_deterministic_mean(inputs, step * dt) for step in range(steps + 1)]

    values = [float(inputs.notional)] * (2 * steps + 1)
    for step in range(steps - 1, -1, -1):
        next_values = values
        current_values: list[float] = []
        for node in range(2 * step + 1):
            j = node - step
            raw_rate = centers[step] + j * dx
            rate = max(raw_rate, 0.0) if inputs.model == "cir" else raw_rate
            drift = inputs.mean_reversion * (inputs.long_term_rate - rate)
            expected = rate + drift * dt
            local_variance = (
                inputs.sigma * inputs.sigma * max(rate, 0.0) * dt
                if inputs.model == "cir"
                else inputs.sigma * inputs.sigma * dt
            )
            next_mid = centers[step + 1] + j * dx
            p_down, p_mid, p_up = _transition_probabilities(
                current_rate=rate,
                expected_rate=expected,
                variance=local_variance,
                next_mid_rate=next_mid,
                dx=dx,
            )
            discount = exp(-rate * dt)
            current_values.append(
                discount
                * (
                    p_down * next_values[node]
                    + p_mid * next_values[node + 1]
                    + p_up * next_values[node + 2]
                )
            )
        values = current_values
    return float(values[0])


__all__ = [
    "ResolvedShortRateBondInputs",
    "resolve_short_rate_bond_inputs",
    "price_short_rate_zero_coupon_bond_analytical",
    "price_vasicek_zero_coupon_bond_analytical",
    "price_cir_zero_coupon_bond_analytical",
    "price_short_rate_zero_coupon_bond_tree",
]
