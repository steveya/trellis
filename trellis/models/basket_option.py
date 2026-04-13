"""Typed helper-backed basket option pricing surfaces."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from datetime import date
import math
from typing import Protocol

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.runtime_contract import wrap_market_state_with_contract
from trellis.core.types import DayCountConvention
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.basket_semantics import (
    ResolvedBasketSemantics,
    resolve_basket_semantics,
)

np = get_numpy()


class BasketOptionSpecLike(Protocol):
    """Minimal basket-option spec surface consumed by the typed helpers."""

    notional: float
    underliers: str
    strike: float
    expiry_date: date
    correlation: str
    weights: str | None
    spots: str | None
    vols: str | None
    dividend_yields: str | None
    basket_style: str
    option_type: str
    day_count: DayCountConvention


@dataclass(frozen=True)
class ResolvedBasketOptionInputs:
    """Typed market and payoff inputs for generic basket-option helpers."""

    semantics: ResolvedBasketSemantics
    weights: tuple[float, ...]
    basket_style: str
    option_type: str
    strike: float
    comparison_target: str | None = None

    @property
    def notional_spots(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.semantics.constituent_spots)

    @property
    def vols(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.semantics.constituent_vols)

    @property
    def carry(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.semantics.constituent_carry)

    @property
    def correlation_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(cell) for cell in row) for row in self.semantics.correlation_matrix)

def resolve_basket_option_inputs(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
) -> ResolvedBasketOptionInputs:
    """Resolve generic basket-option market inputs from typed state and spec fields."""
    target = str(comparison_target or "").strip().lower() or None
    underliers = _parse_name_vector(getattr(spec, "underliers", None))
    if not underliers:
        spots = tuple((market_state.underlier_spots or {}).keys())
        underliers = tuple(str(item) for item in spots[:2])
    if len(underliers) < 2:
        raise ValueError("Basket option helpers require at least two underliers")

    correlation_source = _correlation_source_descriptor(
        getattr(spec, "correlation", None),
        n_assets=len(underliers),
    )
    market_state_for_resolution = market_state
    market_state_updates: dict[str, object] = {}
    if correlation_source is not None:
        model_parameters = dict(getattr(market_state, "model_parameters", None) or {})
        model_parameters["correlation_source"] = correlation_source
        market_state_updates["model_parameters"] = model_parameters
    # Generic basket proof lanes should read the shocked plain vol surface when
    # it exists; unrelated local-vol overlays in the synthetic snapshot should
    # not suppress volatility sensitivity checks for analytical/MC/FFT helpers.
    if (
        getattr(market_state, "vol_surface", None) is not None
        and getattr(market_state, "local_vol_surface", None) is not None
    ):
        market_state_updates["local_vol_surface"] = None
        market_state_updates["local_vol_surfaces"] = {}
    if market_state_updates:
        market_state_for_resolution = _replace_market_state_like(
            market_state,
            **market_state_updates,
        )
    semantics = resolve_basket_semantics(
        market_state_for_resolution,
        constituents=",".join(underliers),
        strike=float(getattr(spec, "strike", 0.0)),
        expiry_date=getattr(spec, "expiry_date"),
        option_type=getattr(spec, "option_type", "call"),
        day_count=getattr(spec, "day_count", None) or DayCountConvention.ACT_365,
    )

    spot_override = _parse_float_vector(getattr(spec, "spots", None), expected=len(underliers))
    vol_override = _parse_float_vector(getattr(spec, "vols", None), expected=len(underliers))
    carry_override = _parse_float_vector(
        getattr(spec, "dividend_yields", None),
        expected=len(underliers),
    )
    if spot_override is not None:
        semantics = replace(semantics, constituent_spots=spot_override)
    if vol_override is not None:
        semantics = replace(semantics, constituent_vols=vol_override)
    if carry_override is not None:
        semantics = replace(semantics, constituent_carry=carry_override)

    basket_style = _normalized_basket_style(
        getattr(spec, "basket_style", None),
        comparison_target=target,
    )
    weights = _resolve_basket_weights(
        getattr(spec, "weights", None),
        expected=len(underliers),
        basket_style=basket_style,
    )
    option_type = _normalized_option_type(getattr(spec, "option_type", "call"))
    return ResolvedBasketOptionInputs(
        semantics=semantics,
        weights=weights,
        basket_style=basket_style,
        option_type=option_type,
        strike=float(getattr(spec, "strike", 0.0)),
        comparison_target=target,
    )


def _replace_market_state_like(market_state: object, **updates):
    """Return a market-state clone while preserving runtime-contract wrappers."""
    if is_dataclass(market_state):
        return replace(market_state, **updates)

    raw_market_state = getattr(market_state, "raw_market_state", None)
    if raw_market_state is not None and is_dataclass(raw_market_state):
        replaced = replace(raw_market_state, **updates)
        return wrap_market_state_with_contract(
            replaced,
            requirements=getattr(market_state, "_requirements", ()),
            context=str(getattr(market_state, "_context", "") or ""),
        )

    if hasattr(market_state, "__dict__"):
        payload = dict(vars(market_state))
        payload.update(updates)
        market_state_type = type(market_state)
        try:
            return market_state_type(**payload)
        except Exception:
            pass

    raise TypeError("basket option helper could not clone market state for correlation override")


def price_basket_option_analytical(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
) -> float:
    """Price a two-asset rainbow or spread option through typed basket inputs."""
    resolved = resolve_basket_option_inputs(
        market_state,
        spec,
        comparison_target=comparison_target,
    )
    semantics = resolved.semantics
    n_assets = len(semantics.constituent_names)
    if n_assets != 2:
        raise ValueError("Analytical basket-option helper currently supports exactly two underliers")
    if semantics.T <= 0.0:
        intrinsic = _terminal_payoff(
            raw_np.asarray([semantics.constituent_spots], dtype=float),
            resolved,
        )[0]
        return float(getattr(spec, "notional", 1.0)) * float(intrinsic)

    if resolved.basket_style == "spread":
        return float(getattr(spec, "notional", 1.0)) * _price_two_asset_spread_kirk(resolved)

    return float(getattr(spec, "notional", 1.0)) * _price_two_asset_basket_quadrature(resolved)


def price_basket_option_monte_carlo(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
    n_paths: int | None = None,
    seed: int = 42,
) -> float:
    """Price a generic terminal basket option through typed multi-asset Monte Carlo."""
    resolved = resolve_basket_option_inputs(
        market_state,
        spec,
        comparison_target=comparison_target,
    )
    semantics = resolved.semantics
    if semantics.T <= 0.0:
        intrinsic = _terminal_payoff(
            raw_np.asarray([semantics.constituent_spots], dtype=float),
            resolved,
        )[0]
        return float(getattr(spec, "notional", 1.0)) * float(intrinsic)

    paths = int(
        n_paths
        or getattr(spec, "n_paths", None)
        or getattr(spec, "n_simulations", None)
        or 40_000
    )
    domestic_rate = _implied_zero_rate(float(semantics.domestic_df), float(semantics.T))
    process = CorrelatedGBM(
        mu=[domestic_rate for _ in semantics.constituent_names],
        sigma=list(resolved.vols),
        corr=[list(row) for row in resolved.correlation_matrix],
        dividend_yield=list(resolved.carry),
    )
    engine = MonteCarloEngine(
        process,
        n_paths=max(paths, 8_192),
        n_steps=1,
        seed=int(seed),
        method="exact",
    )

    def payoff_fn(simulated_paths):
        terminal = raw_np.asarray(simulated_paths[:, -1, :], dtype=float)
        return _terminal_payoff(terminal, resolved)

    result = engine.price(
        raw_np.asarray(semantics.constituent_spots, dtype=float),
        float(semantics.T),
        payoff_fn,
        discount_rate=domestic_rate,
        return_paths=False,
    )
    return float(getattr(spec, "notional", 1.0)) * float(result["price"])


def price_basket_option_transform_proxy(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
) -> float:
    """Stabilize transform-lane basket spreads through the typed spread kernel.

    The transform proof lane currently needs a deterministic typed implementation
    identity, not a free-form generated spread adapter. For two-asset spread
    tasks, reuse the same typed spread kernel as the analytical lane until a
    dedicated checked 2D transform helper exists.
    """
    return float(
        price_basket_option_analytical(
            market_state,
            spec,
            comparison_target=comparison_target or "fft_spread_2d",
        )
    )


def _price_two_asset_spread_kirk(resolved: ResolvedBasketOptionInputs) -> float:
    semantics = resolved.semantics
    F1, F2 = _forwards_from_resolved(semantics)
    strike = float(resolved.strike)
    if strike < 0.0:
        raise ValueError("Spread-option strike must be non-negative for Kirk pricing")
    denom = F2 + strike
    if denom <= 0.0:
        raise ValueError("Spread-option denominator must be positive for Kirk pricing")

    sigma1 = float(resolved.vols[0])
    sigma2 = float(resolved.vols[1])
    rho = float(resolved.correlation_matrix[0][1])
    ratio = F2 / denom
    effective_variance = (
        sigma1 * sigma1
        - 2.0 * rho * sigma1 * sigma2 * ratio
        + sigma2 * sigma2 * ratio * ratio
    )
    effective_vol = float(np.sqrt(max(effective_variance, 0.0)))
    if effective_vol <= 0.0 or semantics.T <= 0.0:
        intrinsic = max(F1 - denom, 0.0)
        if resolved.option_type == "put":
            intrinsic = max(denom - F1, 0.0)
        return float(semantics.domestic_df) * float(intrinsic)

    sqrt_t = float(np.sqrt(float(semantics.T)))
    d1 = (float(np.log(F1 / denom)) + 0.5 * effective_vol * effective_vol * float(semantics.T)) / (
        effective_vol * sqrt_t
    )
    d2 = d1 - effective_vol * sqrt_t
    call = float(
        semantics.domestic_df
        * (
            F1 * _normal_cdf(d1)
            - denom * _normal_cdf(d2)
        )
    )
    if resolved.option_type == "call":
        return call
    return float(call - float(semantics.domestic_df) * (F1 - denom))


def _price_two_asset_basket_quadrature(
    resolved: ResolvedBasketOptionInputs,
    *,
    n_points: int = 32,
) -> float:
    semantics = resolved.semantics
    if len(semantics.constituent_names) != 2:
        raise ValueError("Quadrature basket helper currently supports exactly two underliers")

    nodes, weights = raw_np.polynomial.hermite.hermgauss(int(n_points))
    z = raw_np.sqrt(2.0) * nodes
    rho = float(resolved.correlation_matrix[0][1])
    sqrt_one_minus_rho = raw_np.sqrt(max(1.0 - rho * rho, 0.0))
    s1, s2 = (float(value) for value in semantics.constituent_spots)
    v1, v2 = (float(value) for value in resolved.vols)
    q1, q2 = (float(value) for value in resolved.carry)
    r = _implied_zero_rate(float(semantics.domestic_df), float(semantics.T))
    drift1 = (r - q1 - 0.5 * v1 * v1) * float(semantics.T)
    drift2 = (r - q2 - 0.5 * v2 * v2) * float(semantics.T)
    scale1 = v1 * raw_np.sqrt(float(semantics.T))
    scale2 = v2 * raw_np.sqrt(float(semantics.T))

    expectation = 0.0
    for i, xi in enumerate(z):
        for j, yj in enumerate(z):
            n1 = xi
            n2 = rho * xi + sqrt_one_minus_rho * yj
            terminal = raw_np.array(
                [
                    [
                        s1 * raw_np.exp(drift1 + scale1 * n1),
                        s2 * raw_np.exp(drift2 + scale2 * n2),
                    ]
                ],
                dtype=float,
            )
            expectation += float(weights[i] * weights[j]) * float(
                _terminal_payoff(terminal, resolved)[0]
            )
    return float(semantics.domestic_df) * expectation / raw_np.pi


def _terminal_payoff(terminal, resolved: ResolvedBasketOptionInputs):
    terminal_arr = raw_np.asarray(terminal, dtype=float)
    style = resolved.basket_style
    if style == "best_of":
        basket = raw_np.max(terminal_arr, axis=-1)
    elif style == "worst_of":
        basket = raw_np.min(terminal_arr, axis=-1)
    elif style == "spread":
        basket = raw_np.dot(terminal_arr, raw_np.asarray(resolved.weights, dtype=float))
    else:
        basket = raw_np.dot(terminal_arr, raw_np.asarray(resolved.weights, dtype=float))
    strike = float(resolved.strike)
    if resolved.option_type == "put":
        return raw_np.maximum(strike - basket, 0.0)
    return raw_np.maximum(basket - strike, 0.0)


def _forwards_from_resolved(semantics: ResolvedBasketSemantics) -> tuple[float, float]:
    if len(semantics.constituent_spots) != 2:
        raise ValueError("Spread helper currently requires exactly two underliers")
    domestic_rate = _implied_zero_rate(float(semantics.domestic_df), float(semantics.T))
    forwards = []
    for spot, carry in zip(semantics.constituent_spots, semantics.constituent_carry, strict=True):
        forwards.append(float(spot) * float(raw_np.exp((domestic_rate - float(carry)) * float(semantics.T))))
    return float(forwards[0]), float(forwards[1])


def _implied_zero_rate(discount_factor: float, T: float) -> float:
    if T <= 0.0:
        return 0.0
    return float(-raw_np.log(max(discount_factor, 1e-16)) / T)


def _normalized_basket_style(
    value: object,
    *,
    comparison_target: str | None,
) -> str:
    target = str(comparison_target or "").strip().lower()
    if "rainbow" in target:
        return "best_of"
    if "spread" in target:
        return "spread"
    style = str(value or "weighted_sum").strip().lower()
    aliases = {
        "best_of_two": "best_of",
        "bestof": "best_of",
        "best": "best_of",
        "worstof": "worst_of",
        "worst": "worst_of",
    }
    return aliases.get(style, style)


def _normalized_option_type(value: object) -> str:
    option_type = str(value or "call").strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unsupported option_type {value!r}; expected 'call' or 'put'")
    return option_type


def _resolve_basket_weights(
    value: object,
    *,
    expected: int,
    basket_style: str,
) -> tuple[float, ...]:
    parsed = _parse_float_vector(value, expected=expected)
    if parsed is not None:
        return parsed
    if basket_style == "spread":
        if expected < 2:
            raise ValueError("Spread basket helper requires at least two underliers")
        return (1.0, -1.0) + tuple(0.0 for _ in range(expected - 2))
    return tuple(1.0 / float(expected) for _ in range(expected))


def _parse_name_vector(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def _parse_float_vector(
    value: object,
    *,
    expected: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (int, float)):
        if expected != 1:
            return tuple(float(value) for _ in range(expected))
        return (float(value),)
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        parsed = tuple(float(item) for item in items)
    else:
        parsed = tuple(float(item) for item in value)
    if len(parsed) != expected:
        raise ValueError(f"Expected {expected} numeric entries, got {len(parsed)}")
    return parsed


def _correlation_source_descriptor(value: object, *, n_assets: int):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return {"kind": "explicit", "value": float(value)}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ";" not in text and "," not in text:
            return {"kind": "explicit", "value": float(text)}
        rows = [
            [float(cell.strip()) for cell in row.split(",") if cell.strip()]
            for row in text.split(";")
            if row.strip()
        ]
        if len(rows) == 1 and len(rows[0]) == 1:
            return {"kind": "explicit", "value": rows[0][0]}
        matrix = tuple(tuple(row) for row in rows)
        if len(matrix) != n_assets or any(len(row) != n_assets for row in matrix):
            raise ValueError(
                f"Expected a {n_assets}x{n_assets} correlation matrix, got {len(matrix)} row(s)"
            )
        return {"kind": "explicit", "matrix": matrix}
    return value


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


__all__ = [
    "BasketOptionSpecLike",
    "ResolvedBasketOptionInputs",
    "price_basket_option_analytical",
    "price_basket_option_monte_carlo",
    "price_basket_option_transform_proxy",
    "resolve_basket_option_inputs",
]
