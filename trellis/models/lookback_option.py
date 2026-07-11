"""Checked fixed-strike lookback option helper surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention


class FixedLookbackOptionSpecLike(Protocol):
    """Spec fields consumed by the fixed-strike lookback helpers."""

    notional: float
    spot: float
    strike: float
    expiry_date: date


@dataclass(frozen=True)
class FixedLookbackMonteCarloResult:
    """Structured result for fixed-strike lookback Monte Carlo pricing."""

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    seed: int | None


def price_equity_fixed_lookback_option_monte_carlo_result(
    market_state: MarketState,
    spec: FixedLookbackOptionSpecLike,
) -> FixedLookbackMonteCarloResult:
    """Price a fixed-strike equity lookback option with GBM Monte Carlo."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Lookback Monte Carlo requires market_state.settlement or as_of")
    if market_state.discount is None:
        raise ValueError("Lookback Monte Carlo requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Lookback Monte Carlo requires market_state.vol_surface")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    notional = float(getattr(spec, "notional", 1.0) or 1.0)
    spot = float(getattr(spec, "spot"))
    strike = float(getattr(spec, "strike"))
    option_type = str(getattr(spec, "option_type", "call") or "call").strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unsupported option_type {option_type!r}")
    lookback_type = str(getattr(spec, "lookback_type", "fixed_strike") or "fixed_strike").strip().lower()
    if lookback_type != "fixed_strike":
        raise ValueError(f"Unsupported lookback_type {lookback_type!r}")
    if spot <= 0.0:
        raise ValueError("Lookback Monte Carlo requires strictly positive spot")

    running_extreme = getattr(spec, "running_extreme", None)
    if running_extreme is None:
        running_extreme = spot
    running_extreme = float(running_extreme)
    if running_extreme <= 0.0:
        raise ValueError("Lookback Monte Carlo requires positive running_extreme")

    if maturity <= 0.0:
        if option_type == "put":
            payoff = max(strike - min(running_extreme, spot), 0.0)
        else:
            payoff = max(max(running_extreme, spot) - strike, 0.0)
        return FixedLookbackMonteCarloResult(
            price=notional * payoff,
            std_error=0.0,
            n_paths=0,
            n_steps=0,
            seed=None if getattr(spec, "seed", None) is None else int(getattr(spec, "seed")),
        )

    rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
    if sigma < 0.0:
        raise ValueError(f"Lookback Monte Carlo requires non-negative volatility, got {sigma}")
    n_paths = max(int(getattr(spec, "n_paths", 80_000) or 80_000), 1)
    n_steps = max(int(getattr(spec, "n_steps", 96) or 96), 1)
    seed = getattr(spec, "seed", 42)

    rng = raw_np.random.default_rng(None if seed is None else int(seed))
    dt = maturity / float(n_steps)
    variance_dt = sigma * sigma * dt
    drift = (rate - 0.5 * sigma * sigma) * dt
    log_spot = raw_np.full(n_paths, raw_np.log(spot), dtype=float)
    if option_type == "put":
        log_extreme = raw_np.full(n_paths, raw_np.log(min(running_extreme, spot)), dtype=float)
    else:
        log_extreme = raw_np.full(n_paths, raw_np.log(max(running_extreme, spot)), dtype=float)

    for _step in range(n_steps):
        previous = log_spot
        normals = rng.standard_normal(n_paths)
        current = previous + drift + sigma * raw_np.sqrt(dt) * normals
        if sigma > 0.0 and dt > 0.0:
            uniforms = raw_np.clip(rng.random(n_paths), 1e-16, 1.0)
            bridge_span = raw_np.sqrt((current - previous) ** 2 - 2.0 * variance_dt * raw_np.log(uniforms))
            if option_type == "put":
                bridge_extreme = 0.5 * (previous + current - bridge_span)
                log_extreme = raw_np.minimum(log_extreme, bridge_extreme)
            else:
                bridge_extreme = 0.5 * (previous + current + bridge_span)
                log_extreme = raw_np.maximum(log_extreme, bridge_extreme)
        else:
            if option_type == "put":
                log_extreme = raw_np.minimum(log_extreme, raw_np.minimum(previous, current))
            else:
                log_extreme = raw_np.maximum(log_extreme, raw_np.maximum(previous, current))
        log_spot = current

    extreme = raw_np.exp(log_extreme)
    if option_type == "put":
        payoffs = raw_np.maximum(strike - extreme, 0.0)
    else:
        payoffs = raw_np.maximum(extreme - strike, 0.0)
    discounted = raw_np.exp(-rate * maturity) * notional * payoffs
    price = float(raw_np.mean(discounted))
    std_error = 0.0 if n_paths <= 1 else float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(n_paths))
    return FixedLookbackMonteCarloResult(
        price=price,
        std_error=std_error,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=None if seed is None else int(seed),
    )


def price_equity_fixed_lookback_option_monte_carlo(
    market_state: MarketState,
    spec: FixedLookbackOptionSpecLike,
) -> float:
    """Return the scalar fixed-strike equity lookback Monte Carlo price."""
    return float(price_equity_fixed_lookback_option_monte_carlo_result(market_state, spec).price)


__all__ = [
    "FixedLookbackMonteCarloResult",
    "FixedLookbackOptionSpecLike",
    "price_equity_fixed_lookback_option_monte_carlo",
    "price_equity_fixed_lookback_option_monte_carlo_result",
]
