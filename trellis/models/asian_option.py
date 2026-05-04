"""Checked arithmetic-Asian Monte Carlo helper surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type


class ArithmeticAsianOptionSpecLike(Protocol):
    """Minimal spec surface for the bounded arithmetic-Asian helper."""

    notional: float
    underlier: str
    strike: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str
    day_count: DayCountConvention
    dividend_yield: float
    n_paths: int
    seed: int | None


@dataclass(frozen=True)
class ArithmeticAsianOptionMonteCarloResult:
    """Structured result for the bounded arithmetic-Asian Monte Carlo lane."""

    price: float
    std_error: float
    n_paths: int
    seed: int | None
    observation_count: int


def price_arithmetic_asian_option_monte_carlo_result(
    market_state: MarketState,
    spec: ArithmeticAsianOptionSpecLike,
) -> ArithmeticAsianOptionMonteCarloResult:
    """Price one expiry-aligned arithmetic Asian through exact GBM date sampling."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Arithmetic Asian helper requires market_state.settlement or as_of")

    observation_dates = tuple(getattr(spec, "observation_dates", ()) or ())
    if not observation_dates:
        raise ValueError("Arithmetic Asian helper requires at least one observation date")
    if tuple(sorted(observation_dates)) != observation_dates:
        raise ValueError("Arithmetic Asian helper requires strictly increasing observation dates")
    if observation_dates[-1] != getattr(spec, "expiry_date"):
        raise ValueError("Arithmetic Asian helper requires the final observation to equal expiry_date")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    observation_times = tuple(
        float(year_fraction(settlement, observation_date, day_count))
        for observation_date in observation_dates
    )
    if any(observation_time <= 0.0 for observation_time in observation_times):
        raise ValueError(
            "Arithmetic Asian helper currently requires all observation dates to be after settlement"
        )

    strike = float(getattr(spec, "strike"))
    maturity = float(observation_times[-1])
    if market_state.discount is None:
        raise ValueError("Arithmetic Asian helper requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Arithmetic Asian helper requires market_state.vol_surface")
    rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
    if sigma < 0.0:
        raise ValueError(f"Arithmetic Asian helper requires non-negative volatility, got {sigma}")

    underlier = str(getattr(spec, "underlier", "")).strip()
    if not underlier:
        raise ValueError("Arithmetic Asian helper requires a non-empty underlier id")
    spot = _resolve_spot(market_state, underlier)
    if spot <= 0.0:
        raise ValueError("Arithmetic Asian helper requires strictly positive spot")

    option_type = normalized_option_type(getattr(spec, "option_type", "call"))
    dividend_yield = float(getattr(spec, "dividend_yield", 0.0) or 0.0)
    notional = float(getattr(spec, "notional", 1.0))
    n_paths = max(int(getattr(spec, "n_paths", 50_000) or 50_000), 1)
    seed = getattr(spec, "seed", 42)

    rng = raw_np.random.default_rng(None if seed is None else int(seed))
    normals = raw_np.asarray(
        rng.standard_normal((n_paths, len(observation_times))),
        dtype=float,
    )
    log_levels = raw_np.full(n_paths, raw_np.log(spot), dtype=float)
    observed = raw_np.empty((n_paths, len(observation_times)), dtype=float)
    prev_time = 0.0
    drift = rate - dividend_yield - 0.5 * sigma * sigma
    for idx, current_time in enumerate(observation_times):
        dt = float(current_time - prev_time)
        if dt <= 0.0:
            raise ValueError("Arithmetic Asian helper requires strictly increasing observation times")
        log_levels = log_levels + drift * dt + sigma * raw_np.sqrt(dt) * normals[:, idx]
        observed[:, idx] = raw_np.exp(log_levels)
        prev_time = current_time

    arithmetic_average = raw_np.mean(observed, axis=1)
    if option_type == "put":
        payoffs = raw_np.maximum(strike - arithmetic_average, 0.0)
    else:
        payoffs = raw_np.maximum(arithmetic_average - strike, 0.0)
    discounted = raw_np.exp(-rate * maturity) * notional * payoffs
    price = float(raw_np.mean(discounted))
    std_error = (
        0.0
        if n_paths <= 1
        else float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(n_paths))
    )
    return ArithmeticAsianOptionMonteCarloResult(
        price=price,
        std_error=std_error,
        n_paths=n_paths,
        seed=None if seed is None else int(seed),
        observation_count=len(observation_dates),
    )


def price_arithmetic_asian_option_monte_carlo(
    market_state: MarketState,
    spec: ArithmeticAsianOptionSpecLike,
) -> float:
    """Return the scalar bounded arithmetic-Asian Monte Carlo price."""
    return float(price_arithmetic_asian_option_monte_carlo_result(market_state, spec).price)


def _resolve_spot(market_state: MarketState, underlier: str) -> float:
    spots = dict(getattr(market_state, "underlier_spots", None) or {})
    if underlier in spots:
        return float(spots[underlier])
    if getattr(market_state, "spot", None) is not None:
        return float(market_state.spot)
    raise ValueError(f"Arithmetic Asian helper could not resolve spot for underlier {underlier!r}")


__all__ = [
    "ArithmeticAsianOptionMonteCarloResult",
    "ArithmeticAsianOptionSpecLike",
    "price_arithmetic_asian_option_monte_carlo",
    "price_arithmetic_asian_option_monte_carlo_result",
]
