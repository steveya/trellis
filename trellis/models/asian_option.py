"""Checked Asian-option Monte Carlo helper surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
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


class AsianOptionMonteCarloSpecLike(Protocol):
    """Broader spec surface used by compatibility wrappers and bounded helpers."""

    notional: float
    strike: float
    expiry_date: date


@dataclass(frozen=True)
class ArithmeticAsianOptionMonteCarloResult:
    """Structured result for the bounded arithmetic-Asian Monte Carlo lane."""

    price: float
    std_error: float
    n_paths: int
    seed: int | None
    observation_count: int


@dataclass(frozen=True)
class ArithmeticAsianOptionAnalyticalResult:
    """Structured result for the bounded arithmetic-Asian analytical lane."""

    price: float
    matched_mean: float
    matched_second_moment: float
    effective_lognormal_vol: float
    observation_count: int


@dataclass(frozen=True)
class AsianOptionMonteCarloResult:
    """Structured result for generic Asian Monte Carlo helper calls."""

    price: float
    std_error: float
    n_paths: int
    seed: int | None
    observation_count: int
    averaging_type: str


def _price_asian_option_monte_carlo_result(
    market_state: MarketState,
    spec: AsianOptionMonteCarloSpecLike,
) -> AsianOptionMonteCarloResult:
    """Price one bounded Asian option through exact GBM observation sampling."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Asian Monte Carlo helper requires market_state.settlement or as_of")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    strike = float(getattr(spec, "strike"))
    expiry_date = getattr(spec, "expiry_date")
    maturity = max(float(year_fraction(settlement, expiry_date, day_count)), 0.0)
    if market_state.discount is None:
        raise ValueError("Asian Monte Carlo helper requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Asian Monte Carlo helper requires market_state.vol_surface")

    averaging_type = str(getattr(spec, "averaging_type", "arithmetic") or "arithmetic").strip().lower()
    if averaging_type not in {"arithmetic", "geometric"}:
        raise ValueError(f"Unsupported Asian averaging_type {averaging_type!r}")

    observation_times = _resolve_observation_times(
        settlement=settlement,
        expiry_date=expiry_date,
        day_count=day_count,
        maturity=maturity,
        spec=spec,
    )
    if maturity <= 0.0:
        return AsianOptionMonteCarloResult(
            price=0.0,
            std_error=0.0,
            n_paths=0,
            seed=None if getattr(spec, "seed", None) is None else int(getattr(spec, "seed", 42)),
            observation_count=len(observation_times),
            averaging_type=averaging_type,
        )

    rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
    if sigma < 0.0:
        raise ValueError(f"Asian Monte Carlo helper requires non-negative volatility, got {sigma}")

    spot = _resolve_spot(market_state, spec)
    if spot <= 0.0:
        raise ValueError("Asian Monte Carlo helper requires strictly positive spot")

    option_type = normalized_option_type(getattr(spec, "option_type", "call"))
    dividend_yield = float(getattr(spec, "dividend_yield", 0.0) or 0.0)
    notional = float(getattr(spec, "notional", 1.0))
    n_paths = max(int(getattr(spec, "n_paths", 50_000) or 50_000), 1)
    seed = getattr(spec, "seed", 42)

    rng = raw_np.random.default_rng(None if seed is None else int(seed))
    log_levels = raw_np.full(n_paths, raw_np.log(spot), dtype=float)
    observed = raw_np.empty((n_paths, len(observation_times)), dtype=float)
    prev_time = 0.0
    drift = rate - dividend_yield - 0.5 * sigma * sigma
    positive_steps = [time for time in observation_times if time > 0.0]
    normals = raw_np.asarray(
        rng.standard_normal((n_paths, len(positive_steps))),
        dtype=float,
    )
    normal_idx = 0
    for idx, current_time in enumerate(observation_times):
        if current_time == 0.0:
            observed[:, idx] = spot
            continue
        dt = float(current_time - prev_time)
        if dt <= 0.0:
            raise ValueError("Asian Monte Carlo helper requires strictly increasing observation times")
        log_levels = log_levels + drift * dt + sigma * raw_np.sqrt(dt) * normals[:, normal_idx]
        observed[:, idx] = raw_np.exp(log_levels)
        prev_time = current_time
        normal_idx += 1

    if averaging_type == "geometric":
        average_value = raw_np.exp(raw_np.mean(raw_np.log(observed), axis=1))
    else:
        average_value = raw_np.mean(observed, axis=1)
    if option_type == "put":
        payoffs = raw_np.maximum(strike - average_value, 0.0)
    else:
        payoffs = raw_np.maximum(average_value - strike, 0.0)
    discounted = raw_np.exp(-rate * maturity) * notional * payoffs
    price = float(raw_np.mean(discounted))
    std_error = (
        0.0
        if n_paths <= 1
        else float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(n_paths))
    )
    return AsianOptionMonteCarloResult(
        price=price,
        std_error=std_error,
        n_paths=n_paths,
        seed=None if seed is None else int(seed),
        observation_count=len(observation_times),
        averaging_type=averaging_type,
    )


def price_asian_option_monte_carlo_result(
    market_state: MarketState,
    spec: AsianOptionMonteCarloSpecLike,
) -> AsianOptionMonteCarloResult:
    """Return a structured Monte Carlo price result for one bounded Asian option."""
    return _price_asian_option_monte_carlo_result(market_state, spec)


def price_arithmetic_asian_option_monte_carlo_result(
    market_state: MarketState,
    spec: ArithmeticAsianOptionSpecLike,
) -> ArithmeticAsianOptionMonteCarloResult:
    """Price one expiry-aligned arithmetic Asian through exact GBM date sampling."""
    result = _price_asian_option_monte_carlo_result(market_state, spec)
    return ArithmeticAsianOptionMonteCarloResult(
        price=result.price,
        std_error=result.std_error,
        n_paths=result.n_paths,
        seed=result.seed,
        observation_count=result.observation_count,
    )


def price_asian_option_monte_carlo(
    market_state: MarketState,
    spec: AsianOptionMonteCarloSpecLike,
) -> float:
    """Return the scalar bounded Asian-option Monte Carlo price."""
    return float(price_asian_option_monte_carlo_result(market_state, spec).price)


def price_arithmetic_asian_option_analytical_result(
    market_state: MarketState,
    spec: ArithmeticAsianOptionSpecLike,
) -> ArithmeticAsianOptionAnalyticalResult:
    """Price one bounded arithmetic Asian through discrete moment matching."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Arithmetic Asian analytical helper requires market_state.settlement or as_of")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    strike = float(getattr(spec, "strike"))
    expiry_date = getattr(spec, "expiry_date")
    maturity = max(float(year_fraction(settlement, expiry_date, day_count)), 0.0)
    if market_state.discount is None:
        raise ValueError("Arithmetic Asian analytical helper requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Arithmetic Asian analytical helper requires market_state.vol_surface")

    observation_times = _resolve_observation_times(
        settlement=settlement,
        expiry_date=expiry_date,
        day_count=day_count,
        maturity=maturity,
        spec=spec,
    )
    spot = _resolve_spot(market_state, spec)
    if spot <= 0.0:
        raise ValueError("Arithmetic Asian analytical helper requires strictly positive spot")

    notional = float(getattr(spec, "notional", 1.0))
    option_type = normalized_option_type(getattr(spec, "option_type", "call"))
    dividend_yield = float(getattr(spec, "dividend_yield", 0.0) or 0.0)
    rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
    if sigma < 0.0:
        raise ValueError(
            f"Arithmetic Asian analytical helper requires non-negative volatility, got {sigma}"
        )

    if maturity <= 0.0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return ArithmeticAsianOptionAnalyticalResult(
            price=notional * intrinsic,
            matched_mean=spot,
            matched_second_moment=spot * spot,
            effective_lognormal_vol=0.0,
            observation_count=len(observation_times),
        )

    matched_mean = _arithmetic_average_first_moment(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        observation_times=observation_times,
    )
    matched_second_moment = _arithmetic_average_second_moment(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        sigma=sigma,
        observation_times=observation_times,
    )
    effective_lognormal_var = max(
        math.log(max(matched_second_moment, 1e-300) / max(matched_mean * matched_mean, 1e-300)),
        0.0,
    )
    price = notional * _discounted_lognormal_option_price(
        matched_mean=matched_mean,
        matched_lognormal_var=effective_lognormal_var,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type=option_type,
    )
    return ArithmeticAsianOptionAnalyticalResult(
        price=price,
        matched_mean=matched_mean,
        matched_second_moment=matched_second_moment,
        effective_lognormal_vol=math.sqrt(effective_lognormal_var / maturity),
        observation_count=len(observation_times),
    )


def price_arithmetic_asian_option_analytical(
    market_state: MarketState,
    spec: ArithmeticAsianOptionSpecLike,
) -> float:
    """Return the scalar bounded arithmetic-Asian analytical approximation."""
    return float(price_arithmetic_asian_option_analytical_result(market_state, spec).price)


def price_arithmetic_asian_option_monte_carlo(
    market_state: MarketState,
    spec: ArithmeticAsianOptionSpecLike,
) -> float:
    """Return the scalar bounded arithmetic-Asian Monte Carlo price."""
    return float(price_arithmetic_asian_option_monte_carlo_result(market_state, spec).price)


def _resolve_spot(market_state: MarketState, spec: AsianOptionMonteCarloSpecLike) -> float:
    explicit_spot = getattr(spec, "spot", None)
    if explicit_spot is not None:
        return float(explicit_spot)
    underlier = str(getattr(spec, "underlier", "") or "").strip()
    spots = dict(getattr(market_state, "underlier_spots", None) or {})
    if underlier in spots:
        return float(spots[underlier])
    if getattr(market_state, "spot", None) is not None:
        return float(market_state.spot)
    raise ValueError(f"Asian Monte Carlo helper could not resolve spot for underlier {underlier!r}")


def _resolve_observation_times(
    *,
    settlement: date,
    expiry_date: date,
    day_count: DayCountConvention,
    maturity: float,
    spec: AsianOptionMonteCarloSpecLike,
) -> tuple[float, ...]:
    observation_dates = tuple(getattr(spec, "observation_dates", ()) or ())
    if observation_dates:
        if tuple(sorted(observation_dates)) != observation_dates:
            raise ValueError("Asian Monte Carlo helper requires strictly increasing observation dates")
        if observation_dates[-1] != expiry_date:
            raise ValueError("Asian Monte Carlo helper requires the final observation to equal expiry_date")
        observation_times = tuple(
            float(year_fraction(settlement, observation_date, day_count))
            for observation_date in observation_dates
        )
        if any(observation_time <= 0.0 for observation_time in observation_times):
            raise ValueError(
                "Asian Monte Carlo helper currently requires all explicit observation dates to be after settlement"
            )
        return observation_times

    n_observations = int(getattr(spec, "n_observations", 0) or 0)
    if n_observations <= 0:
        raise ValueError("Asian Monte Carlo helper requires observation_dates or a positive n_observations")
    if n_observations == 1:
        return (0.0,)
    return tuple(float(value) for value in raw_np.linspace(0.0, maturity, n_observations))


def _arithmetic_average_first_moment(
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    observation_times: tuple[float, ...],
) -> float:
    count = len(observation_times)
    if count == 0:
        raise ValueError("Arithmetic Asian helper requires at least one observation time")
    growth = raw_np.exp((rate - dividend_yield) * raw_np.asarray(observation_times, dtype=float))
    return float(spot * raw_np.mean(growth))


def _arithmetic_average_second_moment(
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    observation_times: tuple[float, ...],
) -> float:
    count = len(observation_times)
    if count == 0:
        raise ValueError("Arithmetic Asian helper requires at least one observation time")
    total = 0.0
    drift = rate - dividend_yield
    for left_time in observation_times:
        for right_time in observation_times:
            total += math.exp(
                drift * (left_time + right_time) + sigma * sigma * min(left_time, right_time)
            )
    return spot * spot * total / float(count * count)


def _discounted_lognormal_option_price(
    *,
    matched_mean: float,
    matched_lognormal_var: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str,
) -> float:
    discount = math.exp(-rate * maturity)
    if strike <= 0.0:
        if option_type == "put":
            return 0.0
        return discount * (matched_mean - strike)
    if matched_lognormal_var <= 1e-14:
        intrinsic = max(matched_mean - strike, 0.0)
        if option_type == "put":
            intrinsic = max(strike - matched_mean, 0.0)
        return discount * intrinsic
    std = math.sqrt(matched_lognormal_var)
    d1 = (math.log(matched_mean / strike) + 0.5 * matched_lognormal_var) / std
    d2 = d1 - std
    if option_type == "put":
        return discount * (strike * _normal_cdf(-d2) - matched_mean * _normal_cdf(-d1))
    return discount * (matched_mean * _normal_cdf(d1) - strike * _normal_cdf(d2))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


__all__ = [
    "AsianOptionMonteCarloResult",
    "AsianOptionMonteCarloSpecLike",
    "ArithmeticAsianOptionAnalyticalResult",
    "ArithmeticAsianOptionMonteCarloResult",
    "ArithmeticAsianOptionSpecLike",
    "price_arithmetic_asian_option_analytical",
    "price_arithmetic_asian_option_analytical_result",
    "price_asian_option_monte_carlo",
    "price_asian_option_monte_carlo_result",
    "price_arithmetic_asian_option_monte_carlo",
    "price_arithmetic_asian_option_monte_carlo_result",
]
