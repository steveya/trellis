"""Variance-swap pricing helpers for deterministic task adapters.

The analytical helper in :mod:`trellis.models.analytical.equity_exotics`
owns log-contract style fair-strike replication.  This module provides the
matching bounded Monte Carlo route for realised log variance under a one-factor
GBM surface binding, so task adapters do not hand-roll path simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention


@dataclass(frozen=True)
class EquityVarianceSwapMonteCarloResult:
    """Structured Monte Carlo result for an equity variance swap."""

    price: float
    fair_strike_variance: float
    standard_error: float
    n_paths: int
    n_steps: int
    seed: int


@dataclass(frozen=True)
class ResolvedEquityVarianceSwapMonteCarloInputs:
    """Resolved scalar inputs for variance-swap Monte Carlo."""

    spot: float
    maturity: float
    rate: float
    dividend_yield: float
    sigma: float
    notional: float
    strike_variance: float
    realized_variance: float
    discount_factor: float
    n_paths: int
    n_steps: int
    seed: int


def resolve_equity_variance_swap_monte_carlo_inputs(
    market_state: MarketState,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> ResolvedEquityVarianceSwapMonteCarloInputs:
    """Resolve a variance-swap MC problem from ``MarketState`` and spec."""
    if market_state.discount is None:
        raise ValueError("Variance swap Monte Carlo pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Variance swap Monte Carlo pricing requires market_state.vol_surface")
    settlement = getattr(market_state, "settlement", None)
    if settlement is None:
        raise ValueError("Variance swap Monte Carlo pricing requires market_state.settlement")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    spot = float(spec.spot)
    sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), spot))
    rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    discount_factor = float(market_state.discount.discount(max(maturity, 0.0)))
    return ResolvedEquityVarianceSwapMonteCarloInputs(
        spot=spot,
        maturity=maturity,
        rate=rate,
        dividend_yield=_carry_rate_from_market_state(market_state, spec),
        sigma=sigma,
        notional=float(getattr(spec, "notional", 1.0) or 1.0),
        strike_variance=float(getattr(spec, "strike_variance", 0.0) or 0.0),
        realized_variance=float(getattr(spec, "realized_variance", 0.0) or 0.0),
        discount_factor=discount_factor,
        n_paths=max(int(n_paths or getattr(spec, "n_paths", 60_000) or 60_000), 2),
        n_steps=max(int(n_steps or getattr(spec, "n_steps", 252) or 252), 1),
        seed=int(seed if seed is not None else getattr(spec, "seed", 42) or 42),
    )


def equity_variance_swap_outputs_monte_carlo(
    market_state: MarketState,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Return price and fair strike variance from the MC variance-swap route."""
    result = price_equity_variance_swap_monte_carlo_result(
        market_state,
        spec,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    return {
        "price": result.price,
        "fair_strike_variance": result.fair_strike_variance,
        "standard_error": result.standard_error,
    }


def price_equity_variance_swap_monte_carlo_result(
    market_state: MarketState,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> EquityVarianceSwapMonteCarloResult:
    """Price a variance swap by simulating annualised realised log variance."""
    resolved = resolve_equity_variance_swap_monte_carlo_inputs(
        market_state,
        spec,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    if resolved.maturity <= 0.0:
        price = resolved.notional * resolved.discount_factor * (
            resolved.realized_variance - resolved.strike_variance
        )
        return EquityVarianceSwapMonteCarloResult(
            price=float(price),
            fair_strike_variance=resolved.realized_variance,
            standard_error=0.0,
            n_paths=resolved.n_paths,
            n_steps=resolved.n_steps,
            seed=resolved.seed,
        )

    rng = raw_np.random.default_rng(resolved.seed)
    dt = resolved.maturity / resolved.n_steps
    sqrt_dt = sqrt(dt)
    drift = (resolved.rate - resolved.dividend_yield - 0.5 * resolved.sigma**2) * dt
    diffusion_scale = resolved.sigma * sqrt_dt
    total = 0.0
    total_sq = 0.0
    remaining = resolved.n_paths
    chunk_size = min(16_384, resolved.n_paths)

    while remaining > 0:
        count = min(chunk_size, remaining)
        half = (count + 1) // 2
        normals = rng.standard_normal((half, resolved.n_steps))
        if count > half:
            normals = raw_np.concatenate((normals, -normals[: count - half]), axis=0)
        increments = drift + diffusion_scale * normals[:count]
        realised = raw_np.sum(increments * increments, axis=1) / resolved.maturity
        total += float(raw_np.sum(realised))
        total_sq += float(raw_np.sum(realised * realised))
        remaining -= count

    mean_realised = total / resolved.n_paths
    variance = max(total_sq / resolved.n_paths - mean_realised * mean_realised, 0.0)
    fair_strike_variance = resolved.realized_variance + mean_realised
    price_samples_scale = resolved.notional * resolved.discount_factor
    price = price_samples_scale * (fair_strike_variance - resolved.strike_variance)
    standard_error = price_samples_scale * sqrt(variance / resolved.n_paths)
    return EquityVarianceSwapMonteCarloResult(
        price=float(price),
        fair_strike_variance=float(fair_strike_variance),
        standard_error=float(standard_error),
        n_paths=resolved.n_paths,
        n_steps=resolved.n_steps,
        seed=resolved.seed,
    )


def price_equity_variance_swap_monte_carlo(
    market_state: MarketState,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> float:
    """Return the scalar present value from the MC variance-swap helper."""
    return float(
        price_equity_variance_swap_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        ).price
    )


def _carry_rate_from_market_state(market_state: MarketState, spec) -> float:
    if getattr(spec, "dividend_yield", None) is not None:
        return float(spec.dividend_yield)
    if getattr(spec, "dividend_rate", None) is not None:
        return float(spec.dividend_rate)
    params = dict(getattr(market_state, "model_parameters", None) or {})
    carry_rates = dict(params.get("underlier_carry_rates") or {})
    if carry_rates:
        return float(next(iter(carry_rates.values())))
    return 0.0


__all__ = [
    "EquityVarianceSwapMonteCarloResult",
    "ResolvedEquityVarianceSwapMonteCarloInputs",
    "equity_variance_swap_outputs_monte_carlo",
    "price_equity_variance_swap_monte_carlo",
    "price_equity_variance_swap_monte_carlo_result",
    "resolve_equity_variance_swap_monte_carlo_inputs",
]
