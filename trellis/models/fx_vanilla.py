"""Semantic-facing helper kit for vanilla FX options."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.fx import (
    ResolvedGarmanKohlhagenInputs,
    garman_kohlhagen_price_raw,
)

np = get_numpy()


class FXVanillaSpecLike(Protocol):
    """Minimal semantic surface consumed by the vanilla FX helper kit."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str
    day_count: DayCountConvention


@dataclass(frozen=True)
class ResolvedFXVanillaInputs:
    """Resolved market and numerical inputs for one vanilla FX route."""

    notional: float
    option_type: str
    garman_kohlhagen: ResolvedGarmanKohlhagenInputs
    domestic_rate: float
    foreign_rate: float


def resolve_fx_vanilla_inputs(
    market_state: MarketState,
    spec: FXVanillaSpecLike,
) -> ResolvedFXVanillaInputs:
    """Resolve spot, domestic/foreign curves, and volatility for one FX option."""
    if market_state.discount is None:
        raise ValueError("market_state.discount is required for FX pricing")
    if not market_state.forecast_curves or spec.foreign_discount_key not in market_state.forecast_curves:
        raise ValueError(
            f"market_state.forecast_curves must contain foreign discount key {spec.foreign_discount_key!r}"
        )

    fx_quote = (market_state.fx_rates or {}).get(spec.fx_pair)
    if fx_quote is not None:
        spot = float(fx_quote.spot)
    elif market_state.spot is not None:
        spot = float(market_state.spot)
    elif market_state.underlier_spots and spec.fx_pair in market_state.underlier_spots:
        spot = float(market_state.underlier_spots[spec.fx_pair])
    else:
        raise ValueError(f"FX spot for pair {spec.fx_pair!r} is not available in market_state")

    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for FX pricing")

    maturity = max(
        float(
            year_fraction(
                settlement,
                spec.expiry_date,
                getattr(spec, "day_count", DayCountConvention.ACT_365),
            )
        ),
        0.0,
    )
    domestic_curve = market_state.discount
    foreign_curve = market_state.forecast_curves[spec.foreign_discount_key]
    domestic_df = float(domestic_curve.discount(maturity))
    foreign_df = float(foreign_curve.discount(maturity))
    domestic_rate = float(domestic_curve.zero_rate(max(maturity, 1e-6))) if maturity > 0.0 else 0.0
    foreign_rate = float(foreign_curve.zero_rate(max(maturity, 1e-6))) if maturity > 0.0 else 0.0
    sigma = (
        float(market_state.vol_surface.black_vol(max(maturity, 1e-6), float(spec.strike)))
        if maturity > 0.0 and market_state.vol_surface is not None
        else 0.0
    )
    if maturity > 0.0 and market_state.vol_surface is None:
        raise ValueError("market_state.vol_surface is required for FX pricing")

    return ResolvedFXVanillaInputs(
        notional=float(spec.notional),
        option_type=str(spec.option_type).strip().strip("'\"").lower(),
        garman_kohlhagen=ResolvedGarmanKohlhagenInputs(
            spot=spot,
            strike=float(spec.strike),
            sigma=sigma,
            T=maturity,
            df_domestic=domestic_df,
            df_foreign=foreign_df,
        ),
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
    )


def price_fx_vanilla_analytical(
    market_state: MarketState,
    spec: FXVanillaSpecLike,
) -> float:
    """Price one vanilla FX option through the checked analytical helper."""
    resolved = resolve_fx_vanilla_inputs(market_state, spec)
    return float(resolved.notional) * float(
        garman_kohlhagen_price_raw(resolved.option_type, resolved.garman_kohlhagen)
    )


def price_fx_vanilla_monte_carlo(
    market_state: MarketState,
    spec: FXVanillaSpecLike,
    *,
    seed: int = 42,
) -> float:
    """Price one vanilla FX option through the bounded FX Monte Carlo helper."""
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.processes.gbm import GBM

    resolved = resolve_fx_vanilla_inputs(market_state, spec)
    gk = resolved.garman_kohlhagen
    if gk.T <= 0.0:
        intrinsic = (
            max(float(gk.spot) - float(gk.strike), 0.0)
            if resolved.option_type == "call"
            else max(float(gk.strike) - float(gk.spot), 0.0)
        )
        return float(resolved.notional) * float(intrinsic)

    process = GBM(mu=resolved.domestic_rate - resolved.foreign_rate, sigma=float(gk.sigma))
    engine = MonteCarloEngine(
        process,
        n_paths=max(int(getattr(spec, "n_paths", 50_000)), 1),
        n_steps=max(int(getattr(spec, "n_steps", 252)), 1),
        seed=int(getattr(spec, "seed", seed)),
        method="exact",
    )

    strike = float(gk.strike)
    notional = float(resolved.notional)
    option_type = resolved.option_type

    def payoff_fn(paths):
        terminal = np.asarray(paths[:, -1], dtype=float)
        if option_type == "call":
            payoffs = np.maximum(terminal - strike, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(strike - terminal, 0.0)
        else:
            raise ValueError(
                f"Unsupported option_type {spec.option_type!r}; expected 'call' or 'put'"
            )
        return payoffs * notional

    result = engine.price(
        float(gk.spot),
        float(gk.T),
        payoff_fn,
        discount_rate=float(resolved.domestic_rate),
        return_paths=False,
    )
    return float(result["price"])


__all__ = [
    "FXVanillaSpecLike",
    "ResolvedFXVanillaInputs",
    "price_fx_vanilla_analytical",
    "price_fx_vanilla_monte_carlo",
    "resolve_fx_vanilla_inputs",
]
