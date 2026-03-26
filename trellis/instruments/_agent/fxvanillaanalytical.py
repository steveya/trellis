"""Deterministic vanilla FX analytical payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.black import garman_kohlhagen_call, garman_kohlhagen_put


@dataclass(frozen=True)
class FXVanillaOptionSpec:
    """Specification for a vanilla FX option under Garman-Kohlhagen."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _resolve_fx_inputs(market_state: MarketState, spec: FXVanillaOptionSpec):
    """Resolve FX spot and domestic/foreign discount factors from MarketState."""
    if market_state.discount is None:
        raise ValueError("market_state.discount is required for FX analytical pricing")
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

    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    domestic_df = float(market_state.discount.discount(T))
    foreign_curve = market_state.forecast_curves[spec.foreign_discount_key]
    foreign_df = float(foreign_curve.discount(T))
    return spot, T, domestic_df, foreign_df


class FXVanillaAnalyticalPayoff:
    """Deterministic thin adapter over the Garman-Kohlhagen library kernel."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "forward_rate", "black_vol", "fx"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        spot, T, domestic_df, foreign_df = _resolve_fx_inputs(market_state, spec)
        if T <= 0:
            intrinsic = max(spot - spec.strike, 0.0) if spec.option_type.lower() == "call" else max(spec.strike - spot, 0.0)
            return float(spec.notional) * float(intrinsic)

        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        option_type = spec.option_type.lower()
        if option_type == "call":
            price = garman_kohlhagen_call(spot, spec.strike, sigma, T, domestic_df, foreign_df)
        elif option_type == "put":
            price = garman_kohlhagen_put(spot, spec.strike, sigma, T, domestic_df, foreign_df)
        else:
            raise ValueError(f"Unsupported option_type {spec.option_type!r}; expected 'call' or 'put'")
        return float(spec.notional) * float(price)
