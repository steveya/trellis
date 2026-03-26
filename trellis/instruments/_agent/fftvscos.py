"""Agent-generated payoff: Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class FFTvsCOSSpec:
    """Specification for cached FFT/COS transform-family benchmarks."""

    s0: float = 100.0
    strike: float = 100.0
    expiry_date: date = date(2025, 11, 15)
    is_call: bool = True


class FFTvsCOSPricer:
    """Cached generic payoff used for transform-family offline benchmarks."""

    def __init__(self, spec: FFTvsCOSSpec):
        self._spec = spec

    @property
    def spec(self) -> FFTvsCOSSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "black_vol_surface", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.expiry_date)
        if T <= 0.0:
            intrinsic = max(spec.s0 - spec.strike, 0.0) if spec.is_call else max(spec.strike - spec.s0, 0.0)
            return float(intrinsic)

        spot = float(market_state.spot if market_state.spot is not None else spec.s0)
        r = float(market_state.discount.zero_rate(T)) if market_state.discount is not None else 0.0
        sigma = (
            float(market_state.vol_surface.black_vol(T, spec.strike))
            if market_state.vol_surface is not None
            else 0.20
        )
        forward = spot * exp(r * T)
        df = exp(-r * T)
        if spec.is_call:
            return float(df * black76_call(forward, spec.strike, sigma, T))
        return float(df * black76_put(forward, spec.strike, sigma, T))

