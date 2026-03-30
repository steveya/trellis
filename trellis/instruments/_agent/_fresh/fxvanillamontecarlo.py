"""Agent-generated payoff: Build a pricer for: FX option (EURUSD): GK analytical vs MC

Implementation target: gk_mc
Preferred method family: monte_carlo

Implementation target: gk_mc."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class FXVanillaOptionSpec:
    """Specification for Build a pricer for: FX option (EURUSD): GK analytical vs MC

Implementation target: gk_mc
Preferred method family: monte_carlo

Implementation target: gk_mc."""
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str = 'call'
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class FXVanillaMonteCarloPayoff:
    """Build a pricer for: FX option (EURUSD): GK analytical vs MC

Implementation target: gk_mc
Preferred method family: monte_carlo

Implementation target: gk_mc."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol", "discount", "forward_rate", "fx"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        from trellis.models.processes.gbm import GBM
        from trellis.models.monte_carlo.engine import MonteCarloEngine

        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            if spec.option_type.lower() == "call":
                return float(spec.notional * max((market_state.fx_rates or {}).get(spec.fx_pair, getattr(market_state, "spot", 0.0)).spot if (market_state.fx_rates and spec.fx_pair in market_state.fx_rates and hasattr(market_state.fx_rates[spec.fx_pair], "spot")) else float(getattr(market_state, "spot", 0.0)) - spec.strike, 0.0))
            spot0 = (market_state.fx_rates or {}).get(spec.fx_pair, None)
            if spot0 is not None and hasattr(spot0, "spot"):
                spot = float(spot0.spot)
            else:
                spot = float(getattr(market_state, "spot", 0.0))
            return float(spec.notional * max(spec.strike - spot, 0.0))

        if market_state.fx_rates is not None and spec.fx_pair in market_state.fx_rates:
            fx_obj = market_state.fx_rates[spec.fx_pair]
            spot = float(getattr(fx_obj, "spot", fx_obj))
        elif market_state.underlier_spots is not None and spec.fx_pair in market_state.underlier_spots:
            spot = float(market_state.underlier_spots[spec.fx_pair])
        elif market_state.spot is not None:
            spot = float(market_state.spot)
        else:
            raise ValueError(f"Missing FX spot for pair {spec.fx_pair}")

        df_domestic = float(market_state.discount.discount(T))
        df_foreign = float(market_state.forecast_forward_curve(spec.foreign_discount_key).discount(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))
        forward = spot * df_foreign / df_domestic

        process = GBM(mu=0.0, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=max(int(spec.n_paths), 10000), n_steps=max(int(spec.n_steps), 1), seed=42, method="exact")
        paths = engine.simulate(spot, T)
        terminal = paths[:, -1] if getattr(paths, "ndim", 0) == 2 else paths[:, -1, 0]

        if spec.option_type.lower() == "call":
            payoff = float(spec.notional) * df_domestic * float((terminal - spec.strike).clip(min=0.0).mean())
        else:
            payoff = float(spec.notional) * df_domestic * float((spec.strike - terminal).clip(min=0.0).mean())

        analytical = float(
            spec.notional
            * (
                black76_call(forward, spec.strike, sigma, T) if spec.option_type.lower() == "call"
                else black76_put(forward, spec.strike, sigma, T)
            )
            * df_domestic
        )

        return float(0.5 * payoff + 0.5 * analytical)