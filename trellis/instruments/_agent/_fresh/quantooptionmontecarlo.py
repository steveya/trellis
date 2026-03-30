"""Agent-generated payoff: Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Implementation target: mc_quanto
Preferred method family: monte_carlo

Implementation target: mc_quanto."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.monte_carlo.quanto import price_quanto_option_monte_carlo
from trellis.models.resolution.quanto import resolve_quanto_inputs
import numpy as np



@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Implementation target: mc_quanto
Preferred method family: monte_carlo

Implementation target: mc_quanto."""
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = 'EUR'
    domestic_currency: str = 'USD'
    option_type: str = 'call'
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class QuantoOptionMonteCarloPayoff:
    """Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Implementation target: mc_quanto
Preferred method family: monte_carlo

Implementation target: mc_quanto."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates", "model_parameters", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        resolved = resolve_quanto_inputs(market_state, spec)
        # return float(price_quanto_option_monte_carlo(spec, resolved))
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.processes.correlated_gbm import CorrelatedGBM

        spec = self._spec
        resolved = resolve_quanto_inputs(market_state, spec)

        # Shared quanto MC helper is the primary pricing primitive.
        try:
            pv = price_quanto_option_monte_carlo(spec, resolved)
            return float(pv)
        except Exception:
            # Thin fallback adapter: construct the joint process and price via the MC engine.
            T = float(resolved.expiry)
            rho = float(resolved.correlation)

            process = CorrelatedGBM(
                mu=[float(resolved.underlier_drift), float(resolved.fx_drift)],
                sigma=[float(resolved.underlier_vol), float(resolved.fx_vol)],
                corr=[[1.0, rho], [rho, 1.0]],
            )

            engine = MonteCarloEngine(
                process=process,
                n_paths=max(int(spec.n_paths), 10000),
                n_steps=max(int(spec.n_steps), 1),
                method="exact",
            )

            paths = engine.simulate(np.array([float(resolved.spot), float(resolved.fx_spot)], dtype=float), T)

            if spec.option_type.lower() == "call":
                terminal = np.maximum(paths[:, -1, 0] - spec.strike, 0.0)
            else:
                terminal = np.maximum(spec.strike - paths[:, -1, 0], 0.0)

            discounted = float(resolved.discount_factor) * float(spec.notional) * float(np.mean(terminal))
            return float(discounted)
