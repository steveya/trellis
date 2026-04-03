"""Agent-generated payoff: Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Construct methods: analytical, monte_carlo
Comparison targets: quanto_bs (analytical), mc_quanto (monte_carlo)
Cross-validation harness:
  internal targets: quanto_bs, mc_quanto
  external targets: quantlib
New component: ['quanto_adjustment', 'cross_currency_mc']

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
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.quanto import price_quanto_option_monte_carlo
from trellis.models.resolution.quanto import resolve_quanto_inputs



@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Construct methods: analytical, monte_carlo
Comparison targets: quanto_bs (analytical), mc_quanto (monte_carlo)
Cross-validation harness:
  internal targets: quanto_bs, mc_quanto
  external targets: quantlib
New component: ['quanto_adjustment', 'cross_currency_mc']

Implementation target: mc_quanto
Preferred method family: monte_carlo

Implementation target: mc_quanto."""
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "'EUR'"
    domestic_currency: str = "'USD'"
    option_type: str = "'call'"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class QuantoOptionMonteCarloPayoff:
    """Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Construct methods: analytical, monte_carlo
Comparison targets: quanto_bs (analytical), mc_quanto (monte_carlo)
Cross-validation harness:
  internal targets: quanto_bs, mc_quanto
  external targets: quantlib
New component: ['quanto_adjustment', 'cross_currency_mc']

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
        return float(price_quanto_option_monte_carlo(spec, resolved))