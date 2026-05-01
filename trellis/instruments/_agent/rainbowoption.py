"""Agent-generated payoff: Build a pricer for: Trellis extension: Bermudan best-of-two rainbow basket

Price a Bermudan best-of-two rainbow call. Return price plus the relevant Greeks and compare internal lower/upper bounds where available.

Construct methods: monte_carlo, rate_tree
Comparison targets: monte_carlo (monte_carlo), rate_tree (rate_tree)

Implementation target: rate_tree
Preferred method family: rate_tree

Implementation target: rate_tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import PricingValue
from trellis.core.types import DayCountConvention
from trellis.execution import price_bermudan_best_of_basket_from_compat_spec



@dataclass(frozen=True)
class RainbowOptionSpec:
    """Specification for Build a pricer for: Trellis extension: Bermudan best-of-two rainbow basket

Price a Bermudan best-of-two rainbow call. Return price plus the relevant Greeks and compare internal lower/upper bounds where available.

Construct methods: monte_carlo, rate_tree
Comparison targets: monte_carlo (monte_carlo), rate_tree (rate_tree)

Implementation target: rate_tree
Preferred method family: rate_tree

Implementation target: rate_tree."""
    underliers: str
    spots: str
    strike: float
    expiry_date: date
    vols: str
    correlation: str
    notional: float = 1.0
    dividend_yields: str | None = None
    basket_style: str = 'best_of'
    option_type: str = 'call'
    observation_dates: tuple[date, ...] | None = None
    exercise_dates: tuple[date, ...] | None = None
    risk_free_rate: float | None = None
    n_paths: int = 4096
    n_steps: int = 96
    seed: int | None = 42
    lattice_n_steps: int = 48
    day_count: DayCountConvention = DayCountConvention.ACT_365


class RainbowOptionPayoff:
    """Build a pricer for: Trellis extension: Bermudan best-of-two rainbow basket

Price a Bermudan best-of-two rainbow call. Return price plus the relevant Greeks and compare internal lower/upper bounds where available.

Construct methods: monte_carlo, rate_tree
Comparison targets: monte_carlo (monte_carlo), rate_tree (rate_tree)

Implementation target: rate_tree
Preferred method family: rate_tree

Implementation target: rate_tree."""

    def __init__(self, spec: RainbowOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> RainbowOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return set()

    def evaluate(self, market_state: MarketState) -> PricingValue:
        spec = self._spec
        return price_bermudan_best_of_basket_from_compat_spec(
            market_state,
            spec,
            method="lattice",
        )
