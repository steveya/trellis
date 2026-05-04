"""Compatibility Asian-option adapter over checked helper code."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.asian_option import price_asian_option_monte_carlo


@dataclass(frozen=True)
class AsianOptionSpec:
    """Legacy Asian-option compatibility spec."""

    notional: float
    spot: float
    strike: float
    expiry_date: date
    averaging_type: str = "arithmetic"
    option_type: str = "call"
    n_observations: int = 12
    day_count: DayCountConvention = DayCountConvention.ACT_365


class AsianOptionPayoff:
    """Thin compatibility wrapper over the checked Asian Monte Carlo helper."""

    def __init__(self, spec: AsianOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> AsianOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        legacy_notional = float(self._spec.notional)
        spot = float(self._spec.spot)
        helper_spec = replace(self._spec, notional=legacy_notional / spot)
        return float(price_asian_option_monte_carlo(market_state, helper_spec))
