"""Nth-to-default basket — reference implementation using copula.

This is the hand-coded reference for copula-based pricing patterns.
The agent uses this as a template for other portfolio credit instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.payoff import PresentValue
from trellis.core.types import DayCountConvention
from trellis.models.copulas.gaussian import GaussianCopula


@dataclass(frozen=True)
class NthToDefaultSpec:
    """Specification for an nth-to-default basket."""

    notional: float
    n_names: int
    n_th: int                       # which default triggers (1=first)
    end_date: date
    correlation: float = 0.3
    recovery: float = 0.4
    day_count: DayCountConvention = DayCountConvention.ACT_360


class NthToDefaultPayoff:
    """Nth-to-default basket priced via Gaussian copula simulation."""

    def __init__(self, spec: NthToDefaultSpec):
        self._spec = spec

    @property
    def spec(self) -> NthToDefaultSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "credit"}

    def evaluate(self, market_state: MarketState) -> PresentValue:
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return PresentValue(0.0)

        # Get hazard rate from credit curve (uniform for all names)
        lam = float(market_state.credit_curve.hazard_rate(T))
        hazard_rates = raw_np.full(spec.n_names, lam)

        # Build correlation matrix (equicorrelation)
        corr = raw_np.full((spec.n_names, spec.n_names), spec.correlation)
        raw_np.fill_diagonal(corr, 1.0)

        copula = GaussianCopula(corr)
        n_paths = 50000
        rng = raw_np.random.default_rng(42)

        # Simulate default times
        default_times = copula.sample_default_times(hazard_rates, n_paths, rng)

        # Count defaults within protection period
        defaults_in_period = raw_np.sum(default_times <= T, axis=1)

        # Nth-to-default triggers if n_th or more names default
        triggered = defaults_in_period >= spec.n_th

        # Protection payment: (1 - recovery) * notional, discounted
        loss_given_default = (1 - spec.recovery) * spec.notional
        df = float(market_state.discount.discount(T))

        # Expected discounted protection payment
        protection_pv = float(raw_np.mean(triggered)) * loss_given_default * df

        return PresentValue(protection_pv)
