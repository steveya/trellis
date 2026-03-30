"""Nth-to-default credit basket priced via copula simulation.

An nth-to-default basket is a credit derivative that pays out when the
nth entity in a group defaults. For example, a 1st-to-default basket on
5 companies pays the holder a loss amount when any one company defaults.
Pricing requires modeling correlated defaults, which is done here using
a Gaussian copula (a standard model for multi-name credit products).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState

from trellis.core.types import DayCountConvention
from trellis.models.copulas.gaussian import GaussianCopula


@dataclass(frozen=True)
class NthToDefaultSpec:
    """Contract terms for an nth-to-default credit basket."""

    notional: float
    n_names: int
    n_th: int                       # which default triggers (1=first)
    end_date: date
    correlation: float = 0.3
    recovery: float = 0.4
    day_count: DayCountConvention = DayCountConvention.ACT_360


class NthToDefaultPayoff:
    """Prices an nth-to-default basket by simulating correlated defaults."""

    def __init__(self, spec: NthToDefaultSpec):
        """Store the basket-credit contract specification."""
        self._spec = spec

    @property
    def spec(self) -> NthToDefaultSpec:
        """Return the immutable nth-to-default specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Needs a discount curve and a credit curve (for default probabilities)."""
        return {"discount", "credit"}

    def evaluate(self, market_state: MarketState) -> float:
        """Simulate correlated defaults and compute the expected discounted loss payment."""
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

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

        return protection_pv
