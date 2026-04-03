"""Agent-generated payoff: Build a pricer for: CDO tranche: Gaussian vs Student-t copula."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class CDOTrancheSpec:
    """Specification for Build a pricer for: CDO tranche: Gaussian vs Student-t copula."""
    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    correlation: float = 0.3
    recovery: float = 0.4
    day_count: DayCountConvention = DayCountConvention.ACT_360


class CDOTranchePayoff:
    """Build a pricer for: CDO tranche: Gaussian vs Student-t copula."""

    def __init__(self, spec: CDOTrancheSpec):
        """Store the generated tranche specification."""
        self._spec = spec

    @property
    def spec(self) -> CDOTrancheSpec:
        """Return the immutable generated tranche specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and credit-survival inputs."""
        return {"credit_curve", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Estimate discounted expected tranche loss from a factor-copula loss distribution."""
        from trellis.core.date_utils import year_fraction
        from trellis.models.copulas.factor import FactorCopula
        import numpy as np

        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0:
            return 0.0

        # Consistent with calibration: marginal default probability from credit curve
        marginal_prob = 1 - float(market_state.credit_curve.survival_probability(T))

        # Build copula with given correlation and portfolio size
        copula = FactorCopula(n_names=spec.n_names, correlation=spec.correlation)
        losses, probs = copula.loss_distribution(marginal_prob)

        # Compute expected tranche loss by integrating over possible default states
        tranche_el = 0.0
        for n_defaults, prob in zip(losses, probs):
            portfolio_loss = n_defaults / spec.n_names
            tranche_loss = max(0, min(portfolio_loss - spec.attachment, spec.detachment - spec.attachment))
            tranche_el += prob * tranche_loss

        # Discount the expected loss to get the present value for the tranche
        df = float(market_state.discount.discount(T))
        tranche_pv = spec.notional * (tranche_el * df)
        return tranche_pv
