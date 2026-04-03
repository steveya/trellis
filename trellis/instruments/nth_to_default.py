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

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState

from trellis.core.types import DayCountConvention
from trellis.models.contingent_cashflows import (
    ProtectionPayment,
    nth_to_default_probability,
    protection_payment_pv,
    terminal_default_probability,
)


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
        return {"discount_curve", "credit_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Simulate correlated defaults and compute the expected discounted loss payment."""
        spec = self._spec
        T = _nth_to_default_horizon(
            market_state.settlement,
            spec.end_date,
            spec.day_count,
        )
        return price_nth_to_default_basket(
            notional=spec.notional,
            n_names=spec.n_names,
            n_th=spec.n_th,
            horizon=T,
            correlation=spec.correlation,
            recovery=spec.recovery,
            credit_curve=market_state.credit_curve,
            discount_curve=market_state.discount,
        )


def price_nth_to_default_basket(
    *,
    notional: float,
    n_names: int,
    n_th: int,
    horizon: float,
    correlation: float,
    recovery: float,
    credit_curve,
    discount_curve,
) -> float:
    """Price a helper-backed nth-to-default basket from curve inputs and contract terms."""
    T = float(horizon)
    if T <= 0:
        return 0.0

    default_prob = terminal_default_probability(credit_curve, T)
    trigger_prob = nth_to_default_probability(
        n_names,
        n_th,
        default_prob,
        correlation,
    )
    df = float(discount_curve.discount(T))
    return float(
        protection_payment_pv(
            ProtectionPayment(
                notional=notional,
                recovery=recovery,
                default_probability=trigger_prob,
                discount_factor=df,
            )
        )
    )


def _nth_to_default_horizon(start: date, end: date, day_count: DayCountConvention) -> float:
    """Normalize same-anniversary maturities to whole-year tenors for helper parity."""
    if (start.month, start.day) == (end.month, end.day):
        return float(end.year - start.year)
    return year_fraction(start, end, day_count)
