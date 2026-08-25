"""Nth-to-default credit basket priced via copula simulation.

An nth-to-default basket is a credit derivative that pays out when the
nth entity in a group defaults. For example, a 1st-to-default basket on
5 companies pays the holder a loss amount when any one company defaults.
Pricing requires modeling correlated defaults, which is done here using
a Gaussian copula (a standard model for multi-name credit products).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from trellis.core.market_state import MarketState

from trellis.core.types import DayCountConvention
from trellis.models.contingent_cashflows import (
    TriggerSettlement,
    exchangeable_ranked_event_expected_weight,
    nth_to_default_probability,
    terminal_default_probability,
    trigger_settlement_pv,
)
from trellis.models.credit_basket_copula import resolve_credit_basket_inputs


@dataclass(frozen=True)
class NthToDefaultSpec:
    """Contract terms for an nth-to-default credit basket."""

    notional: float
    n_names: int
    n_th: int                       # which default triggers (1=first)
    end_date: date
    basket_names: tuple[str, ...] = ()
    basket_weights: tuple[float, ...] = ()
    correlation: float = 0.3
    recovery: float = 0.4
    spread: float | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_360


class NthToDefaultPayoff:
    """Price terminal name-weighted nth-to-default protection analytically."""

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
        """Compute the expected discounted terminal ranked-loss settlement."""
        spec = self._spec
        resolved = resolve_credit_basket_inputs(market_state, spec)
        trigger_probability = nth_to_default_probability(
            resolved.n_names,
            int(spec.n_th),
            resolved.default_probability,
            resolved.correlation,
        )
        expected_weight = exchangeable_ranked_event_expected_weight(
            trigger_probability,
            event_weights=resolved.exposure_weights,
        )
        return trigger_settlement_pv(
            TriggerSettlement(
                amount=resolved.notional * (1.0 - resolved.recovery),
                discount_factor=resolved.discount_factor,
                trigger_weight=expected_weight,
            )
        )

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        """Return price and parallel one-basis-point spread CS01 when quoted."""
        price = float(self.evaluate(market_state))
        if self._spec.spread is None:
            return {"price": price}
        bumped = replace(self._spec, spread=float(self._spec.spread) + 1.0e-4)
        bumped_price = float(type(self)(bumped).evaluate(market_state))
        return {"price": price, "spread_cs01": bumped_price - price}


def price_nth_to_default_basket(
    *,
    notional: float,
    n_names: int,
    n_th: int,
    horizon: float | None = None,
    correlation: float = 0.3,
    recovery: float = 0.4,
    credit_curve=None,
    discount_curve=None,
    maturity: float | None = None,
    default_prob: float | None = None,
) -> float:
    """Price a helper-backed nth-to-default basket from curve inputs and contract terms."""
    T = float(horizon if horizon is not None else maturity if maturity is not None else 0.0)
    if T <= 0:
        return 0.0

    if default_prob is None:
        if credit_curve is None:
            raise ValueError("credit_curve is required when default_prob is not supplied")
        default_prob = terminal_default_probability(credit_curve, T)
    trigger_prob = nth_to_default_probability(
        n_names,
        n_th,
        float(default_prob),
        correlation,
    )
    df = 1.0 if discount_curve is None else float(discount_curve.discount(T))
    expected_weight = exchangeable_ranked_event_expected_weight(
        trigger_prob,
        event_weights=(1.0,) * int(n_names),
    )
    return float(
        trigger_settlement_pv(
            TriggerSettlement(
                amount=float(notional) * (1.0 - float(recovery)),
                discount_factor=df,
                trigger_weight=expected_weight,
            )
        )
    )
