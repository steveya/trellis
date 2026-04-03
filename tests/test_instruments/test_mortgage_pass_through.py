"""Tests for the thin prepayment-driven mortgage pass-through wrapper."""

from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.mortgage_pass_through import (
    MortgagePassThroughPayoff,
    MortgagePassThroughSpec,
)


SETTLE = date(2024, 11, 15)


def _market_state(rate: float = 0.04) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate),
    )


def test_requirements():
    spec = MortgagePassThroughSpec(
        balance=250_000.0,
        mortgage_rate=0.06,
        pass_through_rate=0.055,
        term_months=360,
    )
    assert MortgagePassThroughPayoff(spec).requirements == {"discount_curve"}


def test_pass_through_returns_positive_present_value():
    spec = MortgagePassThroughSpec(
        balance=250_000.0,
        mortgage_rate=0.06,
        pass_through_rate=0.055,
        term_months=360,
        psa_speed=1.0,
    )
    pv = MortgagePassThroughPayoff(spec).evaluate(_market_state())
    assert pv > 0.0
