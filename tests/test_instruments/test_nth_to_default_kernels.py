"""Tests that nth-to-default reuses the shared credit-event kernels."""

from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.nth_to_default import (
    NthToDefaultPayoff,
    NthToDefaultSpec,
    price_nth_to_default_basket,
)


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.04),
        credit_curve=CreditCurve.flat(0.02),
    )


def test_nth_to_default_uses_shared_credit_kernels(monkeypatch):
    import trellis.instruments.nth_to_default as nth_module

    calls: dict[str, object] = {}

    def fake_terminal_default_probability(credit_curve, horizon):
        calls["terminal"] = (credit_curve, horizon)
        return 0.18

    def fake_nth_to_default_probability(n_names, n_th, marginal_default_prob, correlation):
        calls["nth"] = (n_names, n_th, marginal_default_prob, correlation)
        return 0.11

    def fake_protection_payment_pv(payment):
        calls["payment"] = payment
        return 12_345.0

    monkeypatch.setattr(nth_module, "terminal_default_probability", fake_terminal_default_probability)
    monkeypatch.setattr(nth_module, "nth_to_default_probability", fake_nth_to_default_probability)
    monkeypatch.setattr(nth_module, "protection_payment_pv", fake_protection_payment_pv)

    spec = NthToDefaultSpec(
        notional=1_000_000.0,
        n_names=5,
        n_th=2,
        end_date=date(2029, 11, 15),
        correlation=0.35,
        recovery=0.4,
    )

    pv = NthToDefaultPayoff(spec).evaluate(_market_state())

    assert pv == 12_345.0
    assert calls["nth"] == (5, 2, 0.18, 0.35)


def test_price_nth_to_default_basket_matches_reference_payoff():
    market_state = _market_state()
    spec = NthToDefaultSpec(
        notional=1_000_000.0,
        n_names=5,
        n_th=2,
        end_date=date(2029, 11, 15),
        correlation=0.35,
        recovery=0.4,
    )

    helper_pv = price_nth_to_default_basket(
        notional=spec.notional,
        n_names=spec.n_names,
        n_th=spec.n_th,
        horizon=5.0,
        correlation=spec.correlation,
        recovery=spec.recovery,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount,
    )
    reference_pv = NthToDefaultPayoff(spec).evaluate(market_state)

    assert helper_pv == reference_pv
