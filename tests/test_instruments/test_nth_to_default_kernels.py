"""Tests that nth-to-default reuses the shared credit-event kernels."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

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

    def fake_resolve_credit_basket_inputs(market_state, spec):
        calls["resolve"] = (market_state, spec)
        return SimpleNamespace(
            n_names=5,
            default_probability=0.18,
            correlation=0.35,
            exposure_weights=(1.0,) * 5,
            notional=1_000_000.0,
            recovery=0.4,
            discount_factor=0.8,
        )

    def fake_nth_to_default_probability(n_names, n_th, marginal_default_prob, correlation):
        calls["nth"] = (n_names, n_th, marginal_default_prob, correlation)
        return 0.11

    def fake_exchangeable_ranked_event_expected_weight(probability, *, event_weights):
        calls["weight"] = (probability, event_weights)
        return probability

    def fake_trigger_settlement_pv(settlement):
        calls["settlement"] = settlement
        return 12_345.0

    monkeypatch.setattr(nth_module, "resolve_credit_basket_inputs", fake_resolve_credit_basket_inputs)
    monkeypatch.setattr(nth_module, "nth_to_default_probability", fake_nth_to_default_probability)
    monkeypatch.setattr(
        nth_module,
        "exchangeable_ranked_event_expected_weight",
        fake_exchangeable_ranked_event_expected_weight,
    )
    monkeypatch.setattr(nth_module, "trigger_settlement_pv", fake_trigger_settlement_pv)

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
    assert calls["weight"] == (0.11, (1.0,) * 5)


def test_public_nth_to_default_supports_weighted_spread_price_and_cs01():
    spec = NthToDefaultSpec(
        notional=5_000_000.0,
        n_names=4,
        n_th=2,
        end_date=date(2029, 11, 15),
        basket_names=("A", "B", "C", "D"),
        basket_weights=(0.4, 0.2, 0.2, 0.2),
        spread=0.025,
        correlation=0.3,
        recovery=0.4,
    )

    outputs = NthToDefaultPayoff(spec).benchmark_outputs(_market_state())

    assert outputs["price"] > 0.0
    assert outputs["spread_cs01"] > 0.0


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


def test_price_nth_to_default_basket_accepts_generated_default_probability_aliases():
    pv = price_nth_to_default_basket(
        notional=1_000_000.0,
        n_names=4,
        n_th=2,
        maturity=5.0,
        default_prob=0.12,
        correlation=0.3,
        recovery=0.4,
    )

    assert pv > 0.0
