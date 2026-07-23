"""Tests for the admitted ranked-observation basket adapter composition."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


def test_ranked_observation_adapter_composes_process_engine_and_state_payoff(monkeypatch):
    from trellis.instruments._agent import priceapayoff as adapter

    settlement = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(0.05),
        underlier_spots={"SPX": 100.0, "NDX": 101.5},
        vol_surface=FlatVol(0.20),
        model_parameters={"correlation_matrix": 0.35},
    )
    spec = adapter.HimalayaBasketSpec(
        underlyings="SPX,NDX",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
        expiry_date=date(2025, 11, 15),
        notional=100.0,
        correlation_key=None,
        rate_index=None,
        strike=0.05,
    )
    captured = {}

    class FakeEngine:
        def __init__(self, process, **kwargs):
            captured["process"] = process
            captured["engine_kwargs"] = kwargs

        def price(self, x0, T, payoff, **kwargs):
            captured["x0"] = tuple(x0)
            captured["T"] = T
            captured["payoff"] = payoff
            captured["price_kwargs"] = kwargs
            return {"price": 0.06}

    monkeypatch.setattr(adapter, "MonteCarloEngine", FakeEngine)

    value = adapter.HimalayaBasketPayoff(spec).evaluate(market_state)

    expected_df = market_state.discount.discount(1.0)
    assert value == pytest.approx(100.0 * expected_df * 0.06)
    assert captured["process"].state_dim == 2
    assert captured["x0"] == pytest.approx((100.0, 101.5))
    assert captured["T"] == pytest.approx(1.0)
    assert captured["engine_kwargs"]["n_steps"] == 252
    assert captured["payoff"].path_requirement.snapshot_steps
    assert captured["price_kwargs"]["storage_policy"] is captured["payoff"].path_requirement


def test_admitted_ranked_observation_source_has_no_product_pricing_authority():
    from pathlib import Path

    source = Path(adapter_path()).read_text()

    assert "price_ranked_observation_basket_monte_carlo" not in source
    assert "RankedObservationBasketMonteCarloPayoff" not in source
    assert "build_ranked_observation_basket_process" not in source
    assert "resolve_basket_semantics(" in source
    assert "CorrelatedGBM(" in source
    assert "MonteCarloEngine(" in source
    assert "build_ranked_observation_basket_state_payoff(" in source


def adapter_path() -> str:
    from trellis.instruments._agent import priceapayoff as adapter

    return adapter.__file__
