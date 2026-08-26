"""T49 raw CDO-tranche adapter regression tests."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.cdotranche import CDOTranchePayoff, CDOTrancheSpec
from trellis.models.credit_basket_copula import price_credit_basket_tranche_result


def _market_state() -> MarketState:
    settlement = date(2024, 11, 15)
    return MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        credit_curve=CreditCurve.flat(0.02, max_tenor=10.0),
    )


def test_t49_checked_student_t_adapter_matches_independent_reference():
    spec = CDOTrancheSpec(
        notional=100_000_000.0,
        n_names=100,
        attachment=0.03,
        detachment=0.07,
        end_date=date(2029, 11, 15),
        n_paths=20_000,
        seed=42,
    )
    market = _market_state()
    reference = price_credit_basket_tranche_result(
        market,
        spec,
        copula_family="student_t",
        degrees_of_freedom=spec.degrees_of_freedom,
        n_paths=spec.n_paths,
        seed=spec.seed,
    )

    outputs = CDOTranchePayoff(spec).benchmark_outputs(market)

    assert outputs["price"] == pytest.approx(reference.price)
    assert outputs["expected_loss_fraction"] == pytest.approx(
        reference.expected_loss_fraction
    )
    assert outputs["fair_spread_bp"] == pytest.approx(reference.fair_spread_bp)
