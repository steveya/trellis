"""T73: authored European swaption comparison proof.

The proof compares three Trellis-native lanes over one exact contract and one
named market regime. External-library parity is intentionally outside this
bounded legacy proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_style_swaption import (
    price_swaption_black76,
    price_swaption_monte_carlo,
    resolve_swaption_monte_carlo_problem,
)
from trellis.models.rate_style_swaption_tree import price_swaption_tree
from trellis.models.vol_surface import FlatVol


VALUATION_DATE = date(2024, 11, 15)
MEAN_REVERSION = 0.05
SIGMA = 0.01


@dataclass(frozen=True)
class _T73SwaptionSpec:
    notional: float = 1_000_000.0
    strike: float = 0.03
    expiry_date: date = date(2025, 11, 15)
    swap_start: date = date(2025, 11, 15)
    swap_end: date = date(2030, 11, 15)
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.THIRTY_360
    float_frequency: Frequency = Frequency.QUARTERLY
    float_day_count: DayCountConvention = DayCountConvention.ACT_360
    model_time_day_count: DayCountConvention = DayCountConvention.THIRTY_360
    rate_index: str = "USD-SOFR-3M"
    is_payer: bool = True
    tree_steps: int = 120
    n_paths: int = 20_000
    n_steps: int = 64
    seed: int = 42


def _t73_market_state() -> MarketState:
    return MarketState(
        as_of=VALUATION_DATE,
        settlement=VALUATION_DATE,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        forecast_curves={
            "USD-SOFR-3M": YieldCurve.flat(0.0425, max_tenor=10.0),
        },
        vol_surface=FlatVol(0.20),
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        },
        model_parameters={
            "family": "hull_white_1f",
            "mean_reversion": MEAN_REVERSION,
            "sigma": SIGMA,
        },
    )


def test_t73_uses_distinct_fixed_and_floating_leg_timelines():
    spec = _T73SwaptionSpec()
    problem = resolve_swaption_monte_carlo_problem(
        _t73_market_state(),
        spec,
        n_steps=spec.n_steps,
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
    )

    assert problem.event_timeline is not None
    settlement = next(
        event
        for event in problem.event_timeline.events
        if event.name == "swaption_settlement"
    )
    assert len(settlement.payload["payment_times"]) == 10
    assert len(settlement.payload["floating_payment_times"]) == 20
    assert len(settlement.payload["floating_accrual_fractions"]) == 20


def test_t73_black_tree_and_seeded_monte_carlo_meet_authored_tolerances():
    market_state = _t73_market_state()
    spec = _T73SwaptionSpec()

    black_price = price_swaption_black76(
        market_state,
        spec,
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
    )
    tree_price = price_swaption_tree(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
        n_steps=spec.tree_steps,
    )
    monte_carlo_price = price_swaption_monte_carlo(
        market_state,
        spec,
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
        n_paths=spec.n_paths,
        n_steps=spec.n_steps,
        seed=spec.seed,
    )

    assert black_price > 0.0
    assert tree_price > 0.0
    assert monte_carlo_price > 0.0
    assert black_price == pytest.approx(tree_price, rel=0.001)
    assert monte_carlo_price == pytest.approx(tree_price, rel=0.03)
