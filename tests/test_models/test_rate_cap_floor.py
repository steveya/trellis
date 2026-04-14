from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trellis.agent.benchmark_contracts import benchmark_spec_overrides
from trellis.agent.financepy_reference import price_financepy_reference
from trellis.agent.task_manifests import load_task_manifest
from trellis.core.market_state import MarketState
from trellis.core.date_utils import build_payment_timeline
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.cap import CapFloorSpec
from trellis.models.rate_cap_floor import (
    price_rate_cap_floor_strip_analytical,
    price_rate_cap_floor_strip_monte_carlo,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)
ROOT = Path(__file__).resolve().parents[2]


def _market_state(rate=0.05, vol=0.20):
    curve = YieldCurve.flat(rate)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        vol_surface=FlatVol(vol),
    )


def _cap_spec(**overrides):
    defaults = dict(
        notional=1_000_000,
        strike=0.04,
        start_date=date(2025, 2, 15),
        end_date=date(2030, 2, 15),
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        rate_index=None,
    )
    defaults.update(overrides)
    return CapFloorSpec(**defaults)


def test_rate_cap_floor_strip_monte_carlo_tracks_analytical_reference():
    market_state = _market_state(rate=0.05, vol=0.20)
    spec = _cap_spec()

    analytical = price_rate_cap_floor_strip_analytical(
        market_state,
        spec,
        instrument_class="cap",
    )
    monte_carlo = price_rate_cap_floor_strip_monte_carlo(
        market_state,
        spec,
        instrument_class="cap",
        n_paths=20000,
        seed=7,
    )

    assert analytical > 0.0
    assert monte_carlo > 0.0
    assert monte_carlo == pytest.approx(analytical, rel=0.20)


def test_rate_cap_floor_strip_helpers_accept_keyword_contract_fields():
    market_state = _market_state(rate=0.05, vol=0.20)
    spec = _cap_spec(rate_index="USD-SOFR-3M")

    analytical_from_spec = price_rate_cap_floor_strip_analytical(
        market_state,
        spec,
        instrument_class="cap",
    )
    monte_carlo_from_spec = price_rate_cap_floor_strip_monte_carlo(
        market_state,
        spec,
        instrument_class="cap",
        n_paths=20000,
        seed=7,
    )

    analytical_from_kwargs = price_rate_cap_floor_strip_analytical(
        market_state=market_state,
        instrument_class="cap",
        notional=spec.notional,
        strike=spec.strike,
        start_date=spec.start_date,
        end_date=spec.end_date,
        frequency=spec.frequency,
        day_count=spec.day_count,
        rate_index=spec.rate_index,
    )
    monte_carlo_from_kwargs = price_rate_cap_floor_strip_monte_carlo(
        market_state=market_state,
        instrument_class="cap",
        notional=spec.notional,
        strike=spec.strike,
        start_date=spec.start_date,
        end_date=spec.end_date,
        frequency=spec.frequency,
        day_count=spec.day_count,
        rate_index=spec.rate_index,
        n_paths=20000,
        seed=7,
    )

    assert analytical_from_kwargs == pytest.approx(analytical_from_spec)
    assert monte_carlo_from_kwargs == pytest.approx(monte_carlo_from_spec)


def test_rate_cap_floor_strip_includes_known_first_fixing_intrinsic():
    market_state = _market_state(rate=0.05, vol=0.20)
    today_start = _cap_spec(start_date=SETTLE, end_date=date(2025, 11, 15))
    later_start = _cap_spec(start_date=date(2025, 2, 15), end_date=date(2025, 11, 15))

    with_first = price_rate_cap_floor_strip_analytical(market_state, today_start, instrument_class="cap")
    without_first = price_rate_cap_floor_strip_analytical(market_state, later_start, instrument_class="cap")

    timeline = build_payment_timeline(
        today_start.start_date,
        later_start.start_date,
        today_start.frequency,
        day_count=today_start.day_count,
        time_origin=market_state.settlement,
        label="cap_floor_known_fixing",
    )
    first_period = timeline[0]
    payment_years = float(first_period.t_payment or 0.0)
    expected_first_caplet = (
        today_start.notional
        * float(first_period.accrual_fraction or 0.0)
        * float(market_state.discount.discount(payment_years))
        * max(
            float(market_state.forward_curve.forward_rate(0.0, payment_years))
            - today_start.strike,
            0.0,
        )
    )

    assert with_first > without_first
    assert with_first - without_first == pytest.approx(expected_first_caplet, rel=1e-6)


def test_rate_cap_floor_strip_matches_financepy_flat_curve_conventions_with_date_aware_curves():
    curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0425,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        forecast_curves={"USD-SOFR-3M": curve},
        vol_surface=FlatVol(0.20),
    )
    spec = _cap_spec(
        start_date=SETTLE,
        end_date=date(2029, 11, 15),
        rate_index="USD-SOFR-3M",
        calendar_name="weekend_only",
        business_day_adjustment="following",
    )

    price = price_rate_cap_floor_strip_analytical(
        market_state,
        spec,
        instrument_class="cap",
    )

    assert price == pytest.approx(26125.903527846524, rel=1e-6)


@pytest.mark.parametrize("task_id", ["F004", "F005"])
def test_rate_cap_floor_strip_honors_benchmark_model_semantics(task_id: str):
    pytest.importorskip("financepy")

    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_BENCHMARK_FINANCEPY.yaml", root=ROOT)
    }
    task = tasks[task_id]
    overrides = benchmark_spec_overrides(task, root=ROOT)
    instrument_class = overrides.pop("instrument_class")
    spec = CapFloorSpec(**overrides)
    curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0425,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        forecast_curves={"USD-SOFR-3M": curve},
        vol_surface=FlatVol(0.20),
        model_parameters={
            "shifted_black_vol": 0.19,
            "shift": 0.01,
            "sabr": {
                "alpha": 0.025,
                "beta": 0.5,
                "rho": -0.2,
                "nu": 0.35,
            },
        },
    )

    trellis_price = price_rate_cap_floor_strip_analytical(
        market_state,
        spec,
        instrument_class=instrument_class or "cap",
    )
    financepy_price = price_financepy_reference(task, root=ROOT)["outputs"]["price"]

    assert trellis_price == pytest.approx(financepy_price, rel=1e-3)
