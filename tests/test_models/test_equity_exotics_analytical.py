from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical import price_equity_cliquet_option_analytical
from trellis.models.monte_carlo.event_aware import price_equity_cliquet_option_monte_carlo
from trellis.models.vol_surface import FlatVol


def _require_financepy() -> None:
    pytest.importorskip("financepy")


def test_cliquet_analytical_matches_financepy_reset_schedule_reference():
    _require_financepy()
    from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_cliquet_option import EquityCliquetOption
    from financepy.utils.date import Date
    from financepy.utils.day_count import DayCountTypes
    from financepy.utils.frequency import FrequencyTypes
    from financepy.utils.global_types import OptionTypes

    valuation_date = date(2024, 11, 15)
    observation_dates = (
        date(2024, 11, 18),
        date(2025, 2, 17),
        date(2025, 5, 19),
        date(2025, 8, 18),
        date(2025, 11, 17),
    )
    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        expiry_date=observation_dates[-1],
        observation_dates=observation_dates,
        option_type="call",
        day_count=DayCountConvention.THIRTY_E_360,
        time_day_count=DayCountConvention.ACT_365,
    )
    market_state = MarketState(
        as_of=valuation_date,
        settlement=valuation_date,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.2),
        model_parameters={"underlier_carry_rates": {"SPX": 0.0}},
    )

    trellis_price = price_equity_cliquet_option_analytical(market_state, spec)

    value_dt = Date(15, 11, 2024)
    financepy_option = EquityCliquetOption(
        value_dt,
        value_dt.add_tenor("12M"),
        OptionTypes.EUROPEAN_CALL,
        FrequencyTypes.QUARTERLY,
        DayCountTypes.THIRTY_E_360,
    )
    financepy_price = financepy_option.value(
        value_dt,
        100.0,
        DiscountCurveFlat(value_dt, 0.05),
        DiscountCurveFlat(value_dt, 0.0),
        BlackScholes(0.2),
    )

    assert trellis_price == pytest.approx(financepy_price, rel=2e-4)


def test_capped_floored_cliquet_analytical_tracks_monte_carlo():
    valuation_date = date(2024, 11, 15)
    observation_dates = (
        date(2025, 2, 17),
        date(2025, 5, 19),
        date(2025, 8, 18),
        date(2025, 11, 17),
    )
    market_state = MarketState(
        as_of=valuation_date,
        settlement=valuation_date,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.2),
        model_parameters={"underlier_carry_rates": {"SPX": 0.0}},
    )
    bounded_spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        expiry_date=observation_dates[-1],
        observation_dates=observation_dates,
        option_type="call",
        local_cap=0.08,
        local_floor=0.0,
        global_cap=0.2,
        global_floor=0.0,
        day_count=DayCountConvention.THIRTY_E_360,
        time_day_count=DayCountConvention.ACT_365,
        quadrature_order=21,
        n_paths=160000,
        seed=17,
    )
    unbounded_spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        expiry_date=observation_dates[-1],
        observation_dates=observation_dates,
        option_type="call",
        day_count=DayCountConvention.THIRTY_E_360,
        time_day_count=DayCountConvention.ACT_365,
    )

    analytical = price_equity_cliquet_option_analytical(market_state, bounded_spec)
    monte_carlo = price_equity_cliquet_option_monte_carlo(
        market_state,
        bounded_spec,
        n_paths=160000,
        seed=17,
    )
    unbounded = price_equity_cliquet_option_analytical(market_state, unbounded_spec)

    assert analytical > 0.0
    assert analytical < unbounded
    assert monte_carlo == pytest.approx(analytical, rel=0.06)


def test_capped_floored_cliquet_analytical_bounds_quadrature_grid():
    valuation_date = date(2024, 11, 15)
    market_state = MarketState(
        as_of=valuation_date,
        settlement=valuation_date,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.2),
    )
    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        expiry_date=date(2025, 11, 17),
        observation_times=(0.25, 0.5, 0.75, 1.0),
        option_type="call",
        local_cap=0.08,
        local_floor=0.0,
        global_cap=0.2,
        global_floor=0.0,
        day_count=DayCountConvention.THIRTY_E_360,
        time_day_count=DayCountConvention.ACT_365,
        quadrature_order=9,
        max_quadrature_nodes=100,
    )

    with pytest.raises(ValueError, match="quadrature grid is too large"):
        price_equity_cliquet_option_analytical(market_state, spec)
