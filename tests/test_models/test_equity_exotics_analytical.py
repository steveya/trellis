from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.cliquetoption import CliquetOptionSpec
from trellis.models.analytical import price_equity_cliquet_option_analytical
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
    spec = CliquetOptionSpec(
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
