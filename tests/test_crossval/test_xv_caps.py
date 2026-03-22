"""XV3: Cap/Floor cross-validation against QuantLib."""

from datetime import date

import numpy as raw_np
import pytest

# --- Trellis ---
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.vol_surface import FlatVol

SETTLE = date(2024, 11, 15)


def trellis_cap_price(rate=0.05, vol=0.20, strike=0.05, years=5):
    spec = CapFloorSpec(
        notional=1_000_000, strike=strike,
        start_date=date(2025, 2, 15),
        end_date=date(2025 + years, 2, 15),
        frequency=Frequency.QUARTERLY,
    )
    ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                      discount=YieldCurve.flat(rate), vol_surface=FlatVol(vol))
    return price_payoff(CapPayoff(spec), ms)


def quantlib_cap_price(rate=0.05, vol=0.20, strike=0.05, years=5):
    import QuantLib as ql

    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today

    start = ql.Date(15, 2, 2025)
    end = ql.Date(15, 2, 2025 + years)

    calendar = ql.NullCalendar()
    schedule = ql.Schedule(
        start, end, ql.Period(ql.Quarterly),
        calendar, ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Forward, False,
    )

    rate_ts = ql.FlatForward(today, rate, ql.Actual360())
    vol_ts = ql.ConstantOptionletVolatility(today, calendar, ql.Unadjusted,
                                              vol, ql.Actual360())

    index = ql.IborIndex("dummy", ql.Period(3, ql.Months), 0,
                          ql.USDCurrency(), calendar,
                          ql.Unadjusted, False, ql.Actual360(),
                          ql.YieldTermStructureHandle(rate_ts))

    cap = ql.Cap(ql.IborLeg([1_000_000], schedule, index), [strike])
    engine = ql.BlackCapFloorEngine(
        ql.YieldTermStructureHandle(rate_ts),
        ql.OptionletVolatilityStructureHandle(vol_ts),
    )
    cap.setPricingEngine(engine)
    return cap.NPV()


class TestCapCrossValidation:

    def test_atm_cap_vs_quantlib(self):
        """ATM cap price: trellis vs QuantLib."""
        trellis_pv = trellis_cap_price(rate=0.05, vol=0.20, strike=0.05)
        ql_pv = quantlib_cap_price(rate=0.05, vol=0.20, strike=0.05)
        # Allow 5% relative tolerance (different schedule/day count handling)
        assert trellis_pv == pytest.approx(ql_pv, rel=0.05), (
            f"Trellis={trellis_pv:.2f}, QL={ql_pv:.2f}"
        )

    def test_itm_cap_vs_quantlib(self):
        """ITM cap (strike < forward)."""
        trellis_pv = trellis_cap_price(rate=0.06, vol=0.20, strike=0.04)
        ql_pv = quantlib_cap_price(rate=0.06, vol=0.20, strike=0.04)
        assert trellis_pv == pytest.approx(ql_pv, rel=0.05)

    def test_vol_sensitivity_agrees(self):
        """Both agree: higher vol → higher cap price."""
        for lib_fn in [trellis_cap_price, quantlib_cap_price]:
            p_low = lib_fn(vol=0.10)
            p_high = lib_fn(vol=0.30)
            assert p_high > p_low
