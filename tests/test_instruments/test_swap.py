"""Tests for SwapPayoff and par_swap_rate."""

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.payoff import Payoff
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.swap import SwapPayoff, SwapSpec, par_swap_rate


SETTLE = date(2024, 11, 15)


def _ms(rate=0.05, forecast_curves=None):
    return MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(rate),
        forecast_curves=forecast_curves,
    )


def _swap_spec(fixed_rate=0.05, **overrides):
    defaults = dict(
        notional=1_000_000,
        fixed_rate=fixed_rate,
        start_date=date(2025, 2, 15),
        end_date=date(2030, 2, 15),
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
    )
    defaults.update(overrides)
    return SwapSpec(**defaults)


class TestSwapPayoff:

    def test_satisfies_protocol(self):
        assert isinstance(SwapPayoff(_swap_spec()), Payoff)

    def test_requirements(self):
        assert SwapPayoff(_swap_spec()).requirements == {"discount", "forward_rate"}

    def test_par_swap_pv_is_zero(self):
        ms = _ms(0.05)
        # Use same day count on both legs to get exact par
        spec = _swap_spec(
            fixed_rate=0.0,
            fixed_day_count=DayCountConvention.ACT_360,
            float_day_count=DayCountConvention.ACT_360,
        )
        rate = par_swap_rate(spec, ms)
        par_spec = _swap_spec(
            fixed_rate=rate,
            fixed_day_count=DayCountConvention.ACT_360,
            float_day_count=DayCountConvention.ACT_360,
        )
        pv = price_payoff(SwapPayoff(par_spec), ms)
        assert pv == pytest.approx(0.0, abs=100.0)  # within $100 on $1M

    def test_payer_vs_receiver(self):
        ms = _ms(0.05)
        payer = SwapPayoff(_swap_spec(fixed_rate=0.04, is_payer=True))
        receiver = SwapPayoff(_swap_spec(fixed_rate=0.04, is_payer=False))
        pv_payer = price_payoff(payer, ms)
        pv_receiver = price_payoff(receiver, ms)
        assert pv_payer == pytest.approx(-pv_receiver, abs=1.0)

    def test_atm_swap_near_zero(self):
        """On flat 5% curve, a swap at ~5% should have PV near zero."""
        ms = _ms(0.05)
        rate = par_swap_rate(_swap_spec(), ms)
        assert rate == pytest.approx(0.05, abs=0.005)

    def test_itm_payer_positive(self):
        """Payer swap with low fixed rate on higher curve → positive PV."""
        ms = _ms(0.06)
        pv = price_payoff(SwapPayoff(_swap_spec(fixed_rate=0.04)), ms)
        assert pv > 0

    def test_otm_payer_negative(self):
        """Payer swap with high fixed rate on lower curve → negative PV."""
        ms = _ms(0.03)
        pv = price_payoff(SwapPayoff(_swap_spec(fixed_rate=0.06)), ms)
        assert pv < 0


class TestMultiCurveSwap:

    def test_multi_curve_par_rate(self):
        """Par rate reflects forecast curve, not discount curve."""
        discount = YieldCurve.flat(0.04)
        forecast = YieldCurve.flat(0.05)

        ms_single = MarketState(
            as_of=SETTLE, settlement=SETTLE, discount=discount,
        )
        ms_multi = MarketState(
            as_of=SETTLE, settlement=SETTLE, discount=discount,
            forecast_curves={"USD-SOFR-3M": forecast},
        )

        spec = _swap_spec(rate_index="USD-SOFR-3M")
        rate_single = par_swap_rate(spec, ms_single)
        rate_multi = par_swap_rate(spec, ms_multi)

        # Multi-curve rate should be higher (forecast at 5% vs discount at 4%)
        assert rate_multi > rate_single
        assert rate_multi == pytest.approx(0.05, abs=0.005)
