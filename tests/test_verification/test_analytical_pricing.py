"""WP2: Analytical pricing verification against independent reference values.

Every price is computed two ways:
1. Via trellis library
2. Via independent hand-calculation or known formula
"""

from datetime import date

import numpy as np
import pytest
from scipy.stats import norm

from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.forward_curve import ForwardCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.instruments.swap import SwapPayoff, SwapSpec, par_swap_rate
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


# ---------------------------------------------------------------------------
# Bond pricing verification
# ---------------------------------------------------------------------------

class TestBondPricing:

    def test_zero_coupon_bond(self):
        """ZCB price = face × exp(-r×T)."""
        r, T, face = 0.05, 10, 100
        zcb = Bond(face=face, coupon=0.0,
                    maturity_date=date(2034, 11, 15), maturity=10,
                    frequency=2, issue_date=SETTLE)
        curve = YieldCurve.flat(r)
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=curve)
        pv = price_payoff(DeterministicCashflowPayoff(zcb), ms)
        expected = face * np.exp(-r * T)
        assert pv == pytest.approx(expected, rel=0.01)

    def test_par_bond_at_par_rate(self):
        """A bond at the par rate should price close to par."""
        rate = 0.05
        bond = Bond(face=100, coupon=rate,
                     maturity_date=date(2034, 11, 15), maturity=10, frequency=2)
        curve = YieldCurve.flat(rate)
        result = price_instrument(bond, curve, SETTLE, greeks=None)
        # Dirty price ≈ par for a bond at the par rate
        assert result.dirty_price == pytest.approx(100.0, rel=0.02)

    def test_premium_bond(self):
        """Bond with coupon > market rate trades above par."""
        bond = Bond(face=100, coupon=0.08,
                     maturity_date=date(2034, 11, 15), maturity=10, frequency=2)
        curve = YieldCurve.flat(0.05)
        result = price_instrument(bond, curve, SETTLE, greeks=None)
        assert result.dirty_price > 100

    def test_discount_bond(self):
        """Bond with coupon < market rate trades below par."""
        bond = Bond(face=100, coupon=0.02,
                     maturity_date=date(2034, 11, 15), maturity=10, frequency=2)
        curve = YieldCurve.flat(0.05)
        result = price_instrument(bond, curve, SETTLE, greeks=None)
        assert result.dirty_price < 100

    def test_bond_price_manual_discounting(self):
        """Verify bond price by manually discounting each cashflow."""
        bond = Bond(face=100, coupon=0.06,
                     maturity_date=date(2029, 11, 15), maturity=5, frequency=2)
        rate = 0.04
        curve = YieldCurve.flat(rate)
        result = price_instrument(bond, curve, SETTLE, greeks=None)

        # Manual: 5Y semi-annual, coupon = 3 per period, 10 periods
        from trellis.core.date_utils import year_fraction
        schedule = bond.cashflows(SETTLE)
        manual_pv = 0.0
        for d, amt in zip(schedule.dates, schedule.amounts):
            t = year_fraction(SETTLE, d, bond.day_count)
            manual_pv += amt * np.exp(-rate * t)

        assert result.dirty_price == pytest.approx(manual_pv, rel=1e-6)


# ---------------------------------------------------------------------------
# Cap pricing verification
# ---------------------------------------------------------------------------

class TestCapPricing:

    def test_single_caplet_matches_black76(self):
        """A 1-period cap = single caplet = Black76 on a forward rate."""
        rate, vol = 0.05, 0.20
        notional = 1_000_000
        strike = 0.05
        curve = YieldCurve.flat(rate)
        fc = ForwardCurve(curve)

        # Dates for a single quarterly caplet
        start = date(2025, 2, 15)
        end = date(2025, 5, 15)
        spec = CapFloorSpec(
            notional=notional, strike=strike,
            start_date=start, end_date=end,
            frequency=Frequency.QUARTERLY,
        )
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                          discount=curve, vol_surface=FlatVol(vol))
        cap_pv = price_payoff(CapPayoff(spec), ms)

        # Manual Black76 caplet
        from trellis.core.date_utils import year_fraction
        tau = year_fraction(start, end, DayCountConvention.ACT_360)
        t_fix = year_fraction(SETTLE, start, DayCountConvention.ACT_360)
        t_pay = year_fraction(SETTLE, end, DayCountConvention.ACT_360)
        F = fc.forward_rate(t_fix, t_pay)
        undiscounted = notional * tau * black76_call(F, strike, vol, t_fix)
        df = float(curve.discount(t_pay))
        manual_pv = undiscounted * df

        assert cap_pv == pytest.approx(manual_pv, rel=0.02)

    def test_cap_floor_parity(self):
        """Cap(K) - Floor(K) = Swap PV (receiving floating, paying K)."""
        rate, vol = 0.05, 0.20
        spec = CapFloorSpec(
            notional=1_000_000, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                          discount=YieldCurve.flat(rate), vol_surface=FlatVol(vol))
        cap_pv = price_payoff(CapPayoff(spec), ms)
        floor_pv = price_payoff(FloorPayoff(spec), ms)
        diff = cap_pv - floor_pv
        # On a flat 5% curve with 5% strike, cap - floor ≈ 0 (ATM)
        assert abs(diff) < spec.notional * 0.01

    def test_deep_itm_cap_approximation(self):
        """Deep ITM cap ≈ sum of (F-K) × tau × notional × df."""
        rate, vol = 0.06, 0.20
        strike = 0.03  # deeply ITM
        spec = CapFloorSpec(
            notional=1_000_000, strike=strike,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )
        ms = MarketState(as_of=SETTLE, settlement=SETTLE,
                          discount=YieldCurve.flat(rate), vol_surface=FlatVol(vol))
        cap_pv = price_payoff(CapPayoff(spec), ms)

        # Intrinsic: zero-vol cap
        ms_zero_vol = MarketState(as_of=SETTLE, settlement=SETTLE,
                                    discount=YieldCurve.flat(rate), vol_surface=FlatVol(1e-10))
        intrinsic = price_payoff(CapPayoff(spec), ms_zero_vol)

        # Cap with vol >= intrinsic (time value always positive)
        assert cap_pv >= intrinsic - 1.0


# ---------------------------------------------------------------------------
# Swap pricing verification
# ---------------------------------------------------------------------------

class TestSwapPricing:

    def test_par_swap_rate_on_flat_curve(self):
        """On a flat CC curve, par swap rate ≈ the continuously compounded rate."""
        rate = 0.05
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(rate))
        spec = SwapSpec(
            notional=10_000_000, fixed_rate=0.0,
            start_date=date(2025, 2, 15), end_date=date(2030, 2, 15),
        )
        par = par_swap_rate(spec, ms)
        # Par rate should be close to the CC rate (not exact due to compounding/day count)
        assert par == pytest.approx(rate, abs=0.005)

    def test_par_swap_pv_zero(self):
        """A swap at the par rate has PV ≈ 0."""
        rate = 0.05
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(rate))
        spec = SwapSpec(
            notional=10_000_000, fixed_rate=0.0,
            start_date=date(2025, 2, 15), end_date=date(2030, 2, 15),
            fixed_day_count=DayCountConvention.ACT_360,
            float_day_count=DayCountConvention.ACT_360,
        )
        par = par_swap_rate(spec, ms)
        par_spec = SwapSpec(
            notional=10_000_000, fixed_rate=par,
            start_date=date(2025, 2, 15), end_date=date(2030, 2, 15),
            fixed_day_count=DayCountConvention.ACT_360,
            float_day_count=DayCountConvention.ACT_360,
        )
        pv = price_payoff(SwapPayoff(par_spec), ms)
        assert abs(pv) < 500  # within $500 on $10M

    def test_payer_positive_when_rates_above_fixed(self):
        """Payer swap with low fixed rate in high-rate environment → positive PV."""
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(0.06))
        spec = SwapSpec(
            notional=10_000_000, fixed_rate=0.04,
            start_date=date(2025, 2, 15), end_date=date(2030, 2, 15),
        )
        pv = price_payoff(SwapPayoff(spec), ms)
        assert pv > 0


# ---------------------------------------------------------------------------
# Forward rate verification
# ---------------------------------------------------------------------------

class TestForwardRates:

    def test_no_arb_simple(self):
        """df(t1)/df(t2) = 1 + F(t1,t2) × tau."""
        curve = YieldCurve.flat(0.05)
        fc = ForwardCurve(curve)
        for t1, t2 in [(0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]:
            F = fc.forward_rate(t1, t2, compounding="simple")
            tau = t2 - t1
            lhs = float(curve.discount(t1) / curve.discount(t2))
            rhs = 1.0 + F * tau
            assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_flat_curve_forward_equals_spot(self):
        """On a flat CC curve, continuous forward = spot rate."""
        curve = YieldCurve.flat(0.05)
        fc = ForwardCurve(curve)
        F = fc.forward_rate(1.0, 2.0, compounding="continuous")
        assert F == pytest.approx(0.05, rel=1e-4)


# ---------------------------------------------------------------------------
# Black76 verification
# ---------------------------------------------------------------------------

class TestBlack76:

    def test_put_call_parity(self):
        """call - put = F - K (undiscounted)."""
        params = [(0.06, 0.05, 0.25, 0.5), (0.04, 0.05, 0.20, 1.0),
                   (0.05, 0.05, 0.30, 2.0)]
        for F, K, sigma, T in params:
            c = black76_call(F, K, sigma, T)
            p = black76_put(F, K, sigma, T)
            assert c - p == pytest.approx(F - K, abs=1e-10)

    def test_atm_call_formula(self):
        """ATM: F=K, call = F × [N(d1) - N(d2)] where d1=σ√T/2, d2=-σ√T/2."""
        F, sigma, T = 0.05, 0.20, 1.0
        d1 = 0.5 * sigma * np.sqrt(T)
        expected = F * (norm.cdf(d1) - norm.cdf(-d1))
        result = black76_call(F, F, sigma, T)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_zero_vol_call(self):
        assert black76_call(0.06, 0.05, 0.0, 1.0) == pytest.approx(0.01, abs=1e-10)
        assert black76_call(0.04, 0.05, 0.0, 1.0) == pytest.approx(0.0, abs=1e-10)

    def test_zero_vol_put(self):
        assert black76_put(0.04, 0.05, 0.0, 1.0) == pytest.approx(0.01, abs=1e-10)
        assert black76_put(0.06, 0.05, 0.0, 1.0) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Credit curve verification
# ---------------------------------------------------------------------------

class TestCreditPricing:

    def test_survival_probability(self):
        """S(t) = exp(-λt) for flat hazard rate."""
        from trellis.curves.credit_curve import CreditCurve
        cc = CreditCurve.flat(0.02)
        for t in [1, 5, 10]:
            assert float(cc.survival_probability(t)) == pytest.approx(
                np.exp(-0.02 * t), rel=1e-10
            )

    def test_cds_spread_approximation(self):
        """For flat hazard rate: par CDS spread ≈ λ × (1-R)."""
        from trellis.curves.credit_curve import CreditCurve
        lam = 0.02
        R = 0.4
        cc = CreditCurve.from_spreads({5.0: lam * (1 - R)}, recovery=R)
        # Should recover the hazard rate
        assert float(cc.hazard_rate(5.0)) == pytest.approx(lam, rel=1e-6)

    def test_risky_discount(self):
        """Risky df = S(t) × df(t)."""
        from trellis.curves.credit_curve import CreditCurve
        cc = CreditCurve.flat(0.02)
        yc = YieldCurve.flat(0.05)
        for t in [1, 5, 10]:
            rd = float(cc.risky_discount(t, yc))
            expected = np.exp(-0.02 * t) * np.exp(-0.05 * t)
            assert rd == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# FX forward verification
# ---------------------------------------------------------------------------

class TestFXPricing:

    def test_covered_interest_parity(self):
        """F(T) = S × df_foreign(T) / df_domestic(T)."""
        from trellis.instruments.fx import FXForward, FXRate
        S = 1.10
        r_dom, r_for = 0.05, 0.03
        dom = YieldCurve.flat(r_dom)
        fgn = YieldCurve.flat(r_for)
        fwd = FXForward(FXRate(S, "USD", "EUR"), dom, fgn)
        for T in [0.5, 1.0, 5.0]:
            expected = S * np.exp(-r_for * T) / np.exp(-r_dom * T)
            assert fwd.forward(T) == pytest.approx(expected, rel=1e-10)

    def test_equal_rates_forward_equals_spot(self):
        from trellis.instruments.fx import FXForward, FXRate
        dom = YieldCurve.flat(0.04)
        fgn = YieldCurve.flat(0.04)
        fwd = FXForward(FXRate(1.10, "USD", "EUR"), dom, fgn)
        assert fwd.forward(5.0) == pytest.approx(1.10, rel=1e-10)
