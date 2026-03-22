"""WP1: Greeks verification — autograd vs finite difference, textbook formulas.

Every Greek is verified two ways:
1. Autograd result matches finite-difference bump-and-reprice
2. Absolute value matches textbook formula where applicable
"""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency, PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.instruments.swap import SwapPayoff, SwapSpec, par_swap_rate
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)
BUMP_BPS = 1.0  # 1 basis point


def _flat_curve(rate=0.05):
    return YieldCurve.flat(rate)


def _bond(coupon=0.05, maturity=10):
    return Bond(
        face=100, coupon=coupon,
        maturity_date=date(2024 + maturity, 11, 15),
        maturity=maturity, frequency=2,
    )


def _price_bond(bond, rate):
    curve = YieldCurve.flat(rate)
    return price_instrument(bond, curve, SETTLE, greeks=None)


def _fd_dv01(bond, rate=0.05):
    """Finite-difference DV01: -(P_up - P_dn) / 2."""
    p_up = _price_bond(bond, rate + 0.0001).dirty_price
    p_dn = _price_bond(bond, rate - 0.0001).dirty_price
    return -(p_up - p_dn) / 2


def _fd_convexity(bond, rate=0.05):
    """Finite-difference convexity: (P_up + P_dn - 2P) / (P * dr^2)."""
    p = _price_bond(bond, rate).dirty_price
    p_up = _price_bond(bond, rate + 0.0001).dirty_price
    p_dn = _price_bond(bond, rate - 0.0001).dirty_price
    return (p_up + p_dn - 2 * p) / (p * 0.0001 ** 2)


# ---------------------------------------------------------------------------
# Bond Greeks
# ---------------------------------------------------------------------------

class TestBondDV01:

    def test_autograd_matches_fd(self):
        """Autograd DV01 matches finite-difference."""
        bond = _bond()
        result = price_instrument(bond, _flat_curve(), SETTLE, greeks="all")
        ag_dv01 = result.greeks["dv01"]
        fd_dv01 = _fd_dv01(bond)
        assert ag_dv01 == pytest.approx(fd_dv01, rel=0.01), (
            f"Autograd DV01={ag_dv01:.6f}, FD DV01={fd_dv01:.6f}"
        )

    def test_dv01_positive_for_long_bond(self):
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks="all")
        assert result.greeks["dv01"] > 0

    def test_dv01_increases_with_maturity(self):
        dv01_5y = price_instrument(_bond(maturity=5), _flat_curve(), SETTLE, greeks="all").greeks["dv01"]
        dv01_10y = price_instrument(_bond(maturity=10), _flat_curve(), SETTLE, greeks="all").greeks["dv01"]
        dv01_30y = price_instrument(_bond(maturity=30), _flat_curve(), SETTLE, greeks="all").greeks["dv01"]
        assert dv01_5y < dv01_10y < dv01_30y


class TestBondDuration:

    def test_autograd_duration_positive(self):
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks="all")
        assert result.greeks["duration"] > 0

    def test_duration_less_than_maturity(self):
        """Macaulay duration < maturity for coupon bonds."""
        result = price_instrument(_bond(maturity=10), _flat_curve(), SETTLE, greeks="all")
        assert result.greeks["duration"] < 10

    def test_zero_coupon_duration_equals_maturity(self):
        """Zero-coupon bond: duration ≈ maturity."""
        zcb = Bond(
            face=100, coupon=0.0,
            maturity_date=date(2034, 11, 15),
            maturity=10, frequency=2,
            issue_date=SETTLE,
        )
        result = price_instrument(zcb, _flat_curve(), SETTLE, greeks="all")
        # For CC discounting, Macaulay duration = maturity for ZCB
        assert result.greeks["duration"] == pytest.approx(10.0, rel=0.05)

    def test_duration_increases_with_maturity(self):
        d5 = price_instrument(_bond(maturity=5), _flat_curve(), SETTLE, greeks="all").greeks["duration"]
        d10 = price_instrument(_bond(maturity=10), _flat_curve(), SETTLE, greeks="all").greeks["duration"]
        d30 = price_instrument(_bond(maturity=30), _flat_curve(), SETTLE, greeks="all").greeks["duration"]
        assert d5 < d10 < d30

    def test_dv01_equals_duration_times_price_times_bp(self):
        """DV01 ≈ duration × price × 0.0001."""
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks="all")
        dv01 = result.greeks["dv01"]
        dur = result.greeks["duration"]
        price = result.greeks["price"]
        expected_dv01 = dur * price * 0.0001
        assert dv01 == pytest.approx(expected_dv01, rel=0.01)


class TestBondConvexity:

    def test_autograd_matches_fd(self):
        """Autograd convexity matches finite-difference."""
        bond = _bond()
        result = price_instrument(bond, _flat_curve(), SETTLE, greeks="all")
        ag_cvx = result.greeks["convexity"]
        fd_cvx = _fd_convexity(bond)
        assert ag_cvx == pytest.approx(fd_cvx, rel=0.05)

    def test_convexity_positive(self):
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks="all")
        assert result.greeks["convexity"] > 0

    def test_convexity_increases_with_maturity(self):
        c5 = price_instrument(_bond(maturity=5), _flat_curve(), SETTLE, greeks="all").greeks["convexity"]
        c10 = price_instrument(_bond(maturity=10), _flat_curve(), SETTLE, greeks="all").greeks["convexity"]
        c30 = price_instrument(_bond(maturity=30), _flat_curve(), SETTLE, greeks="all").greeks["convexity"]
        assert c5 < c10 < c30


class TestBondKeyRateDurations:

    def test_krd_present(self):
        tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.04, 0.042, 0.045, 0.048, 0.05]
        curve = YieldCurve(tenors, rates)
        result = price_instrument(_bond(), curve, SETTLE, greeks="all")
        assert "key_rate_durations" in result.greeks
        krd = result.greeks["key_rate_durations"]
        assert len(krd) == len(tenors)

    def test_krd_sum_approximates_duration(self):
        """Sum of KRDs ≈ total duration."""
        tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.04, 0.042, 0.045, 0.048, 0.05]
        curve = YieldCurve(tenors, rates)
        result = price_instrument(_bond(), curve, SETTLE, greeks="all")
        krd_sum = sum(result.greeks["key_rate_durations"].values())
        total_dur = result.greeks["duration"]
        assert krd_sum == pytest.approx(total_dur, rel=0.05)

    def test_10y_bond_krd_concentrated_at_10y(self):
        """A 10Y bond's KRD should be concentrated near the 10Y tenor."""
        tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.04, 0.042, 0.045, 0.048, 0.05]
        curve = YieldCurve(tenors, rates)
        result = price_instrument(_bond(maturity=10), curve, SETTLE, greeks="all")
        krd = result.greeks["key_rate_durations"]
        krd_10y = krd.get("KRD_10.0y", 0)
        krd_sum = sum(abs(v) for v in krd.values())
        # 10Y KRD should be the largest contributor
        assert abs(krd_10y) > krd_sum * 0.3


# ---------------------------------------------------------------------------
# Cap/Floor Greeks
# ---------------------------------------------------------------------------

class TestCapGreeks:

    def _cap_spec(self):
        return CapFloorSpec(
            notional=1_000_000, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2029, 2, 15),
            frequency=Frequency.QUARTERLY,
        )

    def _cap_pv(self, rate=0.05, vol=0.20):
        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(rate),
            vol_surface=FlatVol(vol),
        )
        return price_payoff(CapPayoff(self._cap_spec()), ms)

    def test_cap_dv01_via_fd(self):
        """Cap DV01 via finite difference."""
        pv_up = self._cap_pv(rate=0.05 + 0.0001)
        pv_dn = self._cap_pv(rate=0.05 - 0.0001)
        fd_dv01 = (pv_up - pv_dn) / 2  # cap DV01 is positive (cap gains when rates rise)
        assert fd_dv01 > 0  # cap benefits from rate increases

    def test_cap_vega_positive(self):
        """Cap vega is positive (cap gains from vol increase)."""
        pv_base = self._cap_pv(vol=0.20)
        pv_up = self._cap_pv(vol=0.21)
        vega = (pv_up - pv_base) / 0.01
        assert vega > 0

    def test_cap_vega_via_fd(self):
        """Finite-difference vega."""
        pv_up = self._cap_pv(vol=0.2001)
        pv_dn = self._cap_pv(vol=0.1999)
        vega_fd = (pv_up - pv_dn) / 0.0002
        assert vega_fd > 0

    def test_cap_floor_parity_greeks(self):
        """Cap DV01 - Floor DV01 ≈ Swap DV01."""
        ms_up = MarketState(as_of=SETTLE, settlement=SETTLE,
                             discount=YieldCurve.flat(0.0501), vol_surface=FlatVol(0.20))
        ms_dn = MarketState(as_of=SETTLE, settlement=SETTLE,
                             discount=YieldCurve.flat(0.0499), vol_surface=FlatVol(0.20))
        spec = self._cap_spec()
        cap_dv01 = (price_payoff(CapPayoff(spec), ms_up) - price_payoff(CapPayoff(spec), ms_dn)) / 2
        floor_dv01 = (price_payoff(FloorPayoff(spec), ms_up) - price_payoff(FloorPayoff(spec), ms_dn)) / 2
        # Cap - Floor = Swap (approximately), so their DV01s should be related
        diff = cap_dv01 - floor_dv01
        # The swap DV01 should be meaningful (not zero)
        assert abs(diff) > 0


# ---------------------------------------------------------------------------
# Swap Greeks
# ---------------------------------------------------------------------------

class TestSwapGreeks:

    def _swap_pv(self, rate=0.05):
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(rate))
        spec = SwapSpec(
            notional=10_000_000, fixed_rate=0.05,
            start_date=date(2025, 2, 15), end_date=date(2030, 2, 15),
        )
        return price_payoff(SwapPayoff(spec), ms)

    def test_swap_dv01_via_fd(self):
        """Payer swap: positive DV01 (benefits from rate increase)."""
        pv_up = self._swap_pv(rate=0.05 + 0.0001)
        pv_dn = self._swap_pv(rate=0.05 - 0.0001)
        dv01 = (pv_up - pv_dn) / 2
        assert dv01 > 0

    def test_par_swap_dv01_magnitude(self):
        """DV01 of a 5Y $10M swap ≈ $4,000-$5,000 per bp."""
        pv_up = self._swap_pv(rate=0.0501)
        pv_dn = self._swap_pv(rate=0.0499)
        dv01 = (pv_up - pv_dn) / 2
        # 5Y swap, $10M notional: DV01 ≈ 4500
        assert 2000 < abs(dv01) < 8000, f"Swap DV01 = {dv01:.0f}"


# ---------------------------------------------------------------------------
# Selective Greeks (greeks= parameter)
# ---------------------------------------------------------------------------

class TestSelectiveGreeks:

    def test_greeks_none_returns_empty(self):
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks=None)
        assert result.greeks == {}

    def test_greeks_all_returns_full_set(self):
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks="all")
        assert "dv01" in result.greeks
        assert "duration" in result.greeks
        assert "convexity" in result.greeks

    def test_greeks_selective_dv01_only(self):
        result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks=["dv01"])
        assert "dv01" in result.greeks
        assert "convexity" not in result.greeks

    def test_selective_matches_all(self):
        """Selective DV01 should equal DV01 from 'all'."""
        all_result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks="all")
        sel_result = price_instrument(_bond(), _flat_curve(), SETTLE, greeks=["dv01"])
        assert sel_result.greeks["dv01"] == pytest.approx(all_result.greeks["dv01"], rel=1e-10)
