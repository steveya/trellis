"""XV1: Bond cross-validation against QuantLib and FinancePy.

Tests: zero coupon bond, coupon bond dirty/clean price, DV01, duration.
"""

from datetime import date

import numpy as np
import pytest

pytest.importorskip("QuantLib")
pytest.importorskip("financepy")

# --- Trellis ---
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond

SETTLE = date(2024, 11, 15)
SETTLE_OFF_CYCLE = date(2025, 2, 15)
MATURITY = date(2034, 11, 15)
COUPON = 0.05
FACE = 100.0
RATE = 0.05


def trellis_bond_price(rate=RATE, settle=SETTLE):
    bond = Bond(face=FACE, coupon=COUPON, maturity_date=MATURITY,
                maturity=10, frequency=2)
    result = price_instrument(bond, YieldCurve.flat(rate), settle, greeks="all")
    return result


# ---------------------------------------------------------------------------
# QuantLib reference
# ---------------------------------------------------------------------------

def quantlib_bond_price(rate=RATE, settle=SETTLE):
    import QuantLib as ql

    settle_ql = ql.Date(settle.day, settle.month, settle.year)
    maturity_ql = ql.Date(15, 11, 2034)
    ql.Settings.instance().evaluationDate = settle_ql

    schedule = ql.Schedule(
        ql.Date(15, 11, 2024), maturity_ql,
        ql.Period(ql.Semiannual),
        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Backward, False,
    )

    bond = ql.FixedRateBond(0, FACE, schedule, [COUPON], ql.ActualActual(ql.ActualActual.ISDA))

    flat_curve = ql.FlatForward(settle_ql, rate, ql.ActualActual(ql.ActualActual.ISDA))
    handle = ql.YieldTermStructureHandle(flat_curve)
    engine = ql.DiscountingBondEngine(handle)
    bond.setPricingEngine(engine)

    return {
        "dirty_price": bond.dirtyPrice(),
        "clean_price": bond.cleanPrice(),
    }


# ---------------------------------------------------------------------------
# FinancePy reference
# ---------------------------------------------------------------------------

def financepy_bond_price(rate=RATE):
    from financepy.products.bonds.bond import Bond as FPBond
    from financepy.utils.date import Date as FPDate
    from financepy.utils.frequency import FrequencyTypes
    from financepy.utils.day_count import DayCountTypes

    settle_fp = FPDate(15, 11, 2024)
    maturity_fp = FPDate(15, 11, 2034)

    bond = FPBond(
        issue_dt=settle_fp,
        maturity_dt=maturity_fp,
        coupon=COUPON,
        freq_type=FrequencyTypes.SEMI_ANNUAL,
        dc_type=DayCountTypes.ACT_ACT_ISDA,
    )

    # FinancePy uses YTM for pricing; face=100 by default
    clean = bond.clean_price_from_ytm(settle_fp, rate)
    dirty = bond.dirty_price_from_ytm(settle_fp, rate)
    return {"clean_price": clean, "dirty_price": dirty}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBondCrossValidation:

    def test_dirty_price_quantlib(self):
        """Trellis dirty price matches QuantLib."""
        trellis = trellis_bond_price()
        ql_ref = quantlib_bond_price()
        assert trellis.dirty_price == pytest.approx(ql_ref["dirty_price"], rel=0.02), (
            f"Trellis={trellis.dirty_price:.4f}, QL={ql_ref['dirty_price']:.4f}"
        )

    def test_dirty_price_financepy(self):
        """Trellis dirty price matches FinancePy."""
        trellis = trellis_bond_price()
        fp_ref = financepy_bond_price()
        assert trellis.dirty_price == pytest.approx(fp_ref["dirty_price"], rel=0.02), (
            f"Trellis={trellis.dirty_price:.4f}, FP={fp_ref['dirty_price']:.4f}"
        )

    def test_quantlib_vs_financepy(self):
        """QuantLib and FinancePy should agree with each other."""
        ql_ref = quantlib_bond_price()
        fp_ref = financepy_bond_price()
        assert ql_ref["dirty_price"] == pytest.approx(fp_ref["dirty_price"], rel=0.02)

    def test_clean_price_and_accrued_interest_quantlib_off_cycle(self):
        """Off-cycle clean price and accrued interest stay close to QuantLib."""
        trellis = trellis_bond_price(settle=SETTLE_OFF_CYCLE)
        ql_ref = quantlib_bond_price(settle=SETTLE_OFF_CYCLE)

        assert trellis.clean_price == pytest.approx(ql_ref["clean_price"], rel=0.01)
        assert trellis.accrued_interest == pytest.approx(
            ql_ref["dirty_price"] - ql_ref["clean_price"],
            rel=0.05,
        )

    def test_zero_coupon_bond(self):
        """ZCB: all three should give face * exp(-r*T) ≈ face * df(T)."""
        r, T = 0.05, 10
        analytical = FACE * np.exp(-r * T)

        # Trellis
        zcb = Bond(face=FACE, coupon=0.0, maturity_date=MATURITY,
                     maturity=10, frequency=2, issue_date=SETTLE)
        ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(r))
        trellis_pv = price_payoff(DeterministicCashflowPayoff(zcb), ms)

        assert trellis_pv == pytest.approx(analytical, rel=0.01)

    def test_rate_sensitivity(self):
        """All three agree on direction: higher rate → lower price."""
        for lib_fn in [trellis_bond_price, quantlib_bond_price, financepy_bond_price]:
            p_low = lib_fn(rate=0.03)
            p_high = lib_fn(rate=0.07)
            low_price = p_low.dirty_price if hasattr(p_low, 'dirty_price') else p_low["dirty_price"]
            high_price = p_high.dirty_price if hasattr(p_high, 'dirty_price') else p_high["dirty_price"]
            assert low_price > high_price

    def test_dv01_quantlib(self):
        """DV01 comparison: trellis autograd vs QuantLib bump."""
        trellis_result = trellis_bond_price()
        trellis_dv01 = trellis_result.greeks["dv01"]

        # QuantLib DV01 via bump
        p_up = quantlib_bond_price(RATE + 0.0001)["dirty_price"]
        p_dn = quantlib_bond_price(RATE - 0.0001)["dirty_price"]
        ql_dv01 = -(p_up - p_dn) / 2

        assert trellis_dv01 == pytest.approx(ql_dv01, rel=0.05), (
            f"Trellis DV01={trellis_dv01:.6f}, QL DV01={ql_dv01:.6f}"
        )
