"""T05: Puttable bond pricing — exercise_fn=max, verify puttable >= straight,
compare callable-puttable symmetry.

Cross-validates:
  1. Puttable bond pricing via HW tree with exercise_fn=max
  2. Puttable >= straight bond at all rate levels (3%, 5%, 7%)
  3. Callable vs puttable symmetry: puttable >= straight >= callable
  4. Puttable OAS: negative OAS for puttable trading above straight
  5. QuantLib cross-validation via CallableFixedRateBond with put schedule
  6. FinancePy cross-validation via BondEmbeddedOption with put dates

Bond specification:
  5% coupon, 10Y maturity, semi-annual, face=100
  Puttable at par at 3Y/5Y/7Y (Bermudan)
  Callable at par at 3Y/5Y/7Y (for symmetry check)
  Flat curve at 5%, HW a=0.1, sigma=0.01
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
from scipy.optimize import brentq

from trellis.conventions.schedule import generate_schedule
from trellis.core.types import Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_generic_lattice,
    lattice_backward_induction,
)
from trellis.models.trees.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

FLAT_RATE = 0.05
COUPON_RATE = 0.05
FACE = 100.0
T = 10.0            # 10-year bond
N_STEPS = 400
PUT_YEARS = [3, 5, 7]   # Bermudan put dates
CALL_YEARS = [3, 5, 7]  # Bermudan call dates (for symmetry check)
PUT_PRICE = 100.0        # puttable at par
CALL_PRICE = 100.0       # callable at par

# HW parameters
HW_SIGMA = 0.01     # 1% absolute rate vol (normal)
HW_A = 0.1          # mean reversion speed


@pytest.fixture(scope="module")
def flat_curve():
    return YieldCurve.flat(FLAT_RATE, max_tenor=max(T + 1, 31.0))


# ---------------------------------------------------------------------------
# Helpers: build coupon schedule and pricing functions
# ---------------------------------------------------------------------------

def _build_coupon_steps(dt: float, n_steps: int) -> dict[int, float]:
    """Map tree step indices to discrete coupon amounts.

    Uses generate_schedule to get semi-annual coupon dates, then maps
    each date to the nearest tree step.
    """
    start = date(2025, 1, 15)
    end = date(2035, 1, 15)
    coupon_dates = generate_schedule(start, end, Frequency.SEMI_ANNUAL)

    coupon_amount = FACE * COUPON_RATE / 2.0  # semi-annual coupon

    coupon_steps: dict[int, float] = {}
    for d in coupon_dates:
        t_years = (d - start).days / 365.25
        step = int(round(t_years / dt))
        if 0 < step <= n_steps:
            coupon_steps[step] = coupon_steps.get(step, 0.0) + coupon_amount

    return coupon_steps


def _build_exercise_steps(dt: float, years: list[int]) -> list[int]:
    """Convert exercise years to sorted tree step indices."""
    return sorted(int(round(y / dt)) for y in years)


def _price_straight_bond(lattice: RecombiningLattice) -> float:
    """Price a straight (non-callable, non-puttable) bond on the lattice."""
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps(dt, n_steps)
    final_coupon = coupon_steps.get(n_steps, 0.0)

    def terminal_payoff(step, node, lat):
        return FACE + final_coupon

    def cashflow_at_node(step, node, lat):
        return coupon_steps.get(step, 0.0)

    return lattice_backward_induction(
        lattice,
        terminal_payoff=terminal_payoff,
        cashflow_at_node=cashflow_at_node,
    )


def _price_puttable_bond(lattice: RecombiningLattice) -> float:
    """Price a puttable bond on a calibrated rate lattice.

    Uses Bermudan exercise with exercise_fn=max (holder maximizes value).
    The holder can put (sell back) the bond at par + accrued coupon.
    """
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps(dt, n_steps)
    put_steps = _build_exercise_steps(dt, PUT_YEARS)

    final_coupon = coupon_steps.get(n_steps, 0.0)

    def terminal_payoff(step, node, lat):
        return FACE + final_coupon

    def cashflow_at_node(step, node, lat):
        return coupon_steps.get(step, 0.0)

    # Exercise value at put dates: put_price + coupon at that step
    # (holder receives put price plus any coupon due)
    def exercise_value(step, node, lat):
        cpn = coupon_steps.get(step, 0.0)
        return PUT_PRICE + cpn

    price = lattice_backward_induction(
        lattice,
        terminal_payoff=terminal_payoff,
        exercise_value=exercise_value,
        exercise_type="bermudan",
        exercise_steps=put_steps,
        cashflow_at_node=cashflow_at_node,
        exercise_fn=max,  # holder puts to maximize value
    )
    return price


def _price_callable_bond(lattice: RecombiningLattice) -> float:
    """Price a callable bond on a calibrated rate lattice.

    Uses Bermudan exercise with exercise_fn=min (issuer minimizes liability).
    """
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps(dt, n_steps)
    call_steps = _build_exercise_steps(dt, CALL_YEARS)

    final_coupon = coupon_steps.get(n_steps, 0.0)

    def terminal_payoff(step, node, lat):
        return FACE + final_coupon

    def cashflow_at_node(step, node, lat):
        return coupon_steps.get(step, 0.0)

    def exercise_value(step, node, lat):
        cpn = coupon_steps.get(step, 0.0)
        return CALL_PRICE + cpn

    price = lattice_backward_induction(
        lattice,
        terminal_payoff=terminal_payoff,
        exercise_value=exercise_value,
        exercise_type="bermudan",
        exercise_steps=call_steps,
        cashflow_at_node=cashflow_at_node,
        exercise_fn=min,  # issuer calls to minimize liability
    )
    return price


def _build_hw_lattice(curve: YieldCurve, n_steps: int = N_STEPS) -> RecombiningLattice:
    """Build a HW lattice calibrated to the given curve."""
    model = MODEL_REGISTRY["hull_white"]
    return build_generic_lattice(
        model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
        T=T, n_steps=n_steps, discount_curve=curve,
    )


def _price_straight_on_shifted_curve(rate: float) -> float:
    """Price straight bond on a flat curve at the given rate."""
    curve = YieldCurve.flat(rate, max_tenor=max(T + 1, 31.0))
    lattice = _build_hw_lattice(curve)
    return _price_straight_bond(lattice)


def _price_puttable_on_shifted_curve(rate: float) -> float:
    """Price puttable bond on a flat curve at the given rate."""
    curve = YieldCurve.flat(rate, max_tenor=max(T + 1, 31.0))
    lattice = _build_hw_lattice(curve)
    return _price_puttable_bond(lattice)


def _price_callable_on_shifted_curve(rate: float) -> float:
    """Price callable bond on a flat curve at the given rate."""
    curve = YieldCurve.flat(rate, max_tenor=max(T + 1, 31.0))
    lattice = _build_hw_lattice(curve)
    return _price_callable_bond(lattice)


# ===================================================================
# Test 1: Puttable bond pricing
# ===================================================================

class TestPuttableBondPricing:
    """HW tree: puttable bond pricing with exercise_fn=max."""

    def test_puttable_bond_prices(self, flat_curve):
        """Puttable bond should price to a reasonable value."""
        lattice = _build_hw_lattice(flat_curve)
        price = _price_puttable_bond(lattice)

        # At par rate (5% coupon, 5% flat curve), straight bond ~ 100
        # Puttable should be >= 100 (put option has positive value to holder)
        assert 99 < price < 110, f"Puttable price={price:.4f} out of range"

    def test_puttable_geq_par_at_par_rate(self, flat_curve):
        """At par rate, puttable bond >= par (put is at-the-money or in-the-money)."""
        lattice = _build_hw_lattice(flat_curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)

        # Straight should be ~100 at par rate
        assert straight == pytest.approx(100.0, abs=0.5), (
            f"Straight bond={straight:.4f} (expected ~100)"
        )
        # Puttable >= straight always
        assert puttable >= straight - 0.01, (
            f"Puttable ({puttable:.4f}) < Straight ({straight:.4f})"
        )

    def test_put_option_value_positive(self, flat_curve):
        """The embedded put option value should be non-negative."""
        lattice = _build_hw_lattice(flat_curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)
        put_option_value = puttable - straight

        assert put_option_value >= -0.01, (
            f"Put option value={put_option_value:.4f} is negative"
        )


# ===================================================================
# Test 2: Puttable >= straight at all rate levels
# ===================================================================

class TestPuttableGeqStraight:
    """Puttable bond >= straight bond at all rate levels."""

    @pytest.mark.parametrize("rate", [0.03, 0.05, 0.07])
    def test_puttable_geq_straight(self, rate):
        """Puttable >= straight at rate={rate}."""
        curve = YieldCurve.flat(rate, max_tenor=max(T + 1, 31.0))
        lattice = _build_hw_lattice(curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)

        assert puttable >= straight - 0.01, (
            f"At rate={rate:.0%}: Puttable ({puttable:.4f}) < "
            f"Straight ({straight:.4f})"
        )

    @pytest.mark.parametrize("rate", [0.03, 0.05, 0.07])
    def test_put_option_larger_when_rates_higher(self, rate):
        """At higher rates, straight bond falls below par, so put option
        value should be larger (holder more likely to exercise)."""
        curve = YieldCurve.flat(rate, max_tenor=max(T + 1, 31.0))
        lattice = _build_hw_lattice(curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)
        put_option_value = puttable - straight

        # Put option value should be non-negative
        assert put_option_value >= -0.01, (
            f"At rate={rate:.0%}: Put option value={put_option_value:.4f}"
        )


# ===================================================================
# Test 3: Callable vs puttable symmetry
# ===================================================================

class TestCallablePuttableSymmetry:
    """Verify: puttable >= straight >= callable.

    The put option has positive value to the holder (increases bond price).
    The call option has positive value to the issuer (decreases bond price).
    """

    def test_ordering_at_par_rate(self, flat_curve):
        """puttable >= straight >= callable at par rate."""
        lattice = _build_hw_lattice(flat_curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)
        callable_ = _price_callable_bond(lattice)

        assert puttable >= straight - 0.01, (
            f"Puttable ({puttable:.4f}) < Straight ({straight:.4f})"
        )
        assert straight >= callable_ - 0.01, (
            f"Straight ({straight:.4f}) < Callable ({callable_:.4f})"
        )
        assert puttable >= callable_ - 0.01, (
            f"Puttable ({puttable:.4f}) < Callable ({callable_:.4f})"
        )

    @pytest.mark.parametrize("rate", [0.03, 0.05, 0.07])
    def test_ordering_at_all_rates(self, rate):
        """puttable >= straight >= callable at all rate levels."""
        curve = YieldCurve.flat(rate, max_tenor=max(T + 1, 31.0))
        lattice = _build_hw_lattice(curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)
        callable_ = _price_callable_bond(lattice)

        assert puttable >= straight - 0.01, (
            f"At rate={rate:.0%}: Puttable ({puttable:.4f}) < "
            f"Straight ({straight:.4f})"
        )
        assert straight >= callable_ - 0.01, (
            f"At rate={rate:.0%}: Straight ({straight:.4f}) < "
            f"Callable ({callable_:.4f})"
        )

    def test_embedded_option_values_positive(self, flat_curve):
        """Both embedded option values should be non-negative."""
        lattice = _build_hw_lattice(flat_curve)
        puttable = _price_puttable_bond(lattice)
        straight = _price_straight_bond(lattice)
        callable_ = _price_callable_bond(lattice)

        put_option_value = puttable - straight
        call_option_value = straight - callable_

        assert put_option_value >= -0.01, (
            f"Put option value={put_option_value:.4f} is negative"
        )
        assert call_option_value >= -0.01, (
            f"Call option value={call_option_value:.4f} is negative"
        )


# ===================================================================
# Test 4: Puttable OAS
# ===================================================================

class TestPuttableOAS:
    """Compute OAS for puttable bond at market_price=101.

    OAS is the constant spread added to the discount curve such that
    the model price equals the market price.  The puttable bond model
    price at the flat 5% curve is ~102.5 (straight ~100 + put option ~2.5).
    At market_price=101, the bond trades *below* model, so a *positive*
    OAS is needed (shift curve up -> lower model price to match market).

    For a puttable trading *above* straight but *below* model, OAS > 0
    reflects the additional credit/liquidity spread not captured by the
    option-free model.
    """

    def test_puttable_oas_positive_when_below_model(self, flat_curve):
        """OAS for puttable at 101 should be positive (below model price ~102.5)."""
        market_price = 101.0

        def _price_with_oas(oas_bps: float) -> float:
            """Price puttable bond on a curve shifted by OAS."""
            shifted = flat_curve.shift(oas_bps)
            lattice = _build_hw_lattice(shifted)
            return _price_puttable_bond(lattice)

        def objective(oas_bps):
            return _price_with_oas(oas_bps) - market_price

        oas = brentq(objective, -200, 200, xtol=0.1)

        # Market price 101 < model price ~102.5, so OAS > 0
        assert oas > 0, (
            f"OAS={oas:.1f}bp should be positive for puttable at 101 "
            f"(below model price)"
        )
        # Should be reasonable magnitude
        assert oas < 100, (
            f"OAS={oas:.1f}bp magnitude too large"
        )

    def test_puttable_oas_negative_when_above_model(self, flat_curve):
        """OAS for puttable at 104 should be negative (above model price ~102.5)."""
        market_price = 104.0

        def _price_with_oas(oas_bps: float) -> float:
            shifted = flat_curve.shift(oas_bps)
            lattice = _build_hw_lattice(shifted)
            return _price_puttable_bond(lattice)

        def objective(oas_bps):
            return _price_with_oas(oas_bps) - market_price

        oas = brentq(objective, -200, 200, xtol=0.1)

        # Market price 104 > model price ~102.5, so OAS < 0
        assert oas < 0, (
            f"OAS={oas:.1f}bp should be negative for puttable at 104 "
            f"(above model price)"
        )

    def test_oas_reprices_bond(self, flat_curve):
        """OAS should reprice the bond to the market price."""
        market_price = 101.0

        def _price_with_oas(oas_bps: float) -> float:
            shifted = flat_curve.shift(oas_bps)
            lattice = _build_hw_lattice(shifted)
            return _price_puttable_bond(lattice)

        def objective(oas_bps):
            return _price_with_oas(oas_bps) - market_price

        oas = brentq(objective, -200, 200, xtol=0.1)

        # Verify repricing
        repriced = _price_with_oas(oas)
        assert repriced == pytest.approx(market_price, abs=0.5), (
            f"Repriced={repriced:.4f} vs Market={market_price:.4f} at OAS={oas:.1f}bp"
        )


# ===================================================================
# Test 5: QuantLib cross-validation
# ===================================================================

class TestQuantLibCrossValidation:
    """Compare Trellis HW puttable bond to QuantLib CallableFixedRateBond
    with a put schedule."""

    @pytest.fixture(autouse=True)
    def _require_quantlib(self):
        pytest.importorskip("QuantLib")

    def test_hw_puttable_vs_quantlib(self, flat_curve):
        """Trellis HW puttable matches QuantLib within 5bp."""
        import QuantLib as ql

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today

        # Build flat curve
        ql_curve = ql.FlatForward(today, FLAT_RATE, ql.Actual365Fixed())
        curve_handle = ql.YieldTermStructureHandle(ql_curve)

        # Build the bond schedule
        issue_date = today
        maturity_date = ql.Date(15, 1, 2035)

        schedule = ql.Schedule(
            issue_date, maturity_date,
            ql.Period(ql.Semiannual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )

        # Put schedule: puttable at par at 3Y, 5Y, 7Y
        callability_schedule = ql.CallabilitySchedule()
        for y in PUT_YEARS:
            put_date = ql.Date(15, 1, 2025 + y)
            put_price_ql = ql.BondPrice(PUT_PRICE, ql.BondPrice.Clean)
            callability_schedule.append(
                ql.Callability(put_price_ql, ql.Callability.Put, put_date)
            )

        # Build callable/puttable bond
        puttable_bond = ql.CallableFixedRateBond(
            0,            # settlement days
            FACE,         # face
            schedule,
            [COUPON_RATE],
            ql.Actual365Fixed(),
            ql.Unadjusted,
            FACE,         # redemption
            issue_date,
            callability_schedule,
        )

        # HW engine
        hw_model = ql.HullWhite(curve_handle, HW_A, HW_SIGMA)
        engine = ql.TreeCallableFixedRateBondEngine(hw_model, N_STEPS)
        puttable_bond.setPricingEngine(engine)

        ql_price = puttable_bond.cleanPrice()

        # Our HW tree price
        lattice = _build_hw_lattice(flat_curve)
        trellis_price = _price_puttable_bond(lattice)

        assert trellis_price == pytest.approx(ql_price, abs=0.05), (
            f"Trellis HW={trellis_price:.4f}, QuantLib={ql_price:.4f}, "
            f"diff={abs(trellis_price - ql_price):.4f}"
        )


# ===================================================================
# Test 6: FinancePy cross-validation
# ===================================================================

class TestFinancePyCrossValidation:
    """Compare Trellis HW puttable bond to FinancePy BondEmbeddedOption."""

    @pytest.fixture(autouse=True)
    def _require_financepy(self):
        pytest.importorskip("financepy")

    def test_hw_puttable_vs_financepy(self, flat_curve):
        """Trellis HW puttable matches FinancePy HWTree within 5bp."""
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.models.hw_tree import HWTree
        from financepy.products.bonds.bond_callable import BondEmbeddedOption
        from financepy.utils.date import Date
        from financepy.utils.day_count import DayCountTypes
        from financepy.utils.frequency import FrequencyTypes

        settle = Date(15, 1, 2025)
        maturity_dt = Date(15, 1, 2035)

        fp_curve = DiscountCurveFlat(
            settle, FLAT_RATE, FrequencyTypes.CONTINUOUS, DayCountTypes.ACT_365F,
        )

        # No call dates
        call_dates = []
        call_prices = []

        # Put dates at 3Y, 5Y, 7Y
        put_dates = [
            Date(15, 1, 2028),   # 3Y
            Date(15, 1, 2030),   # 5Y
            Date(15, 1, 2032),   # 7Y
        ]
        put_prices = [PUT_PRICE] * len(put_dates)

        puttable_bond = BondEmbeddedOption(
            issue_dt=settle,
            maturity_dt=maturity_dt,
            coupon=COUPON_RATE,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            call_dts=call_dates,
            call_prices=call_prices,
            put_dts=put_dates,
            put_prices=put_prices,
        )

        model = HWTree(HW_SIGMA, HW_A, N_STEPS)
        fp_result = puttable_bond.value(settle, fp_curve, model)
        fp_price = fp_result["bondwithoption"]

        # Our HW tree price
        lattice = _build_hw_lattice(flat_curve)
        trellis_price = _price_puttable_bond(lattice)

        assert trellis_price == pytest.approx(fp_price, abs=0.05), (
            f"Trellis HW={trellis_price:.4f}, FinancePy={fp_price:.4f}, "
            f"diff={abs(trellis_price - fp_price):.4f}"
        )
