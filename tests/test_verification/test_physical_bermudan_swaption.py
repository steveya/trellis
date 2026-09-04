"""Independent stochastic verification for the strict physical Bermudan route."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.conventions.calendar import BusinessDayAdjustment, WEEKEND_ONLY
from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.types import Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_swap_tail import (
    ExerciseSwapStart,
    FixedLegConvention,
    FloatingLegConvention,
    NamedRateCurve,
    PhysicalBermudanSwapTailSpec,
    price_physical_bermudan_swaption_lattice,
)
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    TERM_STRUCTURE_TARGET,
    UNIFORM_ADDITIVE_MESH,
    build_lattice,
)
from trellis.models.trees.models import MODEL_REGISTRY


def _trellis_price(*, n_steps: int) -> float:
    valuation_date = date(2024, 1, 1)
    exercise_dates = (date(2025, 1, 1), date(2026, 1, 1))
    maturity_date = date(2028, 1, 1)
    curve = YieldCurve.flat(0.03, max_tenor=10.0)
    fixed = FixedLegConvention(
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.THIRTY_360,
        calendar=WEEKEND_ONLY,
        business_day_adjustment=BusinessDayAdjustment.UNADJUSTED,
        stub_type=StubType.SHORT_LAST,
        roll_convention=RollConvention.NONE,
        payment_lag_business_days=0,
    )
    floating = FloatingLegConvention(
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        calendar=WEEKEND_ONLY,
        business_day_adjustment=BusinessDayAdjustment.UNADJUSTED,
        stub_type=StubType.SHORT_LAST,
        roll_convention=RollConvention.NONE,
        reset_lag_business_days=0,
        fixing_lag_business_days=0,
        payment_lag_business_days=0,
        rate_index="USD-SOFR-3M",
        compounding="simple",
        gearing=1.0,
        spread=0.0,
    )
    spec = PhysicalBermudanSwapTailSpec(
        valuation_date=valuation_date,
        exercise_swap_starts=tuple(
            ExerciseSwapStart(exercise_date, exercise_date)
            for exercise_date in exercise_dates
        ),
        common_maturity_date=maturity_date,
        notional=1_000_000.0,
        fixed_rate=0.035,
        currency="USD",
        option_side="payer",
        settlement_style="physical",
        fixed_convention=fixed,
        floating_convention=floating,
        discount_curve_name="USD-OIS",
        forecast_curve_name="USD-SOFR-3M",
        projection_policy="static_additive_forward_basis",
        model_day_count=DayCountConvention.ACT_365,
        model_time_calendar=WEEKEND_ONLY,
        max_lattice_date_error_days=3.0,
    )
    horizon = (maturity_date - valuation_date).days / 365.0
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        MODEL_REGISTRY["hull_white"],
        calibration_target=TERM_STRUCTURE_TARGET(curve),
        r0=0.03,
        sigma=0.01,
        a=0.05,
        T=horizon,
        n_steps=n_steps,
    )
    return float(
        price_physical_bermudan_swaption_lattice(
            lattice,
            spec,
            discount_curve=NamedRateCurve("USD-OIS", curve),
            forecast_curve=NamedRateCurve("USD-SOFR-3M", curve),
        )
    )


def _quantlib_price(*, n_steps: int) -> float:
    ql = pytest.importorskip("QuantLib")
    with ql.SavedSettings():
        return _quantlib_price_with_saved_settings(ql=ql, n_steps=n_steps)


def _quantlib_price_with_saved_settings(*, ql, n_steps: int) -> float:
    """Build the independent oracle inside a restored QuantLib settings scope."""

    evaluation_date = ql.Date(1, ql.January, 2024)
    ql.Settings.instance().evaluationDate = evaluation_date
    day_count = ql.Actual365Fixed()
    curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, 0.03, day_count, ql.Continuous)
    )
    calendar = ql.WeekendsOnly()
    start = ql.Date(1, ql.January, 2025)
    maturity = ql.Date(1, ql.January, 2028)
    fixed_schedule = ql.Schedule(
        start,
        maturity,
        ql.Period(ql.Semiannual),
        calendar,
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Forward,
        False,
    )
    floating_schedule = ql.Schedule(
        start,
        maturity,
        ql.Period(ql.Quarterly),
        calendar,
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Forward,
        False,
    )
    index = ql.IborIndex(
        "USD-SOFR-3M",
        ql.Period(ql.Quarterly),
        0,
        ql.USDCurrency(),
        calendar,
        ql.Unadjusted,
        False,
        ql.Actual360(),
        curve,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        1_000_000.0,
        fixed_schedule,
        0.035,
        ql.Thirty360(ql.Thirty360.USA),
        floating_schedule,
        index,
        0.0,
        ql.Actual360(),
    )
    exercise = ql.BermudanExercise(
        [
            ql.Date(1, ql.January, 2025),
            ql.Date(1, ql.January, 2026),
        ]
    )
    swaption = ql.Swaption(swap, exercise)
    model = ql.HullWhite(curve, 0.05, 0.01)
    swaption.setPricingEngine(ql.TreeSwaptionEngine(model, n_steps))
    return float(swaption.NPV())


def test_stochastic_multi_exercise_price_matches_quantlib_tree():
    """Cross-check conditional bonds and holder-max rollback independently."""
    n_steps = 512

    trellis_price = _trellis_price(n_steps=n_steps)
    quantlib_price = _quantlib_price(n_steps=n_steps)

    assert trellis_price > 0.0
    assert trellis_price == pytest.approx(quantlib_price, rel=0.01, abs=50.0)
