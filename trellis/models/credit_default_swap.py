"""Reusable single-name CDS schedule and pricing helpers.

These functions provide a stable deterministic surface for single-name CDS
routes so adapters do not have to reconstruct schedule periods, spread
normalization, or premium/protection leg timing by hand.
"""

from __future__ import annotations

from datetime import date
from math import ceil
from typing import Protocol

from scipy.optimize import brentq

from trellis.core.date_utils import build_period_schedule, year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention, EventSchedule, Frequency, SchedulePeriod
from trellis.conventions.calendar import BusinessDayAdjustment, Calendar, WEEKEND_ONLY
from trellis.conventions.schedule import RollConvention, StubType
from trellis.models.contingent_cashflows import (
    CouponAccrual,
    ProtectionPayment,
    coupon_cashflow_pv,
    interval_default_probability_from_survival,
    protection_payment_pv,
)

np = get_numpy()
_CDS_CURVE_DAY_COUNT = DayCountConvention.ACT_365
_CDS_PROTECTION_STEPS_PER_YEAR = 25


class CreditCurveLike(Protocol):
    """Curve interface required by the CDS helpers."""

    def survival_probability(self, t: float) -> float:
        """Return survival probability to time ``t``."""
        ...


class DiscountCurveLike(Protocol):
    """Discount interface required by the CDS helpers."""

    def discount(self, t: float) -> float:
        """Return discount factor to time ``t``."""
        ...


def normalize_cds_running_spread(spread_quote: float) -> float:
    """Normalize CDS running-spread quotes to decimal form.

    Trellis task text often quotes running spreads in basis points. Treat
    values greater than ``1.0`` as basis-point quotes.
    """
    spread = float(spread_quote)
    if spread > 1.0:
        spread *= 1e-4
    return spread


def normalize_cds_upfront_quote(upfront_quote: float) -> float:
    """Normalize CDS upfront quotes to a decimal fraction of notional.

    Values with absolute magnitude greater than ``1.0`` are treated as upfront
    points. For example, ``5.25`` means ``5.25%`` of notional.
    """
    upfront = float(upfront_quote)
    if abs(upfront) > 1.0:
        upfront *= 1e-2
    return upfront


def _price_cds_with_decimal_running_spread(
    *,
    notional: float,
    spread: float,
    recovery: float,
    schedule: EventSchedule,
    credit_curve: CreditCurveLike,
    discount_curve: DiscountCurveLike,
) -> float:
    """Price a single-name CDS from a decimal running spread."""
    periods = _require_period_measurements(schedule)
    if not periods or notional == 0.0:
        return 0.0
    valuation_origin = schedule.time_origin or schedule.start_date

    premium_leg = 0.0
    protection_leg = 0.0
    accrued_to_valuation = 0.0
    accrued_on_default = 0.0

    for period in periods:
        accrual = float(period.accrual_fraction)
        t_start = _curve_time(valuation_origin, period.start_date)
        t_end = _curve_time(valuation_origin, period.end_date)
        t_pay = _curve_time(valuation_origin, period.payment_date)
        if t_end <= 0.0:
            continue
        survival = float(credit_curve.survival_probability(t_end))
        discount = float(discount_curve.discount(t_pay))

        premium_leg += coupon_cashflow_pv(
            CouponAccrual(
                notional=notional,
                rate=spread,
                accrual=accrual,
                discount_factor=discount,
                weight=survival,
            )
        )
        accrued_to_valuation += (
            notional
            * spread
            * accrual
            * _elapsed_coupon_fraction(period, valuation_origin=valuation_origin)
        )
        period_protection, period_accrued_default = _integrated_default_leg_terms(
            notional=notional,
            spread=spread,
            recovery=recovery,
            accrual_fraction=accrual,
            period_start=t_start,
            period_end=t_end,
            credit_curve=credit_curve,
            discount_curve=discount_curve,
        )
        protection_leg += period_protection
        accrued_on_default += period_accrued_default

    return float(protection_leg - premium_leg - accrued_on_default + accrued_to_valuation)


def build_cds_schedule(
    start_date: date,
    end_date: date,
    frequency: Frequency,
    day_count: DayCountConvention,
    *,
    time_origin: date | None = None,
    calendar: Calendar | None = None,
    business_day_adjustment: BusinessDayAdjustment | None = None,
    roll_convention: RollConvention = RollConvention.NONE,
    stub: StubType = StubType.SHORT_LAST,
    payment_lag_days: int = 0,
) -> EventSchedule:
    """Build the canonical single-name CDS schedule.

    The CDS helpers use ``start_date`` as the schedule time origin so
    analytical and Monte Carlo routes can share the same event times.
    """
    origin = start_date if time_origin is None else time_origin
    return build_period_schedule(
        start_date,
        end_date,
        frequency,
        day_count=day_count,
        time_origin=origin,
        calendar=calendar or WEEKEND_ONLY,
        bda=business_day_adjustment or BusinessDayAdjustment.FOLLOWING,
        roll_convention=roll_convention,
        stub=stub,
        payment_lag_days=payment_lag_days,
    )


def interval_default_probability(
    credit_curve: CreditCurveLike,
    t_start: float,
    t_end: float,
) -> float:
    """Return the conditional default probability over ``[t_start, t_end]``."""
    survival_start = float(credit_curve.survival_probability(max(float(t_start), 0.0)))
    survival_end = float(credit_curve.survival_probability(max(float(t_end), 0.0)))
    return interval_default_probability_from_survival(survival_start, survival_end)


def _require_period_measurements(schedule: EventSchedule) -> tuple:
    """Return schedule periods after verifying they carry time/accrual fields."""
    missing = [
        idx
        for idx, period in enumerate(schedule.periods)
        if period.accrual_fraction is None
        or period.t_end is None
        or period.t_payment is None
    ]
    if missing:
        raise ValueError(
            "CDS pricing helpers require EventSchedule periods with "
            "accrual_fraction, t_end, and t_payment populated; "
            f"missing for periods {missing}"
        )
    return schedule.periods


def _curve_time(origin: date, target: date) -> float:
    """Return the curve/discounter time coordinate for one schedule date."""
    return float(year_fraction(origin, target, _CDS_CURVE_DAY_COUNT))


def _elapsed_coupon_fraction(
    period: SchedulePeriod,
    *,
    valuation_origin: date,
) -> float:
    """Return the clean-accrual fraction already earned at valuation."""
    if period.start_date >= valuation_origin or period.end_date <= valuation_origin:
        return 0.0
    total_days = max((period.end_date - period.start_date).days, 1)
    elapsed_days = max((valuation_origin - period.start_date).days, 0)
    return min(max(elapsed_days / total_days, 0.0), 1.0)


def _integrated_default_leg_terms(
    *,
    notional: float,
    spread: float,
    recovery: float,
    accrual_fraction: float,
    period_start: float,
    period_end: float,
    credit_curve: CreditCurveLike,
    discount_curve: DiscountCurveLike,
    steps_per_year: int = _CDS_PROTECTION_STEPS_PER_YEAR,
) -> tuple[float, float]:
    """Integrate protection and accrued-on-default terms over one period."""
    start = max(float(period_start), 0.0)
    end = max(float(period_end), 0.0)
    if end <= start:
        return 0.0, 0.0

    n_steps = max(int(ceil((end - start) * max(int(steps_per_year), 1))), 1)
    dt = (end - start) / n_steps
    total_interval = max(float(period_end) - float(period_start), 1e-12)

    protection_leg = 0.0
    accrued_on_default = 0.0
    for step in range(n_steps):
        t_start = start + step * dt
        t_end = start + (step + 1) * dt
        t_mid = 0.5 * (t_start + t_end)
        survival_start = float(credit_curve.survival_probability(t_start))
        survival_end = float(credit_curve.survival_probability(t_end))
        default_prob = max(0.0, survival_start - survival_end)
        if default_prob <= 0.0:
            continue
        discount = float(discount_curve.discount(t_mid))
        protection_leg += protection_payment_pv(
            ProtectionPayment(
                notional=notional,
                recovery=recovery,
                default_probability=default_prob,
                discount_factor=discount,
            )
        )
        accrued_fraction_elapsed = max(
            0.0,
            min((t_mid - float(period_start)) / total_interval, 1.0),
        )
        accrued_on_default += coupon_cashflow_pv(
            CouponAccrual(
                notional=notional,
                rate=spread,
                accrual=accrual_fraction * accrued_fraction_elapsed,
                discount_factor=discount,
                weight=default_prob,
            )
        )
    return protection_leg, accrued_on_default


def price_cds_analytical(
    *,
    notional: float,
    spread_quote: float,
    recovery: float,
    schedule: EventSchedule,
    credit_curve: CreditCurveLike,
    discount_curve: DiscountCurveLike,
) -> float:
    """Price a single-name CDS from deterministic survival probabilities."""
    spread = normalize_cds_running_spread(spread_quote)
    return _price_cds_with_decimal_running_spread(
        notional=notional,
        spread=spread,
        recovery=recovery,
        schedule=schedule,
        credit_curve=credit_curve,
        discount_curve=discount_curve,
    )


def solve_cds_par_spread_analytical(
    *,
    notional: float,
    recovery: float,
    schedule: EventSchedule,
    credit_curve: CreditCurveLike,
    discount_curve: DiscountCurveLike,
    tol: float = 1e-10,
    upper_spread: float = 5.0,
) -> float:
    """Return the decimal par running spread that zeros the analytical CDS PV."""
    if notional == 0.0:
        return 0.0
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if upper_spread <= 0.0:
        raise ValueError("upper_spread must be positive")

    def objective(spread: float) -> float:
        return _price_cds_with_decimal_running_spread(
            notional=notional,
            spread=float(spread),
            recovery=recovery,
            schedule=schedule,
            credit_curve=credit_curve,
            discount_curve=discount_curve,
        )

    lower = 0.0
    lower_value = objective(lower)
    if abs(lower_value) <= tol:
        return 0.0
    if lower_value < 0.0:
        raise ValueError("CDS par-spread solve expected non-negative PV at zero running spread")

    upper = float(upper_spread)
    upper_value = objective(upper)
    while upper_value > 0.0 and upper < 100.0:
        upper *= 2.0
        upper_value = objective(upper)
    if upper_value > 0.0:
        raise ValueError("CDS par-spread solve could not bracket a zero PV spread")
    return float(brentq(objective, lower, upper, xtol=tol))


def price_cds_monte_carlo(
    *,
    notional: float,
    spread_quote: float,
    recovery: float,
    schedule: EventSchedule,
    credit_curve: CreditCurveLike,
    discount_curve: DiscountCurveLike,
    n_paths: int = 50000,
    seed: int = 42,
) -> float:
    """Price a single-name CDS by interval default-time sampling."""
    periods = _require_period_measurements(schedule)
    spread = normalize_cds_running_spread(spread_quote)
    if not periods or notional == 0.0:
        return 0.0
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    alive = np.ones(n_paths, dtype=bool)
    premium_leg = 0.0
    protection_leg = 0.0
    accrued_on_default = 0.0
    accrued_to_valuation = 0.0
    valuation_origin = schedule.time_origin or schedule.start_date

    for period in periods:
        accrual = float(period.accrual_fraction)
        t_start = _curve_time(valuation_origin, period.start_date)
        t_end = _curve_time(valuation_origin, period.end_date)
        t_pay = _curve_time(valuation_origin, period.payment_date)
        if t_end <= 0.0:
            continue

        accrued_to_valuation += (
            notional
            * spread
            * accrual
            * _elapsed_coupon_fraction(period, valuation_origin=valuation_origin)
        )

        start = max(t_start, 0.0)
        end = max(t_end, 0.0)
        if end > start:
            n_steps = max(int(ceil((end - start) * _CDS_PROTECTION_STEPS_PER_YEAR)), 1)
            dt = (end - start) / n_steps
            total_interval = max(t_end - t_start, 1e-12)
            for step in range(n_steps):
                step_start = start + step * dt
                step_end = start + (step + 1) * dt
                step_mid = 0.5 * (step_start + step_end)
                default_prob = interval_default_probability(credit_curve, step_start, step_end)
                if default_prob <= 0.0:
                    continue
                default_in_step = alive & (rng.uniform(size=n_paths) < default_prob)
                if np.any(default_in_step):
                    discount = float(discount_curve.discount(step_mid))
                    protection_leg += protection_payment_pv(
                        ProtectionPayment(
                            notional=notional,
                            recovery=recovery,
                            default_probability=float(np.mean(default_in_step)),
                            discount_factor=discount,
                        )
                    )
                    accrued_fraction_elapsed = max(
                        0.0,
                        min((step_mid - t_start) / total_interval, 1.0),
                    )
                    accrued_on_default += coupon_cashflow_pv(
                        CouponAccrual(
                            notional=notional,
                            rate=spread,
                            accrual=accrual * accrued_fraction_elapsed,
                            discount_factor=discount,
                            weight=float(np.mean(default_in_step)),
                        )
                    )
                    alive = alive & (~default_in_step)

        premium_leg += coupon_cashflow_pv(
            CouponAccrual(
                notional=notional,
                rate=spread,
                accrual=accrual,
                discount_factor=float(discount_curve.discount(t_pay)),
                weight=float(np.mean(alive)),
            )
        )

    return float(protection_leg - premium_leg - accrued_on_default + accrued_to_valuation)
