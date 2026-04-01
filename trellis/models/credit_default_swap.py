"""Reusable single-name CDS schedule and pricing helpers.

These functions provide a stable deterministic surface for single-name CDS
routes so adapters do not have to reconstruct schedule periods, spread
normalization, or premium/protection leg timing by hand.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from trellis.core.date_utils import build_period_schedule
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention, EventSchedule, Frequency

np = get_numpy()


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


def build_cds_schedule(
    start_date: date,
    end_date: date,
    frequency: Frequency,
    day_count: DayCountConvention,
    *,
    time_origin: date | None = None,
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
    )


def interval_default_probability(
    credit_curve: CreditCurveLike,
    t_start: float,
    t_end: float,
) -> float:
    """Return the conditional default probability over ``[t_start, t_end]``."""
    survival_start = float(credit_curve.survival_probability(max(float(t_start), 0.0)))
    survival_end = float(credit_curve.survival_probability(max(float(t_end), 0.0)))
    if survival_start <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - survival_end / survival_start))


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
    periods = _require_period_measurements(schedule)
    spread = normalize_cds_running_spread(spread_quote)
    if not periods or notional == 0.0:
        return 0.0

    premium_leg = 0.0
    protection_leg = 0.0
    survival_prev = 1.0

    for period in periods:
        accrual = float(period.accrual_fraction)
        t_end = float(period.t_end)
        t_pay = float(period.t_payment)
        survival = float(credit_curve.survival_probability(t_end))
        default_prob = max(0.0, survival_prev - survival)
        discount = float(discount_curve.discount(t_pay))

        premium_leg += notional * spread * accrual * discount * survival
        protection_leg += notional * (1.0 - recovery) * default_prob * discount
        survival_prev = survival

    return float(protection_leg - premium_leg)


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
    t_prev = 0.0

    for period in periods:
        accrual = float(period.accrual_fraction)
        t_end = float(period.t_end)
        t_pay = float(period.t_payment)
        discount = float(discount_curve.discount(t_pay))
        default_prob = interval_default_probability(credit_curve, t_prev, t_end)
        default_in_interval = alive & (rng.uniform(size=n_paths) < default_prob)
        alive = alive & (~default_in_interval)

        premium_leg += notional * spread * accrual * discount * float(np.mean(alive))
        protection_leg += (
            notional
            * (1.0 - recovery)
            * discount
            * float(np.mean(default_in_interval))
        )
        t_prev = t_end

    return float(protection_leg - premium_leg)
