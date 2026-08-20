"""Reusable event primitives and contingent cashflow kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb
from typing import Protocol

from scipy import integrate
from scipy.stats import norm

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention, EventSchedule, SchedulePeriod

np = get_numpy()


class CreditCurveLike(Protocol):
    """Curve interface required by the default-probability helpers."""

    def survival_probability(self, t: float) -> float:
        """Return survival probability to time ``t``."""
        ...


@dataclass(frozen=True)
class DefaultEventInterval:
    """One bounded interval in a first-event integration grid."""

    period_index: int
    start_time: float
    end_time: float
    settlement_time: float
    period_fraction_elapsed: float


@dataclass(frozen=True)
class DefaultEventGrid:
    """Schedule-aligned intervals for deterministic or sampled first events.

    ``period_interval_stops[i]`` is the exclusive interval stop for schedule
    period ``i``. Together with the previous stop, it lets callers assemble
    period cashflows without reconstructing the event partition.
    """

    periods: tuple[SchedulePeriod, ...]
    intervals: tuple[DefaultEventInterval, ...]
    period_interval_stops: tuple[int, ...]
    period_payment_times: tuple[float, ...]
    elapsed_period_fractions: tuple[float, ...]


@dataclass(frozen=True)
class FirstEventWeights:
    """Unconditional first-event and post-interval survival weights."""

    event_weights: tuple[float, ...]
    survival_weights: tuple[float, ...]


@dataclass(frozen=True)
class CouponAccrual:
    """One coupon or premium accrual emission."""

    notional: float
    rate: float
    accrual: float
    discount_factor: float
    weight: float = 1.0
    sign: float = 1.0


@dataclass(frozen=True)
class ProtectionPayment:
    """One protection payment driven by default probability."""

    notional: float
    recovery: float
    default_probability: float
    discount_factor: float
    sign: float = 1.0


@dataclass(frozen=True)
class PrincipalPayment:
    """One principal or amortization payment."""

    scheduled_principal: float
    prepaid_principal: float = 0.0
    discount_factor: float = 1.0
    sign: float = 1.0


@dataclass(frozen=True)
class TriggerSettlement:
    """One simple trigger/rebate settlement."""

    amount: float
    discount_factor: float = 1.0
    trigger_weight: float = 1.0
    sign: float = 1.0


@dataclass(frozen=True)
class PrepaymentStep:
    """One prepayment update step."""

    beginning_balance: float
    scheduled_interest: float
    scheduled_principal: float
    prepaid_principal: float
    total_principal: float
    remaining_balance: float
    smm: float


def build_default_event_grid(
    schedule: EventSchedule,
    *,
    curve_day_count: DayCountConvention = DayCountConvention.ACT_365,
    steps_per_year: int = 25,
) -> DefaultEventGrid:
    """Partition measured schedule periods into a reusable first-event grid.

    Coupon accrual remains on each period's declared day-count convention.
    Curve, survival, and discount times use ``curve_day_count`` so callers do
    not accidentally reuse premium-accrual fractions as model times.
    """
    if schedule.time_origin is None:
        raise ValueError("default-event grids require schedule.time_origin")
    if schedule.day_count is None:
        raise ValueError("default-event grids require schedule.day_count")
    if steps_per_year <= 0:
        raise ValueError("steps_per_year must be positive")

    missing = [
        index
        for index, period in enumerate(schedule.periods)
        if period.accrual_fraction is None
    ]
    if missing:
        raise ValueError(
            "default-event grids require accrual_fraction on every period; "
            f"missing for periods {missing}"
        )

    origin = schedule.time_origin
    intervals: list[DefaultEventInterval] = []
    stops: list[int] = []
    payment_times: list[float] = []
    elapsed_fractions: list[float] = []

    for period_index, period in enumerate(schedule.periods):
        raw_start = float(year_fraction(origin, period.start_date, curve_day_count))
        raw_end = float(year_fraction(origin, period.end_date, curve_day_count))
        start = max(raw_start, 0.0)
        end = max(raw_end, 0.0)
        payment_times.append(
            max(float(year_fraction(origin, period.payment_date, curve_day_count)), 0.0)
        )

        if period.start_date < origin < period.end_date:
            elapsed_accrual = float(
                year_fraction(
                    period.start_date,
                    origin,
                    schedule.day_count,
                    ref_start=period.start_date,
                    ref_end=period.end_date,
                    frequency=schedule.frequency,
                )
            )
            full_accrual = float(period.accrual_fraction)
            elapsed_fractions.append(
                min(max(elapsed_accrual / max(full_accrual, 1e-12), 0.0), 1.0)
            )
        else:
            elapsed_fractions.append(0.0)

        if end > start:
            step_count = max(int(ceil((end - start) * steps_per_year)), 1)
            step_size = (end - start) / step_count
            total_period_time = max(raw_end - raw_start, 1e-12)
            for step in range(step_count):
                interval_start = start + step * step_size
                interval_end = start + (step + 1) * step_size
                settlement_time = 0.5 * (interval_start + interval_end)
                period_fraction_elapsed = min(
                    max((settlement_time - raw_start) / total_period_time, 0.0),
                    1.0,
                )
                intervals.append(
                    DefaultEventInterval(
                        period_index=period_index,
                        start_time=interval_start,
                        end_time=interval_end,
                        settlement_time=settlement_time,
                        period_fraction_elapsed=period_fraction_elapsed,
                    )
                )
        stops.append(len(intervals))

    return DefaultEventGrid(
        periods=tuple(schedule.periods),
        intervals=tuple(intervals),
        period_interval_stops=tuple(stops),
        period_payment_times=tuple(payment_times),
        elapsed_period_fractions=tuple(elapsed_fractions),
    )


def conditional_event_probabilities_from_curve(
    credit_curve: CreditCurveLike,
    intervals: tuple[DefaultEventInterval, ...],
) -> tuple[float, ...]:
    """Return conditional event probabilities for ordered grid intervals."""
    return tuple(
        interval_default_probability_from_survival(
            float(credit_curve.survival_probability(interval.start_time)),
            float(credit_curve.survival_probability(interval.end_time)),
        )
        for interval in intervals
    )


def expected_first_event_weights(
    conditional_probabilities: tuple[float, ...],
) -> FirstEventWeights:
    """Propagate conditional probabilities into exact first-event weights."""
    alive = 1.0
    event_weights: list[float] = []
    survival_weights: list[float] = []
    for probability in conditional_probabilities:
        conditional_probability = max(0.0, min(float(probability), 1.0))
        event_weight = alive * conditional_probability
        alive = max(alive - event_weight, 0.0)
        event_weights.append(event_weight)
        survival_weights.append(alive)
    return FirstEventWeights(tuple(event_weights), tuple(survival_weights))


def sample_first_event_weights(
    conditional_probabilities: tuple[float, ...],
    *,
    n_paths: int,
    seed: int,
) -> FirstEventWeights:
    """Estimate first-event weights with one persistent alive-state simulation."""
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    alive = np.ones(int(n_paths), dtype=bool)
    event_weights: list[float] = []
    survival_weights: list[float] = []
    for probability in conditional_probabilities:
        conditional_probability = max(0.0, min(float(probability), 1.0))
        event = alive & (rng.uniform(size=int(n_paths)) < conditional_probability)
        event_weights.append(float(np.mean(event)))
        alive = alive & (~event)
        survival_weights.append(float(np.mean(alive)))
    return FirstEventWeights(tuple(event_weights), tuple(survival_weights))


def interval_default_probability_from_survival(
    survival_start: float,
    survival_end: float,
) -> float:
    """Return conditional default probability from survival ratios."""
    survival_start = max(float(survival_start), 0.0)
    survival_end = max(float(survival_end), 0.0)
    if survival_start <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - survival_end / survival_start))


def terminal_default_probability(
    credit_curve: CreditCurveLike,
    horizon: float,
) -> float:
    """Return the default probability over ``[0, horizon]``."""
    horizon = max(float(horizon), 0.0)
    survival = float(credit_curve.survival_probability(horizon))
    return max(0.0, min(1.0, 1.0 - survival))


def coupon_cashflow_pv(coupon: CouponAccrual) -> float:
    """Return discounted coupon/premium PV with explicit sign and weight."""
    return (
        float(coupon.sign)
        * float(coupon.notional)
        * float(coupon.rate)
        * float(coupon.accrual)
        * float(coupon.discount_factor)
        * float(coupon.weight)
    )


def protection_payment_pv(payment: ProtectionPayment) -> float:
    """Return discounted protection-payment PV."""
    loss_given_default = max(0.0, 1.0 - float(payment.recovery))
    return (
        float(payment.sign)
        * float(payment.notional)
        * loss_given_default
        * float(payment.default_probability)
        * float(payment.discount_factor)
    )


def principal_payment_pv(payment: PrincipalPayment) -> float:
    """Return discounted principal/amortization PV."""
    total_principal = float(payment.scheduled_principal) + float(payment.prepaid_principal)
    return float(payment.sign) * total_principal * float(payment.discount_factor)


def trigger_settlement_pv(settlement: TriggerSettlement) -> float:
    """Return discounted trigger/rebate settlement PV."""
    return (
        float(settlement.sign)
        * float(settlement.amount)
        * float(settlement.discount_factor)
        * float(settlement.trigger_weight)
    )


def project_prepayment_step(
    *,
    beginning_balance: float,
    scheduled_interest: float,
    scheduled_principal: float,
    smm: float,
) -> PrepaymentStep:
    """Advance one scheduled-principal plus prepayment step."""
    balance = max(float(beginning_balance), 0.0)
    scheduled_interest = max(float(scheduled_interest), 0.0)
    scheduled_principal = max(0.0, min(float(scheduled_principal), balance))
    smm = max(0.0, min(float(smm), 1.0))

    balance_after_schedule = max(balance - scheduled_principal, 0.0)
    prepaid_principal = min(balance_after_schedule * smm, balance_after_schedule)
    total_principal = scheduled_principal + prepaid_principal
    remaining_balance = max(balance - total_principal, 0.0)

    return PrepaymentStep(
        beginning_balance=balance,
        scheduled_interest=scheduled_interest,
        scheduled_principal=scheduled_principal,
        prepaid_principal=prepaid_principal,
        total_principal=total_principal,
        remaining_balance=remaining_balance,
        smm=smm,
    )


def nth_to_default_probability(
    n_names: int,
    n_th: int,
    marginal_default_prob: float,
    correlation: float,
) -> float:
    """Return the probability that at least ``n_th`` names default."""
    n_names = int(n_names)
    n_th = int(n_th)
    if n_names <= 0:
        raise ValueError("n_names must be positive")
    if n_th <= 0 or n_th > n_names:
        raise ValueError("n_th must lie in [1, n_names]")

    p_def = max(0.0, min(1.0, float(marginal_default_prob)))
    rho = max(0.0, min(float(correlation), 0.999999))

    if rho <= 1e-8:
        return max(
            0.0,
            min(
                1.0,
                1.0
                - sum(
                    comb(n_names, j) * (p_def ** j) * ((1.0 - p_def) ** (n_names - j))
                    for j in range(n_th)
                ),
            ),
        )

    p_thr = norm.ppf(max(1e-9, min(1.0 - 1e-9, p_def)))
    sq_rho = rho ** 0.5
    sq_1mr = (1.0 - rho) ** 0.5

    def integrand(z: float) -> float:
        conditional_prob = norm.cdf((p_thr - sq_rho * z) / sq_1mr)
        triggered = 1.0 - sum(
            comb(n_names, j)
            * (conditional_prob ** j)
            * ((1.0 - conditional_prob) ** (n_names - j))
            for j in range(n_th)
        )
        return float(triggered) * float(norm.pdf(z))

    result, _ = integrate.quad(integrand, -8.0, 8.0)
    return max(0.0, min(1.0, float(result)))


__all__ = [
    "CouponAccrual",
    "DefaultEventGrid",
    "DefaultEventInterval",
    "FirstEventWeights",
    "PrepaymentStep",
    "PrincipalPayment",
    "ProtectionPayment",
    "TriggerSettlement",
    "build_default_event_grid",
    "conditional_event_probabilities_from_curve",
    "coupon_cashflow_pv",
    "expected_first_event_weights",
    "interval_default_probability_from_survival",
    "nth_to_default_probability",
    "principal_payment_pv",
    "project_prepayment_step",
    "protection_payment_pv",
    "sample_first_event_weights",
    "terminal_default_probability",
    "trigger_settlement_pv",
]
