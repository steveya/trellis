"""Generic short-rate fixed-income helper kits for event-aware claims.

This module owns reusable coupon/event/control assembly for bounded
single-state fixed-income claims with embedded call/put schedules. Product
wrappers such as callable bonds should stay thin and delegate schedule,
exercise, lattice, and PDE event preparation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.core.date_utils import (
    build_payment_timeline,
    coerce_contract_timeline_from_dates,
    normalize_explicit_dates,
    year_fraction,
)
from trellis.core.types import DayCountConvention, Frequency, TimelineRole
import trellis.models.trees.algebra as lattice_algebra
from trellis.models.pde.event_aware import EventAwarePDEEventBucket, EventAwarePDETransform
from trellis.models.trees.control import (
    lattice_steps_from_timeline,
    resolve_lattice_exercise_policy_from_control_style,
)


class DiscountCurveLike(Protocol):
    """Discount interface required by short-rate fixed-income helpers."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...


class FixedIncomeMarketStateLike(Protocol):
    """Market-state interface required by fixed-income helper kits."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None


class FixedIncomeScheduleSpecLike(Protocol):
    """Coupon schedule fields shared by bounded fixed-income claims."""

    notional: float
    coupon: float
    start_date: date
    end_date: date
    frequency: Frequency
    day_count: DayCountConvention


class EmbeddedFixedIncomeSpecLike(FixedIncomeScheduleSpecLike, Protocol):
    """Embedded exercise fields shared by callable/putable fixed-income claims."""

    call_dates: object
    call_price: float
    put_dates: object
    put_price: float


@dataclass(frozen=True)
class EmbeddedFixedIncomeExerciseConfig:
    """Resolved embedded exercise semantics for callable or puttable claims."""

    schedule_dates: tuple[date, ...]
    exercise_price_cash: float
    exercise_style: str
    control_style: str
    reference_bound: str
    projection_kind: str


@dataclass(frozen=True)
class FixedIncomeCouponCashflow:
    """One coupon cashflow on the fixed-income event timeline."""

    payment_date: date
    amount: float
    accrual_fraction: float
    time_to_payment: float


@dataclass(frozen=True)
class EmbeddedFixedIncomeEventTimeline:
    """Shared event timeline for bounded embedded fixed-income claims."""

    settlement: date
    coupon_cashflows: tuple[FixedIncomeCouponCashflow, ...]
    exercise: EmbeddedFixedIncomeExerciseConfig
    terminal_coupon_cash: float
    terminal_redemption_cash: float


def settlement_date_for_fixed_income_claim(
    market_state: FixedIncomeMarketStateLike,
    spec: FixedIncomeScheduleSpecLike,
) -> date:
    """Resolve the settlement anchor for fixed-income claim helpers."""
    return getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None) or spec.start_date


def quoted_embedded_exercise_price_to_cash(
    spec: EmbeddedFixedIncomeSpecLike,
    field_name: str,
) -> float:
    """Convert quoted call/put prices from 100-par quotes into cash terms."""
    quoted_price = float(getattr(spec, field_name))
    return quoted_price / 100.0 * float(spec.notional)


def resolve_embedded_fixed_income_exercise_config(
    spec: EmbeddedFixedIncomeSpecLike,
) -> EmbeddedFixedIncomeExerciseConfig:
    """Resolve bounded callable/putable exercise semantics from one spec."""
    has_call_dates = hasattr(spec, "call_dates")
    has_put_dates = hasattr(spec, "put_dates")
    if has_call_dates and has_put_dates:
        raise ValueError("Embedded bond spec must define either call_dates or put_dates, not both")
    if has_call_dates:
        return EmbeddedFixedIncomeExerciseConfig(
            schedule_dates=tuple(normalize_explicit_dates(getattr(spec, "call_dates"))),
            exercise_price_cash=quoted_embedded_exercise_price_to_cash(spec, "call_price"),
            exercise_style="issuer_call",
            control_style="issuer_min",
            reference_bound="upper",
            projection_kind="project_min",
        )
    if has_put_dates:
        return EmbeddedFixedIncomeExerciseConfig(
            schedule_dates=tuple(normalize_explicit_dates(getattr(spec, "put_dates"))),
            exercise_price_cash=quoted_embedded_exercise_price_to_cash(spec, "put_price"),
            exercise_style="holder_put",
            control_style="holder_max",
            reference_bound="lower",
            projection_kind="project_max",
        )
    raise ValueError("Embedded bond spec must define call_dates or put_dates")


def build_fixed_income_coupon_cashflows(
    spec: FixedIncomeScheduleSpecLike,
    *,
    settlement: date,
    include_terminal: bool = False,
) -> tuple[FixedIncomeCouponCashflow, ...]:
    """Build the coupon cashflows on the fixed-income timeline."""
    payment_timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=settlement,
        label="embedded_fixed_income_coupon_timeline",
    )
    cashflows: list[FixedIncomeCouponCashflow] = []
    for period in payment_timeline:
        payment_date = period.payment_date
        if payment_date <= settlement:
            continue
        if not include_terminal and payment_date >= spec.end_date:
            continue
        coupon_cash = float(spec.notional) * float(spec.coupon) * float(period.accrual_fraction or 0.0)
        cashflows.append(
            FixedIncomeCouponCashflow(
                payment_date=payment_date,
                amount=float(coupon_cash),
                accrual_fraction=float(period.accrual_fraction or 0.0),
                time_to_payment=max(float(period.t_payment or 0.0), 0.0),
            )
        )
    return tuple(cashflows)


def final_fixed_income_coupon_amount(
    spec: FixedIncomeScheduleSpecLike,
    *,
    settlement: date,
) -> float:
    """Return the maturity coupon amount carried into terminal redemption."""
    return float(
        sum(
            cashflow.amount
            for cashflow in build_fixed_income_coupon_cashflows(
                spec,
                settlement=settlement,
                include_terminal=True,
            )
            if cashflow.payment_date == spec.end_date
        )
    )


def matured_embedded_fixed_income_value(
    spec: FixedIncomeScheduleSpecLike,
    *,
    settlement: date,
) -> float:
    """Return the immediate settlement value once the fixed-income claim matures."""
    if settlement > spec.end_date:
        return 0.0
    return float(spec.notional) + final_fixed_income_coupon_amount(spec, settlement=settlement)


def build_embedded_fixed_income_event_timeline(
    spec: EmbeddedFixedIncomeSpecLike,
    *,
    settlement: date,
) -> EmbeddedFixedIncomeEventTimeline:
    """Build the shared coupon/exercise timeline for embedded fixed-income claims."""
    exercise = resolve_embedded_fixed_income_exercise_config(spec)
    terminal_coupon_cash = final_fixed_income_coupon_amount(spec, settlement=settlement)
    return EmbeddedFixedIncomeEventTimeline(
        settlement=settlement,
        coupon_cashflows=build_fixed_income_coupon_cashflows(spec, settlement=settlement, include_terminal=False),
        exercise=exercise,
        terminal_coupon_cash=float(terminal_coupon_cash),
        terminal_redemption_cash=float(spec.notional) + float(terminal_coupon_cash),
    )


def build_embedded_fixed_income_coupon_step_map(
    event_timeline: EmbeddedFixedIncomeEventTimeline,
    *,
    dt: float,
    n_steps: int,
) -> dict[int, float]:
    """Map coupon cashflows onto lattice steps."""
    coupon_by_step: dict[int, float] = {}
    for cashflow in event_timeline.coupon_cashflows:
        step = int(round(float(cashflow.time_to_payment) / float(dt)))
        if step <= 0 or step > int(n_steps):
            continue
        coupon_by_step[step] = coupon_by_step.get(step, 0.0) + float(cashflow.amount)
    return coupon_by_step


def build_embedded_fixed_income_exercise_policy(
    event_timeline: EmbeddedFixedIncomeEventTimeline,
    *,
    day_count: DayCountConvention,
    dt: float,
    n_steps: int,
):
    """Resolve callable/putable exercise dates into a lattice control policy."""
    exercise_dates = tuple(
        exercise_date
        for exercise_date in event_timeline.exercise.schedule_dates
        if event_timeline.settlement < exercise_date
    )
    if not exercise_dates:
        return resolve_lattice_exercise_policy_from_control_style(
            event_timeline.exercise.control_style,
            exercise_steps=(),
            exercise_style=event_timeline.exercise.exercise_style,
        )
    exercise_timeline = coerce_contract_timeline_from_dates(
        exercise_dates,
        role=TimelineRole.EXERCISE,
        day_count=day_count,
        time_origin=event_timeline.settlement,
        label=f"{event_timeline.exercise.exercise_style}_timeline",
    )
    exercise_steps = lattice_steps_from_timeline(
        exercise_timeline,
        dt=dt,
        n_steps=n_steps,
    )
    return resolve_lattice_exercise_policy_from_control_style(
        event_timeline.exercise.control_style,
        exercise_steps=exercise_steps,
        exercise_style=event_timeline.exercise.exercise_style,
    )


def compile_embedded_fixed_income_lattice_contract_spec(
    spec: EmbeddedFixedIncomeSpecLike,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
) -> lattice_algebra.LatticeContractSpec:
    """Compile one callable/putable fixed-income claim onto the lattice substrate."""
    event_timeline = build_embedded_fixed_income_event_timeline(spec, settlement=settlement)
    coupon_by_step = build_embedded_fixed_income_coupon_step_map(
        event_timeline,
        dt=dt,
        n_steps=n_steps,
    )
    exercise_policy = build_embedded_fixed_income_exercise_policy(
        event_timeline,
        day_count=spec.day_count,
        dt=dt,
        n_steps=n_steps,
    )
    terminal_coupon = coupon_by_step.get(n_steps, 0.0)
    exercise = event_timeline.exercise
    return lattice_algebra.LatticeContractSpec(
        claim=lattice_algebra.LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice, obs: float(spec.notional) + float(terminal_coupon),
            node_cashflow_fn=lambda step, node, lattice, obs: float(coupon_by_step.get(step, 0.0)),
            observable_requirements=("rate",),
        ),
        control=lattice_algebra.LatticeControlSpec(
            objective=exercise.control_style,
            exercise_steps=exercise_policy.exercise_steps,
            exercise_value_fn=(
                lambda step, node, lattice, obs: float(exercise.exercise_price_cash)
                + float(coupon_by_step.get(step, 0.0))
            ),
        ),
        metadata={"coupon_by_step": coupon_by_step},
    )


def build_embedded_fixed_income_pde_event_buckets(
    event_timeline: EmbeddedFixedIncomeEventTimeline,
    *,
    day_count: DayCountConvention,
    maturity_date: date,
) -> tuple[EventAwarePDEEventBucket, ...]:
    """Compile coupon and exercise timelines into generic PDE event buckets."""
    coupon_by_date = {
        cashflow.payment_date: cashflow.amount
        for cashflow in event_timeline.coupon_cashflows
    }
    exercise_dates = {
        exercise_date
        for exercise_date in event_timeline.exercise.schedule_dates
        if event_timeline.settlement < exercise_date < maturity_date
    }
    all_event_dates = sorted(set(coupon_by_date).union(exercise_dates))
    buckets: list[EventAwarePDEEventBucket] = []
    for event_date in all_event_dates:
        event_time = max(
            float(year_fraction(event_timeline.settlement, event_date, day_count)),
            0.0,
        )
        transforms: list[EventAwarePDETransform] = []
        coupon_cash = float(coupon_by_date.get(event_date, 0.0))
        if coupon_cash:
            transforms.append(
                EventAwarePDETransform(
                    kind="add_cashflow",
                    payload=coupon_cash,
                    label="coupon_cashflow",
                )
            )
        if event_date in exercise_dates:
            transforms.append(
                EventAwarePDETransform(
                    kind=event_timeline.exercise.projection_kind,
                    payload=float(event_timeline.exercise.exercise_price_cash) + coupon_cash,
                    label="embedded_exercise_projection",
                )
            )
        if transforms:
            buckets.append(
                EventAwarePDEEventBucket(
                    time=event_time,
                    transforms=tuple(transforms),
                    label=event_date.isoformat(),
                )
            )
    return tuple(buckets)


def present_value_fixed_coupon_bond(
    market_state: FixedIncomeMarketStateLike,
    spec: FixedIncomeScheduleSpecLike,
    *,
    settlement: date,
) -> float:
    """Reference straight-bond PV for bounded fixed-income claims."""
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("fixed-income present-value helper requires market_state.discount")
    pv = 0.0
    for cashflow in build_fixed_income_coupon_cashflows(
        spec,
        settlement=settlement,
        include_terminal=True,
    ):
        if cashflow.payment_date == spec.end_date:
            continue
        pv += float(cashflow.amount) * float(discount_curve.discount(cashflow.time_to_payment))
    maturity = year_fraction(settlement, spec.end_date, spec.day_count)
    pv += matured_embedded_fixed_income_value(spec, settlement=settlement) * float(
        discount_curve.discount(maturity)
    )
    return float(pv)


__all__ = [
    "EmbeddedFixedIncomeEventTimeline",
    "EmbeddedFixedIncomeExerciseConfig",
    "EmbeddedFixedIncomeSpecLike",
    "FixedIncomeCouponCashflow",
    "FixedIncomeMarketStateLike",
    "FixedIncomeScheduleSpecLike",
    "build_embedded_fixed_income_event_timeline",
    "build_embedded_fixed_income_coupon_step_map",
    "build_embedded_fixed_income_exercise_policy",
    "build_embedded_fixed_income_pde_event_buckets",
    "build_fixed_income_coupon_cashflows",
    "compile_embedded_fixed_income_lattice_contract_spec",
    "final_fixed_income_coupon_amount",
    "matured_embedded_fixed_income_value",
    "present_value_fixed_coupon_bond",
    "quoted_embedded_exercise_price_to_cash",
    "resolve_embedded_fixed_income_exercise_config",
    "settlement_date_for_fixed_income_claim",
]
