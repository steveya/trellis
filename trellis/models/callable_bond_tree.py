"""Reusable callable-bond tree helpers.

This module lifts callable fixed-income schedule/control logic out of generated
routes so the DSL can target stable Trellis primitives instead of rebuilding the
same coupon and exercise mapping inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from trellis.core.date_utils import (
    build_payment_timeline,
    coerce_contract_timeline_from_dates,
    normalize_explicit_dates,
    year_fraction,
)
from trellis.core.types import TimelineRole
import trellis.models.trees.algebra as lattice_algebra
import trellis.models.trees.models as tree_models
from trellis.models.trees.control import (
    lattice_step_from_time,
    lattice_steps_from_timeline,
    resolve_lattice_exercise_policy_from_control_style,
)
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_lattice,
    price_on_lattice,
)


@dataclass(frozen=True)
class _EmbeddedBondExerciseConfig:
    """Resolved embedded-option semantics for callable or puttable bonds."""

    schedule_dates: object
    exercise_price: float
    exercise_style: str
    control_style: str
    reference_bound: str


def build_callable_bond_lattice(
    market_state,
    spec,
    *,
    model: str = "hull_white",
    mean_reversion: float = 0.1,
    n_steps: int | None = None,
) -> RecombiningLattice:
    """Build the calibrated tree used to price a callable bond."""
    settlement = _settlement_date(market_state, spec)
    maturity = year_fraction(settlement, spec.end_date, spec.day_count)
    if maturity <= 0.0:
        raise ValueError("Callable bond maturity must be after settlement")

    r0 = float(market_state.discount.zero_rate(max(maturity / 2.0, 1e-6)))
    black_vol = float(market_state.vol_surface.black_vol(max(maturity / 2.0, 1e-6), max(r0, 1e-6)))
    step_count = int(n_steps or min(200, max(50, int(maturity * 50))))
    model_key = str(model).strip().lower()

    tree_model = tree_models.MODEL_REGISTRY[model_key]
    sigma = black_vol if tree_model.vol_type == "lognormal" else black_vol * max(abs(r0), 1e-6)
    return build_lattice(
        lattice_algebra.BINOMIAL_1F_TOPOLOGY,
        lattice_algebra.UNIFORM_ADDITIVE_MESH,
        tree_model.as_lattice_model_spec(),
        calibration_target=lattice_algebra.TERM_STRUCTURE_TARGET(market_state.discount),
        r0=r0,
        sigma=sigma,
        a=mean_reversion,
        T=maturity,
        n_steps=step_count,
    )


def build_callable_bond_coupon_map(
    spec,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
) -> dict[int, float]:
    """Map scheduled coupon payments onto lattice steps."""
    coupon_by_step: dict[int, float] = {}
    payment_timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=settlement,
        label="callable_bond_coupon_timeline",
    )

    for period in payment_timeline:
        if period.payment_date <= settlement:
            continue
        step = lattice_step_from_time(
            period,
            dt=dt,
            n_steps=n_steps,
            allow_terminal_step=True,
        )
        if step is None:
            continue
        coupon = float(spec.notional) * float(spec.coupon) * float(period.accrual_fraction or 0.0)
        coupon_by_step[step] = coupon_by_step.get(step, 0.0) + coupon
    return coupon_by_step


def build_callable_bond_exercise_policy(
    spec,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
):
    """Resolve callable or puttable exercise dates into a lattice policy."""
    exercise = _resolve_embedded_bond_exercise_config(spec)
    exercise_dates = tuple(
        exercise_date
        for exercise_date in normalize_explicit_dates(exercise.schedule_dates)
        if settlement < exercise_date <= spec.end_date
    )
    if not exercise_dates:
        return resolve_lattice_exercise_policy_from_control_style(
            exercise.control_style,
            exercise_steps=(),
            exercise_style=exercise.exercise_style,
        )

    exercise_timeline = coerce_contract_timeline_from_dates(
        exercise_dates,
        role=TimelineRole.EXERCISE,
        day_count=spec.day_count,
        time_origin=settlement,
        label=f"{exercise.exercise_style}_timeline",
    )
    exercise_steps = lattice_steps_from_timeline(
        exercise_timeline,
        dt=dt,
        n_steps=n_steps,
    )
    return resolve_lattice_exercise_policy_from_control_style(
        exercise.control_style,
        exercise_steps=exercise_steps,
        exercise_style=exercise.exercise_style,
    )


def compile_callable_bond_contract_spec(
    spec,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
) -> lattice_algebra.LatticeContractSpec:
    """Compile callable- or puttable-bond coupons and exercise into a lattice contract."""
    exercise = _resolve_embedded_bond_exercise_config(spec)
    coupon_by_step = build_callable_bond_coupon_map(
        spec,
        settlement=settlement,
        dt=dt,
        n_steps=n_steps,
    )
    exercise_policy = build_callable_bond_exercise_policy(
        spec,
        settlement=settlement,
        dt=dt,
        n_steps=n_steps,
    )
    terminal_coupon = coupon_by_step.get(n_steps, 0.0)
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
                lambda step, node, lattice, obs: float(exercise.exercise_price)
                + float(coupon_by_step.get(step, 0.0))
            ),
        ),
        metadata={"coupon_by_step": coupon_by_step},
    )


def price_callable_bond_on_lattice(
    lattice: RecombiningLattice,
    *,
    spec=None,
    settlement: date | None = None,
    contract_spec: lattice_algebra.LatticeContractSpec | None = None,
) -> float:
    """Price a callable bond on a pre-built lattice."""
    if contract_spec is None:
        if spec is None or settlement is None:
            raise ValueError("Provide either contract_spec or both spec and settlement")
        contract_spec = compile_callable_bond_contract_spec(
            spec,
            settlement=settlement,
            dt=lattice.dt,
            n_steps=lattice.n_steps,
        )
    return float(price_on_lattice(lattice, contract_spec))


def price_callable_bond_tree(
    market_state,
    spec,
    *,
    model: str = "hull_white",
    mean_reversion: float = 0.1,
    n_steps: int | None = None,
) -> float:
    """Build the requested callable- or puttable-bond tree and return the holder PV."""
    settlement = _settlement_date(market_state, spec)
    lattice = build_callable_bond_lattice(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        n_steps=n_steps,
    )
    tree_price = price_callable_bond_on_lattice(
        lattice,
        spec=spec,
        settlement=settlement,
    )
    straight_price = straight_bond_present_value(market_state, spec, settlement=settlement)
    exercise = _resolve_embedded_bond_exercise_config(spec)
    if exercise.reference_bound == "lower":
        return max(tree_price, straight_price)
    return min(tree_price, straight_price)


def straight_bond_present_value(market_state, spec, *, settlement: date) -> float:
    """Reference straight-bond PV used to cap callable-bond values."""
    pv = 0.0
    payment_timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=settlement,
        label="callable_bond_coupon_timeline",
    )
    for period in payment_timeline:
        if period.payment_date <= settlement:
            continue
        tau = float(period.accrual_fraction or 0.0)
        t_pay = float(period.t_payment or 0.0)
        pv += float(spec.notional) * float(spec.coupon) * tau * float(
            market_state.discount.discount(t_pay)
        )

    maturity = year_fraction(settlement, spec.end_date, spec.day_count)
    pv += float(spec.notional) * float(market_state.discount.discount(maturity))
    return float(pv)


def _settlement_date(market_state, spec) -> date:
    return market_state.settlement or market_state.as_of or spec.start_date


def _resolve_embedded_bond_exercise_config(spec) -> _EmbeddedBondExerciseConfig:
    has_call_dates = hasattr(spec, "call_dates")
    has_put_dates = hasattr(spec, "put_dates")
    if has_call_dates and has_put_dates:
        raise ValueError("Embedded bond spec must define either call_dates or put_dates, not both")
    if has_call_dates:
        return _EmbeddedBondExerciseConfig(
            schedule_dates=getattr(spec, "call_dates"),
            exercise_price=_quoted_exercise_price_to_cash(spec, "call_price"),
            exercise_style="issuer_call",
            control_style="issuer_min",
            reference_bound="upper",
        )
    if has_put_dates:
        return _EmbeddedBondExerciseConfig(
            schedule_dates=getattr(spec, "put_dates"),
            exercise_price=_quoted_exercise_price_to_cash(spec, "put_price"),
            exercise_style="holder_put",
            control_style="holder_max",
            reference_bound="lower",
        )
    raise ValueError("Embedded bond spec must define call_dates or put_dates")


def _quoted_exercise_price_to_cash(spec, field_name: str) -> float:
    quoted_price = float(getattr(spec, field_name))
    return quoted_price / 100.0 * float(spec.notional)
