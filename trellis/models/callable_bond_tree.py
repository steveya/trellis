"""Reusable callable-bond tree helpers.

This module lifts callable fixed-income schedule/control logic out of generated
routes so the DSL can target stable Trellis primitives instead of rebuilding the
same coupon and exercise mapping inline.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from trellis.core.date_utils import (
    build_exercise_timeline_from_dates,
    build_payment_timeline,
    year_fraction,
)
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    TERM_STRUCTURE_TARGET,
    UNIFORM_ADDITIVE_MESH,
    LatticeContractSpec,
    LatticeControlSpec,
    LatticeLinearClaimSpec,
)
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
from trellis.models.trees.models import MODEL_REGISTRY


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

    tree_model = MODEL_REGISTRY[model_key]
    sigma = black_vol if tree_model.vol_type == "lognormal" else black_vol * max(abs(r0), 1e-6)
    return build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        tree_model.as_lattice_model_spec(),
        calibration_target=TERM_STRUCTURE_TARGET(market_state.discount),
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
    """Resolve issuer-call exercise dates into a lattice policy."""
    call_dates = _normalized_call_dates(spec.call_dates)
    if not call_dates:
        return resolve_lattice_exercise_policy_from_control_style(
            "issuer_min",
            exercise_steps=(),
            exercise_style="issuer_call",
        )

    exercise_timeline = build_exercise_timeline_from_dates(
        [d for d in call_dates if settlement < d <= spec.end_date],
        day_count=spec.day_count,
        time_origin=settlement,
        label="callable_bond_call_timeline",
    )
    exercise_steps = lattice_steps_from_timeline(
        exercise_timeline,
        dt=dt,
        n_steps=n_steps,
    )
    return resolve_lattice_exercise_policy_from_control_style(
        "issuer_min",
        exercise_steps=exercise_steps,
        exercise_style="issuer_call",
    )


def compile_callable_bond_contract_spec(
    spec,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
) -> LatticeContractSpec:
    """Compile callable-bond coupons and exercise into a lattice contract."""
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
    return LatticeContractSpec(
        claim=LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice, obs: float(spec.notional) + float(terminal_coupon),
            node_cashflow_fn=lambda step, node, lattice, obs: float(coupon_by_step.get(step, 0.0)),
            observable_requirements=("rate",),
        ),
        control=LatticeControlSpec(
            objective="issuer_min",
            exercise_steps=exercise_policy.exercise_steps,
            exercise_value_fn=(
                lambda step, node, lattice, obs: float(spec.call_price) + float(coupon_by_step.get(step, 0.0))
            ),
        ),
        metadata={"coupon_by_step": coupon_by_step},
    )


def price_callable_bond_on_lattice(
    lattice: RecombiningLattice,
    *,
    spec=None,
    settlement: date | None = None,
    contract_spec: LatticeContractSpec | None = None,
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
    """Build the requested callable-bond tree and return the holder PV."""
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
    return min(tree_price, straight_bond_present_value(market_state, spec, settlement=settlement))


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


def _normalized_call_dates(call_dates: str | Iterable[date]) -> list[date]:
    if isinstance(call_dates, str):
        return [date.fromisoformat(item.strip()) for item in call_dates.split(",") if item.strip()]
    return sorted(date.fromisoformat(item) if isinstance(item, str) else item for item in call_dates)


def _settlement_date(market_state, spec) -> date:
    return market_state.settlement or market_state.as_of or spec.start_date
