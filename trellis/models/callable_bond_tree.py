"""Reusable callable-bond tree helpers.

This module lifts callable fixed-income schedule/control logic out of generated
routes so the DSL can target stable Trellis primitives instead of rebuilding the
same coupon and exercise mapping inline.
"""

from __future__ import annotations

from datetime import date

from trellis.core.date_utils import (
    year_fraction,
)
import trellis.models.trees.algebra as lattice_algebra
import trellis.models.trees.models as tree_models
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_lattice,
    price_on_lattice,
)
from trellis.models.hull_white_parameters import resolve_hull_white_parameters
from trellis.models.short_rate_fixed_income import (
    build_embedded_fixed_income_coupon_step_map,
    build_embedded_fixed_income_event_timeline,
    build_embedded_fixed_income_exercise_policy,
    compile_embedded_fixed_income_lattice_contract_spec,
    present_value_fixed_coupon_bond,
    settlement_date_for_fixed_income_claim,
)


def build_callable_bond_lattice(
    market_state,
    spec,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> RecombiningLattice:
    """Build the calibrated tree used to price a callable bond."""
    settlement = settlement_date_for_fixed_income_claim(market_state, spec)
    maturity = year_fraction(settlement, spec.end_date, spec.day_count)
    if maturity <= 0.0:
        raise ValueError("Callable bond maturity must be after settlement")

    r0 = float(market_state.discount.zero_rate(max(maturity / 2.0, 1e-6)))
    default_sigma = None
    if market_state.vol_surface is not None:
        black_vol = float(market_state.vol_surface.black_vol(max(maturity / 2.0, 1e-6), max(r0, 1e-6)))
        default_sigma = black_vol * max(abs(r0), 1e-6)
    step_count = int(n_steps or min(200, max(50, int(maturity * 50))))
    model_key = str(model).strip().lower()

    tree_model = tree_models.MODEL_REGISTRY[model_key]
    resolved_mean_reversion, resolved_sigma = resolve_hull_white_parameters(
        market_state,
        mean_reversion=mean_reversion,
        sigma=sigma,
        default_mean_reversion=0.1,
        default_sigma=default_sigma if tree_model.vol_type != "lognormal" else black_vol if market_state.vol_surface is not None else None,
    )
    return build_lattice(
        lattice_algebra.BINOMIAL_1F_TOPOLOGY,
        lattice_algebra.UNIFORM_ADDITIVE_MESH,
        tree_model.as_lattice_model_spec(),
        calibration_target=lattice_algebra.TERM_STRUCTURE_TARGET(market_state.discount),
        r0=r0,
        sigma=resolved_sigma,
        a=resolved_mean_reversion,
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
    event_timeline = build_embedded_fixed_income_event_timeline(spec, settlement=settlement)
    return build_embedded_fixed_income_coupon_step_map(
        event_timeline,
        dt=dt,
        n_steps=n_steps,
    )


def build_callable_bond_exercise_policy(
    spec,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
):
    """Resolve callable or puttable exercise dates into a lattice policy."""
    event_timeline = build_embedded_fixed_income_event_timeline(spec, settlement=settlement)
    return build_embedded_fixed_income_exercise_policy(
        event_timeline,
        maturity_date=spec.end_date,
        day_count=spec.day_count,
        dt=dt,
        n_steps=n_steps,
    )


def compile_callable_bond_contract_spec(
    spec,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
) -> lattice_algebra.LatticeContractSpec:
    """Compile callable- or puttable-bond coupons and exercise into a lattice contract."""
    return compile_embedded_fixed_income_lattice_contract_spec(
        spec,
        settlement=settlement,
        dt=dt,
        n_steps=n_steps,
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
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> float:
    """Build the requested callable- or puttable-bond tree and return the holder PV."""
    settlement = settlement_date_for_fixed_income_claim(market_state, spec)
    lattice = build_callable_bond_lattice(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
        n_steps=n_steps,
    )
    tree_price = price_callable_bond_on_lattice(
        lattice,
        spec=spec,
        settlement=settlement,
    )
    straight_price = straight_bond_present_value(market_state, spec, settlement=settlement)
    exercise = build_embedded_fixed_income_event_timeline(spec, settlement=settlement).exercise
    if exercise.reference_bound == "lower":
        return max(tree_price, straight_price)
    return min(tree_price, straight_price)


def straight_bond_present_value(market_state, spec, *, settlement: date) -> float:
    """Reference straight-bond PV used to cap callable-bond values."""
    return present_value_fixed_coupon_bond(
        market_state,
        spec,
        settlement=settlement,
    )
