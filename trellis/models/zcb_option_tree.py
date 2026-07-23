"""Compatibility wrappers for zero-coupon-bond option lattice pricing.

Generated construction uses the shared claim resolver, generic calibrated
lattice algebra, time-step mapping, and partial-horizon rollback directly.
"""

from __future__ import annotations

from typing import Protocol

import trellis.models.trees.models as tree_models

from trellis.models.trees.control import lattice_step_from_time
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_generic_lattice,
    lattice_backward_induction,
    lattice_backward_induction_result,
)
from trellis.models.resolution.short_rate_claims import (
    DiscountBondClaimSpecLike,
    ResolvedDiscountBondClaim,
    ShortRateClaimMarketStateLike,
    resolve_discount_bond_claim_inputs,
)


class ZCBOptionTreeMarketStateLike(ShortRateClaimMarketStateLike, Protocol):
    """Market-state interface required by the tree helpers."""


class ZCBOptionSpecLike(DiscountBondClaimSpecLike, Protocol):
    """Spec fields consumed by the tree helpers."""


def _resolve_tree_claim(
    market_state: ZCBOptionTreeMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    model: str,
    mean_reversion: float | None,
    sigma: float | None,
) -> ResolvedDiscountBondClaim:
    """Resolve one discount-bond option claim for tree construction."""
    return resolve_discount_bond_claim_inputs(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
        default_mean_reversion=0.1,
    )


def build_zcb_option_lattice(
    market_state: ZCBOptionTreeMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> RecombiningLattice:
    """Build the calibrated rate tree used by a ZCB option."""
    claim = _resolve_tree_claim(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("ZCB option tree pricing requires market_state.discount")

    step_count = int(n_steps or min(400, max(100, int(claim.bond_maturity_time * 24))))
    return build_generic_lattice(
        tree_models.MODEL_REGISTRY[str(model).strip().lower()],
        r0=float(claim.regime.initial_rate),
        sigma=float(claim.regime.sigma),
        a=float(claim.regime.mean_reversion),
        T=float(claim.bond_maturity_time),
        n_steps=step_count,
        discount_curve=discount_curve,
    )


def price_zcb_option_on_lattice(
    lattice: RecombiningLattice,
    *,
    claim: ResolvedDiscountBondClaim,
) -> float:
    """Price a European option on a ZCB using generic partial-horizon rollback."""
    expiry_step = lattice_step_from_time(
        claim.expiry_time,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
        allow_step_zero=True,
    )
    bond_step = lattice_step_from_time(
        claim.bond_maturity_time,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
        allow_step_zero=True,
    )
    if expiry_step is None or bond_step is None:
        raise ValueError("ZCB option dates must lie on the calibrated lattice horizon")
    if expiry_step >= bond_step:
        raise ValueError("ZCB option expiry step must precede bond maturity step")

    bond_result = lattice_backward_induction_result(
        lattice,
        terminal_value=1.0,
        terminal_step=bond_step,
        observation_steps=(expiry_step,),
    )
    bond_values = bond_result.observation_at(expiry_step).post_control_values
    payoff_sign = -1.0 if claim.option_type == "put" else 1.0
    return float(lattice_backward_induction(
        lattice,
        terminal_payoff=lambda step, node, lattice_: (
            claim.notional
            * max(
                payoff_sign * (bond_values[node] - claim.strike_unit),
                0.0,
            )
        ),
        terminal_step=expiry_step,
    ))


def price_zcb_option_tree(
    market_state: ZCBOptionTreeMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> float:
    """Build the requested tree and return the ZCB option PV."""
    claim = _resolve_tree_claim(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    lattice = build_zcb_option_lattice(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
        n_steps=n_steps,
    )
    return price_zcb_option_on_lattice(
        lattice,
        claim=claim,
    )


__all__ = [
    "build_zcb_option_lattice",
    "price_zcb_option_on_lattice",
    "price_zcb_option_tree",
]
