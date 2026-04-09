"""Stable zero-coupon-bond option tree helpers."""

from __future__ import annotations

from typing import Protocol

import trellis.models.trees.models as tree_models

from trellis.models.trees.lattice import RecombiningLattice, build_generic_lattice
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
    """Price a European option on a ZCB using a pre-built calibrated lattice."""
    exp_step = int(round(claim.expiry_time / lattice.dt))
    bond_step = int(round(claim.bond_maturity_time / lattice.dt))
    if bond_step > lattice.n_steps:
        raise ValueError(
            f"Lattice horizon {lattice.n_steps} is shorter than bond maturity step {bond_step}"
        )

    zcb_values = [1.0] * lattice.n_nodes(bond_step)
    for step in range(bond_step - 1, exp_step - 1, -1):
        new_vals = [0.0] * lattice.n_nodes(step)
        for node in range(lattice.n_nodes(step)):
            discount = lattice.get_discount(step, node)
            probs = lattice.get_probabilities(step, node)
            children = lattice.child_indices(step, node)
            new_vals[node] = discount * sum(
                p * zcb_values[child] for p, child in zip(probs, children)
            )
        zcb_values = new_vals

    payoff = []
    for node in range(lattice.n_nodes(exp_step)):
        bond_unit_price = float(zcb_values[node])
        if claim.option_type == "put":
            payoff.append(max(claim.strike_unit - bond_unit_price, 0.0) * claim.notional)
        else:
            payoff.append(max(bond_unit_price - claim.strike_unit, 0.0) * claim.notional)

    values = payoff
    for step in range(exp_step - 1, -1, -1):
        new_vals = [0.0] * lattice.n_nodes(step)
        for node in range(lattice.n_nodes(step)):
            discount = lattice.get_discount(step, node)
            probs = lattice.get_probabilities(step, node)
            children = lattice.child_indices(step, node)
            new_vals[node] = discount * sum(
                p * values[child] for p, child in zip(probs, children)
            )
        values = new_vals
    return float(values[0])


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
