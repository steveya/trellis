"""Stable zero-coupon-bond option tree helpers."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import trellis.models.trees.models as tree_models

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.trees.lattice import RecombiningLattice, build_generic_lattice
from trellis.models.zcb_option import (
    VolSurfaceLike,
    normalize_zcb_option_strike,
)


class DiscountCurveLike(Protocol):
    """Discount interface required by the tree helpers."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return the zero rate to time ``t``."""
        ...


class ZCBOptionTreeMarketStateLike(Protocol):
    """Market-state interface required by the tree helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None


class ZCBOptionSpecLike(Protocol):
    """Spec fields consumed by the tree helpers."""

    notional: float
    strike: float
    expiry_date: date
    bond_maturity_date: date


def build_zcb_option_lattice(
    market_state: ZCBOptionTreeMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float = 0.1,
    n_steps: int | None = None,
) -> RecombiningLattice:
    """Build the calibrated rate tree used by a ZCB option."""
    settlement = _settlement_date(market_state, spec)
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("ZCB option tree pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("ZCB option tree pricing requires market_state.vol_surface")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    t_exp = year_fraction(settlement, spec.expiry_date, day_count)
    t_bond = year_fraction(settlement, spec.bond_maturity_date, day_count)
    if t_bond <= 0.0:
        raise ValueError("bond_maturity_date must be after settlement")
    if t_bond <= t_exp:
        raise ValueError("bond_maturity_date must be after expiry_date")

    strike_unit = normalize_zcb_option_strike(spec.strike, spec.notional)
    r0 = float(discount_curve.zero_rate(max(min(t_exp, t_bond), 1e-6)))
    sigma = float(market_state.vol_surface.black_vol(max(t_exp, 1e-6), strike_unit))
    step_count = int(n_steps or min(400, max(100, int(t_bond * 24))))
    return build_generic_lattice(
        tree_models.MODEL_REGISTRY[str(model).strip().lower()],
        r0=r0,
        sigma=sigma,
        a=float(mean_reversion),
        T=float(t_bond),
        n_steps=step_count,
        discount_curve=discount_curve,
    )


def price_zcb_option_on_lattice(
    lattice: RecombiningLattice,
    *,
    spec: ZCBOptionSpecLike,
    settlement: date,
) -> float:
    """Price a European option on a ZCB using a pre-built calibrated lattice."""
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    t_exp = year_fraction(settlement, spec.expiry_date, day_count)
    t_bond = year_fraction(settlement, spec.bond_maturity_date, day_count)
    exp_step = int(round(t_exp / lattice.dt))
    bond_step = int(round(t_bond / lattice.dt))
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

    strike = float(spec.strike)
    notional = float(spec.notional)
    option_type = _resolve_option_type(spec)
    unit_strike = normalize_zcb_option_strike(strike, notional)
    payoff = []
    for node in range(lattice.n_nodes(exp_step)):
        bond_unit_price = float(zcb_values[node])
        if option_type == "put":
            payoff.append(max(unit_strike - bond_unit_price, 0.0) * notional)
        else:
            payoff.append(max(bond_unit_price - unit_strike, 0.0) * notional)

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
    mean_reversion: float = 0.1,
    n_steps: int | None = None,
) -> float:
    """Build the requested tree and return the ZCB option PV."""
    settlement = _settlement_date(market_state, spec)
    lattice = build_zcb_option_lattice(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        n_steps=n_steps,
    )
    return price_zcb_option_on_lattice(
        lattice,
        spec=spec,
        settlement=settlement,
    )


def _resolve_option_type(spec) -> str:
    option_type = getattr(spec, "option_type", None)
    if option_type is not None:
        normalized = str(option_type).strip().lower()
        if normalized in {"call", "put"}:
            return normalized
        if normalized == "payer":
            return "put"
        if normalized == "receiver":
            return "call"
    if hasattr(spec, "is_call"):
        return "call" if bool(spec.is_call) else "put"
    if hasattr(spec, "is_payer"):
        return "put" if bool(spec.is_payer) else "call"
    return "call"


def _settlement_date(market_state, spec) -> date:
    settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
    if settlement is None:
        return spec.expiry_date
    return settlement


__all__ = [
    "build_zcb_option_lattice",
    "price_zcb_option_on_lattice",
    "price_zcb_option_tree",
]
