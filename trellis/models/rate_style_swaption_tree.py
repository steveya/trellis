"""Stable rate-tree helpers for single-exercise rate-style swaptions."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from trellis.core.market_state import MarketState
from trellis.models.bermudan_swaption_tree import (
    BermudanSwaptionTreeSpec,
    price_bermudan_swaption_tree,
)
from trellis.models.rate_style_swaption import resolve_swaption_curve_basis_spread


class EuropeanRateTreeSwaptionSpecLike(Protocol):
    """Protocol for single-exercise swaption specs used by the tree helper."""

    notional: float
    strike: float
    expiry_date: object
    swap_start: object
    swap_end: object
    swap_frequency: object
    day_count: object
    rate_index: str | None
    is_payer: bool


def build_swaption_tree_spec(
    spec: EuropeanRateTreeSwaptionSpecLike,
) -> BermudanSwaptionTreeSpec:
    """Map a European rate-style swaption onto the one-exercise tree helper surface."""
    if spec.swap_start != spec.expiry_date:
        raise ValueError(
            "Rate-tree swaption helper requires swap_start == expiry_date for the "
            "single-exercise forward-start contract."
        )
    return BermudanSwaptionTreeSpec(
        notional=float(spec.notional),
        strike=float(spec.strike),
        exercise_dates=(spec.expiry_date,),
        swap_end=spec.swap_end,
        swap_frequency=spec.swap_frequency,
        day_count=spec.day_count,
        rate_index=spec.rate_index,
        is_payer=bool(spec.is_payer),
    )


def price_swaption_tree(
    market_state: MarketState,
    spec: EuropeanRateTreeSwaptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> float:
    """Price a single-exercise rate-style swaption on the checked-in rate tree."""
    tree_spec = build_swaption_tree_spec(spec)
    curve_basis_spread = resolve_swaption_curve_basis_spread(market_state, spec)
    if curve_basis_spread:
        tree_spec = replace(
            tree_spec,
            strike=float(tree_spec.strike) - float(curve_basis_spread),
        )
    return float(
        price_bermudan_swaption_tree(
            market_state,
            tree_spec,
            model=model,
            mean_reversion=mean_reversion,
            sigma=sigma,
            n_steps=n_steps,
        )
    )


__all__ = [
    "EuropeanRateTreeSwaptionSpecLike",
    "build_swaption_tree_spec",
    "price_swaption_tree",
]
