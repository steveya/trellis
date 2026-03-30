"""Shared market-input resolution helpers for generated pricing routes."""

from trellis.models.resolution.basket_semantics import (
    BasketSpecLike,
    ResolvedBasketSemantics,
    resolve_basket_semantics,
)
from trellis.models.resolution.quanto import (
    ResolvedQuantoInputs,
    resolve_quanto_correlation,
    resolve_quanto_foreign_curve,
    resolve_quanto_inputs,
    resolve_quanto_underlier_spot,
)

__all__ = [
    "ResolvedQuantoInputs",
    "BasketSpecLike",
    "ResolvedBasketSemantics",
    "resolve_basket_semantics",
    "resolve_quanto_correlation",
    "resolve_quanto_foreign_curve",
    "resolve_quanto_inputs",
    "resolve_quanto_underlier_spot",
]
