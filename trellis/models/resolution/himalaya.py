"""Compatibility wrapper around the generic basket semantics resolver."""

from __future__ import annotations

from trellis.models.resolution.basket_semantics import (
    BasketSpecLike as HimalayaSpecLike,
    ResolvedBasketSemantics as ResolvedHimalayaInputs,
    resolve_basket_semantics as resolve_himalaya_inputs,
)

__all__ = [
    "HimalayaSpecLike",
    "ResolvedHimalayaInputs",
    "resolve_himalaya_inputs",
]
