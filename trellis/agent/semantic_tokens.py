"""Shared structural semantic-family tokens.

These constants let product-specific semantic concepts compile onto a smaller
set of reusable computational abstractions without scattering raw strings
through the compiler, validator, and route registry.
"""

from __future__ import annotations

from collections.abc import Iterable


EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY = "event_triggered_two_legged_contract"
SCHEDULED_LEG_PLUS_TRIGGER_LEG_RULE = "scheduled_leg_plus_trigger_leg"
SCHEDULED_PAYMENTS_AND_TRIGGER_SETTLEMENT_RULE = (
    "scheduled_payments_and_trigger_settlement"
)
SCHEDULED_LEG_OBLIGATION_ID = "scheduled_leg_cashflow"
TRIGGER_LEG_OBLIGATION_ID = "trigger_leg_cashflow"

_CREDIT_DEFAULT_SWAP_COMPAT_LABEL = "credit_default_swap"
_CREDIT_DEFAULT_SWAP_PRODUCT_LABEL = "cds"
_CREDIT_DEFAULT_SWAP_INSTRUMENTS = frozenset({"cds", "credit_default_swap"})


def _normalize_semantic_label(value: object) -> str:
    """Normalize semantic labels before applying compatibility display rules."""
    return str(value or "").strip().lower().replace(" ", "_")


def _is_credit_default_swap_surface(instrument: object) -> bool:
    """Return whether prompt-facing display should use the CDS compatibility label."""
    return _normalize_semantic_label(instrument) in _CREDIT_DEFAULT_SWAP_INSTRUMENTS


def prompt_display_payoff_family(*, instrument: object, payoff_family: object) -> str:
    """Return a stable prompt-facing payoff-family label."""
    normalized = str(payoff_family or "").strip()
    if (
        _is_credit_default_swap_surface(instrument)
        and normalized == EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY
    ):
        return _CREDIT_DEFAULT_SWAP_PRODUCT_LABEL
    return normalized


def prompt_display_route_family(*, instrument: object, route_family: object) -> str:
    """Return a stable prompt-facing route-family label."""
    normalized = str(route_family or "").strip()
    if (
        _is_credit_default_swap_surface(instrument)
        and normalized == EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY
    ):
        return _CREDIT_DEFAULT_SWAP_COMPAT_LABEL
    return normalized


def prompt_display_route_families(
    *,
    instrument: object,
    route_families: Iterable[object],
) -> tuple[str, ...]:
    """Return stable prompt-facing route-family labels."""
    return tuple(
        prompt_display_route_family(
            instrument=instrument,
            route_family=family,
        )
        for family in route_families
    )
