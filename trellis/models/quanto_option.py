"""Semantic-facing helper kit for single-underlier quanto options."""

from __future__ import annotations

from typing import Protocol

from trellis.core.market_state import MarketState
from trellis.models.analytical.quanto import price_quanto_option_analytical
from trellis.models.monte_carlo.quanto import price_quanto_option_monte_carlo
from trellis.models.resolution.quanto import (
    QuantoSpecLike,
    ResolvedQuantoInputs,
    resolve_quanto_inputs,
)


class QuantoOptionSpecLike(QuantoSpecLike, Protocol):
    """Minimal semantic surface consumed by the semantic-facing quanto helpers."""

    notional: float
    strike: float
    option_type: str
    n_paths: int
    n_steps: int
    seed: int


def resolve_quanto_option_inputs(
    market_state: MarketState,
    spec: QuantoOptionSpecLike,
) -> ResolvedQuantoInputs:
    """Resolve one semantic-facing quanto option into the shared market inputs."""
    return resolve_quanto_inputs(market_state, spec)


def price_quanto_option_analytical_from_market_state(
    market_state: MarketState,
    spec: QuantoOptionSpecLike,
) -> float:
    """Resolve and price one quanto option analytically."""
    resolved = resolve_quanto_option_inputs(market_state, spec)
    return float(price_quanto_option_analytical(spec, resolved))


def price_quanto_option_monte_carlo_from_market_state(
    market_state: MarketState,
    spec: QuantoOptionSpecLike,
) -> float:
    """Resolve and price one quanto option through the shared MC helper."""
    resolved = resolve_quanto_option_inputs(market_state, spec)
    return float(price_quanto_option_monte_carlo(spec, resolved))


__all__ = [
    "QuantoOptionSpecLike",
    "price_quanto_option_analytical_from_market_state",
    "price_quanto_option_monte_carlo_from_market_state",
    "resolve_quanto_option_inputs",
]
