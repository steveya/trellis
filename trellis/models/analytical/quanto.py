"""Shared analytical helper for single-name quanto options."""

from __future__ import annotations

from typing import Protocol

from trellis.models.black import black76_call, black76_put
from trellis.models.analytical.support import (
    discounted_value,
    normalized_option_type,
    quanto_adjusted_forward,
    terminal_intrinsic,
)
from trellis.models.resolution.quanto import ResolvedQuantoInputs


class QuantoAnalyticalSpecLike(Protocol):
    """Minimal spec surface consumed by the shared quanto analytical kernel."""

    notional: float
    strike: float
    option_type: str


def price_quanto_option_raw(
    spec: QuantoAnalyticalSpecLike,
    resolved: ResolvedQuantoInputs,
) -> float:
    """Raw quanto pricing kernel over resolved market inputs."""
    option_type = normalized_option_type(spec.option_type)
    strike = spec.strike
    notional = spec.notional

    if resolved.T <= 0.0:
        return notional * terminal_intrinsic(
            option_type,
            spot=resolved.spot,
            strike=strike,
        )

    quanto_forward = quanto_adjusted_forward(
        spot=resolved.spot,
        domestic_df=resolved.domestic_df,
        foreign_df=resolved.foreign_df,
        corr=resolved.corr,
        sigma_underlier=resolved.sigma_underlier,
        sigma_fx=resolved.sigma_fx,
        T=resolved.T,
    )
    if option_type == "call":
        option_value = black76_call(
            quanto_forward,
            strike,
            resolved.sigma_underlier,
            resolved.T,
        )
    elif option_type == "put":
        option_value = black76_put(
            quanto_forward,
            strike,
            resolved.sigma_underlier,
            resolved.T,
        )
    return discounted_value(
        option_value,
        resolved.domestic_df,
        scale=notional,
    )


def price_quanto_option_analytical(
    spec: QuantoAnalyticalSpecLike,
    resolved: ResolvedQuantoInputs,
) -> float:
    """Compatibility wrapper around the raw quanto pricing kernel."""
    return price_quanto_option_raw(spec, resolved)


__all__ = [
    "QuantoAnalyticalSpecLike",
    "price_quanto_option_analytical",
    "price_quanto_option_raw",
]
