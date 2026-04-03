"""Resolved-input analytical helpers for FX vanilla pricing."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.models.analytical.support import (
    discounted_value,
    forward_from_discount_factors,
    normalized_option_type,
    terminal_vanilla_from_basis,
)
from trellis.models.black import (
    black76_asset_or_nothing_call,
    black76_asset_or_nothing_put,
    black76_cash_or_nothing_call,
    black76_cash_or_nothing_put,
)


@dataclass(frozen=True)
class ResolvedGarmanKohlhagenInputs:
    """Resolved market and contract terms for an FX vanilla option."""

    spot: float
    strike: float
    sigma: float
    T: float
    df_domestic: float
    df_foreign: float


def garman_kohlhagen_price_raw(
    option_type: str,
    resolved: ResolvedGarmanKohlhagenInputs,
) -> float:
    """Raw Garman-Kohlhagen kernel over resolved market inputs."""
    normalized_type = normalized_option_type(option_type)
    forward = forward_from_discount_factors(
        spot=resolved.spot,
        domestic_df=resolved.df_domestic,
        foreign_df=resolved.df_foreign,
    )
    if normalized_type == "call":
        asset_value = black76_asset_or_nothing_call(
            forward,
            resolved.strike,
            resolved.sigma,
            resolved.T,
        )
        cash_value = black76_cash_or_nothing_call(
            forward,
            resolved.strike,
            resolved.sigma,
            resolved.T,
        )
    else:
        asset_value = black76_asset_or_nothing_put(
            forward,
            resolved.strike,
            resolved.sigma,
            resolved.T,
        )
        cash_value = black76_cash_or_nothing_put(
            forward,
            resolved.strike,
            resolved.sigma,
            resolved.T,
        )
    return discounted_value(
        terminal_vanilla_from_basis(
            normalized_type,
            asset_value=asset_value,
            cash_value=cash_value,
            strike=resolved.strike,
        ),
        resolved.df_domestic,
    )


def garman_kohlhagen_call_raw(resolved: ResolvedGarmanKohlhagenInputs) -> float:
    """Raw call kernel for domestic-currency FX vanilla pricing."""
    return garman_kohlhagen_price_raw("call", resolved)


def garman_kohlhagen_put_raw(resolved: ResolvedGarmanKohlhagenInputs) -> float:
    """Raw put kernel for domestic-currency FX vanilla pricing."""
    return garman_kohlhagen_price_raw("put", resolved)


__all__ = [
    "ResolvedGarmanKohlhagenInputs",
    "garman_kohlhagen_call_raw",
    "garman_kohlhagen_price_raw",
    "garman_kohlhagen_put_raw",
]
