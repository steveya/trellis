"""Forward-construction helpers for analytical routes."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy

np = get_numpy()


def forward_from_discount_factors(
    *,
    spot: float,
    domestic_df: float,
    foreign_df: float,
) -> float:
    """Return the standard carry-adjusted forward from spot and discount factors."""
    return spot * foreign_df / domestic_df


def forward_from_carry_rate(*, spot: float, carry_rate: float, T: float) -> float:
    """Return the forward implied by continuous carry over ``T``."""
    return spot * np.exp(carry_rate * T)


def forward_from_dividend_yield(
    *,
    spot: float,
    domestic_rate: float,
    dividend_yield: float,
    T: float,
) -> float:
    """Return the standard equity forward under continuous dividend yield."""
    return forward_from_carry_rate(
        spot=spot,
        carry_rate=domestic_rate - dividend_yield,
        T=T,
    )
