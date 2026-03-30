"""Discounting and rate-normalization helpers for analytical routes."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy

np = get_numpy()


def safe_time_fraction(T: float) -> float:
    """Clamp analytical horizons to a non-negative time fraction."""
    return np.maximum(T, 0.0)


def implied_zero_rate(discount_factor: float, T: float) -> float:
    """Infer a continuously compounded zero rate from a discount factor."""
    T = safe_time_fraction(T)
    safe_discount_factor = np.maximum(discount_factor, 1e-16)
    return np.where(T <= 0.0, 0.0, -np.log(safe_discount_factor) / T)


def discount_factor_from_zero_rate(rate: float, T: float) -> float:
    """Return the continuously compounded discount factor for ``rate`` and ``T``."""
    T = safe_time_fraction(T)
    return np.where(T <= 0.0, 1.0, np.exp(-rate * T))


def continuous_rate_from_simple_rate(rate: float, T: float) -> float:
    """Convert a simple-compounded rate into its continuous counterpart."""
    T = safe_time_fraction(T)
    growth = 1.0 + rate * T
    if np.any(growth <= 0.0):
        raise ValueError("Simple-rate growth factor must remain positive")
    return np.where(T <= 0.0, rate, np.log(growth) / T)


def simple_rate_from_discount_factor(discount_factor: float, T: float) -> float:
    """Infer a simple-compounded rate from a discount factor and horizon."""
    T = safe_time_fraction(T)
    safe_discount_factor = np.maximum(discount_factor, 1e-16)
    return np.where(T <= 0.0, 0.0, (1.0 / safe_discount_factor - 1.0) / T)


def forward_discount_ratio(*, domestic_df: float, foreign_df: float) -> float:
    """Return the discount-factor bridge between domestic and foreign carry."""
    return foreign_df / domestic_df


def discounted_value(value: float, discount_factor: float, *, scale: float = 1.0) -> float:
    """Scale and discount an undiscounted analytical value."""
    return scale * discount_factor * value
