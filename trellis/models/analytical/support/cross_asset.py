"""Cross-asset analytical transforms."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.analytical.support.discounting import safe_time_fraction
from trellis.models.analytical.support.forwards import forward_from_discount_factors

np = get_numpy()


def effective_covariance_term(*, corr: float, sigma_1: float, sigma_2: float) -> float:
    """Return the instantaneous covariance term implied by vols and correlation."""
    return corr * sigma_1 * sigma_2


def exchange_option_effective_vol(*, sigma_1: float, sigma_2: float, corr: float) -> float:
    """Return the Margrabe-style effective volatility for an exchange ratio."""
    variance = (
        sigma_1 * sigma_1
        + sigma_2 * sigma_2
        - 2.0 * effective_covariance_term(
            corr=corr,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
        )
    )
    return np.sqrt(np.maximum(variance, 0.0))


def foreign_to_domestic_forward_bridge(
    *,
    spot: float,
    domestic_df: float,
    foreign_df: float,
) -> float:
    """Map a foreign-carry spot process into a domestic forward level."""
    return forward_from_discount_factors(
        spot=spot,
        domestic_df=domestic_df,
        foreign_df=foreign_df,
    )


def quanto_adjusted_forward(
    *,
    spot: float,
    domestic_df: float,
    foreign_df: float,
    corr: float,
    sigma_underlier: float,
    sigma_fx: float,
    T: float,
) -> float:
    """Return the quanto-adjusted forward under domestic payout semantics."""
    forward = foreign_to_domestic_forward_bridge(
        spot=spot,
        domestic_df=domestic_df,
        foreign_df=foreign_df,
    )
    T = safe_time_fraction(T)
    adjustment = np.exp(
        -effective_covariance_term(
            corr=corr,
            sigma_1=sigma_underlier,
            sigma_2=sigma_fx,
        )
        * T
    )
    return forward * adjustment
