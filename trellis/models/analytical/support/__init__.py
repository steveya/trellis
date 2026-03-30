"""Reusable analytical building blocks for route kernels and subproblems."""

from trellis.models.analytical.support.cross_asset import (
    effective_covariance_term,
    exchange_option_effective_vol,
    foreign_to_domestic_forward_bridge,
    quanto_adjusted_forward,
)
from trellis.models.analytical.support.discounting import (
    continuous_rate_from_simple_rate,
    discount_factor_from_zero_rate,
    discounted_value,
    forward_discount_ratio,
    implied_zero_rate,
    safe_time_fraction,
    simple_rate_from_discount_factor,
)
from trellis.models.analytical.support.forwards import (
    forward_from_carry_rate,
    forward_from_discount_factors,
    forward_from_dividend_yield,
)
from trellis.models.analytical.support.payoffs import (
    asset_or_nothing_intrinsic,
    call_put_parity_gap,
    cash_or_nothing_intrinsic,
    normalized_option_type,
    terminal_vanilla_from_basis,
    terminal_intrinsic,
)

__all__ = [
    "asset_or_nothing_intrinsic",
    "call_put_parity_gap",
    "cash_or_nothing_intrinsic",
    "continuous_rate_from_simple_rate",
    "discount_factor_from_zero_rate",
    "discounted_value",
    "effective_covariance_term",
    "exchange_option_effective_vol",
    "foreign_to_domestic_forward_bridge",
    "forward_discount_ratio",
    "forward_from_carry_rate",
    "forward_from_discount_factors",
    "forward_from_dividend_yield",
    "implied_zero_rate",
    "normalized_option_type",
    "quanto_adjusted_forward",
    "safe_time_fraction",
    "simple_rate_from_discount_factor",
    "terminal_vanilla_from_basis",
    "terminal_intrinsic",
]
