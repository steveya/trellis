"""Analytical (closed-form) pricing formulae."""

from trellis.models.analytical.support import (
    asset_or_nothing_intrinsic,
    call_put_parity_gap,
    cash_or_nothing_intrinsic,
    continuous_rate_from_simple_rate,
    discount_factor_from_zero_rate,
    discounted_value,
    effective_covariance_term,
    exchange_option_effective_vol,
    foreign_to_domestic_forward_bridge,
    forward_discount_ratio,
    forward_from_carry_rate,
    forward_from_discount_factors,
    forward_from_dividend_yield,
    implied_zero_rate,
    normalized_option_type,
    quanto_adjusted_forward,
    safe_time_fraction,
    simple_rate_from_discount_factor,
    terminal_vanilla_from_basis,
    terminal_intrinsic,
)
from trellis.models.analytical.jamshidian import (
    ResolvedJamshidianInputs,
    zcb_option_hw,
    zcb_option_hw_raw,
)
from trellis.models.analytical.barrier import (
    ResolvedBarrierInputs,
    barrier_option_price,
    barrier_image_raw,
    barrier_regime_selector_raw,
    down_and_out_call,
    down_and_out_call_raw,
    down_and_in_call,
    down_and_in_call_raw,
    rebate_raw,
    vanilla_call_raw,
)
from trellis.models.analytical.equity_exotics import (
    ResolvedEquityAnalyticalInputs,
    equity_variance_swap_outputs_analytical,
    price_equity_cliquet_option_analytical,
    price_equity_chooser_option_analytical,
    price_equity_compound_option_analytical,
    price_equity_digital_option_analytical,
    price_equity_fixed_lookback_option_analytical,
    price_equity_variance_swap_analytical,
)
__all__ = [
    "zcb_option_hw",
    "barrier_option_price",
    "barrier_image_raw",
    "barrier_regime_selector_raw",
    "down_and_out_call",
    "down_and_out_call_raw",
    "down_and_in_call",
    "down_and_in_call_raw",
    "rebate_raw",
    "ResolvedBarrierInputs",
    "ResolvedEquityAnalyticalInputs",
    "equity_variance_swap_outputs_analytical",
    "price_equity_cliquet_option_analytical",
    "vanilla_call_raw",
    "price_equity_chooser_option_analytical",
    "price_equity_compound_option_analytical",
    "price_equity_digital_option_analytical",
    "price_equity_fixed_lookback_option_analytical",
    "price_equity_variance_swap_analytical",
    "price_quanto_option_analytical",
    "price_quanto_option_raw",
    "ResolvedJamshidianInputs",
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
    "zcb_option_hw_raw",
]


def __getattr__(name: str):
    if name in {"price_quanto_option_analytical", "price_quanto_option_raw"}:
        from trellis.models.analytical.quanto import (
            price_quanto_option_analytical as _price_quanto_option_analytical,
            price_quanto_option_raw as _price_quanto_option_raw,
        )

        globals().update(
            {
                "price_quanto_option_analytical": _price_quanto_option_analytical,
                "price_quanto_option_raw": _price_quanto_option_raw,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
