"""Semantic-facing vanilla-equity transform helpers.

This module is a thin compatibility wrapper over the reusable single-state
diffusion transform family helper. Generated adapters should bind transform
pricing here instead of rebuilding characteristic functions, market
conventions, or put/call parity glue inline.
"""

from __future__ import annotations

from trellis.models.transforms.single_state_diffusion import (
    ResolvedSingleStateTransformInputs as ResolvedEquityTransformInputs,
    SingleStateTransformResult as VanillaEquityTransformResult,
    price_single_state_terminal_claim_transform_result,
    resolve_single_state_terminal_claim_transform_inputs,
)
from trellis.models.resolution.single_state_diffusion import (
    SingleStateDiffusionMarketStateLike,
    SingleStateDiffusionSpecLike,
    gbm_log_ratio_char_fn,
    gbm_log_spot_char_fn,
    put_from_call_parity,
    terminal_intrinsic_from_resolved,
)


def resolve_vanilla_equity_transform_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> ResolvedEquityTransformInputs:
    """Resolve transform inputs from market state and a vanilla option spec."""
    return resolve_single_state_terminal_claim_transform_inputs(
        market_state,
        spec,
        method=method,
        fft_alpha=fft_alpha,
        fft_points=fft_points,
        fft_eta=fft_eta,
        cos_points=cos_points,
        cos_truncation=cos_truncation,
    )


def price_vanilla_equity_option_transform_result(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> VanillaEquityTransformResult:
    """Return a structured transform price result for a vanilla European option."""
    return price_single_state_terminal_claim_transform_result(
        market_state,
        spec,
        method=method,
        fft_alpha=fft_alpha,
        fft_points=fft_points,
        fft_eta=fft_eta,
        cos_points=cos_points,
        cos_truncation=cos_truncation,
        intrinsic_fn=terminal_intrinsic_from_resolved,
        fft_log_spot_char_fn=gbm_log_spot_char_fn,
        cos_log_ratio_char_fn=gbm_log_ratio_char_fn,
        put_from_call_parity_fn=put_from_call_parity,
    )


def price_vanilla_equity_option_transform(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> float:
    """Return a scalar transform price for a vanilla European equity option."""
    return float(
        price_vanilla_equity_option_transform_result(
            market_state,
            spec,
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
        ).price
    )


__all__ = [
    "ResolvedEquityTransformInputs",
    "VanillaEquityTransformResult",
    "price_vanilla_equity_option_transform",
    "price_vanilla_equity_option_transform_result",
    "resolve_vanilla_equity_transform_inputs",
]
