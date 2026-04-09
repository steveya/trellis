"""Semantic-facing vanilla-equity transform helpers.

This module wraps the raw FFT and COS kernels with a stable helper surface that
starts from market state and a plain European equity-option spec. Generated
adapters should bind transform pricing here instead of rebuilding characteristic
functions, market conventions, or put/call parity glue inline.
"""

from __future__ import annotations

from dataclasses import dataclass

from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price
from trellis.models.resolution.single_state_diffusion import (
    ResolvedSingleStateDiffusionInputs,
    SingleStateDiffusionMarketStateLike,
    SingleStateDiffusionSpecLike,
    gbm_log_ratio_char_fn,
    gbm_log_spot_char_fn,
    put_from_call_parity,
    resolve_single_state_diffusion_inputs,
    terminal_intrinsic_from_resolved,
)


@dataclass(frozen=True)
class ResolvedEquityTransformInputs(ResolvedSingleStateDiffusionInputs):
    """Resolved market inputs and transform controls."""
    method: str
    fft_alpha: float
    fft_points: int
    fft_eta: float
    cos_points: int
    cos_truncation: float


@dataclass(frozen=True)
class VanillaEquityTransformResult:
    """Structured transform-pricing result."""

    price: float
    method: str
    sigma: float
    maturity: float


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
    resolved_base = resolve_single_state_diffusion_inputs(market_state, spec)
    resolved_method = _normalized_transform_method(
        method if method is not None else getattr(spec, "transform_method", "fft")
    )

    return ResolvedEquityTransformInputs(
        **resolved_base.__dict__,
        method=resolved_method,
        fft_alpha=float(
            fft_alpha if fft_alpha is not None else getattr(spec, "fft_alpha", 1.5)
        ),
        fft_points=max(
            int(fft_points if fft_points is not None else getattr(spec, "fft_points", 4096)),
            32,
        ),
        fft_eta=float(fft_eta if fft_eta is not None else getattr(spec, "fft_eta", 0.25)),
        cos_points=max(
            int(cos_points if cos_points is not None else getattr(spec, "cos_points", 256)),
            16,
        ),
        cos_truncation=float(
            cos_truncation
            if cos_truncation is not None
            else getattr(spec, "cos_truncation", 10.0)
        ),
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
    resolved = resolve_vanilla_equity_transform_inputs(
        market_state,
        spec,
        method=method,
        fft_alpha=fft_alpha,
        fft_points=fft_points,
        fft_eta=fft_eta,
        cos_points=cos_points,
        cos_truncation=cos_truncation,
    )

    if resolved.maturity <= 0.0:
        intrinsic = terminal_intrinsic_from_resolved(resolved.spot, resolved)
        return VanillaEquityTransformResult(
            price=float(resolved.notional) * float(intrinsic),
            method=resolved.method,
            sigma=resolved.sigma,
            maturity=resolved.maturity,
        )

    if resolved.method == "fft":
        call_price = fft_price(
            gbm_log_spot_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            alpha=resolved.fft_alpha,
            N=resolved.fft_points,
            eta=resolved.fft_eta,
        )
        raw_price = (
            call_price
            if resolved.option_type == "call"
            else put_from_call_parity(call_price, resolved)
        )
    else:
        raw_price = cos_price(
            gbm_log_ratio_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            N=resolved.cos_points,
            L=resolved.cos_truncation,
            option_type=resolved.option_type,
        )

    return VanillaEquityTransformResult(
        price=float(resolved.notional) * float(raw_price),
        method=resolved.method,
        sigma=resolved.sigma,
        maturity=resolved.maturity,
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

def _normalized_transform_method(value: object) -> str:
    method = str(value or "fft").strip().lower().replace("-", "_")
    aliases = {
        "carr_madan": "fft",
        "fang_oosterlee": "cos",
    }
    method = aliases.get(method, method)
    if method not in {"fft", "cos"}:
        raise ValueError(f"Unsupported transform method {value!r}")
    return method
