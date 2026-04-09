"""Reusable transform helpers for single-state diffusion claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price
from trellis.models.resolution.single_state_diffusion import (
    ResolvedSingleStateDiffusionInputs,
    SingleStateDiffusionMarketStateLike,
    SingleStateDiffusionSpecLike,
    resolve_single_state_diffusion_inputs,
)


@dataclass(frozen=True)
class ResolvedSingleStateTransformInputs(ResolvedSingleStateDiffusionInputs):
    """Resolved market inputs and transform controls."""

    method: str
    fft_alpha: float
    fft_points: int
    fft_eta: float
    cos_points: int
    cos_truncation: float


@dataclass(frozen=True)
class SingleStateTransformResult:
    """Structured transform-pricing result."""

    price: float
    method: str
    sigma: float
    maturity: float


def resolve_single_state_terminal_claim_transform_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> ResolvedSingleStateTransformInputs:
    """Resolve transform controls and market inputs for one single-state claim."""
    resolved_base = resolve_single_state_diffusion_inputs(market_state, spec)
    resolved_method = _normalized_transform_method(
        method if method is not None else getattr(spec, "transform_method", "fft")
    )

    return ResolvedSingleStateTransformInputs(
        **resolved_base.__dict__,
        method=resolved_method,
        fft_alpha=float(fft_alpha if fft_alpha is not None else getattr(spec, "fft_alpha", 1.5)),
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


def price_single_state_terminal_claim_transform_result(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
    intrinsic_fn: Callable[[float, ResolvedSingleStateTransformInputs], float],
    fft_log_spot_char_fn: Callable[[ResolvedSingleStateTransformInputs], Callable[[complex], complex]],
    cos_log_ratio_char_fn: Callable[[ResolvedSingleStateTransformInputs], Callable[[complex], complex]],
    put_from_call_parity_fn: Callable[[float, ResolvedSingleStateTransformInputs], float],
) -> SingleStateTransformResult:
    """Price one bounded single-state terminal claim through the transform family helper."""
    resolved = resolve_single_state_terminal_claim_transform_inputs(
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
        return SingleStateTransformResult(
            price=float(resolved.notional) * float(intrinsic_fn(resolved.spot, resolved)),
            method=resolved.method,
            sigma=resolved.sigma,
            maturity=resolved.maturity,
        )

    if resolved.method == "fft":
        call_price = fft_price(
            fft_log_spot_char_fn(resolved),
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
            else put_from_call_parity_fn(call_price, resolved)
        )
    else:
        raw_price = cos_price(
            cos_log_ratio_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            N=resolved.cos_points,
            L=resolved.cos_truncation,
            option_type=resolved.option_type,
        )

    return SingleStateTransformResult(
        price=float(resolved.notional) * float(raw_price),
        method=resolved.method,
        sigma=resolved.sigma,
        maturity=resolved.maturity,
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


__all__ = [
    "ResolvedSingleStateTransformInputs",
    "SingleStateTransformResult",
    "price_single_state_terminal_claim_transform_result",
    "resolve_single_state_terminal_claim_transform_inputs",
]
