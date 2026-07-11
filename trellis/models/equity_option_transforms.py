"""Semantic-facing vanilla-equity transform helpers.

This module is a thin compatibility wrapper over the reusable single-state
diffusion transform family helper. Generated adapters should bind transform
pricing here instead of rebuilding characteristic functions, market
conventions, or put/call parity glue inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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


@dataclass(frozen=True)
class DigitalEquityTransformResult:
    """Structured transform result for a digital equity option."""

    price: float
    method: str
    bump_size: float
    payout_type: str


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


def price_equity_digital_option_transform_result(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    strike_bump: float | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> DigitalEquityTransformResult:
    """Price a cash-or-asset digital option from FFT/COS vanilla prices.

    The helper keeps digital transform targets on a checked transform surface
    without duplicating payoff-coefficient code in generated adapters. Cash
    digitals use the strike derivative of the transform-priced vanilla claim;
    asset digitals are recovered from the vanilla/digital parity identities.
    """
    option_type = str(getattr(spec, "option_type", "call") or "call").strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unsupported option_type {option_type!r}")
    payout_type = str(
        getattr(spec, "payout_type", "cash_or_nothing") or "cash_or_nothing"
    ).strip().lower()
    if payout_type not in {"cash_or_nothing", "asset_or_nothing"}:
        raise ValueError(f"Unsupported payout_type {payout_type!r}")

    strike = float(getattr(spec, "strike"))
    if strike <= 0.0:
        raise ValueError("Digital transform pricing requires a positive strike")
    bump = float(strike_bump if strike_bump is not None else max(1e-3, 1e-3 * strike))
    lower_strike = max(strike - bump, 1e-8)
    upper_strike = strike + bump
    actual_bump = upper_strike - lower_strike

    base_kwargs = dict(
        method=method,
        fft_alpha=fft_alpha,
        fft_points=fft_points,
        fft_eta=fft_eta,
        cos_points=cos_points,
        cos_truncation=cos_truncation,
    )
    lower_spec = _unit_notional_spec(spec, strike=lower_strike, option_type=option_type)
    upper_spec = _unit_notional_spec(spec, strike=upper_strike, option_type=option_type)
    at_spec = _unit_notional_spec(spec, strike=strike, option_type=option_type)

    lower = price_vanilla_equity_option_transform_result(market_state, lower_spec, **base_kwargs)
    upper = price_vanilla_equity_option_transform_result(market_state, upper_spec, **base_kwargs)
    derivative = (upper.price - lower.price) / actual_bump
    if option_type == "call":
        cash_unit = max(-float(derivative), 0.0)
    else:
        cash_unit = max(float(derivative), 0.0)

    cash_payoff = float(getattr(spec, "cash_payoff", 1.0) or 1.0)
    if payout_type == "cash_or_nothing":
        unit_price = cash_payoff * cash_unit
    else:
        vanilla = price_vanilla_equity_option_transform_result(market_state, at_spec, **base_kwargs).price
        if option_type == "call":
            unit_price = vanilla + strike * cash_unit
        else:
            unit_price = strike * cash_unit - vanilla
        unit_price = max(float(unit_price), 0.0)

    return DigitalEquityTransformResult(
        price=float(getattr(spec, "notional", 1.0) or 1.0) * float(unit_price),
        method=lower.method,
        bump_size=actual_bump,
        payout_type=payout_type,
    )


def price_equity_digital_option_transform(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    method: str | None = None,
    strike_bump: float | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> float:
    """Return a scalar FFT/COS transform price for a digital equity option."""
    return float(
        price_equity_digital_option_transform_result(
            market_state,
            spec,
            method=method,
            strike_bump=strike_bump,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
        ).price
    )


def _unit_notional_spec(spec: object, **overrides: object) -> SimpleNamespace:
    fields = {
        "spot",
        "strike",
        "expiry_date",
        "option_type",
        "day_count",
        "time_day_count",
        "dividend_yield",
        "dividend_rate",
        "transform_method",
        "fft_alpha",
        "fft_points",
        "fft_eta",
        "cos_points",
        "cos_truncation",
    }
    values = {name: getattr(spec, name) for name in fields if hasattr(spec, name)}
    values["notional"] = 1.0
    values.update(overrides)
    return SimpleNamespace(**values)


__all__ = [
    "DigitalEquityTransformResult",
    "ResolvedEquityTransformInputs",
    "VanillaEquityTransformResult",
    "price_equity_digital_option_transform",
    "price_equity_digital_option_transform_result",
    "price_vanilla_equity_option_transform",
    "price_vanilla_equity_option_transform_result",
    "resolve_vanilla_equity_transform_inputs",
]
