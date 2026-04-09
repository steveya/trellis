"""Stable zero-coupon-bond option helpers.

This module gives the DSL a checked-in helper surface for European options on
zero-coupon bonds under the Hull-White / Jamshidian contract. Generated
adapters should bind dates, strike units, and market-state access here instead
of rebuilding those semantics inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.models.analytical.jamshidian import (
    ResolvedJamshidianInputs,
    zcb_option_hw_raw,
)
from trellis.models.resolution.short_rate_claims import (
    DiscountBondClaimSpecLike,
    ResolvedDiscountBondClaim,
    ShortRateClaimMarketStateLike,
    normalize_discount_bond_strike,
    resolve_discount_bond_claim_inputs,
    resolve_discount_bond_option_type,
)


class VolSurfaceLike(Protocol):
    """Volatility interface required by the ZCB option helpers."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class ZCBOptionMarketStateLike(ShortRateClaimMarketStateLike, Protocol):
    """Market-state interface required by the ZCB option helpers."""

    as_of: date | None
    settlement: date | None
    vol_surface: VolSurfaceLike | None


class ZCBOptionSpecLike(DiscountBondClaimSpecLike, Protocol):
    """Spec fields consumed by the ZCB option helpers."""

    notional: float
    strike: float
    expiry_date: date
    bond_maturity_date: date


@dataclass(frozen=True)
class ResolvedZCBOptionInputs:
    """Resolved market inputs for a Jamshidian ZCB option."""

    notional: float
    option_type: str
    jamshidian: ResolvedJamshidianInputs


def normalize_zcb_option_strike(strike_quote: float, notional: float) -> float:
    """Backward-compatible alias for the shared discount-bond strike normalizer."""
    return normalize_discount_bond_strike(strike_quote, notional)


def _build_resolved_jamshidian_inputs(
    claim: ResolvedDiscountBondClaim,
) -> ResolvedZCBOptionInputs:
    """Project shared short-rate claim inputs onto Jamshidian kernel inputs."""
    if claim.expiry_time <= 0.0:
        return ResolvedZCBOptionInputs(
            notional=claim.notional,
            option_type=claim.option_type,
            jamshidian=ResolvedJamshidianInputs(
                discount_factor_expiry=1.0,
                discount_factor_bond=claim.discount_factor_bond,
                strike=claim.strike_unit,
                T_exp=0.0,
                T_bond=max(claim.bond_maturity_time, 0.0),
                sigma=0.0,
                a=float(claim.regime.mean_reversion),
            ),
        )
    return ResolvedZCBOptionInputs(
        notional=claim.notional,
        option_type=claim.option_type,
        jamshidian=ResolvedJamshidianInputs(
            discount_factor_expiry=claim.discount_factor_expiry,
            discount_factor_bond=claim.discount_factor_bond,
            strike=claim.strike_unit,
            T_exp=claim.expiry_time,
            T_bond=claim.bond_maturity_time,
            sigma=float(claim.regime.sigma),
            a=float(claim.regime.mean_reversion),
        ),
    )


def resolve_zcb_option_hw_inputs(
    market_state: ZCBOptionMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    mean_reversion: float | None = None,
) -> ResolvedZCBOptionInputs:
    """Resolve dates, strike units, vol, and discount factors for Jamshidian."""
    claim = resolve_discount_bond_claim_inputs(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=mean_reversion,
        default_mean_reversion=0.1,
    )
    return _build_resolved_jamshidian_inputs(claim)


def price_zcb_option_jamshidian(
    market_state: ZCBOptionMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    mean_reversion: float | None = None,
) -> float:
    """Price a European zero-coupon-bond option with Jamshidian/Hull-White."""
    resolved = resolve_zcb_option_hw_inputs(
        market_state,
        spec,
        mean_reversion=mean_reversion,
    )
    if resolved.jamshidian.T_exp <= 0.0:
        intrinsic = _terminal_intrinsic(
            resolved.option_type,
            bond_unit_price=resolved.jamshidian.discount_factor_bond,
            strike=resolved.jamshidian.strike,
        )
        return float(resolved.notional * intrinsic)

    raw = zcb_option_hw_raw(resolved.jamshidian)
    return float(resolved.notional * float(raw[resolved.option_type]))


def _resolve_option_type(spec) -> str:
    """Backward-compatible alias for the shared discount-bond option-type resolver."""
    return resolve_discount_bond_option_type(spec)


def _terminal_intrinsic(option_type: str, *, bond_unit_price: float, strike: float) -> float:
    if option_type == "put":
        return max(float(strike) - float(bond_unit_price), 0.0)
    return max(float(bond_unit_price) - float(strike), 0.0)


def _settlement_date(market_state, spec) -> date:
    settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
    if settlement is None:
        return spec.expiry_date
    return settlement


__all__ = [
    "ResolvedZCBOptionInputs",
    "normalize_zcb_option_strike",
    "resolve_zcb_option_hw_inputs",
    "price_zcb_option_jamshidian",
]
