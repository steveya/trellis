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

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.jamshidian import (
    ResolvedJamshidianInputs,
    zcb_option_hw_raw,
)


class DiscountCurveLike(Protocol):
    """Discount interface required by the ZCB option helpers."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the ZCB option helpers."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class ZCBOptionMarketStateLike(Protocol):
    """Market-state interface required by the ZCB option helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None


class ZCBOptionSpecLike(Protocol):
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
    """Normalize strike quotes to unit-face form.

    T01-style task text and generated specs sometimes use ``63`` as the strike
    on ``100`` face rather than ``0.63`` per unit face. Treat those as
    equivalent when the notional/face is available.
    """
    strike = float(strike_quote)
    face = abs(float(notional))
    if abs(strike) > 1.0 and face > 1.0:
        strike /= face
    return strike


def resolve_zcb_option_hw_inputs(
    market_state: ZCBOptionMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    mean_reversion: float = 0.1,
) -> ResolvedZCBOptionInputs:
    """Resolve dates, strike units, vol, and discount factors for Jamshidian."""
    settlement = _settlement_date(market_state, spec)
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("ZCB option pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("ZCB option pricing requires market_state.vol_surface")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    t_exp = year_fraction(settlement, spec.expiry_date, day_count)
    t_bond = year_fraction(settlement, spec.bond_maturity_date, day_count)
    if t_exp <= 0.0:
        return ResolvedZCBOptionInputs(
            notional=float(spec.notional),
            option_type=_resolve_option_type(spec),
            jamshidian=ResolvedJamshidianInputs(
                discount_factor_expiry=1.0,
                discount_factor_bond=float(discount_curve.discount(max(t_bond, 0.0))),
                strike=normalize_zcb_option_strike(spec.strike, spec.notional),
                T_exp=max(t_exp, 0.0),
                T_bond=max(t_bond, 0.0),
                sigma=0.0,
                a=float(mean_reversion),
            ),
        )
    if t_bond <= t_exp:
        raise ValueError("bond_maturity_date must be after expiry_date for ZCB options")

    strike_unit = normalize_zcb_option_strike(spec.strike, spec.notional)
    sigma = float(market_state.vol_surface.black_vol(max(t_exp, 1e-6), strike_unit))

    return ResolvedZCBOptionInputs(
        notional=float(spec.notional),
        option_type=_resolve_option_type(spec),
        jamshidian=ResolvedJamshidianInputs(
            discount_factor_expiry=float(discount_curve.discount(t_exp)),
            discount_factor_bond=float(discount_curve.discount(t_bond)),
            strike=strike_unit,
            T_exp=float(t_exp),
            T_bond=float(t_bond),
            sigma=sigma,
            a=float(mean_reversion),
        ),
    )


def price_zcb_option_jamshidian(
    market_state: ZCBOptionMarketStateLike,
    spec: ZCBOptionSpecLike,
    *,
    mean_reversion: float = 0.1,
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
    """Resolve call/put semantics from modern or legacy spec fields."""
    option_type = getattr(spec, "option_type", None)
    if option_type is not None:
        normalized = str(option_type).strip().lower()
        if normalized in {"call", "put"}:
            return normalized
        if normalized == "payer":
            return "put"
        if normalized == "receiver":
            return "call"
    if hasattr(spec, "is_call"):
        return "call" if bool(spec.is_call) else "put"
    if hasattr(spec, "is_payer"):
        # Historical compatibility: earlier task scaffolds reused payer/receiver
        # wording for bond options and mapped payer -> put on the bond forward.
        return "put" if bool(spec.is_payer) else "call"
    return "call"


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
