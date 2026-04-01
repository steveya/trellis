"""Stable analytical helpers for rate-style swaptions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.calibration.rates import swaption_terms


class RateStyleSwaptionSpecLike(Protocol):
    """Protocol for rate-style swaption specs consumed by analytical helpers."""

    notional: float
    strike: float
    swap_end: date
    swap_frequency: Frequency
    day_count: DayCountConvention
    rate_index: str | None
    is_payer: bool


class EuropeanSwaptionSpecLike(RateStyleSwaptionSpecLike, Protocol):
    """Protocol for European swaption specs."""

    expiry_date: date
    swap_start: date


class BermudanSwaptionLowerBoundSpecLike(RateStyleSwaptionSpecLike, Protocol):
    """Protocol for Bermudan-style specs that expose exercise dates."""

    exercise_dates: str | Iterable[date | str]


@dataclass(frozen=True)
class ResolvedSwaptionBlack76Inputs:
    """Resolved market and contract terms for one European Black76 swaption."""

    expiry_date: date
    expiry_years: float
    annuity: float
    forward_swap_rate: float
    strike: float
    vol: float
    notional: float
    is_payer: bool
    payment_count: int


@dataclass(frozen=True)
class _EuropeanSwaptionView:
    """Internal European swaption view used by the shared term builder."""

    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency
    day_count: DayCountConvention
    rate_index: str | None
    is_payer: bool


def _normalized_exercise_dates(raw: str | Iterable[date | str]) -> tuple[date, ...]:
    """Return a sorted tuple of unique exercise dates."""
    if isinstance(raw, str):
        items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    else:
        items = list(raw)
    normalized: list[date] = []
    for item in items:
        if isinstance(item, date):
            normalized.append(item)
        else:
            normalized.append(date.fromisoformat(str(item).strip()))
    return tuple(sorted(dict.fromkeys(normalized)))


def _resolve_expiry_date(
    spec: RateStyleSwaptionSpecLike,
    *,
    expiry_date: date | None,
) -> date:
    """Resolve the European exercise date for the pricing helper."""
    if expiry_date is not None:
        return expiry_date
    spec_expiry = getattr(spec, "expiry_date", None)
    if isinstance(spec_expiry, date):
        return spec_expiry
    spec_swap_start = getattr(spec, "swap_start", None)
    if isinstance(spec_swap_start, date):
        return spec_swap_start
    exercise_dates = getattr(spec, "exercise_dates", None)
    if exercise_dates is not None:
        normalized = _normalized_exercise_dates(exercise_dates)
        if normalized:
            return normalized[0]
    raise ValueError("Rate-style swaption helper could not resolve an expiry date.")


def resolve_swaption_black76_inputs(
    market_state: MarketState,
    spec: RateStyleSwaptionSpecLike,
    *,
    expiry_date: date | None = None,
) -> ResolvedSwaptionBlack76Inputs:
    """Resolve one European swaption view onto Black76 inputs."""
    if market_state.discount is None:
        raise ValueError("Rate-style swaption Black76 pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Rate-style swaption Black76 pricing requires market_state.vol_surface")

    expiry = _resolve_expiry_date(spec, expiry_date=expiry_date)
    european_spec = _EuropeanSwaptionView(
        notional=float(spec.notional),
        strike=float(spec.strike),
        expiry_date=expiry,
        swap_start=expiry,
        swap_end=spec.swap_end,
        swap_frequency=spec.swap_frequency,
        day_count=spec.day_count,
        rate_index=spec.rate_index,
        is_payer=bool(spec.is_payer),
    )
    expiry_years, annuity, forward_swap_rate, payment_count = swaption_terms(
        european_spec,
        market_state,
    )
    vol = float(
        market_state.vol_surface.black_vol(
            max(float(expiry_years), 1e-8),
            max(abs(float(spec.strike)), 1e-8),
        )
    )
    return ResolvedSwaptionBlack76Inputs(
        expiry_date=expiry,
        expiry_years=float(expiry_years),
        annuity=float(annuity),
        forward_swap_rate=float(forward_swap_rate),
        strike=float(spec.strike),
        vol=vol,
        notional=float(spec.notional),
        is_payer=bool(spec.is_payer),
        payment_count=int(payment_count),
    )


def price_swaption_black76(
    market_state: MarketState,
    spec: EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike,
    *,
    expiry_date: date | None = None,
) -> float:
    """Price a single-exercise rate-style swaption with Black76."""
    resolved = resolve_swaption_black76_inputs(
        market_state,
        spec,
        expiry_date=expiry_date,
    )
    if resolved.expiry_years <= 0.0 or resolved.annuity <= 0.0 or resolved.payment_count <= 0:
        return 0.0
    option_value = (
        black76_call(
            resolved.forward_swap_rate,
            resolved.strike,
            resolved.vol,
            resolved.expiry_years,
        )
        if resolved.is_payer
        else black76_put(
            resolved.forward_swap_rate,
            resolved.strike,
            resolved.vol,
            resolved.expiry_years,
        )
    )
    return float(resolved.notional * resolved.annuity * float(option_value))


def price_bermudan_swaption_black76_lower_bound(
    market_state: MarketState,
    spec: BermudanSwaptionLowerBoundSpecLike,
) -> float:
    """Return the final-exercise European Black76 lower bound.

    The Bermudan task comparator is defined as the European swaption that may
    only exercise on the final Bermudan date. That keeps the comparison stable
    and guarantees it remains a lower bound to the Bermudan tree price in the
    checked-in T04 contract.
    """
    exercise_dates = tuple(
        exercise_date
        for exercise_date in _normalized_exercise_dates(spec.exercise_dates)
        if market_state.settlement < exercise_date < spec.swap_end
    )
    if not exercise_dates:
        return 0.0
    return float(
        price_swaption_black76(
            market_state,
            spec,
            expiry_date=exercise_dates[-1],
        )
    )


__all__ = [
    "BermudanSwaptionLowerBoundSpecLike",
    "EuropeanSwaptionSpecLike",
    "RateStyleSwaptionSpecLike",
    "ResolvedSwaptionBlack76Inputs",
    "price_bermudan_swaption_black76_lower_bound",
    "price_swaption_black76",
    "resolve_swaption_black76_inputs",
]
