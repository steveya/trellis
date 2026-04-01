"""Rates calibration helpers for cap/floor and swaption Black-vol fits.

These helpers solve the most common desk-style rates calibration problems in a
multi-curve setting:

* cap/floor quotes calibrated to a flat Black volatility
* European swaption quotes calibrated to a flat Black volatility

The helpers keep the curve-selection provenance from ``MarketState`` in the
result object so downstream traces and replay tools can explain which discount
and forecast curves were used in the calibration run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Callable, Literal, Protocol

from scipy.optimize import brentq

from trellis.core.date_utils import build_payment_timeline, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import FlatVol


class SwaptionLike(Protocol):
    """Protocol for the swaption-like inputs used by the calibration helpers."""

    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency
    day_count: DayCountConvention
    rate_index: str | None
    is_payer: bool


@dataclass(frozen=True)
class RatesCalibrationResult:
    """Structured result for a rates Black-vol calibration."""

    instrument_family: str
    instrument_kind: str
    target_price: float
    calibrated_vol: float
    repriced_price: float
    residual: float
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


def _build_provenance(
    market_state: MarketState,
    *,
    rate_index: str | None = None,
    vol_surface_name: str | None = None,
    correlation_source: str | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable provenance payload for calibration runs."""
    provenance: dict[str, object] = {
        "selected_curve_names": dict(market_state.selected_curve_names or {}),
    }
    market_provenance = getattr(market_state, "market_provenance", None)
    if market_provenance:
        provenance["market_provenance"] = dict(market_provenance)
    if rate_index is not None:
        provenance["rate_index"] = rate_index
    if vol_surface_name is not None:
        provenance["vol_surface_name"] = vol_surface_name
    if correlation_source is not None:
        provenance["correlation_source"] = correlation_source
    return provenance


def _implied_flat_vol(
    price_fn: Callable[[float], float],
    target_price: float,
    *,
    lower: float = 0.0,
    upper: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Solve for the flat Black volatility that reproduces ``target_price``."""
    if lower < 0.0:
        raise ValueError("lower volatility bound must be non-negative")
    if upper <= lower:
        raise ValueError("upper volatility bound must be greater than lower")

    target_price = float(target_price)
    low_price = float(price_fn(lower))
    low_residual = low_price - target_price
    if abs(low_residual) <= tol:
        return float(lower)

    high_price = float(price_fn(upper))
    high_residual = high_price - target_price
    if abs(high_residual) <= tol:
        return float(upper)

    if low_residual * high_residual > 0:
        raise ValueError(
            "target price is not bracketed by the supplied volatility range; "
            f"price(lower={lower})={low_price:.10g}, price(upper={upper})={high_price:.10g}, "
            f"target={target_price:.10g}"
        )

    return float(
        brentq(lambda vol: float(price_fn(float(vol))) - target_price, lower, upper, xtol=tol)
    )


def _cap_floor_summary(spec: CapFloorSpec, market_state: MarketState) -> dict[str, object]:
    """Return a compact summary of the cap/floor calibration inputs."""
    timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=market_state.settlement,
        label="cap_floor_calibration_timeline",
    )
    first_fix = None
    first_pay = None
    last_pay = None
    if timeline:
        first_fix = timeline[0].t_start
        first_pay = timeline[0].t_payment
        last_pay = timeline[-1].t_payment
    return {
        "period_count": len(timeline),
        "frequency": spec.frequency.name,
        "day_count": getattr(spec.day_count, "name", str(spec.day_count)),
        "rate_index": spec.rate_index,
        "first_fix_years": float(first_fix) if first_fix is not None else None,
        "first_pay_years": float(first_pay) if first_pay is not None else None,
        "last_pay_years": float(last_pay) if last_pay is not None else None,
    }


def swaption_terms(
    spec: SwaptionLike,
    market_state: MarketState,
) -> tuple[float, float, float, int]:
    """Return expiry, annuity, forward swap rate, and payment count."""
    timeline = build_payment_timeline(
        spec.swap_start,
        spec.swap_end,
        spec.swap_frequency,
        day_count=spec.day_count,
        time_origin=market_state.settlement,
        label="swaption_underlier_timeline",
    )
    if not timeline:
        return 0.0, 0.0, 0.0, 0

    fwd_curve = market_state.forecast_forward_curve(spec.rate_index)
    annuity = 0.0
    float_pv = 0.0
    payment_count = 0

    for period in timeline:
        if period.end_date <= market_state.settlement:
            continue
        tau = float(period.accrual_fraction or 0.0)
        t_start = float(period.t_start or 0.0)
        t_end = float(period.t_end or 0.0)
        t_start = max(t_start, 1e-6)
        df = float(market_state.discount.discount(t_end))
        fwd = float(fwd_curve.forward_rate(t_start, t_end))
        annuity += tau * df
        float_pv += fwd * tau * df
        payment_count += 1

    expiry = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    swap_rate = float_pv / annuity if annuity > 0.0 else 0.0
    return float(expiry), float(annuity), float(swap_rate), payment_count


def _swaption_black76_price(
    spec: SwaptionLike,
    market_state: MarketState,
    vol: float,
) -> tuple[float, dict[str, object]]:
    """Return the Black76 swaption PV and a summary of the assembled terms."""
    T, annuity, swap_rate, payment_count = swaption_terms(spec, market_state)
    if T <= 0.0 or annuity <= 0.0:
        return 0.0, {
            "expiry_years": float(T),
            "annuity": float(annuity),
            "forward_swap_rate": float(swap_rate),
            "payment_count": payment_count,
            "option_type": "payer" if spec.is_payer else "receiver",
        }

    if spec.is_payer:
        option_value = black76_call(swap_rate, spec.strike, vol, T)
    else:
        option_value = black76_put(swap_rate, spec.strike, vol, T)

    pv = spec.notional * annuity * float(option_value)
    return float(pv), {
        "expiry_years": float(T),
        "annuity": float(annuity),
        "forward_swap_rate": float(swap_rate),
        "payment_count": payment_count,
        "option_type": "payer" if spec.is_payer else "receiver",
    }


def calibrate_cap_floor_black_vol(
    spec: CapFloorSpec,
    market_state: MarketState,
    target_price: float,
    *,
    kind: Literal["cap", "floor"] = "cap",
    vol_lower: float = 0.0,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    vol_surface_name: str | None = None,
    correlation_source: str | None = None,
) -> RatesCalibrationResult:
    """Calibrate a flat Black volatility to a cap or floor price quote.

    The helper preserves the selected curve provenance from ``market_state``
    and adds optional source labels for the volatility and correlation inputs.
    """
    if kind not in {"cap", "floor"}:
        raise ValueError(f"kind must be 'cap' or 'floor', got {kind!r}")

    payoff = CapPayoff(spec) if kind == "cap" else FloorPayoff(spec)

    def price_at(vol: float) -> float:
        scenario = replace(market_state, vol_surface=FlatVol(float(vol)))
        return float(payoff.evaluate(scenario))

    calibrated_vol = _implied_flat_vol(
        price_at,
        target_price,
        lower=vol_lower,
        upper=vol_upper,
        tol=tol,
    )
    repriced_price = price_at(calibrated_vol)
    provenance = _build_provenance(
        market_state,
        rate_index=spec.rate_index,
        vol_surface_name=vol_surface_name,
        correlation_source=correlation_source,
    )
    return RatesCalibrationResult(
        instrument_family="cap_floor",
        instrument_kind=kind,
        target_price=float(target_price),
        calibrated_vol=float(calibrated_vol),
        repriced_price=float(repriced_price),
        residual=float(repriced_price - target_price),
        provenance=provenance,
        summary=_cap_floor_summary(spec, market_state),
    )


def calibrate_swaption_black_vol(
    spec: SwaptionLike,
    market_state: MarketState,
    target_price: float,
    *,
    vol_lower: float = 0.0,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    vol_surface_name: str | None = None,
    correlation_source: str | None = None,
) -> RatesCalibrationResult:
    """Calibrate a flat Black volatility to a European swaption price quote."""

    def price_at(vol: float) -> float:
        pv, _summary = _swaption_black76_price(spec, market_state, float(vol))
        return pv

    calibrated_vol = _implied_flat_vol(
        price_at,
        target_price,
        lower=vol_lower,
        upper=vol_upper,
        tol=tol,
    )
    repriced_price, summary = _swaption_black76_price(spec, market_state, calibrated_vol)
    provenance = _build_provenance(
        market_state,
        rate_index=spec.rate_index,
        vol_surface_name=vol_surface_name,
        correlation_source=correlation_source,
    )
    summary["rate_index"] = spec.rate_index
    summary["strike"] = float(spec.strike)
    summary["notional"] = float(spec.notional)
    return RatesCalibrationResult(
        instrument_family="swaption",
        instrument_kind="payer" if spec.is_payer else "receiver",
        target_price=float(target_price),
        calibrated_vol=float(calibrated_vol),
        repriced_price=float(repriced_price),
        residual=float(repriced_price - target_price),
        provenance=provenance,
        summary=summary,
    )


__all__ = [
    "RatesCalibrationResult",
    "SwaptionLike",
    "calibrate_cap_floor_black_vol",
    "calibrate_swaption_black_vol",
    "swaption_terms",
]
