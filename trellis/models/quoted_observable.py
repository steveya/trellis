"""Checked helpers for bounded quoted-observable spread products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Protocol

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention


class CurveQuoteSpreadSpecLike(Protocol):
    notional: float
    curve_id: str
    lhs_coordinate: object
    rhs_coordinate: object
    convention: str
    expiry_date: date
    day_count: DayCountConvention


class SurfaceQuoteSpreadSpecLike(Protocol):
    notional: float
    surface_id: str
    lhs_coordinate: object
    rhs_coordinate: object
    convention: str
    expiry_date: date
    day_count: DayCountConvention


@dataclass(frozen=True)
class QuotedObservableSpreadResult:
    price: float
    lhs_quote: float
    rhs_quote: float
    discount_factor: float


def price_curve_quote_spread_analytical_result(
    market_state: MarketState,
    spec: CurveQuoteSpreadSpecLike,
) -> QuotedObservableSpreadResult:
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Curve-quote spread helper requires market_state.settlement or as_of")
    if market_state.discount is None:
        raise ValueError("Curve-quote spread helper requires market_state.discount")
    curve = _resolve_curve_source(market_state, str(getattr(spec, "curve_id", "") or ""))
    expiry_date = getattr(spec, "expiry_date")
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, expiry_date, day_count)), 0.0)
    lhs_quote = _resolve_curve_quote(curve, getattr(spec, "lhs_coordinate"), getattr(spec, "convention", ""))
    rhs_quote = _resolve_curve_quote(curve, getattr(spec, "rhs_coordinate"), getattr(spec, "convention", ""))
    discount_factor = 1.0 if maturity <= 0.0 else float(market_state.discount.discount(maturity))
    price = float(getattr(spec, "notional", 1.0)) * discount_factor * (lhs_quote - rhs_quote)
    return QuotedObservableSpreadResult(
        price=price,
        lhs_quote=lhs_quote,
        rhs_quote=rhs_quote,
        discount_factor=discount_factor,
    )


def price_curve_quote_spread_analytical(
    market_state: MarketState,
    spec: CurveQuoteSpreadSpecLike,
) -> float:
    return float(price_curve_quote_spread_analytical_result(market_state, spec).price)


def price_surface_quote_spread_analytical_result(
    market_state: MarketState,
    spec: SurfaceQuoteSpreadSpecLike,
) -> QuotedObservableSpreadResult:
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Surface-quote spread helper requires market_state.settlement or as_of")
    if market_state.discount is None:
        raise ValueError("Surface-quote spread helper requires market_state.discount")
    surface = _resolve_surface_source(market_state, str(getattr(spec, "surface_id", "") or ""))
    expiry_date = getattr(spec, "expiry_date")
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, expiry_date, day_count)), 0.0)
    lhs_quote = _resolve_surface_quote(
        surface,
        market_state,
        getattr(spec, "lhs_coordinate"),
        getattr(spec, "convention", ""),
    )
    rhs_quote = _resolve_surface_quote(
        surface,
        market_state,
        getattr(spec, "rhs_coordinate"),
        getattr(spec, "convention", ""),
    )
    discount_factor = 1.0 if maturity <= 0.0 else float(market_state.discount.discount(maturity))
    price = float(getattr(spec, "notional", 1.0)) * discount_factor * (lhs_quote - rhs_quote)
    return QuotedObservableSpreadResult(
        price=price,
        lhs_quote=lhs_quote,
        rhs_quote=rhs_quote,
        discount_factor=discount_factor,
    )


def price_surface_quote_spread_analytical(
    market_state: MarketState,
    spec: SurfaceQuoteSpreadSpecLike,
) -> float:
    return float(price_surface_quote_spread_analytical_result(market_state, spec).price)


def _resolve_curve_source(market_state: MarketState, curve_id: str):
    forecast_curves = dict(getattr(market_state, "forecast_curves", None) or {})
    if curve_id and curve_id in forecast_curves:
        return forecast_curves[curve_id]
    selected_discount_name = str(market_state.selected_curve_name("discount_curve") or "")
    if (
        curve_id
        and selected_discount_name
        and curve_id == selected_discount_name
        and market_state.discount is not None
    ):
        return market_state.discount
    if curve_id:
        raise ValueError(f"Curve-quote spread helper could not resolve curve {curve_id!r}")
    if market_state.discount is not None:
        return market_state.discount
    raise ValueError(f"Curve-quote spread helper could not resolve curve {curve_id!r}")


def _resolve_curve_quote(curve, coordinate: object, convention: str) -> float:
    normalized_convention = str(convention or "").strip().lower()
    if hasattr(coordinate, "tenor"):
        tenor_years = _tenor_to_years(str(getattr(coordinate, "tenor")))
        if normalized_convention == "par_rate":
            return _par_rate_from_discount_curve(curve, tenor_years)
        if normalized_convention == "zero_rate":
            return float(curve.zero_rate(tenor_years))
        raise ValueError(f"Unsupported curve quote convention {convention!r} for tenor coordinate")
    if hasattr(coordinate, "start_tenor") and hasattr(coordinate, "end_tenor"):
        if normalized_convention not in {"forward_rate", "forward"}:
            raise ValueError(
                f"Unsupported curve quote convention {convention!r} for forward-rate interval"
            )
        start = _tenor_to_years(str(getattr(coordinate, "start_tenor")))
        end = _tenor_to_years(str(getattr(coordinate, "end_tenor")))
        if end <= start:
            raise ValueError("Forward-rate interval requires end_tenor > start_tenor")
        return _forward_rate_from_discount_curve(curve, start, end)
    raise ValueError("Curve-quote spread helper requires a tenor or forward-rate interval coordinate")


def _resolve_surface_source(market_state: MarketState, surface_id: str):
    if market_state.vol_surface is None:
        raise ValueError("Surface-quote spread helper requires market_state.vol_surface")
    if not surface_id:
        return market_state.vol_surface
    selected_names = {
        str(name)
        for name in (
            market_state.selected_curve_name("vol_surface"),
            market_state.selected_curve_name("black_vol_surface"),
        )
        if name
    }
    if surface_id in selected_names:
        return market_state.vol_surface
    raise ValueError(f"Surface-quote spread helper could not resolve surface {surface_id!r}")


def _resolve_surface_quote(surface, market_state: MarketState, coordinate: object, convention: str) -> float:
    normalized_convention = str(convention or "").strip().lower()
    if normalized_convention != "black_vol":
        raise ValueError(
            f"Surface-quote spread helper supports only 'black_vol' convention, got {convention!r}"
        )
    if hasattr(coordinate, "delta") or hasattr(coordinate, "delta_style"):
        raise ValueError("Surface-quote spread helper currently supports only VolPoint coordinates")
    if not (
        hasattr(coordinate, "option_tenor")
        and hasattr(coordinate, "strike")
        and hasattr(coordinate, "strike_style")
    ):
        raise ValueError("Surface-quote spread helper requires a VolPoint-style coordinate")
    expiry = _tenor_to_years(str(getattr(coordinate, "option_tenor")))
    strike_style = str(getattr(coordinate, "strike_style") or "").strip().lower()
    raw_strike = float(getattr(coordinate, "strike"))
    if strike_style == "moneyness":
        spot = float(getattr(market_state, "spot", None) or 0.0)
        if spot <= 0.0:
            raise ValueError("Surface-quote spread helper requires market_state.spot for moneyness points")
        strike = raw_strike * spot
    elif strike_style in {"strike", "absolute"}:
        strike = raw_strike
    else:
        raise ValueError(
            f"Surface-quote spread helper supports only 'moneyness' or 'strike', got {strike_style!r}"
        )
    return float(surface.black_vol(max(expiry, 1e-6), strike))


def _par_rate_from_discount_curve(curve, tenor_years: float) -> float:
    if tenor_years <= 0.0:
        return 0.0
    payment_times = _annual_payment_times(tenor_years)
    accruals = _payment_accruals(payment_times)
    annuity = sum(
        accrual * float(curve.discount(payment_time))
        for accrual, payment_time in zip(accruals, payment_times)
    )
    if annuity <= 0.0:
        raise ValueError("Curve-quote spread helper requires positive fixed-leg annuity")
    maturity_df = float(curve.discount(tenor_years))
    return (1.0 - maturity_df) / annuity


def _forward_rate_from_discount_curve(curve, start: float, end: float) -> float:
    start_df = float(curve.discount(start))
    end_df = float(curve.discount(end))
    accrual = end - start
    if accrual <= 0.0:
        raise ValueError("Forward-rate interval requires positive accrual")
    return math.log(start_df / max(end_df, 1e-300)) / accrual


def _annual_payment_times(tenor_years: float) -> tuple[float, ...]:
    whole_years = int(math.floor(tenor_years))
    times = [float(index) for index in range(1, whole_years + 1)]
    if not times or abs(times[-1] - tenor_years) > 1e-12:
        times.append(float(tenor_years))
    return tuple(times)


def _payment_accruals(payment_times: tuple[float, ...]) -> tuple[float, ...]:
    accruals: list[float] = []
    previous = 0.0
    for payment_time in payment_times:
        accruals.append(float(payment_time - previous))
        previous = float(payment_time)
    return tuple(accruals)


def _tenor_to_years(token: str) -> float:
    text = str(token or "").strip().upper()
    if not text:
        raise ValueError("Quoted-observable helper requires a non-empty tenor token")
    unit = text[-1]
    value = float(text[:-1])
    if unit == "Y":
        return value
    if unit == "M":
        return value / 12.0
    if unit == "W":
        return value / 52.0
    if unit == "D":
        return value / 365.0
    raise ValueError(f"Unsupported tenor token {token!r}")


__all__ = [
    "CurveQuoteSpreadSpecLike",
    "QuotedObservableSpreadResult",
    "SurfaceQuoteSpreadSpecLike",
    "price_curve_quote_spread_analytical",
    "price_curve_quote_spread_analytical_result",
    "price_surface_quote_spread_analytical",
    "price_surface_quote_spread_analytical_result",
]
