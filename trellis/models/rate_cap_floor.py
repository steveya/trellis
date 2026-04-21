"""Exact analytical and Monte Carlo helpers for rate cap/floor strips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trellis.conventions.calendar import BusinessDayAdjustment, Calendar, WEEKEND_ONLY
from trellis.core.date_utils import build_payment_timeline
from trellis.conventions.day_count import DayCountConvention
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import Frequency
from trellis.instruments.cap import CapFloorSpec
from trellis.models.black import black76_call, black76_put
from trellis.models.processes.sabr import SABRProcess
from trellis.core.date_utils import year_fraction

np = get_numpy()


@dataclass(frozen=True)
class CapFloorPeriod:
    """One explicit period in a cap/floor strip schedule."""

    start_date: date
    end_date: date
    payment_date: date
    fixing_date: date | None = None


@dataclass(frozen=True)
class _CapFloorTerm:
    """Resolved one surviving cap/floor strip period."""

    fixing_years: float
    payment_years: float
    accrual_fraction: float
    discount_factor: float
    forward_rate: float
    volatility: float
    event_name: str
    option_expiry_years: float
    intrinsic_only: bool = False


def _normalize_instrument_class(instrument_class: str) -> Literal["cap", "floor"]:
    normalized = str(instrument_class or "").strip().lower()
    if normalized not in {"cap", "floor"}:
        raise ValueError(f"instrument_class must be 'cap' or 'floor', got {instrument_class!r}")
    return normalized  # type: ignore[return-value]


def _resolve_forward_curve(market_state: MarketState, spec: CapFloorSpec):
    forward_curve = None
    if spec.rate_index:
        try:
            forward_curve = market_state.forecast_forward_curve(spec.rate_index)
        except Exception:
            forward_curve = None
    if forward_curve is None:
        forward_curve = getattr(market_state, "forward_curve", None)
    if forward_curve is None:
        raise ValueError("Cap/floor strip pricing requires a forward curve")
    return forward_curve


def _discount_factor_for_period(market_state: MarketState, payment_date, payment_years: float) -> float:
    discount_curve = market_state.discount
    discount_date = getattr(discount_curve, "discount_date", None)
    if callable(discount_date):
        return float(discount_date(payment_date))
    return float(discount_curve.discount(payment_years))


def _forward_rate_for_period(forward_curve, start_date, payment_date, fixing_years: float, payment_years: float, day_count) -> float:
    forward_rate_dates = getattr(forward_curve, "forward_rate_dates", None)
    if callable(forward_rate_dates):
        try:
            return float(forward_rate_dates(start_date, payment_date, day_count=day_count))
        except AttributeError:
            pass
    return float(forward_curve.forward_rate(fixing_years, payment_years))


def _uses_date_aware_curve_conventions(market_state: MarketState, forward_curve) -> bool:
    discount_date = getattr(market_state.discount, "discount_date", None)
    if callable(discount_date):
        return True
    underlying_curve = getattr(forward_curve, "_curve", None)
    if underlying_curve is None:
        return False
    return callable(getattr(underlying_curve, "discount_date", None)) or callable(
        getattr(underlying_curve, "forward_rate_dates", None)
    )


def _option_expiry_years_for_observation_date(
    spec: CapFloorSpec,
    observation_date: date,
    *,
    fixing_years: float,
    use_date_aware: bool,
) -> float:
    if not use_date_aware:
        return max(fixing_years, 0.0)
    return max(year_fraction(spec.start_date, observation_date, DayCountConvention.ACT_365), 0.0)


def _cap_floor_model(spec: CapFloorSpec) -> str:
    return str(getattr(spec, "model", None) or "black").strip().lower() or "black"


def _cap_floor_option_inputs(
    market_state: MarketState,
    spec: CapFloorSpec,
    term: _CapFloorTerm,
) -> tuple[float, float, float]:
    model = _cap_floor_model(spec)
    if model == "shifted_black":
        params = dict(getattr(market_state, "model_parameters", None) or {})
        shift = float(getattr(spec, "shift", None) or params.get("shift") or 0.0)
        shifted_black_vol = params.get("shifted_black_vol")
        volatility = float(shifted_black_vol if shifted_black_vol is not None else term.volatility)
        return (
            max(term.forward_rate + shift, 1e-12),
            max(spec.strike + shift, 1e-12),
            volatility,
        )
    if model == "sabr":
        params = dict(getattr(market_state, "model_parameters", None) or {})
        sabr = dict(getattr(spec, "sabr", None) or params.get("sabr") or {})
        if sabr:
            process = SABRProcess(
                float(sabr["alpha"]),
                float(sabr["beta"]),
                float(sabr["rho"]),
                float(sabr["nu"]),
            )
            forward = max(term.forward_rate, 1e-12)
            strike = max(spec.strike, 1e-12)
            volatility = float(process.implied_vol(forward, strike, max(term.option_expiry_years, 1e-6)))
            return forward, strike, volatility
    return term.forward_rate, spec.strike, term.volatility


def _calendar_from_name(name: str | None) -> Calendar | None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return None
    if normalized == "weekend_only":
        return WEEKEND_ONLY
    raise ValueError(f"Unsupported calendar_name {name!r}")


def _business_day_adjustment_from_name(name: str | None) -> BusinessDayAdjustment:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return BusinessDayAdjustment.UNADJUSTED
    return BusinessDayAdjustment(normalized)


def _resolve_cap_floor_terms(
    market_state: MarketState,
    spec: CapFloorSpec,
    *,
    periods: tuple[CapFloorPeriod, ...] | None = None,
) -> tuple[_CapFloorTerm, ...]:
    if getattr(market_state, "discount", None) is None:
        raise ValueError("Cap/floor strip pricing requires market_state.discount")
    if getattr(market_state, "vol_surface", None) is None:
        raise ValueError("Cap/floor strip pricing requires market_state.vol_surface")

    forward_curve = _resolve_forward_curve(market_state, spec)
    use_date_aware = _uses_date_aware_curve_conventions(market_state, forward_curve)

    if periods is None:
        timeline = build_payment_timeline(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            calendar=_calendar_from_name(getattr(spec, "calendar_name", None)),
            bda=_business_day_adjustment_from_name(getattr(spec, "business_day_adjustment", None)),
            day_count=spec.day_count,
            time_origin=market_state.settlement,
            label="rate_cap_floor_strip",
        )
        normalized_periods = tuple(
            CapFloorPeriod(
                start_date=period.start_date,
                end_date=period.end_date,
                payment_date=period.payment_date,
                fixing_date=period.start_date,
            )
            for period in timeline
        )
    else:
        normalized_periods = tuple(periods)

    terms: list[_CapFloorTerm] = []
    for index, period in enumerate(normalized_periods):
        if period.payment_date <= market_state.settlement:
            continue
        fixing_anchor = period.fixing_date or period.start_date
        fixing_years = max(
            year_fraction(market_state.settlement, fixing_anchor, DayCountConvention.ACT_365),
            0.0,
        )
        accrual_fraction = float(year_fraction(period.start_date, period.end_date, spec.day_count))
        payment_years = max(
            year_fraction(market_state.settlement, period.payment_date, DayCountConvention.ACT_365),
            fixing_years,
        )
        end_years = max(
            year_fraction(market_state.settlement, period.end_date, DayCountConvention.ACT_365),
            max(fixing_years, 1e-6),
        )
        forward_rate = _forward_rate_for_period(
            forward_curve,
            period.start_date,
            period.end_date,
            fixing_years,
            end_years,
            spec.day_count,
        )
        discount_factor = _discount_factor_for_period(
            market_state,
            period.payment_date,
            payment_years,
        )
        intrinsic_only = fixing_years <= 0.0
        option_expiry_years = _option_expiry_years_for_observation_date(
            spec,
            fixing_anchor,
            fixing_years=fixing_years,
            use_date_aware=use_date_aware,
        )
        volatility = 0.0 if intrinsic_only else float(
            market_state.vol_surface.black_vol(max(option_expiry_years, 1e-6), spec.strike)
        )
        terms.append(
            _CapFloorTerm(
                fixing_years=max(fixing_years, 0.0),
                payment_years=payment_years,
                accrual_fraction=accrual_fraction,
                discount_factor=discount_factor,
                forward_rate=forward_rate,
                volatility=volatility,
                event_name=f"caplet_fixing_{index}",
                option_expiry_years=option_expiry_years,
                intrinsic_only=intrinsic_only,
            )
        )
    return tuple(terms)


def _coerce_cap_floor_spec(
    spec: CapFloorSpec | None,
    *,
    notional: float | None = None,
    strike: float | None = None,
    start_date=None,
    end_date=None,
    frequency=None,
    day_count=None,
    rate_index: str | None = None,
    calendar_name: str | None = None,
    business_day_adjustment: str | None = None,
    model: str | None = None,
    shift: float | None = None,
    sabr: dict[str, float] | None = None,
) -> CapFloorSpec:
    """Accept either a concrete spec object or expanded cap/floor keyword fields.

    Explicit keyword arguments always win. This matters for fresh-generated
    adapters, which often pass a lightweight ``AgentCapSpec`` plus explicit
    pricing-mode kwargs such as ``model="sabr"``.
    """
    resolved_notional = notional if notional is not None else getattr(spec, "notional", None)
    resolved_strike = strike if strike is not None else getattr(spec, "strike", None)
    resolved_start_date = start_date if start_date is not None else getattr(spec, "start_date", None)
    resolved_end_date = end_date if end_date is not None else getattr(spec, "end_date", None)
    resolved_frequency = frequency if frequency is not None else getattr(spec, "frequency", None)
    resolved_day_count = day_count if day_count is not None else getattr(spec, "day_count", None)
    resolved_rate_index = rate_index if rate_index is not None else getattr(spec, "rate_index", None)
    resolved_calendar_name = (
        calendar_name
        if calendar_name is not None
        else getattr(spec, "calendar_name", None)
    )
    resolved_business_day_adjustment = (
        business_day_adjustment
        if business_day_adjustment is not None
        else getattr(spec, "business_day_adjustment", None)
    )
    resolved_model = model if model is not None else getattr(spec, "model", None)
    resolved_shift = shift if shift is not None else getattr(spec, "shift", None)
    resolved_sabr = sabr if sabr is not None else getattr(spec, "sabr", None)

    missing = [
        name
        for name, value in {
            "notional": resolved_notional,
            "strike": resolved_strike,
            "start_date": resolved_start_date,
            "end_date": resolved_end_date,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Cap/floor strip pricing requires either `spec` or explicit "
            f"fields for {', '.join(missing)}"
        )
    return CapFloorSpec(
        notional=float(resolved_notional),
        strike=float(resolved_strike),
        start_date=resolved_start_date,
        end_date=resolved_end_date,
        frequency=resolved_frequency if resolved_frequency is not None else Frequency.QUARTERLY,
        day_count=resolved_day_count if resolved_day_count is not None else DayCountConvention.ACT_360,
        rate_index=resolved_rate_index,
        calendar_name=resolved_calendar_name,
        business_day_adjustment=resolved_business_day_adjustment,
        model=resolved_model,
        shift=resolved_shift,
        sabr=dict(resolved_sabr) if resolved_sabr else None,
    )


def price_rate_cap_floor_strip_analytical(
    market_state: MarketState,
    spec: CapFloorSpec | None = None,
    *,
    instrument_class: str = "cap",
    periods: tuple[CapFloorPeriod, ...] | None = None,
    notional: float | None = None,
    strike: float | None = None,
    start_date=None,
    end_date=None,
    frequency=None,
    day_count=None,
    rate_index: str | None = None,
    calendar_name: str | None = None,
    business_day_adjustment: str | None = None,
    model: str | None = None,
    shift: float | None = None,
    sabr: dict[str, float] | None = None,
) -> float:
    """Price a cap/floor strip as a sum of discounted Black-76 caplets/floorlets."""
    spec = _coerce_cap_floor_spec(
        spec,
        notional=notional,
        strike=strike,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        day_count=day_count,
        rate_index=rate_index,
        calendar_name=calendar_name,
        business_day_adjustment=business_day_adjustment,
        model=model,
        shift=shift,
        sabr=sabr,
    )
    kind = _normalize_instrument_class(instrument_class)
    terms = _resolve_cap_floor_terms(market_state, spec, periods=periods)
    if not terms:
        return 0.0

    kernel = black76_call if kind == "cap" else black76_put
    pv = 0.0
    for term in terms:
        model_forward, model_strike, model_volatility = _cap_floor_option_inputs(
            market_state,
            spec,
            term,
        )
        if term.intrinsic_only:
            option_value = max(
                model_forward - model_strike,
                0.0,
            ) if kind == "cap" else max(
                model_strike - model_forward,
                0.0,
            )
        else:
            option_value = float(
                kernel(
                    model_forward,
                    model_strike,
                    model_volatility,
                    term.option_expiry_years,
                )
            )
        pv += spec.notional * term.accrual_fraction * term.discount_factor * option_value
    return float(pv)


def price_rate_cap_floor_strip_monte_carlo(
    market_state: MarketState,
    spec: CapFloorSpec | None = None,
    *,
    instrument_class: str = "cap",
    periods: tuple[CapFloorPeriod, ...] | None = None,
    n_paths: int = 10_000,
    seed: int | None = None,
    n_steps: int = 0,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    notional: float | None = None,
    strike: float | None = None,
    start_date=None,
    end_date=None,
    frequency=None,
    day_count=None,
    rate_index: str | None = None,
    calendar_name: str | None = None,
    business_day_adjustment: str | None = None,
    discount_curve=None,
    forward_curve=None,
    vol=None,
) -> float:
    """Price a cap/floor strip by Monte Carlo on the caplet-strip forward surface."""
    del n_steps, mean_reversion, sigma, discount_curve, forward_curve, vol
    spec = _coerce_cap_floor_spec(
        spec,
        notional=notional,
        strike=strike,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        day_count=day_count,
        rate_index=rate_index,
        calendar_name=calendar_name,
        business_day_adjustment=business_day_adjustment,
    )
    kind = _normalize_instrument_class(instrument_class)
    terms = _resolve_cap_floor_terms(market_state, spec, periods=periods)
    if not terms:
        return 0.0

    direction = 1.0 if kind == "cap" else -1.0
    half_paths = max((int(n_paths) + 1) // 2, 1)
    rng = np.random.RandomState(seed if seed is not None else 12345)

    pv = np.zeros(int(n_paths), dtype=float)
    for term in terms:
        if term.intrinsic_only:
            intrinsic = np.full(
                int(n_paths),
                max(direction * (term.forward_rate - float(spec.strike)), 0.0),
                dtype=float,
            )
        else:
            diffusion = term.volatility * np.sqrt(term.fixing_years)
            draws = rng.standard_normal(half_paths)
            antithetic = np.concatenate((draws, -draws), axis=0)[: int(n_paths)]
            simulated_forward = term.forward_rate * np.exp(
                -0.5 * term.volatility * term.volatility * term.fixing_years
                + diffusion * antithetic
            )
            intrinsic = np.maximum(direction * (simulated_forward - float(spec.strike)), 0.0)
        pv = pv + (
            float(spec.notional)
            * term.accrual_fraction
            * term.discount_factor
            * intrinsic
        )

    return float(np.mean(pv))


__all__ = [
    "CapFloorPeriod",
    "CapFloorSpec",
    "price_rate_cap_floor_strip_analytical",
    "price_rate_cap_floor_strip_monte_carlo",
]
