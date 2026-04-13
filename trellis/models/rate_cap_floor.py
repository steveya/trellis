"""Exact analytical and Monte Carlo helpers for rate cap/floor strips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trellis.core.date_utils import build_payment_timeline
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.instruments.cap import CapFloorSpec
from trellis.models.black import black76_call, black76_put

np = get_numpy()


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


def _resolve_cap_floor_terms(
    market_state: MarketState,
    spec: CapFloorSpec,
) -> tuple[_CapFloorTerm, ...]:
    if getattr(market_state, "discount", None) is None:
        raise ValueError("Cap/floor strip pricing requires market_state.discount")
    if getattr(market_state, "vol_surface", None) is None:
        raise ValueError("Cap/floor strip pricing requires market_state.vol_surface")

    timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=market_state.settlement,
        label="rate_cap_floor_strip",
    )
    forward_curve = _resolve_forward_curve(market_state, spec)

    terms: list[_CapFloorTerm] = []
    for index, period in enumerate(timeline):
        if period.payment_date <= market_state.settlement:
            continue
        fixing_years = float(period.t_start or 0.0)
        if fixing_years <= 0.0:
            continue
        payment_years = max(float(period.t_payment or fixing_years), fixing_years)
        accrual_fraction = float(period.accrual_fraction or 0.0)
        forward_rate = float(forward_curve.forward_rate(fixing_years, payment_years))
        discount_factor = float(market_state.discount.discount(payment_years))
        volatility = float(market_state.vol_surface.black_vol(max(fixing_years, 1e-6), spec.strike))
        terms.append(
            _CapFloorTerm(
                fixing_years=max(fixing_years, 1e-6),
                payment_years=payment_years,
                accrual_fraction=accrual_fraction,
                discount_factor=discount_factor,
                forward_rate=forward_rate,
                volatility=volatility,
                event_name=f"caplet_fixing_{index}",
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
) -> CapFloorSpec:
    """Accept either a concrete spec object or expanded cap/floor keyword fields."""
    if spec is not None:
        return spec
    missing = [
        name
        for name, value in {
            "notional": notional,
            "strike": strike,
            "start_date": start_date,
            "end_date": end_date,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Cap/floor strip pricing requires either `spec` or explicit "
            f"fields for {', '.join(missing)}"
        )
    return CapFloorSpec(
        notional=float(notional),
        strike=float(strike),
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        day_count=day_count,
        rate_index=rate_index,
    )


def price_rate_cap_floor_strip_analytical(
    market_state: MarketState,
    spec: CapFloorSpec | None = None,
    *,
    instrument_class: str = "cap",
    notional: float | None = None,
    strike: float | None = None,
    start_date=None,
    end_date=None,
    frequency=None,
    day_count=None,
    rate_index: str | None = None,
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
    )
    kind = _normalize_instrument_class(instrument_class)
    terms = _resolve_cap_floor_terms(market_state, spec)
    if not terms:
        return 0.0

    kernel = black76_call if kind == "cap" else black76_put
    pv = 0.0
    for term in terms:
        option_value = float(kernel(term.forward_rate, spec.strike, term.volatility, term.fixing_years))
        pv += spec.notional * term.accrual_fraction * term.discount_factor * option_value
    return float(pv)


def price_rate_cap_floor_strip_monte_carlo(
    market_state: MarketState,
    spec: CapFloorSpec | None = None,
    *,
    instrument_class: str = "cap",
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
    )
    kind = _normalize_instrument_class(instrument_class)
    terms = _resolve_cap_floor_terms(market_state, spec)
    if not terms:
        return 0.0

    direction = 1.0 if kind == "cap" else -1.0
    half_paths = max((int(n_paths) + 1) // 2, 1)
    rng = np.random.RandomState(seed if seed is not None else 12345)

    pv = np.zeros(int(n_paths), dtype=float)
    for term in terms:
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
    "CapFloorSpec",
    "price_rate_cap_floor_strip_analytical",
    "price_rate_cap_floor_strip_monte_carlo",
]
