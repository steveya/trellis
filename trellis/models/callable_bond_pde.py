"""Stable callable-bond PDE helpers on the generic event-aware rollback lane.

This module gives the event-aware PDE route a bounded callable-bond surface for
issuer-call fixed-income products under a one-factor Hull-White short-rate
model. Generated adapters should bind market-state access and delegate the
coupon/call schedule assembly here instead of rebuilding event buckets inline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as raw_np

from trellis.core.date_utils import build_payment_timeline, normalize_explicit_dates, year_fraction
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.callable_bond_tree import straight_bond_present_value
from trellis.models.hull_white_parameters import resolve_hull_white_parameters
from trellis.models.pde.event_aware import (
    EventAwarePDEBoundarySpec,
    EventAwarePDEEventBucket,
    EventAwarePDEGridSpec,
    EventAwarePDEOperatorSpec,
    EventAwarePDEProblem,
    EventAwarePDEProblemSpec,
    EventAwarePDETransform,
    build_event_aware_pde_problem,
    interpolate_pde_values,
    solve_event_aware_pde,
)
from trellis.models.pde.grid import Grid


class DiscountCurveLike(Protocol):
    """Discount interface required by the callable-bond PDE helper."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return a continuously compounded zero rate to time ``t``."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the callable-bond PDE helper."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class CallableBondPDEMarketStateLike(Protocol):
    """Market-state interface required by the callable-bond PDE helper."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None
    model_parameters: Mapping[str, object] | None
    model_parameter_sets: Mapping[str, Mapping[str, object]] | None


class CallableBondSpecLike(Protocol):
    """Spec fields consumed by the callable-bond PDE helper."""

    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: Iterable[date]
    call_price: float
    frequency: Frequency
    day_count: DayCountConvention


@dataclass(frozen=True)
class InlineCallableBondPDESpec:
    """Concrete callable-bond spec used for flat keyword helper calls."""

    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: tuple[date, ...]
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


@dataclass(frozen=True)
class ResolvedCallableBondPDEInputs:
    """Resolved market inputs and numerical settings for callable-bond PDE."""

    notional: float
    coupon: float
    maturity: float
    mean_reversion: float
    sigma: float
    r0: float
    terminal_redemption: float
    call_price_cash: float
    theta: float
    n_r: int
    n_t: int
    r_min: float
    r_max: float


def resolve_callable_bond_pde_inputs(
    market_state: CallableBondPDEMarketStateLike,
    spec: CallableBondSpecLike,
    *,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    theta: float = 0.5,
    n_r: int | None = None,
    n_t: int | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> ResolvedCallableBondPDEInputs:
    """Resolve callable-bond PDE inputs from market state and spec."""
    settlement = _settlement_date(market_state, spec)
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("Callable-bond PDE pricing requires market_state.discount")

    maturity = max(float(year_fraction(settlement, spec.end_date, spec.day_count)), 0.0)
    if maturity <= 0.0:
        return ResolvedCallableBondPDEInputs(
            notional=float(spec.notional),
            coupon=float(spec.coupon),
            maturity=0.0,
            mean_reversion=float(mean_reversion or 0.1),
            sigma=float(sigma or 0.0),
            r0=0.0,
            terminal_redemption=float(spec.notional),
            call_price_cash=_quoted_call_price_to_cash(spec),
            theta=float(theta),
            n_r=max(int(n_r or 101), 5),
            n_t=max(int(n_t or 1), 1),
            r_min=float(r_min if r_min is not None else -0.10),
            r_max=float(r_max if r_max is not None else 0.20),
        )

    r0 = float(discount_curve.zero_rate(max(maturity / 2.0, 1e-6)))
    default_sigma = (
        _default_hull_white_sigma(
            market_state,
            maturity=maturity,
            r0=r0,
        )
        if sigma is None
        else None
    )
    resolved_mean_reversion, resolved_sigma = resolve_hull_white_parameters(
        market_state,
        mean_reversion=mean_reversion,
        sigma=sigma,
        default_mean_reversion=0.1,
        default_sigma=default_sigma,
    )
    terminal_redemption = float(spec.notional) + _final_coupon_amount(spec, settlement=settlement)
    resolved_n_r = max(int(n_r or 201), 5)
    resolved_n_t = max(int(n_t or max(200, int(round(max(maturity, 1e-6) * 50)))), 1)
    resolved_r_min, resolved_r_max = _rate_bounds(
        r0=r0,
        sigma=resolved_sigma,
        maturity=maturity,
        r_min=r_min,
        r_max=r_max,
    )
    return ResolvedCallableBondPDEInputs(
        notional=float(spec.notional),
        coupon=float(spec.coupon),
        maturity=maturity,
        mean_reversion=float(resolved_mean_reversion),
        sigma=float(resolved_sigma),
        r0=r0,
        terminal_redemption=float(terminal_redemption),
        call_price_cash=_quoted_call_price_to_cash(spec),
        theta=float(theta),
        n_r=resolved_n_r,
        n_t=resolved_n_t,
        r_min=resolved_r_min,
        r_max=resolved_r_max,
    )


def build_callable_bond_pde_problem(
    market_state: CallableBondPDEMarketStateLike,
    spec: CallableBondSpecLike,
    *,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    theta: float = 0.5,
    n_r: int | None = None,
    n_t: int | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> tuple[ResolvedCallableBondPDEInputs, EventAwarePDEProblem]:
    """Build the event-aware callable-bond PDE problem on the shared substrate."""
    resolved = resolve_callable_bond_pde_inputs(
        market_state,
        spec,
        mean_reversion=mean_reversion,
        sigma=sigma,
        theta=theta,
        n_r=n_r,
        n_t=n_t,
        r_min=r_min,
        r_max=r_max,
    )
    if resolved.maturity <= 0.0:
        raise ValueError("Callable-bond PDE problem assembly requires positive maturity")

    discount_curve = market_state.discount
    assert discount_curve is not None  # narrowed by resolve_callable_bond_pde_inputs
    theta_fn = _hull_white_alpha_from_discount_curve(
        discount_curve,
        mean_reversion=resolved.mean_reversion,
        sigma=resolved.sigma,
    )
    problem = build_event_aware_pde_problem(
        EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=resolved.r_min,
                x_max=resolved.r_max,
                n_x=resolved.n_r,
                maturity=resolved.maturity,
                n_t=resolved.n_t,
            ),
            operator_spec=EventAwarePDEOperatorSpec(
                family="hull_white_1f",
                sigma=resolved.sigma,
                mean_reversion=resolved.mean_reversion,
                theta_fn=theta_fn,
                r0=resolved.r0,
            ),
            terminal_condition=lambda r_grid: raw_np.full_like(
                r_grid,
                resolved.terminal_redemption,
                dtype=float,
            ),
            boundary_spec=EventAwarePDEBoundarySpec(
                lower=_lower_short_rate_boundary_discount,
                upper=_upper_short_rate_boundary_discount,
                post_step_policy="linear_extrapolation",
            ),
            event_buckets=_callable_bond_event_buckets(
                spec,
                settlement=settlement_or_asof(market_state, spec),
                call_price_cash=resolved.call_price_cash,
            ),
            theta=resolved.theta,
        )
    )
    return resolved, problem


def solve_callable_bond_pde_surface(
    market_state: CallableBondPDEMarketStateLike,
    spec: CallableBondSpecLike,
    *,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    theta: float = 0.5,
    n_r: int | None = None,
    n_t: int | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> tuple[ResolvedCallableBondPDEInputs, Grid, raw_np.ndarray]:
    """Solve the callable-bond event-aware PDE surface."""
    resolved, problem = build_callable_bond_pde_problem(
        market_state,
        spec,
        mean_reversion=mean_reversion,
        sigma=sigma,
        theta=theta,
        n_r=n_r,
        n_t=n_t,
        r_min=r_min,
        r_max=r_max,
    )
    surface = solve_event_aware_pde(problem)
    return resolved, problem.grid, surface


def price_callable_bond_pde(
    market_state: CallableBondPDEMarketStateLike,
    spec: CallableBondSpecLike | None = None,
    *,
    notional: float | None = None,
    par: float | None = None,
    coupon: float | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    call_dates: Iterable[date] | None = None,
    call_price: float | None = None,
    frequency: Frequency | None = None,
    day_count: DayCountConvention | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    theta: float = 0.5,
    n_r: int | None = None,
    n_t: int | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
) -> float:
    """Price a callable bond with the bounded event-aware Hull-White PDE helper.

    The helper accepts either a typed ``spec`` object or a flat keyword bundle
    containing the callable-bond fields. The keyword form keeps generated
    route adapters thin when they bind request fields directly.
    """
    resolved_spec = _coerce_callable_bond_spec(
        spec,
        notional=notional,
        par=par,
        coupon=coupon,
        start_date=start_date,
        end_date=end_date,
        call_dates=call_dates,
        call_price=call_price,
        frequency=frequency,
        day_count=day_count,
    )
    settlement = _settlement_date(market_state, resolved_spec)
    if settlement >= resolved_spec.end_date:
        return float(_matured_callable_bond_value(resolved_spec, settlement=settlement))

    resolved, grid, surface = solve_callable_bond_pde_surface(
        market_state,
        resolved_spec,
        mean_reversion=mean_reversion,
        sigma=sigma,
        theta=theta,
        n_r=n_r,
        n_t=n_t,
        r_min=r_min,
        r_max=r_max,
    )
    price = float(interpolate_pde_values(surface, grid.x, resolved.r0))
    straight_price = straight_bond_present_value(
        market_state,
        resolved_spec,
        settlement=settlement,
    )
    return min(price, float(straight_price))


def _coerce_callable_bond_spec(
    spec: CallableBondSpecLike | None,
    *,
    notional: float | None,
    par: float | None,
    coupon: float | None,
    start_date: date | None,
    end_date: date | None,
    call_dates: Iterable[date] | None,
    call_price: float | None,
    frequency: Frequency | None,
    day_count: DayCountConvention | None,
) -> CallableBondSpecLike:
    """Resolve either typed-spec or flat keyword inputs into one spec object."""

    def current(name: str, default: object | None = None) -> object | None:
        if spec is None:
            return default
        return getattr(spec, name, default)

    resolved_notional = notional if notional is not None else par
    if resolved_notional is None:
        resolved_notional = current("notional")
    resolved_coupon = coupon if coupon is not None else current("coupon")
    resolved_start = start_date if start_date is not None else current("start_date")
    resolved_end = end_date if end_date is not None else current("end_date")
    resolved_call_dates = call_dates if call_dates is not None else current("call_dates")
    resolved_call_price = call_price if call_price is not None else current("call_price", 100.0)
    resolved_frequency = frequency if frequency is not None else current(
        "frequency",
        Frequency.SEMI_ANNUAL,
    )
    resolved_day_count = day_count if day_count is not None else current(
        "day_count",
        DayCountConvention.ACT_365,
    )

    missing: list[str] = []
    if resolved_notional is None:
        missing.append("notional")
    if resolved_coupon is None:
        missing.append("coupon")
    if resolved_start is None:
        missing.append("start_date")
    if resolved_end is None:
        missing.append("end_date")
    if resolved_call_dates is None:
        missing.append("call_dates")
    if missing:
        raise TypeError(
            "price_callable_bond_pde requires either spec or flat keyword fields: "
            + ", ".join(missing)
        )

    if (
        spec is not None
        and notional is None
        and par is None
        and coupon is None
        and start_date is None
        and end_date is None
        and call_dates is None
        and call_price is None
        and frequency is None
        and day_count is None
    ):
        return spec

    return InlineCallableBondPDESpec(
        notional=float(resolved_notional),
        coupon=float(resolved_coupon),
        start_date=resolved_start,
        end_date=resolved_end,
        call_dates=tuple(normalize_explicit_dates(resolved_call_dates)),
        call_price=float(resolved_call_price),
        frequency=resolved_frequency,
        day_count=resolved_day_count,
    )


def _callable_bond_event_buckets(
    spec: CallableBondSpecLike,
    *,
    settlement: date,
    call_price_cash: float,
) -> tuple[EventAwarePDEEventBucket, ...]:
    """Build coupon and issuer-call event buckets in deterministic time order."""
    coupon_by_date = _coupon_amounts_by_date(spec, settlement=settlement)
    call_dates = {
        call_date
        for call_date in normalize_explicit_dates(spec.call_dates)
        if settlement < call_date < spec.end_date
    }
    all_event_dates = sorted(set(coupon_by_date).union(call_dates))

    buckets: list[EventAwarePDEEventBucket] = []
    for event_date in all_event_dates:
        event_time = max(float(year_fraction(settlement, event_date, spec.day_count)), 0.0)
        transforms: list[EventAwarePDETransform] = []
        coupon_cash = float(coupon_by_date.get(event_date, 0.0))
        if coupon_cash:
            transforms.append(
                EventAwarePDETransform(
                    kind="add_cashflow",
                    payload=coupon_cash,
                    label="coupon_cashflow",
                )
            )
        if event_date in call_dates:
            transforms.append(
                EventAwarePDETransform(
                    kind="project_min",
                    payload=call_price_cash + coupon_cash,
                    label="issuer_call_projection",
                )
            )
        if transforms:
            buckets.append(
                EventAwarePDEEventBucket(
                    time=event_time,
                    transforms=tuple(transforms),
                    label=event_date.isoformat(),
                )
            )
    return tuple(buckets)


def _coupon_amounts_by_date(
    spec: CallableBondSpecLike,
    *,
    settlement: date,
) -> dict[date, float]:
    """Map non-terminal coupon payment dates to cash coupon amounts."""
    coupon_by_date: dict[date, float] = {}
    payment_timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=settlement,
        label="callable_bond_coupon_timeline",
    )
    for period in payment_timeline:
        if period.payment_date <= settlement or period.payment_date >= spec.end_date:
            continue
        coupon = float(spec.notional) * float(spec.coupon) * float(period.accrual_fraction or 0.0)
        coupon_by_date[period.payment_date] = coupon_by_date.get(period.payment_date, 0.0) + coupon
    return coupon_by_date


def _final_coupon_amount(
    spec: CallableBondSpecLike,
    *,
    settlement: date,
) -> float:
    """Return the maturity coupon amount included in the terminal value."""
    payment_timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=settlement,
        label="callable_bond_coupon_timeline",
    )
    coupon = 0.0
    for period in payment_timeline:
        if period.payment_date == spec.end_date:
            coupon += float(spec.notional) * float(spec.coupon) * float(period.accrual_fraction or 0.0)
    return float(coupon)


def _matured_callable_bond_value(spec: CallableBondSpecLike, *, settlement: date) -> float:
    """Return the immediate settlement value once the callable bond has matured."""
    if settlement > spec.end_date:
        return 0.0
    return float(spec.notional) + _final_coupon_amount(spec, settlement=settlement)


def _quoted_call_price_to_cash(spec: CallableBondSpecLike) -> float:
    """Convert the quoted callable-bond price from 100-par quote to cash terms."""
    return float(spec.call_price) / 100.0 * float(spec.notional)


def _default_hull_white_sigma(
    market_state: CallableBondPDEMarketStateLike,
    *,
    maturity: float,
    r0: float,
) -> float:
    """Infer a bounded Hull-White sigma default from the attached Black vol surface."""
    if market_state.vol_surface is None:
        raise ValueError(
            "Callable-bond PDE pricing requires market_state.vol_surface unless sigma is provided explicitly"
        )
    black_vol = float(
        market_state.vol_surface.black_vol(
            max(maturity / 2.0, 1e-6),
            max(abs(r0), 1e-6),
        )
    )
    return float(black_vol * max(abs(r0), 1e-6))


def _rate_bounds(
    *,
    r0: float,
    sigma: float,
    maturity: float,
    r_min: float | None,
    r_max: float | None,
) -> tuple[float, float]:
    """Return a bounded short-rate grid around the current short-rate seed."""
    if r_min is not None and r_max is not None:
        return float(r_min), float(r_max)
    span = max(0.10, 6.0 * float(sigma) * raw_np.sqrt(max(float(maturity), 1e-6)), 2.0 * abs(float(r0)) + 0.05)
    lower = float(r_min) if r_min is not None else min(float(r0) - span, -0.10)
    upper = float(r_max) if r_max is not None else max(float(r0) + span, 0.20)
    return lower, upper


def _lower_short_rate_boundary_discount(t: float, values: raw_np.ndarray, grid: Grid) -> float:
    """Lower short-rate boundary discounting rule used before extrapolation."""
    if not raw_np.isfinite(t):
        raise ValueError("Boundary evaluation time must be finite")
    if values.shape[0] != grid.x.shape[0]:
        raise ValueError("Boundary values must align with the short-rate grid")
    return float(values[0] * raw_np.exp(-float(grid.x[0]) * grid.dt))


def _upper_short_rate_boundary_discount(t: float, values: raw_np.ndarray, grid: Grid) -> float:
    """Upper short-rate boundary discounting rule used before extrapolation."""
    if not raw_np.isfinite(t):
        raise ValueError("Boundary evaluation time must be finite")
    if values.shape[0] != grid.x.shape[0]:
        raise ValueError("Boundary values must align with the short-rate grid")
    return float(values[-1] * raw_np.exp(-float(grid.x[-1]) * grid.dt))


def _hull_white_alpha_from_discount_curve(
    discount_curve: DiscountCurveLike,
    *,
    mean_reversion: float,
    sigma: float,
) -> callable:
    """Approximate the Hull-White long-run mean alpha(t) from the discount curve.

    This bounded proof slice deliberately uses the curve zero rate as a stable
    proxy for the instantaneous forward curve. That is exact for flat curves and
    materially more stable than second-differencing discount factors on the
    synthetic/test curve surfaces used by the live build loop.
    """
    a = float(mean_reversion)
    sigma_sq_term = float(sigma) ** 2 / (2.0 * a * a)

    def alpha_fn(t: float) -> float:
        t = max(float(t), 1e-6)
        forward = float(discount_curve.zero_rate(t))
        return float(
            forward
            + sigma_sq_term * (1.0 - raw_np.exp(-2.0 * a * t))
        )

    return alpha_fn


def _settlement_date(market_state: CallableBondPDEMarketStateLike, spec: CallableBondSpecLike) -> date:
    settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
    if settlement is None:
        return spec.start_date
    return settlement


def settlement_or_asof(market_state: CallableBondPDEMarketStateLike, spec: CallableBondSpecLike) -> date:
    """Return the callable-bond settlement anchor used for schedule timing."""
    return _settlement_date(market_state, spec)


__all__ = [
    "ResolvedCallableBondPDEInputs",
    "build_callable_bond_pde_problem",
    "price_callable_bond_pde",
    "resolve_callable_bond_pde_inputs",
    "solve_callable_bond_pde_surface",
]
