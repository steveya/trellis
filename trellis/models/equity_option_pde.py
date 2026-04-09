"""Stable vanilla-equity PDE helpers.

This module gives the DSL a checked-in helper surface for plain European
vanilla equity options priced with the one-dimensional theta-method PDE route.
Generated adapters should bind market-state access and theta selection here
instead of rebuilding grid assembly, boundary conditions, and interpolation
inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention
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

np = get_numpy()


class DiscountCurveLike(Protocol):
    """Discount interface required by the PDE helpers."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the PDE helpers."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class EquityPDEMarketStateLike(Protocol):
    """Market-state interface required by the PDE helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None


class VanillaEquityOptionSpecLike(Protocol):
    """Spec fields consumed by the PDE helpers."""

    notional: float
    spot: float
    strike: float
    expiry_date: date


class EventAwareEquityOptionSpecLike(VanillaEquityOptionSpecLike, Protocol):
    """Optional early-exercise fields consumed by the event-aware PDE helper."""

    exercise_style: str
    exercise_dates: tuple[date, ...] | list[date]


@dataclass(frozen=True)
class ResolvedEquityPDEInputs:
    """Resolved market inputs and numerical settings for a vanilla PDE route."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    sigma: float
    option_type: str
    theta: float
    s_max: float
    n_x: int
    n_t: int
    exercise_style: str = "european"
    exercise_times: tuple[float, ...] = ()


def resolve_vanilla_equity_pde_inputs(
    market_state: EquityPDEMarketStateLike,
    spec: VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> ResolvedEquityPDEInputs:
    """Resolve vanilla European PDE inputs from market state and spec."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for PDE pricing")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = float(year_fraction(settlement, spec.expiry_date, day_count))
    maturity = max(maturity, 0.0)

    strike = float(spec.strike)
    spot = float(spec.spot)
    option_type = str(getattr(spec, "option_type", "call")).strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unsupported option_type {option_type!r}")
    exercise_style = str(getattr(spec, "exercise_style", "european")).strip().lower()
    if exercise_style not in {"european", "american", "bermudan"}:
        raise ValueError(f"Unsupported exercise_style {exercise_style!r}")

    if maturity <= 0.0:
        rate = 0.0
        sigma = 0.0
    else:
        if market_state.discount is None:
            raise ValueError("equity PDE pricing requires market_state.discount")
        if market_state.vol_surface is None:
            raise ValueError("equity PDE pricing requires market_state.vol_surface")
        df = float(market_state.discount.discount(maturity))
        if df <= 0.0:
            raise ValueError(f"Invalid discount factor at T={maturity}: {df}")
        rate = float(-np.log(df) / maturity)
        sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
        if sigma < 0.0:
            raise ValueError(f"Invalid Black volatility at T={maturity}, K={strike}: {sigma}")

    default_s_max = max(float(s_max_multiplier) * max(spot, 1e-12), 2.0 * max(strike, 1e-12))
    s_max = max(float(getattr(spec, "s_max", 0.0) or 0.0), default_s_max, 1e-6)
    resolved_n_x = int(getattr(spec, "n_x", n_x or 201))
    resolved_n_t = int(getattr(spec, "n_t", n_t or max(200, int(round(max(maturity, 1e-6) * 252)))))
    exercise_times = _resolve_exercise_times(
        settlement,
        maturity=maturity,
        spec=spec,
        exercise_style=exercise_style,
        day_count=day_count,
    )

    return ResolvedEquityPDEInputs(
        notional=float(spec.notional),
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        sigma=sigma,
        option_type=option_type,
        theta=float(theta),
        s_max=s_max,
        n_x=max(resolved_n_x, 5),
        n_t=max(resolved_n_t, 1),
        exercise_style=exercise_style,
        exercise_times=exercise_times,
    )


def build_event_aware_equity_pde_problem(
    market_state: EquityPDEMarketStateLike,
    spec: EventAwareEquityOptionSpecLike | VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> tuple[ResolvedEquityPDEInputs, EventAwarePDEProblem]:
    """Build a bounded event-aware equity PDE problem on the shared substrate."""
    resolved = resolve_vanilla_equity_pde_inputs(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )
    if resolved.maturity <= 0.0:
        raise ValueError("PDE problem assembly requires positive maturity")

    lower_bc_fn, upper_bc_fn = _boundary_conditions(
        option_type=resolved.option_type,
        strike=resolved.strike,
        rate=resolved.rate,
        maturity=resolved.maturity,
        s_max=resolved.s_max,
    )
    problem = build_event_aware_pde_problem(
        EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0,
                x_max=resolved.s_max,
                n_x=resolved.n_x,
                maturity=resolved.maturity,
                n_t=resolved.n_t,
            ),
            operator_spec=EventAwarePDEOperatorSpec(
                family="black_scholes_1d",
                sigma=resolved.sigma,
                r=resolved.rate,
            ),
            terminal_condition=lambda x: _terminal_payoff(x, resolved.option_type, resolved.strike),
            boundary_spec=EventAwarePDEBoundarySpec(
                lower=lower_bc_fn,
                upper=upper_bc_fn,
            ),
            event_buckets=_exercise_event_buckets_for_equity(resolved),
            theta=resolved.theta,
        )
    )
    return resolved, problem


def build_vanilla_equity_pde_problem(
    market_state: EquityPDEMarketStateLike,
    spec: VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> tuple[ResolvedEquityPDEInputs, EventAwarePDEProblem]:
    """Build the grid/operator/terminal-condition bundle for vanilla PDE pricing."""
    return build_event_aware_equity_pde_problem(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )


def solve_vanilla_equity_option_pde_surface(
    market_state: EquityPDEMarketStateLike,
    spec: VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> tuple[ResolvedEquityPDEInputs, Grid, raw_np.ndarray]:
    """Solve the theta-method PDE surface for a vanilla European option."""
    resolved, problem = build_vanilla_equity_pde_problem(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )
    values = solve_event_aware_pde(problem)
    return resolved, problem.grid, values


def solve_event_aware_equity_option_pde_surface(
    market_state: EquityPDEMarketStateLike,
    spec: EventAwareEquityOptionSpecLike | VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> tuple[ResolvedEquityPDEInputs, Grid, raw_np.ndarray]:
    """Solve an event-aware equity PDE surface on the shared rollback substrate."""
    resolved, problem = build_event_aware_equity_pde_problem(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )
    values = solve_event_aware_pde(problem)
    return resolved, problem.grid, values


def price_vanilla_equity_option_pde(
    market_state: EquityPDEMarketStateLike,
    spec: VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> float:
    """Price a vanilla European option via the checked-in theta-method helper."""
    return price_event_aware_equity_option_pde(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )


def price_event_aware_equity_option_pde(
    market_state: EquityPDEMarketStateLike,
    spec: EventAwareEquityOptionSpecLike | VanillaEquityOptionSpecLike,
    *,
    theta: float = 0.5,
    n_x: int | None = None,
    n_t: int | None = None,
    s_max_multiplier: float = 4.0,
) -> float:
    """Price a bounded event-aware equity option via the shared PDE substrate."""
    resolved = resolve_vanilla_equity_pde_inputs(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )
    if resolved.maturity <= 0.0:
        intrinsic = _terminal_intrinsic(
            resolved.option_type,
            spot=resolved.spot,
            strike=resolved.strike,
        )
        return float(resolved.notional * intrinsic)

    _, grid, values = solve_event_aware_equity_option_pde_surface(
        market_state,
        spec,
        theta=theta,
        n_x=n_x,
        n_t=n_t,
        s_max_multiplier=s_max_multiplier,
    )
    price = interpolate_pde_values(values, grid.x, resolved.spot)
    return float(resolved.notional * price)


def _boundary_conditions(*, option_type: str, strike: float, rate: float, maturity: float, s_max: float):
    if option_type == "put":
        lower_bc = lambda t: float(strike * raw_np.exp(-rate * (maturity - t)))
        upper_bc = lambda t: 0.0
    else:
        lower_bc = lambda t: 0.0
        upper_bc = lambda t: float(s_max - strike * raw_np.exp(-rate * (maturity - t)))
    return lower_bc, upper_bc


def _resolve_exercise_times(
    settlement: date,
    *,
    maturity: float,
    spec: EventAwareEquityOptionSpecLike | VanillaEquityOptionSpecLike,
    exercise_style: str,
    day_count: DayCountConvention,
) -> tuple[float, ...]:
    if exercise_style != "bermudan":
        return ()

    raw_dates = tuple(getattr(spec, "exercise_dates", ()) or ())
    if not raw_dates:
        raise ValueError("Bermudan equity PDE pricing requires spec.exercise_dates")

    times: list[float] = []
    for item in raw_dates:
        exercise_time = float(year_fraction(settlement, item, day_count))
        if 0.0 <= exercise_time < maturity and exercise_time not in times:
            times.append(exercise_time)
    return tuple(sorted(times))


def _exercise_event_buckets_for_equity(
    resolved: ResolvedEquityPDEInputs,
) -> tuple:
    if resolved.exercise_style == "european":
        return ()

    if resolved.exercise_style == "american":
        dt = resolved.maturity / max(resolved.n_t, 1)
        event_times = tuple(step * dt for step in range(resolved.n_t))
    else:
        event_times = resolved.exercise_times

    if not event_times:
        return ()

    intrinsic_payload = lambda x, _values, _t: _terminal_payoff(x, resolved.option_type, resolved.strike)
    return tuple(
        EventAwarePDEEventBucket(
            time=float(event_time),
            transforms=(
                EventAwarePDETransform(
                    kind="project_max",
                    payload=intrinsic_payload,
                    label="holder_exercise",
                ),
            ),
            label="holder_exercise",
        )
        for event_time in event_times
        if 0.0 <= float(event_time) < resolved.maturity
    )


def _terminal_intrinsic(option_type: str, *, spot: float, strike: float) -> float:
    if option_type == "put":
        return max(float(strike) - float(spot), 0.0)
    return max(float(spot) - float(strike), 0.0)


def _terminal_payoff(spots: raw_np.ndarray, option_type: str, strike: float) -> raw_np.ndarray:
    if option_type == "put":
        return raw_np.maximum(float(strike) - spots, 0.0)
    return raw_np.maximum(spots - float(strike), 0.0)


__all__ = [
    "ResolvedEquityPDEInputs",
    "build_event_aware_equity_pde_problem",
    "build_vanilla_equity_pde_problem",
    "price_event_aware_equity_option_pde",
    "price_vanilla_equity_option_pde",
    "resolve_vanilla_equity_pde_inputs",
    "solve_event_aware_equity_option_pde_surface",
    "solve_vanilla_equity_option_pde_surface",
]
