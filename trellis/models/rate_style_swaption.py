"""Stable analytical helpers for rate-style swaptions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.core.date_utils import build_payment_timeline, normalize_explicit_dates, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import ContractTimeline, DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.calibration.rates import swaption_terms
from trellis.models.monte_carlo.event_aware import (
    EventAwareMonteCarloProblem,
    EventAwareMonteCarloProblemSpec,
    EventAwareMonteCarloEvent,
    build_discounted_swap_pv_payload,
    build_event_aware_monte_carlo_problem,
    build_short_rate_discount_reducer,
    price_event_aware_monte_carlo,
    resolve_hull_white_monte_carlo_process_inputs,
)


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

    exercise_dates: ContractTimeline | Iterable[date | str]


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
    return normalize_explicit_dates(raw)


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
    mean_reversion: float | None = None,
    sigma: float | None = None,
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
        swap_start=getattr(spec, "swap_start", None) or expiry,
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
        _resolve_swaption_black76_vol(
            market_state,
            european_spec,
            expiry_years=float(expiry_years),
            mean_reversion=mean_reversion,
            sigma=sigma,
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


def price_swaption_black76_raw(
    resolved: ResolvedSwaptionBlack76Inputs,
) -> float:
    """Raw Black76 kernel over resolved swaption inputs."""
    if (
        resolved.expiry_years <= 0.0
        or resolved.annuity <= 0.0
        or resolved.payment_count <= 0
    ):
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
    return resolved.notional * resolved.annuity * option_value


def price_swaption_black76(
    market_state: MarketState,
    spec: EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike,
    *,
    expiry_date: date | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Price a single-exercise rate-style swaption with Black76."""
    resolved = resolve_swaption_black76_inputs(
        market_state,
        spec,
        expiry_date=expiry_date,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    return float(price_swaption_black76_raw(resolved))


def resolve_swaption_curve_basis_spread(
    market_state: MarketState,
    spec: EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike,
) -> float:
    """Return the constant discount-vs-forecast par-rate basis used by family helpers."""
    if market_state.discount is None:
        raise ValueError("Rate-style swaption basis resolution requires market_state.discount")

    settlement = getattr(market_state, "settlement", None) or market_state.as_of
    payment_timeline = tuple(
        period
        for period in build_payment_timeline(
            spec.swap_start,
            spec.swap_end,
            spec.swap_frequency,
            day_count=spec.day_count,
            time_origin=settlement,
            label="rate_style_swaption_curve_basis",
        )
        if period.end_date > settlement
    )
    if not payment_timeline:
        return 0.0

    expiry_years = max(
        float(year_fraction(settlement, _resolve_expiry_date(spec, expiry_date=None), spec.day_count)),
        1e-6,
    )
    forward_curve = None
    rate_index = getattr(spec, "rate_index", None)
    if rate_index and hasattr(market_state, "forecast_forward_curve"):
        forward_curve = market_state.forecast_forward_curve(rate_index)
    if forward_curve is None:
        forward_curve = getattr(market_state, "forward_curve", None)
    if forward_curve is None:
        return 0.0

    payload = build_discounted_swap_pv_payload(
        payment_timeline=payment_timeline,
        discount_curve=market_state.discount,
        forward_curve=forward_curve,
        exercise_time=expiry_years,
        discount_reducer_name="curve_basis_only",
        mean_reversion=0.0,
        strike=float(spec.strike),
        notional=float(spec.notional),
        is_payer=bool(spec.is_payer),
        anchor_short_rate=0.0,
    )
    return float(payload.get("curve_basis_spread", 0.0))


def _resolve_swaption_black76_vol(
    market_state: MarketState,
    spec: EuropeanSwaptionSpecLike,
    *,
    expiry_years: float,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Resolve the analytical swaption vol under either market or comparison regime."""
    comparison_params = _resolve_swaption_hull_white_comparison_parameters(
        market_state,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    if comparison_params is None:
        return float(
            market_state.vol_surface.black_vol(
                max(float(expiry_years), 1e-8),
                max(abs(float(spec.strike)), 1e-8),
            )
        )

    resolved_mean_reversion, resolved_sigma = comparison_params
    from trellis.models.calibration.rates import calibrate_swaption_black_vol
    from trellis.models.rate_style_swaption_tree import price_swaption_tree

    comparison_price = float(
        price_swaption_tree(
            market_state,
            spec,
            model="hull_white",
            mean_reversion=float(resolved_mean_reversion),
            sigma=float(resolved_sigma),
        )
    )
    result = calibrate_swaption_black_vol(
        spec,
        market_state,
        comparison_price,
    )
    return float(result.calibrated_vol)


def _resolve_swaption_hull_white_comparison_parameters(
    market_state: MarketState,
    *,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> tuple[float, float] | None:
    """Return explicit or materialized Hull-White parameters for comparison normalization."""
    resolved_mean_reversion = None if mean_reversion is None else float(mean_reversion)
    resolved_sigma = None if sigma is None else float(sigma)
    if resolved_mean_reversion is not None and resolved_sigma is not None:
        return resolved_mean_reversion, resolved_sigma

    model_payload = dict(getattr(market_state, "model_parameters", None) or {})
    if not model_payload:
        return None

    family = str(
        model_payload.get("family")
        or model_payload.get("model_family")
        or model_payload.get("model")
        or model_payload.get("parameter_set_name")
        or ""
    ).strip().lower()
    if family not in {"hull_white", "hull_white_1f"}:
        return None

    if resolved_mean_reversion is None:
        for key in ("mean_reversion", "a"):
            if key in model_payload:
                resolved_mean_reversion = float(model_payload[key])
                break
    if resolved_sigma is None and "sigma" in model_payload:
        resolved_sigma = float(model_payload["sigma"])
    if resolved_mean_reversion is None or resolved_sigma is None:
        return None
    return resolved_mean_reversion, resolved_sigma


def resolve_swaption_monte_carlo_problem(
    market_state: MarketState,
    spec: EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike,
    *,
    n_steps: int = 64,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> EventAwareMonteCarloProblem:
    """Resolve a European rate-style swaption onto the shared MC runtime surface."""
    if market_state.discount is None:
        raise ValueError("Rate-style swaption Monte Carlo pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Rate-style swaption Monte Carlo pricing requires market_state.vol_surface")

    resolved = resolve_swaption_black76_inputs(market_state, spec)
    settlement = getattr(market_state, "settlement", None) or market_state.as_of
    payment_timeline = tuple(
        period
        for period in build_payment_timeline(
            resolved.expiry_date,
            spec.swap_end,
            spec.swap_frequency,
            day_count=spec.day_count,
            time_origin=settlement,
            label="rate_style_swaption_monte_carlo",
        )
        if period.end_date > settlement
    )
    if not payment_timeline:
        raise ValueError("Rate-style swaption Monte Carlo pricing requires payments after expiry")

    process_spec, initial_state = resolve_hull_white_monte_carlo_process_inputs(
        market_state,
        option_horizon=max(float(resolved.expiry_years), 1e-6),
        strike=float(spec.strike),
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    forward_curve = None
    rate_index = getattr(spec, "rate_index", None)
    if rate_index and hasattr(market_state, "forecast_forward_curve"):
        forward_curve = market_state.forecast_forward_curve(rate_index)
    if forward_curve is None:
        forward_curve = getattr(market_state, "forward_curve", None)

    settlement_payload = build_discounted_swap_pv_payload(
        payment_timeline=payment_timeline,
        discount_curve=market_state.discount,
        forward_curve=forward_curve,
        exercise_time=float(resolved.expiry_years),
        discount_reducer_name="discount_to_expiry",
        mean_reversion=float(process_spec.mean_reversion or 0.0),
        strike=float(spec.strike),
        notional=float(spec.notional),
        is_payer=bool(spec.is_payer),
        anchor_short_rate=float(initial_state),
    )
    return build_event_aware_monte_carlo_problem(
        EventAwareMonteCarloProblemSpec(
            process_spec=process_spec,
            initial_state=float(initial_state),
            maturity=float(resolved.expiry_years),
            n_steps=max(int(n_steps), 1),
            path_requirement_kind="event_replay",
            reducer_kind="compiled_schedule_payoff",
            path_reducers=(
                build_short_rate_discount_reducer(
                    name="discount_to_expiry",
                    maturity=float(resolved.expiry_years),
                ),
            ),
            settlement_event="swaption_settlement",
            event_specs=(
                EventAwareMonteCarloEvent(
                    time=float(resolved.expiry_years),
                    name="swaption_observation",
                    kind="observation",
                ),
                EventAwareMonteCarloEvent(
                    time=float(resolved.expiry_years),
                    name="swaption_settlement",
                    kind="settlement",
                    priority=1,
                    payload=settlement_payload,
                ),
            ),
        )
    )


def price_swaption_monte_carlo(
    market_state: MarketState,
    spec: EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike,
    *,
    n_paths: int = 10_000,
    seed: int | None = None,
    n_steps: int = 64,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Price a single-exercise European rate-style swaption on the shared MC runtime."""
    problem = resolve_swaption_monte_carlo_problem(
        market_state,
        spec,
        n_steps=n_steps,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    result = price_event_aware_monte_carlo(
        problem,
        n_paths=n_paths,
        seed=seed,
        return_paths=False,
    )
    return float(result["price"])


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
    resolved = resolve_swaption_black76_inputs(
        market_state,
        spec,
        expiry_date=exercise_dates[-1],
    )
    return float(price_swaption_black76_raw(resolved))


__all__ = [
    "BermudanSwaptionLowerBoundSpecLike",
    "EuropeanSwaptionSpecLike",
    "RateStyleSwaptionSpecLike",
    "ResolvedSwaptionBlack76Inputs",
    "price_bermudan_swaption_black76_lower_bound",
    "price_swaption_monte_carlo",
    "price_swaption_black76_raw",
    "price_swaption_black76",
    "resolve_swaption_curve_basis_spread",
    "resolve_swaption_black76_inputs",
    "resolve_swaption_monte_carlo_problem",
]
