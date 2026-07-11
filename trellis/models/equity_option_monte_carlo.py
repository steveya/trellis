"""Stable vanilla-equity Monte Carlo helpers.

This module gives the compiler a semantic-facing helper kit for plain European
equity options priced with bounded Monte Carlo schemes. Generated adapters
should bind market-state access and comparison-target controls here instead of
rebuilding GBM construction, scheme wiring, or variance-reduction glue inline.
"""

from __future__ import annotations

import numpy as raw_np

from trellis.models.monte_carlo.single_state_diffusion import (
    ResolvedSingleStateMonteCarloInputs as ResolvedEquityMonteCarloInputs,
    SingleStateMonteCarloResult as VanillaEquityMonteCarloResult,
    build_single_state_terminal_claim_monte_carlo_problem,
    price_single_state_terminal_claim_monte_carlo_result,
    resolve_single_state_monte_carlo_inputs,
)
from trellis.models.monte_carlo.event_aware import EventAwareMonteCarloProblem
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.lsm import longstaff_schwartz_result
from trellis.models.resolution.single_state_diffusion import (
    SingleStateDiffusionMarketStateLike,
    SingleStateDiffusionSpecLike,
    terminal_intrinsic_from_resolved,
)
from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.processes.gbm import GBM


def resolve_vanilla_equity_monte_carlo_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> ResolvedEquityMonteCarloInputs:
    """Resolve vanilla-equity MC inputs from market state and a product spec."""
    return resolve_single_state_monte_carlo_inputs(
        market_state,
        spec,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )


def build_vanilla_equity_monte_carlo_problem(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> tuple[ResolvedEquityMonteCarloInputs, EventAwareMonteCarloProblem]:
    """Build the bounded event-aware MC problem for a vanilla European option."""
    return build_single_state_terminal_claim_monte_carlo_problem(
        market_state,
        spec,
        terminal_payoff=_terminal_payoff,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )


def price_vanilla_equity_option_monte_carlo_result(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> VanillaEquityMonteCarloResult:
    """Return a structured vanilla-equity Monte Carlo price result."""
    return price_single_state_terminal_claim_monte_carlo_result(
        market_state,
        spec,
        terminal_payoff=_terminal_payoff,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        control_variate_values=lambda terminal, _: raw_np.asarray(terminal, dtype=float),
        control_variate_expected=lambda resolved: float(
            resolved.spot
            * raw_np.exp((resolved.rate - resolved.dividend_yield) * resolved.maturity)
        ),
    )


def price_vanilla_equity_option_monte_carlo(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> float:
    """Return the scalar present value of a vanilla-equity Monte Carlo price."""
    return float(
        price_vanilla_equity_option_monte_carlo_result(
            market_state,
            spec,
            scheme=scheme,
            variance_reduction=variance_reduction,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        ).price
    )


def price_american_equity_option_lsm_monte_carlo(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    basis: str = "laguerre",
) -> float:
    """Price an American/Bermudan vanilla equity option with LSM on exact GBM paths."""
    resolved = resolve_vanilla_equity_monte_carlo_inputs(
        market_state,
        spec,
        scheme="exact",
        variance_reduction="none",
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    if resolved.maturity <= 0.0:
        return float(
            resolved.notional
            * terminal_intrinsic_from_resolved(resolved.spot, resolved)
        )

    exercise_steps = _lsm_exercise_steps(market_state, spec, resolved)
    process = GBM(
        mu=resolved.rate - resolved.dividend_yield,
        sigma=resolved.sigma,
    )
    engine = MonteCarloEngine(
        process,
        n_paths=resolved.n_paths,
        n_steps=resolved.n_steps,
        seed=resolved.seed,
        method="exact",
    )
    paths = engine.simulate(resolved.spot, resolved.maturity)
    dt = resolved.maturity / resolved.n_steps

    result = longstaff_schwartz_result(
        paths,
        exercise_steps,
        lambda spots: _terminal_payoff(spots, resolved),
        discount_rate=resolved.rate,
        dt=dt,
        basis_fn=_lsm_basis_fn(resolved, basis),
    )
    return float(resolved.notional * result.price_lower)


def _terminal_payoff(terminal, resolved: ResolvedEquityMonteCarloInputs) -> raw_np.ndarray:
    terminal_arr = raw_np.asarray(terminal, dtype=float)
    return raw_np.asarray(terminal_intrinsic_from_resolved(terminal_arr, resolved), dtype=float)


def _lsm_basis_fn(resolved: ResolvedEquityMonteCarloInputs, basis: str):
    normalized = str(basis or "laguerre").strip().lower()
    if normalized not in {"laguerre", "scaled_laguerre", "polynomial"}:
        raise ValueError(f"Unsupported LSM basis {basis!r}")
    if normalized == "polynomial":
        return None

    scale = max(float(resolved.strike), 1e-12)

    def scaled_laguerre(spots):
        x = raw_np.asarray(spots, dtype=float) / scale
        return raw_np.column_stack([
            raw_np.ones_like(x),
            1.0 - x,
            0.5 * (x * x - 4.0 * x + 2.0),
        ])

    return scaled_laguerre


def _lsm_exercise_steps(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    resolved: ResolvedEquityMonteCarloInputs,
) -> list[int]:
    style = str(getattr(spec, "exercise_style", "american") or "american").strip().lower()
    if style == "european":
        return [resolved.n_steps]
    if style == "american":
        return list(range(1, resolved.n_steps + 1))
    if style != "bermudan":
        raise ValueError(f"Unsupported exercise_style {style!r}")

    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Bermudan LSM pricing requires market_state settlement or as_of")
    dates = tuple(getattr(spec, "exercise_dates", ()) or ())
    if not dates:
        raise ValueError("Bermudan LSM pricing requires spec.exercise_dates")
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    steps: set[int] = set()
    for item in dates:
        exercise_time = float(year_fraction(settlement, item, day_count))
        if exercise_time < 0.0 or exercise_time > resolved.maturity:
            continue
        step = int(round(exercise_time / max(resolved.maturity, 1e-12) * resolved.n_steps))
        steps.add(min(max(step, 1), resolved.n_steps))
    if not steps:
        raise ValueError("Bermudan LSM pricing resolved no valid exercise dates")
    return sorted(steps)


__all__ = [
    "ResolvedEquityMonteCarloInputs",
    "VanillaEquityMonteCarloResult",
    "build_vanilla_equity_monte_carlo_problem",
    "price_american_equity_option_lsm_monte_carlo",
    "price_vanilla_equity_option_monte_carlo",
    "price_vanilla_equity_option_monte_carlo_result",
    "resolve_vanilla_equity_monte_carlo_inputs",
]
