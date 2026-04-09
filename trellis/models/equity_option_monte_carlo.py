"""Stable vanilla-equity Monte Carlo helpers.

This module gives the compiler a semantic-facing helper kit for plain European
equity options priced with bounded Monte Carlo schemes. Generated adapters
should bind market-state access and comparison-target controls here instead of
rebuilding GBM construction, scheme wiring, or variance-reduction glue inline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as raw_np

from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.event_aware import (
    EventAwareMonteCarloProblem,
    EventAwareMonteCarloProblemSpec,
    EventAwareMonteCarloProcessSpec,
    build_event_aware_monte_carlo_problem,
    price_event_aware_monte_carlo,
)
from trellis.models.monte_carlo.schemes import Antithetic, Euler, Exact, LogEuler, Milstein
from trellis.models.monte_carlo.variance_reduction import control_variate
from trellis.models.resolution.single_state_diffusion import (
    ResolvedSingleStateDiffusionInputs,
    SingleStateDiffusionMarketStateLike,
    SingleStateDiffusionSpecLike,
    resolve_single_state_diffusion_inputs,
    terminal_intrinsic_from_resolved,
)


@dataclass(frozen=True)
class ResolvedEquityMonteCarloInputs(ResolvedSingleStateDiffusionInputs):
    """Resolved market inputs and numerical controls for vanilla MC pricing."""
    n_paths: int
    n_steps: int
    seed: int | None
    scheme: str
    variance_reduction: str


@dataclass(frozen=True)
class VanillaEquityMonteCarloResult:
    """Structured vanilla-equity Monte Carlo pricing result."""

    price: float
    std_error: float
    n_paths: int
    scheme: str
    variance_reduction: str
    control_variate_beta: float | None = None


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
    resolved_base = resolve_single_state_diffusion_inputs(market_state, spec)
    resolved_scheme = _normalized_scheme(
        scheme if scheme is not None else getattr(spec, "scheme", "exact")
    )
    resolved_variance_reduction = _normalized_variance_reduction(
        variance_reduction
        if variance_reduction is not None
        else getattr(spec, "variance_reduction", "none")
    )

    return ResolvedEquityMonteCarloInputs(
        **resolved_base.__dict__,
        n_paths=max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 50_000)), 1),
        n_steps=max(int(n_steps if n_steps is not None else getattr(spec, "n_steps", 252)), 1),
        seed=seed if seed is not None else getattr(spec, "seed", 42),
        scheme=resolved_scheme,
        variance_reduction=resolved_variance_reduction,
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
    resolved = resolve_vanilla_equity_monte_carlo_inputs(
        market_state,
        spec,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    problem = build_event_aware_monte_carlo_problem(
        EventAwareMonteCarloProblemSpec(
            process_spec=EventAwareMonteCarloProcessSpec(
                family="gbm_1d",
                risk_free_rate=resolved.rate,
                dividend_yield=resolved.dividend_yield,
                sigma=resolved.sigma,
                simulation_method="exact" if resolved.scheme == "log_euler" else resolved.scheme,
            ),
            initial_state=resolved.spot,
            maturity=resolved.maturity,
            n_steps=resolved.n_steps,
            discount_rate=resolved.rate,
            path_requirement_kind="terminal_only",
            reducer_kind="terminal_payoff",
            terminal_payoff=lambda terminal: _terminal_payoff(terminal, resolved),
        )
    )
    return resolved, problem


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
    resolved, problem = build_vanilla_equity_monte_carlo_problem(
        market_state,
        spec,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    if resolved.maturity <= 0.0:
        intrinsic = _terminal_payoff(raw_np.asarray([resolved.spot], dtype=float), resolved)[0]
        return VanillaEquityMonteCarloResult(
            price=float(resolved.notional) * float(intrinsic),
            std_error=0.0,
            n_paths=0,
            scheme=resolved.scheme,
            variance_reduction=resolved.variance_reduction,
        )

    if resolved.variance_reduction == "none" and resolved.scheme in {"exact", "euler"}:
        result = price_event_aware_monte_carlo(
            problem,
            n_paths=resolved.n_paths,
            seed=resolved.seed,
            return_paths=False,
        )
        return VanillaEquityMonteCarloResult(
            price=float(resolved.notional) * float(result["price"]),
            std_error=float(resolved.notional) * float(result["std_error"]),
            n_paths=int(result["n_paths"]),
            scheme=resolved.scheme,
            variance_reduction=resolved.variance_reduction,
        )

    scheme_obj = _scheme_object(resolved.scheme)
    if resolved.variance_reduction == "antithetic":
        if resolved.n_paths % 2 != 0:
            raise ValueError("antithetic variance reduction requires an even n_paths")
        scheme_obj = Antithetic(scheme_obj)

    engine = MonteCarloEngine(
        problem.process,
        n_paths=resolved.n_paths,
        n_steps=resolved.n_steps,
        seed=resolved.seed,
        scheme=scheme_obj,
    )

    if resolved.variance_reduction == "control_variate":
        paths = engine.simulate(problem.initial_state, problem.maturity)
        terminal = raw_np.asarray(paths[:, -1], dtype=float)
        discounted_payoffs = (
            raw_np.exp(-resolved.rate * resolved.maturity)
            * resolved.notional
            * _terminal_payoff(terminal, resolved)
        )
        control_values = terminal
        control_expected = resolved.spot * raw_np.exp(
            (resolved.rate - resolved.dividend_yield) * resolved.maturity
        )
        result = control_variate(
            discounted_payoffs,
            control_values,
            float(control_expected),
        )
        return VanillaEquityMonteCarloResult(
            price=float(result["price"]),
            std_error=float(result["std_error"]),
            n_paths=resolved.n_paths,
            scheme=resolved.scheme,
            variance_reduction=resolved.variance_reduction,
            control_variate_beta=float(result["beta"]),
        )

    if resolved.variance_reduction == "antithetic":
        paths = engine.simulate(problem.initial_state, problem.maturity)
        terminal = raw_np.asarray(paths[:, -1], dtype=float)
        payoffs = resolved.notional * _terminal_payoff(terminal, resolved)
        half = resolved.n_paths // 2
        averaged = 0.5 * (payoffs[:half] + payoffs[half:])
        discounted = raw_np.exp(-resolved.rate * resolved.maturity) * averaged
        return VanillaEquityMonteCarloResult(
            price=float(raw_np.mean(discounted)),
            std_error=float(raw_np.std(discounted) / raw_np.sqrt(half)),
            n_paths=resolved.n_paths,
            scheme=resolved.scheme,
            variance_reduction=resolved.variance_reduction,
        )

    result = engine.price(
        problem.initial_state,
        problem.maturity,
        lambda paths: resolved.notional * _terminal_payoff(paths[:, -1], resolved),
        discount_rate=problem.discount_rate,
        storage_policy=problem.path_requirement,
        return_paths=False,
    )
    return VanillaEquityMonteCarloResult(
        price=float(result["price"]),
        std_error=float(result["std_error"]),
        n_paths=int(result["n_paths"]),
        scheme=resolved.scheme,
        variance_reduction=resolved.variance_reduction,
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

def _normalized_scheme(value: object) -> str:
    scheme = str(value or "exact").strip().lower().replace("-", "_")
    aliases = {
        "plain": "exact",
        "plain_mc": "exact",
        "antithetic_mc": "exact",
        "control_variate_mc": "exact",
    }
    scheme = aliases.get(scheme, scheme)
    if scheme not in {"euler", "milstein", "exact", "log_euler"}:
        raise ValueError(f"Unsupported Monte Carlo scheme {value!r}")
    return scheme


def _normalized_variance_reduction(value: object) -> str:
    reduction = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "plain": "none",
        "plain_mc": "none",
        "antithetic_mc": "antithetic",
        "control_variate_mc": "control_variate",
    }
    reduction = aliases.get(reduction, reduction)
    if reduction not in {"none", "antithetic", "control_variate"}:
        raise ValueError(f"Unsupported variance reduction mode {value!r}")
    return reduction


def _scheme_object(scheme: str):
    if scheme == "euler":
        return Euler()
    if scheme == "milstein":
        return Milstein()
    if scheme == "log_euler":
        return LogEuler()
    return Exact()


def _terminal_payoff(terminal, resolved: ResolvedEquityMonteCarloInputs) -> raw_np.ndarray:
    terminal_arr = raw_np.asarray(terminal, dtype=float)
    return raw_np.asarray(terminal_intrinsic_from_resolved(terminal_arr, resolved), dtype=float)


__all__ = [
    "ResolvedEquityMonteCarloInputs",
    "VanillaEquityMonteCarloResult",
    "build_vanilla_equity_monte_carlo_problem",
    "price_vanilla_equity_option_monte_carlo",
    "price_vanilla_equity_option_monte_carlo_result",
    "resolve_vanilla_equity_monte_carlo_inputs",
]
