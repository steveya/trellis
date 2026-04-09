"""Reusable event-aware Monte Carlo helpers for single-state diffusion claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
)


@dataclass(frozen=True)
class ResolvedSingleStateMonteCarloInputs(ResolvedSingleStateDiffusionInputs):
    """Resolved market inputs and numerical controls for one single-state MC claim."""

    n_paths: int
    n_steps: int
    seed: int | None
    scheme: str
    variance_reduction: str


@dataclass(frozen=True)
class SingleStateMonteCarloResult:
    """Structured result for a bounded single-state Monte Carlo claim."""

    price: float
    std_error: float
    n_paths: int
    scheme: str
    variance_reduction: str
    control_variate_beta: float | None = None


def resolve_single_state_terminal_claim_monte_carlo_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> ResolvedSingleStateMonteCarloInputs:
    """Resolve one bounded single-state Monte Carlo contract from semantics."""
    resolved_base = resolve_single_state_diffusion_inputs(market_state, spec)
    resolved_scheme = _normalized_scheme(
        scheme if scheme is not None else getattr(spec, "scheme", "exact")
    )
    resolved_variance_reduction = _normalized_variance_reduction(
        variance_reduction
        if variance_reduction is not None
        else getattr(spec, "variance_reduction", "none")
    )

    return ResolvedSingleStateMonteCarloInputs(
        **resolved_base.__dict__,
        n_paths=max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 50_000)), 1),
        n_steps=max(int(n_steps if n_steps is not None else getattr(spec, "n_steps", 252)), 1),
        seed=seed if seed is not None else getattr(spec, "seed", 42),
        scheme=resolved_scheme,
        variance_reduction=resolved_variance_reduction,
    )


def build_single_state_terminal_claim_monte_carlo_problem_from_resolved(
    resolved: ResolvedSingleStateMonteCarloInputs,
    *,
    terminal_payoff: Callable[[raw_np.ndarray, ResolvedSingleStateMonteCarloInputs], raw_np.ndarray],
    process_family: str = "gbm_1d",
    local_vol_surface: Callable | None = None,
) -> EventAwareMonteCarloProblem:
    """Build the event-aware runtime problem for one single-state terminal claim."""
    return build_event_aware_monte_carlo_problem(
        EventAwareMonteCarloProblemSpec(
            process_spec=EventAwareMonteCarloProcessSpec(
                family=str(process_family).strip(),
                risk_free_rate=resolved.rate,
                dividend_yield=resolved.dividend_yield,
                sigma=resolved.sigma,
                local_vol_surface=local_vol_surface,
                simulation_method="exact" if resolved.scheme == "log_euler" else resolved.scheme,
            ),
            initial_state=resolved.spot,
            maturity=resolved.maturity,
            n_steps=resolved.n_steps,
            discount_rate=resolved.rate,
            path_requirement_kind="terminal_only",
            reducer_kind="terminal_payoff",
            terminal_payoff=lambda terminal: terminal_payoff(terminal, resolved),
        )
    )


def build_single_state_terminal_claim_monte_carlo_problem(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    terminal_payoff: Callable[[raw_np.ndarray, ResolvedSingleStateMonteCarloInputs], raw_np.ndarray],
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    process_family: str = "gbm_1d",
    local_vol_surface: Callable | None = None,
) -> tuple[ResolvedSingleStateMonteCarloInputs, EventAwareMonteCarloProblem]:
    """Resolve one single-state terminal claim and build its event-aware MC problem."""
    resolved = resolve_single_state_terminal_claim_monte_carlo_inputs(
        market_state,
        spec,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    problem = build_single_state_terminal_claim_monte_carlo_problem_from_resolved(
        resolved,
        terminal_payoff=terminal_payoff,
        process_family=process_family,
        local_vol_surface=local_vol_surface,
    )
    return resolved, problem


def price_single_state_terminal_claim_monte_carlo_result(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
    *,
    terminal_payoff: Callable[[raw_np.ndarray, ResolvedSingleStateMonteCarloInputs], raw_np.ndarray],
    scheme: str | None = None,
    variance_reduction: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    process_family: str = "gbm_1d",
    local_vol_surface: Callable | None = None,
    control_variate_values: Callable[[raw_np.ndarray, ResolvedSingleStateMonteCarloInputs], raw_np.ndarray]
    | None = None,
    control_variate_expected: Callable[[ResolvedSingleStateMonteCarloInputs], float] | None = None,
) -> SingleStateMonteCarloResult:
    """Price one bounded single-state terminal claim through the generic MC family helper."""
    resolved, problem = build_single_state_terminal_claim_monte_carlo_problem(
        market_state,
        spec,
        terminal_payoff=terminal_payoff,
        scheme=scheme,
        variance_reduction=variance_reduction,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        process_family=process_family,
        local_vol_surface=local_vol_surface,
    )

    if resolved.maturity <= 0.0:
        intrinsic = terminal_payoff(raw_np.asarray([resolved.spot], dtype=float), resolved)[0]
        return SingleStateMonteCarloResult(
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
        return SingleStateMonteCarloResult(
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
        if control_variate_values is None or control_variate_expected is None:
            raise ValueError(
                "control variate pricing requires control_variate_values and control_variate_expected"
            )
        paths = engine.simulate(problem.initial_state, problem.maturity)
        terminal = raw_np.asarray(paths[:, -1], dtype=float)
        discounted_payoffs = (
            raw_np.exp(-resolved.rate * resolved.maturity)
            * resolved.notional
            * terminal_payoff(terminal, resolved)
        )
        control_values = raw_np.asarray(control_variate_values(terminal, resolved), dtype=float)
        result = control_variate(
            discounted_payoffs,
            control_values,
            float(control_variate_expected(resolved)),
        )
        return SingleStateMonteCarloResult(
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
        payoffs = resolved.notional * terminal_payoff(terminal, resolved)
        half = resolved.n_paths // 2
        averaged = 0.5 * (payoffs[:half] + payoffs[half:])
        discounted = raw_np.exp(-resolved.rate * resolved.maturity) * averaged
        return SingleStateMonteCarloResult(
            price=float(raw_np.mean(discounted)),
            std_error=float(raw_np.std(discounted) / raw_np.sqrt(half)),
            n_paths=resolved.n_paths,
            scheme=resolved.scheme,
            variance_reduction=resolved.variance_reduction,
        )

    result = engine.price(
        problem.initial_state,
        problem.maturity,
        lambda paths: resolved.notional * terminal_payoff(paths[:, -1], resolved),
        discount_rate=problem.discount_rate,
        storage_policy=problem.path_requirement,
        return_paths=False,
    )
    return SingleStateMonteCarloResult(
        price=float(result["price"]),
        std_error=float(result["std_error"]),
        n_paths=int(result["n_paths"]),
        scheme=resolved.scheme,
        variance_reduction=resolved.variance_reduction,
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


__all__ = [
    "ResolvedSingleStateMonteCarloInputs",
    "SingleStateMonteCarloResult",
    "build_single_state_terminal_claim_monte_carlo_problem",
    "build_single_state_terminal_claim_monte_carlo_problem_from_resolved",
    "price_single_state_terminal_claim_monte_carlo_result",
    "resolve_single_state_terminal_claim_monte_carlo_inputs",
]
