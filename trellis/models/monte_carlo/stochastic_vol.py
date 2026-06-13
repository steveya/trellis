"""Stochastic-volatility Monte Carlo helper contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as raw_np

from trellis.models.analytical.support import terminal_intrinsic
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement
from trellis.models.monte_carlo.schemes import Euler, HestonQuadraticExponential
from trellis.models.processes.heston import HestonRuntimeBinding
from trellis.models.transforms.heston import resolve_heston_transform_inputs


@dataclass(frozen=True)
class ResolvedHestonMonteCarloInputs:
    """Resolved market, model, and numerical controls for Heston MC pricing."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    process_family: str
    initial_state: raw_np.ndarray
    correlation: float
    variance_scheme: str
    n_paths: int
    n_steps: int
    seed: int | None
    payoff_reducer: str
    validation_bundle: str
    runtime_binding: HestonRuntimeBinding


@dataclass(frozen=True)
class HestonMonteCarloProblem:
    """Executable Heston Monte Carlo problem contract."""

    process: object
    initial_state: raw_np.ndarray
    maturity: float
    n_steps: int
    discount_rate: float
    path_requirement: MonteCarloPathRequirement
    terminal_payoff: Callable[[raw_np.ndarray], raw_np.ndarray]
    scheme: object
    process_family: str
    variance_scheme: str
    payoff_reducer: str
    validation_bundle: str


@dataclass(frozen=True)
class HestonMonteCarloResult:
    """Structured result for a bounded Heston Monte Carlo price."""

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    process_family: str
    variance_scheme: str
    payoff_reducer: str
    validation_bundle: str
    model_parameters: dict[str, object]
    runtime_binding: dict[str, object]


def resolve_heston_monte_carlo_inputs(
    market_state,
    spec,
    *,
    scheme: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> ResolvedHestonMonteCarloInputs:
    """Resolve a vanilla European Heston MC pricing contract."""
    resolved = resolve_heston_transform_inputs(
        market_state,
        spec,
        method="fft",
        mu=mu,
        parameter_set_name=parameter_set_name,
    )
    variance_scheme = _normalized_heston_mc_scheme(
        scheme
        if scheme is not None
        else getattr(spec, "variance_scheme", getattr(spec, "scheme", "heston_qe"))
    )
    process = resolved.runtime_binding.process
    return ResolvedHestonMonteCarloInputs(
        notional=resolved.notional,
        spot=resolved.spot,
        strike=resolved.strike,
        maturity=resolved.maturity,
        rate=resolved.rate,
        dividend_yield=resolved.dividend_yield,
        option_type=resolved.option_type,
        process_family="heston",
        initial_state=raw_np.asarray([resolved.spot, process.v0], dtype=float),
        correlation=float(process.rho),
        variance_scheme=variance_scheme,
        n_paths=max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 50_000)), 1),
        n_steps=max(int(n_steps if n_steps is not None else getattr(spec, "n_steps", 252)), 1),
        seed=seed if seed is not None else getattr(spec, "seed", 42),
        payoff_reducer="terminal_vanilla_option",
        validation_bundle="heston:monte_carlo",
        runtime_binding=resolved.runtime_binding,
    )


def build_heston_monte_carlo_problem_from_resolved(
    resolved: ResolvedHestonMonteCarloInputs,
) -> HestonMonteCarloProblem:
    """Build an executable Heston MC problem from resolved inputs."""
    return HestonMonteCarloProblem(
        process=resolved.runtime_binding.process,
        initial_state=raw_np.asarray(resolved.initial_state, dtype=float),
        maturity=resolved.maturity,
        n_steps=resolved.n_steps,
        discount_rate=resolved.rate,
        path_requirement=MonteCarloPathRequirement.terminal_only(),
        terminal_payoff=lambda terminal_spot: _terminal_payoff(terminal_spot, resolved),
        scheme=_scheme_object(resolved.variance_scheme),
        process_family=resolved.process_family,
        variance_scheme=resolved.variance_scheme,
        payoff_reducer=resolved.payoff_reducer,
        validation_bundle=resolved.validation_bundle,
    )


def build_heston_monte_carlo_problem(
    market_state,
    spec,
    *,
    scheme: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> tuple[ResolvedHestonMonteCarloInputs, HestonMonteCarloProblem]:
    """Resolve and build one bounded Heston Monte Carlo problem."""
    resolved = resolve_heston_monte_carlo_inputs(
        market_state,
        spec,
        scheme=scheme,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mu=mu,
        parameter_set_name=parameter_set_name,
    )
    return resolved, build_heston_monte_carlo_problem_from_resolved(resolved)


def price_heston_option_monte_carlo_result(
    market_state,
    spec,
    *,
    scheme: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> HestonMonteCarloResult:
    """Return a structured Heston Monte Carlo price result."""
    resolved, problem = build_heston_monte_carlo_problem(
        market_state,
        spec,
        scheme=scheme,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mu=mu,
        parameter_set_name=parameter_set_name,
    )

    if resolved.maturity <= 0.0:
        intrinsic = _terminal_payoff(
            raw_np.asarray([resolved.spot], dtype=float),
            resolved,
        )[0]
        return _result_from_price(
            resolved,
            price=float(intrinsic),
            std_error=0.0,
            n_paths=0,
        )

    engine = MonteCarloEngine(
        problem.process,
        n_paths=resolved.n_paths,
        n_steps=resolved.n_steps,
        seed=resolved.seed,
        scheme=problem.scheme,
    )
    paths = engine.simulate(problem.initial_state, problem.maturity)
    terminal_spot = raw_np.asarray(paths[:, -1, 0], dtype=float)
    payoffs = raw_np.asarray(problem.terminal_payoff(terminal_spot), dtype=float)
    discounted = raw_np.exp(-problem.discount_rate * problem.maturity) * payoffs
    return _result_from_price(
        resolved,
        price=float(raw_np.mean(discounted)),
        std_error=float(raw_np.std(discounted) / raw_np.sqrt(len(discounted))),
        n_paths=resolved.n_paths,
    )


def price_heston_option_monte_carlo(
    market_state,
    spec,
    *,
    scheme: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> float:
    """Return the scalar Heston Monte Carlo present value."""
    return float(
        price_heston_option_monte_carlo_result(
            market_state,
            spec,
            scheme=scheme,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            mu=mu,
            parameter_set_name=parameter_set_name,
        ).price
    )


def _result_from_price(
    resolved: ResolvedHestonMonteCarloInputs,
    *,
    price: float,
    std_error: float,
    n_paths: int,
) -> HestonMonteCarloResult:
    runtime_payload = resolved.runtime_binding.to_payload()
    return HestonMonteCarloResult(
        price=float(price),
        std_error=float(std_error),
        n_paths=int(n_paths),
        n_steps=int(resolved.n_steps),
        process_family=resolved.process_family,
        variance_scheme=resolved.variance_scheme,
        payoff_reducer=resolved.payoff_reducer,
        validation_bundle=resolved.validation_bundle,
        model_parameters=dict(resolved.runtime_binding.model_parameters),
        runtime_binding=runtime_payload,
    )


def _terminal_payoff(
    terminal_spot: raw_np.ndarray,
    resolved: ResolvedHestonMonteCarloInputs,
) -> raw_np.ndarray:
    return raw_np.asarray(
        resolved.notional
        * terminal_intrinsic(
            resolved.option_type,
            spot=raw_np.asarray(terminal_spot, dtype=float),
            strike=resolved.strike,
        ),
        dtype=float,
    )


def _normalized_heston_mc_scheme(value: object) -> str:
    scheme = str(value or "heston_qe").strip().lower().replace("-", "_")
    aliases = {
        "qe": "heston_qe",
        "qe_heston": "heston_qe",
        "andersen_qe": "heston_qe",
        "quadratic_exponential": "heston_qe",
        "heston_quadratic_exponential": "heston_qe",
        "euler_heston": "euler",
        "heston_euler": "euler",
        "heston_mc": "euler",
        "mc_heston": "euler",
    }
    scheme = aliases.get(scheme, scheme)
    if scheme not in {"euler", "heston_qe"}:
        raise ValueError(f"Unsupported Heston Monte Carlo scheme {value!r}")
    return scheme


def _scheme_object(variance_scheme: str):
    if variance_scheme == "heston_qe":
        return HestonQuadraticExponential()
    return Euler()


__all__ = [
    "HestonMonteCarloProblem",
    "HestonMonteCarloResult",
    "ResolvedHestonMonteCarloInputs",
    "build_heston_monte_carlo_problem",
    "build_heston_monte_carlo_problem_from_resolved",
    "price_heston_option_monte_carlo",
    "price_heston_option_monte_carlo_result",
    "resolve_heston_monte_carlo_inputs",
]
