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
    resolve_single_state_terminal_claim_monte_carlo_inputs,
)
from trellis.models.resolution.single_state_diffusion import (
    SingleStateDiffusionMarketStateLike,
    SingleStateDiffusionSpecLike,
    terminal_intrinsic_from_resolved,
)


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
    return resolve_single_state_terminal_claim_monte_carlo_inputs(
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
) -> tuple[ResolvedEquityMonteCarloInputs, object]:
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
