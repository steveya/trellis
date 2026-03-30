"""Shared Monte Carlo helpers for single-name quanto routes."""

from __future__ import annotations

from typing import Protocol

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.quanto import ResolvedQuantoInputs

np = get_numpy()


class QuantoMonteCarloSpecLike(Protocol):
    """Minimal spec surface consumed by the shared quanto Monte Carlo helper."""

    notional: float
    strike: float
    option_type: str
    n_paths: int
    n_steps: int
    seed: int


def _implied_zero_rate(discount_factor: float, T: float) -> float:
    """Convert a discount factor into a continuously compounded zero rate."""
    if T <= 0.0:
        return 0.0
    return -float(np.log(max(float(discount_factor), 1e-16))) / float(T)


def build_quanto_mc_process(resolved: ResolvedQuantoInputs) -> CorrelatedGBM:
    """Build the joint underlier/FX process for a quanto Monte Carlo route."""
    if resolved.T <= 0.0:
        raise ValueError("Quanto Monte Carlo process requires positive time to expiry")

    domestic_rate = _implied_zero_rate(resolved.domestic_df, resolved.T)
    foreign_rate = _implied_zero_rate(resolved.foreign_df, resolved.T)
    mu_underlier = (
        domestic_rate
        - foreign_rate
        - float(resolved.corr * resolved.sigma_underlier * resolved.sigma_fx)
    )
    return CorrelatedGBM(
        mu=[mu_underlier, domestic_rate],
        sigma=[float(resolved.sigma_underlier), float(resolved.sigma_fx)],
        corr=[
            [1.0, float(resolved.corr)],
            [float(resolved.corr), 1.0],
        ],
    )


def build_quanto_mc_initial_state(resolved: ResolvedQuantoInputs):
    """Return the joint initial underlier/FX state used for simulation."""
    return np.array([float(resolved.spot), float(resolved.fx_spot)], dtype=float)


def recommended_quanto_mc_engine_kwargs(
    spec: QuantoMonteCarloSpecLike,
    resolved: ResolvedQuantoInputs,
) -> dict[str, object]:
    """Return deterministic engine controls for shared quanto MC routes."""
    recommended_steps = max(100, int(np.ceil(float(resolved.T) * 252.0)))
    return {
        "n_paths": max(int(spec.n_paths), 4096),
        "n_steps": max(int(spec.n_steps), recommended_steps),
        "seed": int(getattr(spec, "seed", 42)),
        "method": "exact",
    }


def terminal_quanto_option_payoff(spec: QuantoMonteCarloSpecLike, paths):
    """Return pathwise terminal intrinsic values from joint underlier/FX paths."""
    normalized = np.asarray(paths, dtype=float)
    if normalized.ndim == 2:
        terminal_spot = normalized[:, -1]
    elif normalized.ndim == 3 and normalized.shape[-1] >= 1:
        terminal_spot = normalized[:, -1, 0]
    else:
        raise ValueError(
            f"Expected Monte Carlo paths with shape (n_paths, n_steps+1[, state_dim]); got {normalized.shape}."
        )

    strike = float(spec.strike)
    option_type = str(spec.option_type).lower()
    if option_type == "put":
        return np.maximum(strike - terminal_spot, 0.0)
    if option_type == "call":
        return np.maximum(terminal_spot - strike, 0.0)
    raise ValueError(
        f"Unsupported option_type {spec.option_type!r}; expected 'call' or 'put'"
    )


def price_quanto_option_monte_carlo(
    spec: QuantoMonteCarloSpecLike,
    resolved: ResolvedQuantoInputs,
) -> float:
    """Price a single-underlier quanto option via joint underlier/FX Monte Carlo."""
    if resolved.T <= 0.0:
        intrinsic = terminal_quanto_option_payoff(
            spec,
            np.asarray([[float(resolved.spot)]], dtype=float),
        )[0]
        return float(spec.notional) * float(intrinsic)

    process = build_quanto_mc_process(resolved)
    engine = MonteCarloEngine(
        process,
        **recommended_quanto_mc_engine_kwargs(spec, resolved),
    )
    paths = engine.simulate(
        build_quanto_mc_initial_state(resolved),
        float(resolved.T),
    )
    payoff_samples = terminal_quanto_option_payoff(spec, paths)
    return (
        float(spec.notional)
        * float(resolved.domestic_df)
        * float(np.mean(np.asarray(payoff_samples, dtype=float)))
    )
