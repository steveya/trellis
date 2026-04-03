"""Shared Monte Carlo helpers for ranked-observation basket routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.core.types import ContractTimeline
from trellis.core.differentiable import get_numpy
from trellis.core.payoff import MonteCarloPathPayoff
from trellis.models.analytical.support import normalized_option_type
from trellis.models.monte_carlo.basket_state import (
    build_basket_path_requirement,
    evaluate_ranked_observation_basket_paths,
    evaluate_ranked_observation_basket_state,
    observation_step_indices,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import StateAwarePayoff
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.basket_semantics import ResolvedBasketSemantics

np = get_numpy()


class RankedObservationBasketSpecLike(Protocol):
    """Minimal spec surface consumed by the shared basket Monte Carlo helper."""

    notional: float
    strike: float
    option_type: str
    expiry_date: date
    constituents: str
    observation_dates: ContractTimeline | tuple[date, ...] | None
    selection_rule: str
    lock_rule: str
    aggregation_rule: str
    observation_frequency: object
    selection_count: int
    day_count: object
    n_paths: int
    n_steps: int
    seed: int
    mc_method: str
    correlation_matrix_key: str | None


def _implied_zero_rate(discount_factor: float, T: float) -> float:
    """Convert a discount factor into a continuously compounded zero rate."""
    if T <= 0.0:
        return 0.0
    return -float(np.log(max(float(discount_factor), 1e-16))) / float(T)


def build_ranked_observation_basket_process(resolved: ResolvedBasketSemantics) -> CorrelatedGBM:
    """Build the joint correlated process used by basket Monte Carlo routes."""
    if resolved.T <= 0.0:
        raise ValueError("Basket Monte Carlo process requires positive time to expiry")

    domestic_rate = _implied_zero_rate(resolved.domestic_df, resolved.T)
    mu = [
        domestic_rate - float(carry)
        for carry in resolved.constituent_carry
    ]
    return CorrelatedGBM(
        mu=mu,
        sigma=[float(value) for value in resolved.constituent_vols],
        corr=[[float(cell) for cell in row] for row in resolved.correlation_matrix],
    )


def build_ranked_observation_basket_initial_state(resolved: ResolvedBasketSemantics):
    """Return the joint initial constituent spot vector used for simulation."""
    return np.array([float(value) for value in resolved.constituent_spots], dtype=float)


def recommended_ranked_observation_basket_mc_engine_kwargs(
    spec: RankedObservationBasketSpecLike,
    resolved: ResolvedBasketSemantics,
) -> dict[str, object]:
    """Return deterministic engine controls for the basket MC route."""
    n_paths = _spec_attr(spec, "n_paths", "n_simulations", default=50_000)
    n_steps = _spec_attr(spec, "n_steps", "n_simulation_steps", default=252)
    recommended_steps = max(
        64,
        int(np.ceil(float(resolved.T) * 252.0)),
        len(resolved.observation_times) * 16 if resolved.observation_times else 64,
    )
    return {
        "n_paths": max(int(n_paths), 4096),
        "n_steps": max(int(n_steps), recommended_steps),
        "seed": int(_spec_attr(spec, "seed", default=42)),
        "method": _spec_attr(spec, "mc_method", "method", default="exact"),
    }


def build_ranked_observation_basket_state_payoff(
    spec: RankedObservationBasketSpecLike,
    resolved: ResolvedBasketSemantics,
) -> StateAwarePayoff:
    """Return a state-aware payoff that can consume reduced basket path state."""
    engine_kwargs = recommended_ranked_observation_basket_mc_engine_kwargs(spec, resolved)
    observation_steps = observation_step_indices(
        resolved.observation_times,
        resolved.T,
        int(engine_kwargs["n_steps"]),
    )
    requirement = build_basket_path_requirement(
        resolved.observation_times,
        resolved.T,
        int(engine_kwargs["n_steps"]),
    )

    def evaluate_paths(paths):
        return terminal_ranked_observation_basket_payoff(spec, paths, resolved)

    def evaluate_state(state):
        return evaluate_ranked_observation_basket_state(
            state,
            resolved.constituent_spots,
            observation_steps,
            selection_rule=resolved.selection_rule,
            lock_rule=resolved.lock_rule,
            aggregation_rule=resolved.aggregation_rule,
            selection_count=resolved.selection_count,
        )

    return StateAwarePayoff(
        path_requirement=requirement,
        evaluate_paths_fn=evaluate_paths,
        evaluate_state_fn=evaluate_state,
        name="ranked_observation_basket_payoff",
    )


def terminal_ranked_observation_basket_payoff(
    spec: RankedObservationBasketSpecLike,
    paths,
    resolved: ResolvedBasketSemantics,
):
    """Return pathwise ranked-observation basket payoffs from joint basket paths."""
    normalized = np.asarray(paths, dtype=float)
    if normalized.ndim == 2:
        normalized = normalized[:, :, np.newaxis]
    if normalized.ndim != 3:
        raise ValueError(
            f"Expected Monte Carlo paths with rank 2 or 3; received shape {normalized.shape}."
        )

    n_paths, n_steps_plus_one, n_assets = normalized.shape
    if n_assets != len(resolved.constituent_spots):
        raise ValueError(
            f"Expected {len(resolved.constituent_spots)} constituent dimensions; got {n_assets}."
        )

    observation_steps = observation_step_indices(
        resolved.observation_times,
        resolved.T,
        n_steps_plus_one - 1,
    )
    if not observation_steps:
        observation_steps = (0,)

    aggregate = evaluate_ranked_observation_basket_paths(
        normalized,
        resolved.constituent_spots,
        observation_steps,
        selection_rule=resolved.selection_rule,
        lock_rule=resolved.lock_rule,
        aggregation_rule=resolved.aggregation_rule,
        selection_count=resolved.selection_count,
    )

    option_type = normalized_option_type(_spec_attr(spec, "option_type", default="call"))
    strike = float(_spec_attr(spec, "strike", default=0.0))
    if option_type == "put":
        return np.maximum(strike - aggregate, 0.0)
    return np.maximum(aggregate - strike, 0.0)


def price_ranked_observation_basket_monte_carlo(
    spec: RankedObservationBasketSpecLike,
    resolved: ResolvedBasketSemantics,
) -> float:
    """Price a ranked-observation basket option via joint Monte Carlo."""
    if resolved.T <= 0.0:
        intrinsic = terminal_ranked_observation_basket_payoff(
            spec,
            np.broadcast_to(
                np.asarray([float(v) for v in resolved.constituent_spots], dtype=float),
                (1, 1, len(resolved.constituent_spots)),
            ),
            resolved,
        )[0]
        return float(_spec_attr(spec, "notional", default=1.0)) * float(intrinsic)

    process = build_ranked_observation_basket_process(resolved)
    engine = MonteCarloEngine(
        process,
        **recommended_ranked_observation_basket_mc_engine_kwargs(spec, resolved),
    )
    payoff = build_ranked_observation_basket_state_payoff(spec, resolved)
    price_result = engine.price(
        build_ranked_observation_basket_initial_state(resolved),
        float(resolved.T),
        payoff,
        discount_rate=0.0,
        storage_policy=payoff.path_requirement,
        return_paths=False,
    )
    return (
        float(_spec_attr(spec, "notional", default=1.0))
        * float(resolved.domestic_df)
        * float(price_result["price"])
    )


def _spec_attr(spec: RankedObservationBasketSpecLike, *names: str, default=None):
    """Return the first available spec attribute, falling back to ``default``."""
    for name in names:
        if hasattr(spec, name):
            value = getattr(spec, name)
            if value is not None:
                return value
    return default


__all__ = [
    "build_ranked_observation_basket_initial_state",
    "build_ranked_observation_basket_process",
    "build_ranked_observation_basket_state_payoff",
    "price_ranked_observation_basket_monte_carlo",
    "recommended_ranked_observation_basket_mc_engine_kwargs",
    "terminal_ranked_observation_basket_payoff",
]
