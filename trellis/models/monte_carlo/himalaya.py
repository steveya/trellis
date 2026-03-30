"""Compatibility wrapper around the generic basket Monte Carlo helpers."""

from __future__ import annotations

from trellis.models.monte_carlo.ranked_observation_payoffs import (
    build_ranked_observation_basket_initial_state as build_himalaya_mc_initial_state,
    build_ranked_observation_basket_process as build_himalaya_mc_process,
    price_ranked_observation_basket_monte_carlo as price_himalaya_option_monte_carlo,
    recommended_ranked_observation_basket_mc_engine_kwargs as recommended_himalaya_mc_engine_kwargs,
    terminal_ranked_observation_basket_payoff as terminal_himalaya_option_payoff,
)
from trellis.models.resolution.basket_semantics import (
    BasketSpecLike as HimalayaMonteCarloSpecLike,
    ResolvedBasketSemantics as ResolvedHimalayaInputs,
)

__all__ = [
    "HimalayaMonteCarloSpecLike",
    "ResolvedHimalayaInputs",
    "build_himalaya_mc_initial_state",
    "build_himalaya_mc_process",
    "price_himalaya_option_monte_carlo",
    "recommended_himalaya_mc_engine_kwargs",
    "terminal_himalaya_option_payoff",
]
