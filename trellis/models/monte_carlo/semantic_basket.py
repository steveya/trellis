"""Generic Monte Carlo payoff adapter for ranked-observation basket routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import MonteCarloPathPayoff
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.monte_carlo.basket_state import build_basket_path_requirement, observation_step_indices
from trellis.models.monte_carlo.ranked_observation_payoffs import (
    build_ranked_observation_basket_initial_state,
    build_ranked_observation_basket_process,
    build_ranked_observation_basket_state_payoff,
    price_ranked_observation_basket_monte_carlo as _price_ranked_observation_basket_monte_carlo,
    recommended_ranked_observation_basket_mc_engine_kwargs,
    terminal_ranked_observation_basket_payoff,
)
from trellis.models.resolution.basket_semantics import (
    ResolvedBasketSemantics,
    resolve_basket_semantics,
)


@dataclass(frozen=True)
class RankedObservationBasketSpec:
    """Specification for a ranked-observation basket option."""

    notional: float
    strike: float
    expiry_date: date
    constituents: str
    observation_dates: str | None = None
    selection_rule: str = "best_of_remaining"
    lock_rule: str = "remove_selected"
    aggregation_rule: str = "average_locked_returns"
    option_type: str = "call"
    observation_frequency: Frequency = Frequency.QUARTERLY
    selection_count: int = 1
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 42
    mc_method: str = "exact"
    correlation_matrix_key: str | None = None


@dataclass(frozen=True)
class RankedObservationBasketPathContract:
    """Typed snapshot/path-state contract for the ranked-observation basket helper."""

    observation_dates: tuple[date, ...]
    observation_times: tuple[float, ...]
    observation_steps: tuple[int, ...]
    snapshot_steps: tuple[int, ...]
    state_tags: tuple[str, ...] = ("pathwise_only", "remaining_pool", "locked_cashflow_state")
    event_kinds: tuple[str, ...] = ("observation", "settlement")
    path_requirement_kind: str = "observation_snapshot_state"


class RankedObservationBasketMonteCarloPayoff(
    MonteCarloPathPayoff[RankedObservationBasketSpec, ResolvedBasketSemantics]
):
    """Deterministic thin adapter over the generic basket Monte Carlo route."""

    @property
    def requirements(self) -> set[str]:
        return {
            "discount_curve",
            "spot",
            "black_vol_surface",
            "model_parameters",
        }

    def resolve_inputs(self, market_state: MarketState) -> ResolvedBasketSemantics:
        return resolve_basket_semantics(market_state, self.spec)

    def evaluate_at_expiry(self, resolved: ResolvedBasketSemantics) -> float:
        constant_paths = build_ranked_observation_basket_initial_state(resolved)[None, None, :]
        payoff = terminal_ranked_observation_basket_payoff(self.spec, constant_paths, resolved)[0]
        return float(self.spec.notional) * float(payoff)

    def build_process(self, resolved: ResolvedBasketSemantics):
        return build_ranked_observation_basket_process(resolved)

    def build_initial_state(self, resolved: ResolvedBasketSemantics):
        return build_ranked_observation_basket_initial_state(resolved)

    def engine_kwargs(self, resolved: ResolvedBasketSemantics) -> dict[str, object]:
        return recommended_ranked_observation_basket_mc_engine_kwargs(self.spec, resolved)

    def path_contract(self, resolved: ResolvedBasketSemantics) -> RankedObservationBasketPathContract:
        """Return the typed snapshot/path-state contract consumed by this route."""
        return build_ranked_observation_basket_path_contract(self.spec, resolved)

    def pathwise_payoff(self, paths, resolved: ResolvedBasketSemantics):
        normalized = self.normalize_paths(paths)
        if normalized.shape[-1] < 1:
            raise ValueError(
                f"Expected joint basket paths with state_dim >= 1; got {normalized.shape}."
            )
        return terminal_ranked_observation_basket_payoff(self.spec, normalized, resolved)

    def evaluate_from_resolved(self, resolved: ResolvedBasketSemantics) -> float:
        return float(price_ranked_observation_basket_monte_carlo(self.spec, resolved))


def price_ranked_observation_basket_monte_carlo(
    spec: RankedObservationBasketSpec,
    resolved: ResolvedBasketSemantics,
) -> float:
    """Price a ranked-observation basket option through the generic helper path."""
    return float(_price_ranked_observation_basket_monte_carlo(spec, resolved))


def build_ranked_observation_basket_path_contract(
    spec: RankedObservationBasketSpec,
    resolved: ResolvedBasketSemantics,
) -> RankedObservationBasketPathContract:
    """Return the typed snapshot/path-state contract for the basket MC helper."""
    engine_kwargs = recommended_ranked_observation_basket_mc_engine_kwargs(spec, resolved)
    n_steps = int(engine_kwargs["n_steps"])
    observation_steps = observation_step_indices(
        resolved.observation_times,
        resolved.T,
        n_steps,
    )
    requirement = build_basket_path_requirement(
        resolved.observation_times,
        resolved.T,
        n_steps,
    )
    return RankedObservationBasketPathContract(
        observation_dates=tuple(resolved.observation_dates),
        observation_times=tuple(float(item) for item in resolved.observation_times),
        observation_steps=tuple(int(item) for item in observation_steps),
        snapshot_steps=tuple(int(item) for item in requirement.snapshot_steps),
    )


__all__ = [
    "RankedObservationBasketPathContract",
    "RankedObservationBasketMonteCarloPayoff",
    "RankedObservationBasketSpec",
    "build_ranked_observation_basket_path_contract",
    "build_ranked_observation_basket_initial_state",
    "build_ranked_observation_basket_process",
    "build_ranked_observation_basket_state_payoff",
    "price_ranked_observation_basket_monte_carlo",
    "recommended_ranked_observation_basket_mc_engine_kwargs",
    "terminal_ranked_observation_basket_payoff",
]
