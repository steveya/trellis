"""Generic Monte Carlo payoff adapter for ranked-observation basket routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.payoff import MonteCarloPathPayoff
from trellis.core.runtime_contract import ContractState, ResolvedInputs, RuntimeContext
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.analytical.support import normalized_option_type
from trellis.models.monte_carlo.basket_state import (
    build_basket_path_requirement,
    evaluate_ranked_observation_basket_paths,
    observation_step_indices,
)
from trellis.models.monte_carlo.ranked_observation_payoffs import (
    build_ranked_observation_basket_initial_state,
    build_ranked_observation_basket_process,
    build_ranked_observation_basket_state_payoff,
    recommended_ranked_observation_basket_mc_engine_kwargs,
)
from trellis.models.resolution.basket_semantics import (
    ResolvedBasketSemantics,
    resolve_basket_semantics,
)

np = get_numpy()


@dataclass(frozen=True)
class RankedObservationBasketSpec:
    """Specification for a ranked-observation basket option."""

    notional: float
    strike: float
    expiry_date: date
    constituents: str
    observation_dates: tuple[date, ...] | None = None
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
    initial_state: ContractState = field(default_factory=ContractState)
    resolved_inputs: ResolvedInputs = field(
        default_factory=lambda: ResolvedInputs(bindings={})
    )
    runtime_context: RuntimeContext = field(default_factory=RuntimeContext)


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
        payoff = self.pathwise_payoff(constant_paths, resolved)[0]
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
        contract = self.path_contract(resolved)
        normalized = self.normalize_paths(paths)
        constituent_spots = tuple(contract.resolved_inputs.require("constituent_spots"))
        if normalized.shape[-1] != len(constituent_spots):
            raise ValueError(
                "Expected joint basket paths aligned with constituent_spots; "
                f"got shape {normalized.shape} for {len(constituent_spots)} constituents."
            )
        aggregate = evaluate_ranked_observation_basket_paths(
            normalized,
            constituent_spots,
            tuple(int(step) for step in contract.initial_state.require_memory("observation_steps")),
            selection_rule=str(contract.initial_state.require_memory("selection_rule")),
            lock_rule=str(contract.initial_state.require_memory("lock_rule")),
            aggregation_rule=str(contract.initial_state.require_memory("aggregation_rule")),
            selection_count=int(contract.initial_state.require_memory("selection_count")),
        )
        strike = float(contract.resolved_inputs.require("strike"))
        option_type = normalized_option_type(
            str(contract.resolved_inputs.require("option_type"))
        )
        if option_type == "put":
            return np.maximum(strike - aggregate, 0.0)
        return np.maximum(aggregate - strike, 0.0)

    def evaluate_from_resolved(self, resolved: ResolvedBasketSemantics) -> float:
        state_payoff = build_ranked_observation_basket_state_payoff(self.spec, resolved)
        process = self.build_process(resolved)
        engine = self.build_engine(process, resolved)
        price_result = engine.price(
            self.build_initial_state(resolved),
            self.time_horizon(resolved),
            state_payoff,
            discount_rate=0.0,
            storage_policy=state_payoff.path_requirement,
            return_paths=False,
        )
        return (
            float(self.spec.notional)
            * float(resolved.domestic_df)
            * float(price_result["price"])
        )


def price_ranked_observation_basket_monte_carlo(
    spec: RankedObservationBasketSpec,
    resolved: ResolvedBasketSemantics,
) -> float:
    """Price a ranked-observation basket option through the generic helper path."""
    return float(RankedObservationBasketMonteCarloPayoff(spec).evaluate_from_resolved(resolved))


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
    observation_steps_tuple = tuple(int(item) for item in observation_steps)
    return RankedObservationBasketPathContract(
        observation_dates=tuple(resolved.observation_dates),
        observation_times=tuple(float(item) for item in resolved.observation_times),
        observation_steps=observation_steps_tuple,
        snapshot_steps=tuple(int(item) for item in requirement.snapshot_steps),
        initial_state=ContractState(
            event_state={"path_requirement_kind": "observation_snapshot_state"},
            contract_memory={
                "selection_rule": resolved.selection_rule,
                "lock_rule": resolved.lock_rule,
                "aggregation_rule": resolved.aggregation_rule,
                "selection_count": int(resolved.selection_count),
                "observation_steps": observation_steps_tuple,
            },
            phase="observation",
            metadata={"event_kinds": ("observation", "settlement")},
        ),
        resolved_inputs=ResolvedInputs(
            bindings={
                "constituent_names": tuple(resolved.constituent_names),
                "constituent_spots": tuple(float(item) for item in resolved.constituent_spots),
                "constituent_vols": tuple(float(item) for item in resolved.constituent_vols),
                "constituent_carry": tuple(float(item) for item in resolved.constituent_carry),
                "correlation_matrix": tuple(
                    tuple(float(cell) for cell in row)
                    for row in resolved.correlation_matrix
                ),
                "domestic_df": float(resolved.domestic_df),
                "observation_times": tuple(float(item) for item in resolved.observation_times),
                "strike": float(spec.strike),
                "option_type": str(spec.option_type),
            },
            requirements=("spot", "black_vol_surface", "discount_curve", "model_parameters"),
            source_kind="basket_semantics",
            metadata={"path_requirement_kind": "observation_snapshot_state"},
        ),
        runtime_context=RuntimeContext(
            phase="observation",
            schedule_role="observation_dates",
            metadata={"observation_count": len(resolved.observation_dates)},
        ),
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
