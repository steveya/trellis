"""Deterministic Himalaya Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import MonteCarloPathPayoff
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.monte_carlo.himalaya import (
    build_himalaya_mc_initial_state,
    build_himalaya_mc_process,
    recommended_himalaya_mc_engine_kwargs,
    terminal_himalaya_option_payoff,
)
from trellis.models.resolution.himalaya import ResolvedHimalayaInputs, resolve_himalaya_inputs


REQUIREMENTS = frozenset(
    {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "spot",
        "model_parameters",
    }
)


@dataclass(frozen=True)
class HimalayaOptionSpec:
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


class HimalayaOptionMonteCarloPayoff(
    MonteCarloPathPayoff[HimalayaOptionSpec, ResolvedHimalayaInputs]
):
    """Deterministic thin adapter over the correlated basket Monte Carlo route."""

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def resolve_inputs(self, market_state: MarketState) -> ResolvedHimalayaInputs:
        return resolve_himalaya_inputs(market_state, self.spec)

    def evaluate_at_expiry(self, resolved: ResolvedHimalayaInputs) -> float:
        constant_paths = build_himalaya_mc_initial_state(resolved)[None, None, :]
        payoff = terminal_himalaya_option_payoff(self.spec, constant_paths, resolved)[0]
        return float(self.spec.notional) * float(payoff)

    def build_process(self, resolved: ResolvedHimalayaInputs):
        return build_himalaya_mc_process(resolved)

    def build_initial_state(self, resolved: ResolvedHimalayaInputs):
        return build_himalaya_mc_initial_state(resolved)

    def engine_kwargs(self, resolved: ResolvedHimalayaInputs) -> dict[str, object]:
        return recommended_himalaya_mc_engine_kwargs(self.spec, resolved)

    def pathwise_payoff(self, paths, resolved: ResolvedHimalayaInputs):
        normalized = self.normalize_paths(paths)
        if normalized.shape[-1] < 1:
            raise ValueError(
                f"Expected joint basket paths with state_dim >= 1; got {normalized.shape}."
            )
        return terminal_himalaya_option_payoff(self.spec, normalized, resolved)
