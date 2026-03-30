"""Deterministic quanto-option Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import MonteCarloPathPayoff
from trellis.core.types import DayCountConvention
from trellis.models.monte_carlo.quanto import (
    build_quanto_mc_initial_state,
    build_quanto_mc_process,
    recommended_quanto_mc_engine_kwargs,
    terminal_quanto_option_payoff,
)
from trellis.models.resolution.quanto import ResolvedQuantoInputs, resolve_quanto_inputs


REQUIREMENTS = frozenset(
    {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "fx_rates",
        "spot",
        "model_parameters",
    }
)


@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for a single-underlier quanto option."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 42
    mc_method: str = "exact"


def _intrinsic_value(option_type: str, spot: float, strike: float) -> float:
    normalized_type = option_type.lower()
    if normalized_type == "call":
        return max(float(spot) - float(strike), 0.0)
    if normalized_type == "put":
        return max(float(strike) - float(spot), 0.0)
    raise ValueError(
        f"Unsupported option_type {option_type!r}; expected 'call' or 'put'"
    )


class QuantoOptionMonteCarloPayoff(
    MonteCarloPathPayoff[QuantoOptionSpec, ResolvedQuantoInputs]
):
    """Deterministic thin adapter over the correlated-GBM quanto MC route."""

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def resolve_inputs(self, market_state: MarketState) -> ResolvedQuantoInputs:
        return resolve_quanto_inputs(market_state, self.spec)

    def evaluate_at_expiry(self, resolved: ResolvedQuantoInputs) -> float:
        return float(self.spec.notional) * _intrinsic_value(
            self.spec.option_type,
            resolved.spot,
            float(self.spec.strike),
        )

    def build_process(self, resolved: ResolvedQuantoInputs):
        return build_quanto_mc_process(resolved)

    def build_initial_state(self, resolved: ResolvedQuantoInputs):
        return build_quanto_mc_initial_state(resolved)

    def engine_kwargs(self, resolved: ResolvedQuantoInputs) -> dict[str, object]:
        return recommended_quanto_mc_engine_kwargs(self.spec, resolved)

    def pathwise_payoff(self, paths, resolved: ResolvedQuantoInputs):
        normalized = self.normalize_paths(paths)
        if normalized.shape[-1] < 2:
            raise ValueError(
                f"Expected joint underlier/FX paths with state_dim >= 2; got {normalized.shape}."
            )
        return terminal_quanto_option_payoff(self.spec, normalized)
