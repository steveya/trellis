"""Deterministic quanto-option Monte Carlo payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import (
    implied_zero_rate,
    normalized_option_type,
    terminal_intrinsic,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import terminal_value_payoff
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.quanto import resolve_quanto_inputs


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
    underlier_id: str | None = None
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    underlier_vol_surface_key: str | None = None
    fx_vol_surface_key: str | None = None
    option_type: str = "call"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 42
    mc_method: str = "exact"


class QuantoOptionMonteCarloPayoff:
    """Quanto MC adapter composed from generic correlated-state primitives."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def evaluate(self, market_state: MarketState) -> float:
        resolved = resolve_quanto_inputs(market_state, self._spec)
        option_type = normalized_option_type(self._spec.option_type)
        if resolved.T <= 0.0:
            return float(self._spec.notional) * float(
                terminal_intrinsic(
                    option_type,
                    spot=resolved.spot,
                    strike=self._spec.strike,
                )
            )

        domestic_rate = float(implied_zero_rate(resolved.domestic_df, resolved.T))
        foreign_rate = float(implied_zero_rate(resolved.foreign_df, resolved.T))
        underlier_drift = (
            domestic_rate
            - foreign_rate
            - resolved.corr * resolved.sigma_underlier * resolved.sigma_fx
        )
        process = CorrelatedGBM(
            mu=[underlier_drift, domestic_rate - foreign_rate],
            sigma=[resolved.sigma_underlier, resolved.sigma_fx],
            corr=[[1.0, resolved.corr], [resolved.corr, 1.0]],
        )
        payoff = terminal_value_payoff(
            lambda terminal: float(self._spec.notional)
            * terminal_intrinsic(
                option_type,
                spot=terminal[..., 0],
                strike=self._spec.strike,
            ),
            name="quanto_terminal",
        )
        engine = MonteCarloEngine(
            process,
            n_paths=max(int(self._spec.n_paths), 1),
            n_steps=max(int(self._spec.n_steps), 1),
            seed=int(self._spec.seed),
            method="exact",
        )
        result = engine.price(
            get_numpy().array([resolved.spot, resolved.fx_spot], dtype=float),
            float(resolved.T),
            payoff,
            discount_rate=domestic_rate,
            return_paths=False,
        )
        return float(result["price"])
