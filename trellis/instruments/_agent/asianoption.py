"""Compatibility Asian-option adapter composed from reusable primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import StateAwarePayoff
from trellis.models.observation_aggregation import (
    WeightedObservationContract,
    weighted_observation_payoff,
)
from trellis.models.processes.gbm import GBM
from trellis.models.resolution.single_state_diffusion import (
    resolve_single_state_diffusion_inputs,
)


@dataclass(frozen=True)
class AsianOptionSpec:
    """Legacy Asian-option compatibility spec."""

    notional: float
    spot: float
    strike: float
    expiry_date: date
    averaging_type: str = "arithmetic"
    option_type: str = "call"
    n_observations: int = 12
    observation_dates: tuple[date, ...] = ()
    dividend_yield: float = 0.0
    n_paths: int = 50_000
    n_steps: int | None = None
    max_grid_steps: int = 4096
    seed: int | None = 42
    day_count: DayCountConvention = DayCountConvention.ACT_365


class AsianOptionPayoff:
    """Compatibility payoff that demonstrates explicit arithmetic averaging."""

    def __init__(self, spec: AsianOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> AsianOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        averaging_type = str(spec.averaging_type or "arithmetic").strip().lower()
        if averaging_type != "arithmetic":
            raise ValueError(
                "Asian compatibility adapter currently requires arithmetic averaging"
            )

        resolved = resolve_single_state_diffusion_inputs(market_state, spec)
        legacy_scale = resolved.notional / resolved.spot
        if resolved.maturity <= 0.0:
            intrinsic = (
                max(resolved.strike - resolved.spot, 0.0)
                if resolved.option_type == "put"
                else max(resolved.spot - resolved.strike, 0.0)
            )
            return float(legacy_scale * intrinsic)

        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError("Asian compatibility adapter requires settlement or as_of")
        observation_dates = tuple(spec.observation_dates or ())
        if observation_dates:
            if tuple(sorted(observation_dates)) != observation_dates:
                raise ValueError("Asian observation dates must be increasing")
            if observation_dates[-1] != spec.expiry_date:
                raise ValueError("Asian final observation must equal expiry_date")
            observation_times = tuple(
                float(year_fraction(settlement, observation_date, spec.day_count))
                for observation_date in observation_dates
            )
            if any(time <= 0.0 for time in observation_times):
                raise ValueError("Asian observations must be after settlement")
        else:
            observation_count = int(spec.n_observations)
            if observation_count <= 0:
                raise ValueError(
                    "Asian compatibility adapter requires observation_dates or "
                    "positive n_observations"
                )
            if observation_count == 1:
                observation_times = (0.0,)
            else:
                observation_times = tuple(
                    resolved.maturity * index / (observation_count - 1)
                    for index in range(observation_count)
                )

        observation_count = len(observation_times)
        observation_contract = WeightedObservationContract(
            observation_times=observation_times,
            weights=(1.0 / observation_count,) * observation_count,
        )
        configured_steps = spec.n_steps
        if configured_steps is None and not observation_dates:
            configured_steps = max(observation_count - 1, 1)
        n_steps = observation_contract.resolve_uniform_grid_steps(
            maturity=resolved.maturity,
            n_steps=(
                None
                if configured_steps is None
                else max(int(configured_steps), 1)
            ),
            min_steps=max(observation_count - 1, 1),
            max_steps=int(spec.max_grid_steps),
        )

        np = get_numpy()
        if resolved.option_type == "put":
            def settlement_fn(average):
                return np.maximum(resolved.strike - average, 0.0)
        else:
            def settlement_fn(average):
                return np.maximum(average - resolved.strike, 0.0)
        payoff: StateAwarePayoff = weighted_observation_payoff(
            observation_contract,
            maturity=resolved.maturity,
            n_steps=n_steps,
            settlement_fn=settlement_fn,
            reducer_name="arithmetic_average",
            name="arithmetic_asian_payoff",
        )
        engine = MonteCarloEngine(
            GBM(
                mu=resolved.rate - resolved.dividend_yield,
                sigma=resolved.sigma,
            ),
            n_paths=max(int(spec.n_paths), 2),
            n_steps=n_steps,
            seed=spec.seed,
            method="exact",
        )
        result = engine.price(
            resolved.spot,
            resolved.maturity,
            payoff,
            discount_rate=resolved.rate,
            return_paths=False,
        )
        return float(legacy_scale * result["price"])
