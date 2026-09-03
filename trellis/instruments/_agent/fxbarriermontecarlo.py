"""Agent-generated payoff: Build a pricer for: Trellis extension: knock-in FX vanilla option

Price a down_and_in FX option on EURUSD with domestic/foreign discounting.

Currency pair: EURUSD.
Option type: call.
Strike: 1.1.
Spot: 1.1.
Barrier: 1.02.
Barrier type: down_and_in.
Expiry date: 2025-11-15.
Notional: 1000000.0.
Monitoring: discrete (252 observations/year).
Rebate: 0.0.
Monte Carlo controls: n_paths=120000, n_steps=252, seed=42.
Foreign discount key: EUR-DISC.

Benchmark product: fx_barrier_option

Construct methods: analytical, monte_carlo
Comparison targets: analytical (analytical), monte_carlo (monte_carlo)

Implementation target: monte_carlo
Preferred method family: monte_carlo
Typed comparison-target contract: {"backend_binding_id":"","contract_id":"comparison-target:P003:monte_carlo:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"monte_carlo","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"monte_carlo","validation_bundle_id":"","variant_parameters":{}}
The executable must exercise every declared variant and spec override. Declare the exercised artifact contract using the canonical shape __trellis_comparison_bindings__ = {"monte_carlo":{"target_contract":{"backend_binding_id":"","contract_id":"comparison-target:P003:monte_carlo:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"monte_carlo","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"monte_carlo","validation_bundle_id":"","variant_parameters":{}}}}.

Implementation target: monte_carlo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import PricingValue
from trellis.core.types import DayCountConvention
from trellis.models.analytical import terminal_intrinsic
from trellis.models.fx_barrier_option import resolve_fx_barrier_inputs
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import BarrierMonitor, MonteCarloPathRequirement, StateAwarePayoff
from trellis.models.processes.gbm import GBM
import numpy as raw_np



@dataclass(frozen=True)
class FXBarrierOptionSpec:
    """Specification for Build a pricer for: Trellis extension: knock-in FX vanilla option

Price a down_and_in FX option on EURUSD with domestic/foreign discounting.

Currency pair: EURUSD.
Option type: call.
Strike: 1.1.
Spot: 1.1.
Barrier: 1.02.
Barrier type: down_and_in.
Expiry date: 2025-11-15.
Notional: 1000000.0.
Monitoring: discrete (252 observations/year).
Rebate: 0.0.
Monte Carlo controls: n_paths=120000, n_steps=252, seed=42.
Foreign discount key: EUR-DISC.

Benchmark product: fx_barrier_option

Construct methods: analytical, monte_carlo
Comparison targets: analytical (analytical), monte_carlo (monte_carlo)

Implementation target: monte_carlo
Preferred method family: monte_carlo
Typed comparison-target contract: {"backend_binding_id":"","contract_id":"comparison-target:P003:monte_carlo:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"monte_carlo","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"monte_carlo","validation_bundle_id":"","variant_parameters":{}}
The executable must exercise every declared variant and spec override. Declare the exercised artifact contract using the canonical shape __trellis_comparison_bindings__ = {"monte_carlo":{"target_contract":{"backend_binding_id":"","contract_id":"comparison-target:P003:monte_carlo:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"monte_carlo","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"monte_carlo","validation_bundle_id":"","variant_parameters":{}}}}.

Implementation target: monte_carlo."""
    notional: float
    strike: float
    barrier: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    barrier_type: str
    monitoring: str
    option_type: str = 'call'
    rebate: float = 0.0
    observations_per_year: int | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 120000
    n_steps: int = 252
    seed: int = 42


class FXBarrierMonteCarloPayoff:
    """Build a pricer for: Trellis extension: knock-in FX vanilla option

Price a down_and_in FX option on EURUSD with domestic/foreign discounting.

Currency pair: EURUSD.
Option type: call.
Strike: 1.1.
Spot: 1.1.
Barrier: 1.02.
Barrier type: down_and_in.
Expiry date: 2025-11-15.
Notional: 1000000.0.
Monitoring: discrete (252 observations/year).
Rebate: 0.0.
Monte Carlo controls: n_paths=120000, n_steps=252, seed=42.
Foreign discount key: EUR-DISC.

Benchmark product: fx_barrier_option

Construct methods: analytical, monte_carlo
Comparison targets: analytical (analytical), monte_carlo (monte_carlo)

Implementation target: monte_carlo
Preferred method family: monte_carlo
Typed comparison-target contract: {"backend_binding_id":"","contract_id":"comparison-target:P003:monte_carlo:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"monte_carlo","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"monte_carlo","validation_bundle_id":"","variant_parameters":{}}
The executable must exercise every declared variant and spec override. Declare the exercised artifact contract using the canonical shape __trellis_comparison_bindings__ = {"monte_carlo":{"target_contract":{"backend_binding_id":"","contract_id":"comparison-target:P003:monte_carlo:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"monte_carlo","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"monte_carlo","validation_bundle_id":"","variant_parameters":{}}}}.

Implementation target: monte_carlo."""

    __trellis_comparison_bindings__ = {'monte_carlo': {'target_contract': {'schema_version': 1, 'contract_id': 'comparison-target:P003:monte_carlo:v1', 'target_id': 'monte_carlo', 'method': 'monte_carlo', 'route_id': '', 'route_family': '', 'backend_binding_id': '', 'variant_parameters': {}, 'spec_overrides': {}, 'validation_bundle_id': '', 'payoff_family': '', 'exercise_style': '', 'model_family': '', 'observation_style': '', 'semantic_axes': {}, 'equivalence_group': '', 'resolution_source': 'legacy_target_inference', 'explicit': False}}}

    def __init__(self, spec: FXBarrierOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXBarrierOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates"}

    def evaluate(self, market_state: MarketState) -> PricingValue:
        spec = self._spec
        resolved = resolve_fx_barrier_inputs(market_state, self._spec)
        direction = "down" if resolved.barrier_type.startswith("down") else "up"
        knock = "in" if resolved.barrier_type.endswith("_in") else "out"
        initial_touched = (
            resolved.spot <= resolved.barrier
            if direction == "down"
            else resolved.spot >= resolved.barrier
        )
        if resolved.maturity <= 0.0:
            active = initial_touched if knock == "in" else not initial_touched
            intrinsic = terminal_intrinsic(
                resolved.option_type,
                spot=resolved.spot,
                strike=resolved.strike,
            )
            payoff_at_expiry = intrinsic if active else resolved.rebate
            return float(resolved.notional * payoff_at_expiry)

        observation_steps = ()
        if resolved.observations_per_year is not None:
            observation_count = max(
                int(round(resolved.maturity * resolved.observations_per_year)),
                1,
            )
            observation_steps = (
                0,
                *tuple(
                    sorted(
                        {
                            max(
                                1,
                                min(
                                    resolved.n_steps,
                                    int(round(index * resolved.n_steps / observation_count)),
                                ),
                            )
                            for index in range(1, observation_count + 1)
                        }
                    )
                ),
            )
        monitor = BarrierMonitor(
            name="barrier",
            level=resolved.barrier,
            direction=direction,
            observation_steps=observation_steps,
        )
        requirement = MonteCarloPathRequirement(barrier_monitors=(monitor,))

        def apply_barrier(terminal, touched):
            intrinsic = terminal_intrinsic(
                resolved.option_type,
                spot=terminal,
                strike=resolved.strike,
            )
            active = touched if knock == "in" else ~touched
            return resolved.notional * raw_np.where(active, intrinsic, resolved.rebate)

        def evaluate_paths(paths):
            observed = paths[:, observation_steps] if observation_steps else paths
            touched = (
                raw_np.any(observed <= resolved.barrier, axis=1)
                if direction == "down"
                else raw_np.any(observed >= resolved.barrier, axis=1)
            )
            return apply_barrier(paths[:, -1], touched)

        def evaluate_state(state):
            return apply_barrier(
                state.terminal_values,
                state.barrier_hit(monitor.name),
            )

        payoff = StateAwarePayoff(
            path_requirement=requirement,
            evaluate_paths_fn=evaluate_paths,
            evaluate_state_fn=evaluate_state,
            name="fx_single_barrier",
        )
        process = GBM(
            mu=resolved.domestic_rate - resolved.foreign_rate,
            sigma=resolved.sigma,
        )
        engine = MonteCarloEngine(
            process,
            n_paths=resolved.n_paths,
            n_steps=resolved.n_steps,
            seed=resolved.seed,
            method="exact",
        )
        result = engine.price(
            resolved.spot,
            resolved.maturity,
            payoff,
            discount_rate=resolved.domestic_rate,
            return_paths=False,
        )
        return float(result["price"])