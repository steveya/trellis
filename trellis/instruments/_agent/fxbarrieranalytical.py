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

Implementation target: analytical
Preferred method family: analytical
Typed comparison-target contract: {"backend_binding_id":"","contract_id":"comparison-target:P003:analytical:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"analytical","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"analytical","validation_bundle_id":"","variant_parameters":{}}
The executable must exercise every declared variant and spec override. Declare the exercised artifact contract using the canonical shape __trellis_comparison_bindings__ = {"analytical":{"target_contract":{"backend_binding_id":"","contract_id":"comparison-target:P003:analytical:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"analytical","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"analytical","validation_bundle_id":"","variant_parameters":{}}}}.

Implementation target: analytical."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import PricingValue
from trellis.core.types import DayCountConvention
from trellis.models.analytical.barrier import barrier_option_price
from trellis.models.fx_barrier_option import resolve_fx_barrier_inputs



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

Implementation target: analytical
Preferred method family: analytical
Typed comparison-target contract: {"backend_binding_id":"","contract_id":"comparison-target:P003:analytical:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"analytical","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"analytical","validation_bundle_id":"","variant_parameters":{}}
The executable must exercise every declared variant and spec override. Declare the exercised artifact contract using the canonical shape __trellis_comparison_bindings__ = {"analytical":{"target_contract":{"backend_binding_id":"","contract_id":"comparison-target:P003:analytical:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"analytical","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"analytical","validation_bundle_id":"","variant_parameters":{}}}}.

Implementation target: analytical."""
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


class FXBarrierAnalyticalPayoff:
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

Implementation target: analytical
Preferred method family: analytical
Typed comparison-target contract: {"backend_binding_id":"","contract_id":"comparison-target:P003:analytical:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"analytical","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"analytical","validation_bundle_id":"","variant_parameters":{}}
The executable must exercise every declared variant and spec override. Declare the exercised artifact contract using the canonical shape __trellis_comparison_bindings__ = {"analytical":{"target_contract":{"backend_binding_id":"","contract_id":"comparison-target:P003:analytical:v1","equivalence_group":"","exercise_style":"","explicit":false,"method":"analytical","model_family":"","observation_style":"","payoff_family":"","resolution_source":"legacy_target_inference","route_family":"","route_id":"","schema_version":1,"semantic_axes":{},"spec_overrides":{},"target_id":"analytical","validation_bundle_id":"","variant_parameters":{}}}}.

Implementation target: analytical."""

    __trellis_comparison_bindings__ = {'analytical': {'target_contract': {'schema_version': 1, 'contract_id': 'comparison-target:P003:analytical:v1', 'target_id': 'analytical', 'method': 'analytical', 'route_id': '', 'route_family': '', 'backend_binding_id': '', 'variant_parameters': {}, 'spec_overrides': {}, 'validation_bundle_id': '', 'payoff_family': '', 'exercise_style': '', 'model_family': '', 'observation_style': '', 'semantic_axes': {}, 'equivalence_group': '', 'resolution_source': 'legacy_target_inference', 'explicit': False}}}

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
        unit_price = barrier_option_price(
            resolved.spot,
            resolved.strike,
            resolved.barrier,
            resolved.domestic_rate,
            resolved.sigma,
            resolved.maturity,
            barrier_type=resolved.barrier_type,
            option_type=resolved.option_type,
            rebate=resolved.rebate,
            q=resolved.foreign_rate,
            observations_per_year=resolved.observations_per_year,
        )
        return float(resolved.notional) * float(unit_price)