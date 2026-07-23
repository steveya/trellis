"""Agent-generated payoff: Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree

Construct methods: pde_solver, rate_tree
Comparison targets: hw_pde_psor (pde_solver), hw_rate_tree (rate_tree)
Cross-validation harness:
  internal targets: hw_pde_psor, hw_rate_tree
  external targets: quantlib
New component: hw_rate_pde_operator

Implementation target: hw_rate_tree
Preferred method family: rate_tree

Implementation target: hw_rate_tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.short_rate_fixed_income import (
    build_embedded_fixed_income_event_timeline,
    compile_embedded_fixed_income_lattice_contract_spec,
    present_value_fixed_coupon_bond,
    settlement_date_for_fixed_income_claim,
)
from trellis.models.short_rate_lattice import resolve_short_rate_lattice_inputs
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    TERM_STRUCTURE_TARGET,
    UNIFORM_ADDITIVE_MESH,
    build_lattice,
    price_on_lattice,
)
from trellis.models.trees.models import MODEL_REGISTRY

@dataclass(frozen=True)
class CallableBondSpec:
    """Specification for Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree

Construct methods: pde_solver, rate_tree
Comparison targets: hw_pde_psor (pde_solver), hw_rate_tree (rate_tree)
Cross-validation harness:
  internal targets: hw_pde_psor, hw_rate_tree
  external targets: quantlib
New component: hw_rate_pde_operator

Implementation target: hw_rate_tree
Preferred method family: rate_tree

Implementation target: hw_rate_tree."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    call_dates: tuple[date, ...]
    call_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class CallableBondPayoff:
    """Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree

Construct methods: pde_solver, rate_tree
Comparison targets: hw_pde_psor (pde_solver), hw_rate_tree (rate_tree)
Cross-validation harness:
  internal targets: hw_pde_psor, hw_rate_tree
  external targets: quantlib
New component: hw_rate_pde_operator

Implementation target: hw_rate_tree
Preferred method family: rate_tree

Implementation target: hw_rate_tree."""

    def __init__(self, spec: CallableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> CallableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        if market_state.discount is None:
            raise ValueError("CallableBondPayoff requires a discount curve in market_state")

        settlement = settlement_date_for_fixed_income_claim(market_state, spec)
        horizon = year_fraction(settlement, spec.end_date, spec.day_count)
        if horizon <= 0.0:
            raise ValueError("Callable bond maturity must be after settlement")

        resolved = resolve_short_rate_lattice_inputs(
            market_state,
            horizon=horizon,
            model="hull_white",
            minimum_steps=50,
            maximum_steps=200,
            steps_per_year=50.0,
        )
        tree_model = MODEL_REGISTRY[resolved.model_name]
        lattice = build_lattice(
            BINOMIAL_1F_TOPOLOGY,
            UNIFORM_ADDITIVE_MESH,
            tree_model.as_lattice_model_spec(),
            calibration_target=TERM_STRUCTURE_TARGET(market_state.discount),
            r0=resolved.r0,
            sigma=resolved.sigma,
            a=resolved.mean_reversion,
            T=resolved.horizon,
            n_steps=resolved.n_steps,
        )
        event_timeline = build_embedded_fixed_income_event_timeline(
            spec,
            settlement=settlement,
        )
        contract = compile_embedded_fixed_income_lattice_contract_spec(
            spec,
            event_timeline=event_timeline,
            expected_control_style="issuer_min",
            dt=lattice.dt,
            n_steps=lattice.n_steps,
        )
        lattice_value = float(price_on_lattice(lattice, contract))
        straight_bond_value = present_value_fixed_coupon_bond(
            market_state,
            spec,
            settlement=settlement,
        )
        return min(lattice_value, straight_bond_value)
