"""Agent-generated payoff: Build a pricer for: American put: equity tree knowledge-light proving

Build a thin adapter for a vanilla American put on an equity underlier by composing the shared lattice algebra.

Construct methods: rate_tree
Comparison targets: rate_tree (rate_tree)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.resolution.single_state_diffusion import (
    resolve_single_state_diffusion_inputs,
    terminal_intrinsic_from_resolved,
)
from trellis.models.trees.algebra import (
    build_lattice,
    compile_lattice_recipe,
    equity_tree,
    price_on_lattice,
    with_control,
)



@dataclass(frozen=True)
class AmericanPutEquityTreeSpec:
    """Specification for Build a pricer for: American put: equity tree knowledge-light proving

Build a thin adapter for a vanilla American put on an equity underlier by composing the shared lattice algebra.

Construct methods: rate_tree
Comparison targets: rate_tree (rate_tree)."""
    underlying: str
    expiry_date: date
    strike: float
    spot: float
    notional: float
    rate_index: str | None
    num_steps: int = 100
    option_type: str = "put"
    day_count: DayCountConvention = DayCountConvention.ACT_365


class AmericanPutEquityTreePayoff:
    """Build a pricer for: American put: equity tree knowledge-light proving

Build a thin adapter for a vanilla American put on an equity underlier by composing the shared lattice algebra.

Construct methods: rate_tree
Comparison targets: rate_tree (rate_tree)."""

    def __init__(self, spec: AmericanPutEquityTreeSpec):
        self._spec = spec

    @property
    def spec(self) -> AmericanPutEquityTreeSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        resolved = resolve_single_state_diffusion_inputs(market_state, spec)
        if resolved.maturity <= 0.0:
            return float(
                resolved.notional
                * terminal_intrinsic_from_resolved(resolved.spot, resolved)
            )

        recipe = with_control(
            equity_tree(
                model_family="crr",
                strike=resolved.strike,
                option_type=resolved.option_type,
            ),
            "american",
        )
        topology, mesh, model, contract = compile_lattice_recipe(recipe)
        lattice = build_lattice(
            topology,
            mesh,
            model,
            spot=resolved.spot,
            rate=resolved.rate,
            dividend_yield=resolved.dividend_yield,
            sigma=resolved.sigma,
            maturity=resolved.maturity,
            n_steps=max(int(spec.num_steps), 1),
        )
        return float(resolved.notional * price_on_lattice(lattice, contract))
