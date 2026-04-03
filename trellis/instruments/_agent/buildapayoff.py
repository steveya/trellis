"""Agent-generated payoff: Build a pricer for: American put: equity tree knowledge-light proving

Build a thin adapter for a vanilla American put on an equity underlier. Use the checked lattice helper for the pricing engine rather than open-coding the rollback.

Construct methods: rate_tree
Comparison targets: rate_tree (rate_tree)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.equity_option_tree import price_vanilla_equity_option_tree
from trellis.models.trees.lattice import build_spot_lattice, lattice_backward_induction



@dataclass(frozen=True)
class AmericanPutEquityTreeSpec:
    """Specification for Build a pricer for: American put: equity tree knowledge-light proving

Build a thin adapter for a vanilla American put on an equity underlier. Use the checked lattice helper for the pricing engine rather than open-coding the rollback.

Construct methods: rate_tree
Comparison targets: rate_tree (rate_tree)."""
    underlying: str
    expiry_date: date
    strike: float
    spot: float
    notional: float
    rate_index: str | None
    num_steps: int = 100
    day_count: DayCountConvention = DayCountConvention.ACT_365


class AmericanPutEquityTreePayoff:
    """Build a pricer for: American put: equity tree knowledge-light proving

Build a thin adapter for a vanilla American put on an equity underlier. Use the checked lattice helper for the pricing engine rather than open-coding the rollback.

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

        raw_price = price_vanilla_equity_option_tree(
            market_state,
            spec,
            model="crr",
            n_steps=spec.num_steps,
        )

        return float(spec.notional * raw_price)