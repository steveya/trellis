"""Deterministic vanilla FX analytical payoff adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.fx import garman_kohlhagen_price_raw
from trellis.models.analytical.fx_vanilla_gk import fx_vanilla_gk_outputs
from trellis.models.fx_vanilla import resolve_fx_vanilla_inputs


@dataclass(frozen=True)
class FXVanillaOptionSpec:
    """Specification for a vanilla FX option under Garman-Kohlhagen."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _resolve_fx_inputs(market_state: MarketState, spec: FXVanillaOptionSpec):
    """Backward-compatible alias for the semantic-facing FX helper resolver."""
    return resolve_fx_vanilla_inputs(market_state, spec).garman_kohlhagen


class FXVanillaAnalyticalPayoff:
    """Deterministic adapter over resolved FX inputs and the raw GK kernel."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates"}

    def evaluate(self, market_state: MarketState) -> float:
        resolved = resolve_fx_vanilla_inputs(market_state, self._spec)
        return float(resolved.notional) * float(
            garman_kohlhagen_price_raw(
                resolved.option_type,
                resolved.garman_kohlhagen,
            )
        )

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        """Return native GK parity outputs for the FinancePy benchmark harness."""
        return dict(fx_vanilla_gk_outputs(market_state, self._spec))
