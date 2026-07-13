"""Compatibility adapter for the quanto analytical payoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import PricingValue
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import (
    discounted_value,
    normalized_option_type,
    quanto_adjusted_forward,
    terminal_intrinsic,
)
from trellis.models.black import black76_call, black76_put
from trellis.models.resolution.quanto import resolve_quanto_inputs


REQUIREMENTS = frozenset(
    {
        "black_vol_surface",
        "discount_curve",
        "forward_curve",
        "fx_rates",
        "model_parameters",
        "spot",
    }
)


@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for the single-name quanto analytical adapter."""

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


class QuantoOptionAnalyticalPayoff:
    """Quanto analytical adapter composed from market and Black primitives."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return REQUIREMENTS

    def evaluate(self, market_state: MarketState) -> PricingValue:
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

        forward = quanto_adjusted_forward(
            spot=resolved.spot,
            domestic_df=resolved.domestic_df,
            foreign_df=resolved.foreign_df,
            corr=resolved.corr,
            sigma_underlier=resolved.sigma_underlier,
            sigma_fx=resolved.sigma_fx,
            T=resolved.T,
        )
        if option_type == "call":
            undiscounted = black76_call(
                forward,
                self._spec.strike,
                resolved.sigma_underlier,
                resolved.T,
            )
        else:
            undiscounted = black76_put(
                forward,
                self._spec.strike,
                resolved.sigma_underlier,
                resolved.T,
            )
        return float(
            discounted_value(
                undiscounted,
                resolved.domestic_df,
                scale=self._spec.notional,
            )
        )
