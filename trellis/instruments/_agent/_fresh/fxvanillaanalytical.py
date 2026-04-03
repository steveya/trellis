"""Agent-generated payoff: Build a pricer for: FX vanilla option: Garman-Kohlhagen vs MC

Construct methods: analytical, monte_carlo
Comparison targets: gk_analytical (analytical), mc_fx_option (monte_carlo)
Cross-validation harness:
  internal targets: gk_analytical, mc_fx_option
  external targets: quantlib, financepy
New component: garman_kohlhagen_formula

Implementation target: gk_analytical
Preferred method family: analytical

Implementation target: gk_analytical."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.fx import garman_kohlhagen_price_raw, ResolvedGarmanKohlhagenInputs
from trellis.core.date_utils import year_fraction
from trellis.models.black import garman_kohlhagen_call, garman_kohlhagen_put


@dataclass(frozen=True)
class FXVanillaOptionSpec:
    """Specification for Build a pricer for: FX vanilla option: Garman-Kohlhagen vs MC

Construct methods: analytical, monte_carlo
Comparison targets: gk_analytical (analytical), mc_fx_option (monte_carlo)
Cross-validation harness:
  internal targets: gk_analytical, mc_fx_option
  external targets: quantlib, financepy
New component: garman_kohlhagen_formula

Implementation target: gk_analytical
Preferred method family: analytical

Implementation target: gk_analytical."""
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str = "'call'"
    day_count: DayCountConvention = DayCountConvention.ACT_365


class FXVanillaAnalyticalPayoff:
    """Build a pricer for: FX vanilla option: Garman-Kohlhagen vs MC

Construct methods: analytical, monte_carlo
Comparison targets: gk_analytical (analytical), mc_fx_option (monte_carlo)
Cross-validation harness:
  internal targets: gk_analytical, mc_fx_option
  external targets: quantlib, financepy
New component: garman_kohlhagen_formula

Implementation target: gk_analytical
Preferred method family: analytical

Implementation target: gk_analytical."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        # Compute time to expiry
        as_of = market_state.as_of
        T = year_fraction(as_of, spec.expiry_date, spec.day_count)

        # Extract scalar spot FX rate (domestic per unit of foreign)
        fx_rate_wrapper = market_state.fx_rates[spec.fx_pair]
        spot = fx_rate_wrapper.spot

        # Domestic discount factor to expiry
        df_domestic = market_state.discount.discount(T)

        # Foreign discount factor to expiry (rf curve identified by foreign_discount_key)
        foreign_curve = market_state.forecast_curves[spec.foreign_discount_key]
        df_foreign = foreign_curve.discount(T)

        # Implied vol from vol surface at (T, strike)
        sigma = market_state.vol_surface.black_vol(T, spec.strike)

        # Normalise option_type: strip surrounding quotes if present
        raw_type = spec.option_type.strip().strip("'\"").lower()

        # Build resolved inputs for Garman-Kohlhagen
        resolved = ResolvedGarmanKohlhagenInputs(
            spot=spot,
            strike=spec.strike,
            T=T,
            sigma=sigma,
            df_domestic=df_domestic,
            df_foreign=df_foreign,
        )

        # Price via Garman-Kohlhagen closed-form kernel (returns undiscounted unit price)
        unit_price = garman_kohlhagen_price_raw(raw_type, resolved)

        # Scale by notional and return PV (discounting is embedded in GK formula)
        return spec.notional * unit_price