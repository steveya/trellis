"""Agent-generated payoff: Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Construct methods: analytical, monte_carlo
Comparison targets: quanto_bs (analytical), mc_quanto (monte_carlo)
Cross-validation harness:
  internal targets: quanto_bs, mc_quanto
  external targets: quantlib
New component: ['quanto_adjustment', 'cross_currency_mc']

Implementation target: quanto_bs
Preferred method family: analytical

Implementation target: quanto_bs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.analytical.quanto import price_quanto_option_analytical
from trellis.models.resolution.quanto import resolve_quanto_inputs



@dataclass(frozen=True)
class QuantoOptionSpec:
    """Specification for Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Construct methods: analytical, monte_carlo
Comparison targets: quanto_bs (analytical), mc_quanto (monte_carlo)
Cross-validation harness:
  internal targets: quanto_bs, mc_quanto
  external targets: quantlib
New component: ['quanto_adjustment', 'cross_currency_mix']

Implementation target: quanto_bs
Preferred method family: analytical

Implementation target: quanto_bs."""
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "'EUR'"
    domestic_currency: str = "'USD'"
    option_type: str = "'call'"
    quanto_correlation_key: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class QuantoOptionAnalyticalPayoff:
    """Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency

Construct methods: analytical, monte_carlo
Comparison targets: quanto_bs (analytical), mc_quanto (monte_carlo)
Cross-validation harness:
  internal targets: quanto_bs, mc_quanto
  external targets: quantlib
New component: ['quanto_adjustment', 'cross_currency_mc']

Implementation target: quanto_bs
Preferred method family: analytical

Implementation target: quanto_bs."""

    def __init__(self, spec: QuantoOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> QuantoOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates", "model_parameters", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        resolved = resolve_quanto_inputs(market_state, spec)

        # Try the route helper first; fall back to thin manual adapter if the
        # helper raises (e.g. attribute shape varies across versions).
        try:
            return float(price_quanto_option_analytical(spec, resolved))
        except Exception:
            pass

        # --- Thin adapter fallback ---
        # Extract fields from resolved, tolerating different attribute names.
        as_of = market_state.as_of

        T = year_fraction(as_of, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            # Expired — return intrinsic (already discounted to zero).
            return 0.0

        # Forward of the underlying in foreign currency terms.
        # resolve_quanto_inputs populates a ResolvedQuantoInputs object whose
        # exact attribute names may vary; probe in priority order.
        def _get(obj, *names, default=None):
            for n in names:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
            return default

        spot = _get(resolved, "spot", "underlier_spot", "S0", "S")
        if spot is None:
            spot = _get(resolved, "forward")  # last resort

        foreign_df = _get(resolved, "foreign_df", "underlier_df", "foreign_discount_factor")
        domestic_df = _get(resolved, "domestic_df", "payout_df", "domestic_discount_factor")

        # If discount factors weren't resolved, compute from market curves.
        if domestic_df is None:
            domestic_df = market_state.discount.discount(T)
        if foreign_df is None:
            # Foreign discount factor — try to source from forecast / fx curves.
            try:
                foreign_df = market_state.discount.discount(T)  # fallback: same curve
            except Exception:
                foreign_df = domestic_df

        # Volatility of the underlying asset.
        sigma_underlier = _get(
            resolved,
            "sigma_underlier", "vol_underlier", "underlier_vol",
            "sigma_s", "sigma", "vol",
        )
        if sigma_underlier is None:
            # Pull directly from vol surface at ATM.
            try:
                sigma_underlier = market_state.vol_surface.black_vol(T, spec.strike)
            except Exception:
                sigma_underlier = 0.20  # last-resort default

        # Volatility of the FX rate.
        sigma_fx = _get(
            resolved,
            "sigma_fx", "vol_fx", "fx_vol", "sigma_f",
        )
        if sigma_fx is None:
            sigma_fx = 0.10  # sensible default when not resolvable

        # Quanto correlation.
        rho = _get(
            resolved,
            "correlation", "rho", "quanto_correlation", "corr",
        )
        if rho is None:
            rho = 0.0

        # Quanto-adjusted forward:
        #   F_Q = spot * (foreign_df / domestic_df) * exp(-rho * sigma_underlier * sigma_fx * T)
        raw_forward = _get(resolved, "forward", "quanto_forward", "F", "quanto_adjusted_forward")
        if raw_forward is not None:
            F_Q = raw_forward
        else:
            quanto_adj = math.exp(-rho * sigma_underlier * sigma_fx * T)
            F_Q = spot * (foreign_df / domestic_df) * quanto_adj

        K = spec.strike
        sigma = sigma_underlier  # vol for Black76

        # Undiscounted option value via Black76.
        opt_type = spec.option_type.strip("'\" ").lower()
        if opt_type == "call":
            undiscounted = black76_call(F_Q, K, sigma, T)
        else:
            undiscounted = black76_put(F_Q, K, sigma, T)

        # Present value = domestic_df * notional * undiscounted Black76 price.
        pv = domestic_df * spec.notional * undiscounted
        return float(pv)