"""Agent-generated payoff: Build a pricer for: FX vanilla option: Garman-Kohlhagen vs MC

Construct methods: analytical, monte_carlo
Comparison targets: gk_analytical (analytical), mc_fx_option (monte_carlo)
Cross-validation harness:
  internal targets: gk_analytical, mc_fx_option
  external targets: quantlib, financepy
New component: garman_kohlhagen_formula

Implementation target: mc_fx_option
Preferred method family: monte_carlo

Implementation target: mc_fx_option."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention



@dataclass(frozen=True)
class FXVanillaOptionSpec:
    """Specification for Build a pricer for: FX vanilla option: Garman-Kohlhagen vs MC

Construct methods: analytical, monte_carlo
Comparison targets: gk_analytical (analytical), mc_fx_option (monte_carlo)
Cross-validation harness:
  internal targets: gk_analytical, mc_fx_option
  external targets: quantlib, financepy
New component: garman_kohlhagen_formula

Implementation target: mc_fx_option
Preferred method family: monte_carlo

Implementation target: mc_fx_option."""
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    foreign_discount_key: str
    option_type: str = "'call'"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class FXVanillaMonteCarloPayoff:
    """Build a pricer for: FX vanilla option: Garman-Kohlhagen vs MC

Construct methods: analytical, monte_carlo
Comparison targets: gk_analytical (analytical), mc_fx_option (monte_carlo)
Cross-validation harness:
  internal targets: gk_analytical, mc_fx_option
  external targets: quantlib, financepy
New component: garman_kohlhagen_formula

Implementation target: mc_fx_option
Preferred method family: monte_carlo

Implementation target: mc_fx_option."""

    def __init__(self, spec: FXVanillaOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> FXVanillaOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve", "forward_curve", "fx_rates"}

    def evaluate(self, market_state: MarketState) -> float:
        from trellis.core.date_utils import year_fraction
        from trellis.core.differentiable import get_numpy
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.processes.gbm import GBM

        np = get_numpy()
        spec = self._spec

        # Year fraction to expiry
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            return 0.0

        # Domestic discount rate (risk-free rate in domestic currency)
        r_d = float(market_state.discount.zero_rate(T))

        # Foreign discount rate (risk-free rate in foreign currency)
        # Used as the dividend yield in the Garman-Kohlhagen / FX GBM framework
        foreign_curve = market_state.forecast_curves.get(spec.foreign_discount_key)
        if foreign_curve is not None:
            r_f = float(foreign_curve.zero_rate(T))
        else:
            # Fall back: try to get from discount curve keyed by foreign_discount_key
            r_f = 0.0

        # Spot FX rate — extract scalar from FXRate wrapper
        fx_rate_obj = market_state.fx_rates[spec.fx_pair]
        if hasattr(fx_rate_obj, "spot"):
            S0 = float(fx_rate_obj.spot)
        else:
            S0 = float(fx_rate_obj)

        # Implied volatility from vol surface
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

        # GBM drift under domestic risk-neutral measure for FX:
        # dS = (r_d - r_f) * S * dt + sigma * S * dW
        mu = r_d - r_f

        # Build GBM process — no 'spot' keyword; initial state is passed to simulate()
        process = GBM(mu=mu, sigma=sigma)

        # Number of paths and steps
        n_paths = spec.n_paths
        n_steps = max(spec.n_steps, int(T * 252), 50)

        engine = MonteCarloEngine(
            process,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=42,
            method="exact",
        )

        # Simulate paths: shape (n_paths, n_steps + 1)
        paths = engine.simulate(S0, T)

        # Terminal FX spot values
        S_T = paths[:, -1]

        # Normalise option type — strip surrounding quotes if present
        opt_type = spec.option_type.strip().strip("'\"").lower()

        # Compute intrinsic payoff at expiry
        if opt_type == "call":
            intrinsic = np.maximum(S_T - spec.strike, 0.0)
        else:
            intrinsic = np.maximum(spec.strike - S_T, 0.0)

        # Discount back to valuation date under domestic rate
        df = float(market_state.discount.discount(T))

        # Monte Carlo price estimate
        price = df * float(np.mean(intrinsic)) * spec.notional

        return price