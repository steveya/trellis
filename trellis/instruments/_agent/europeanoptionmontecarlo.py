"""Agent-generated payoff: Build a pricer for: Heston smile: FFT vs COS vs MC implied vol surface

Extract the implied volatility smile surface under the Heston
stochastic volatility model for SPX.
Use the market snapshot (as_of 2024-11-15) which provides:
  - SPX spot (S=5890)
  - Heston parameters: kappa=2.0, theta=0.04, xi=0.48, rho=-0.68, v0=0.04
  - USD OIS discount curve
  - A reference implied vol smile surface (6 expiries x 7 strikes)
Price European calls across strikes K=[5400, 5600, 5800, 5900, 6000,
6200, 6400] and maturities T=[0.25, 0.5, 1.0, 2.0] years.
Method 1: Heston FFT (Carr-Madan with the Heston characteristic function).
Method 2: Heston COS method.
Method 3: Heston MC (QE scheme, 200k paths).
From each set of prices, extract the implied vol surface by inverting
Black-Scholes.  All three surfaces should agree within 0.5 vol points.
The resulting smile should show negative skew (lower strikes have
higher IV) consistent with rho=-0.68.

Construct methods: fft_pricing, monte_carlo
Comparison targets: heston_fft (fft_pricing), heston_cos (fft_pricing), heston_mc (monte_carlo)
Cross-validation harness:
  internal targets: heston_fft, heston_cos, heston_mc
  external targets: quantlib
New component: implied_vol_surface_extraction

Implementation target: heston_mc
Preferred method family: monte_carlo

Implementation target: heston_mc."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.equity_option_monte_carlo import price_vanilla_equity_option_monte_carlo



@dataclass(frozen=True)
class EuropeanOptionSpec:
    """Specification for Build a pricer for: Heston smile: FFT vs COS vs MC implied vol surface

Extract the implied volatility smile surface under the Heston
stochastic volatility model for SPX.
Use the market snapshot (as_of 2024-11-15) which provides:
  - SPX spot (S=5890)
  - Heston parameters: kappa=2.0, theta=0.04, xi=0.48, rho=-0.68, v0=0.04
  - USD OIS discount curve
  - A reference implied vol smile surface (6 expiries x 7 strikes)
Price European calls across strikes K=[5400, 5600, 5800, 5900, 6000,
6200, 6400] and maturities T=[0.25, 0.5, 1.0, 2.0] years.
Method 1: Heston FFT (Carr-Madan with the Heston characteristic function).
Method 2: Heston COS method.
Method 3: Heston MC (QE scheme, 200k paths).
From each set of prices, extract the implied vol surface by inverting
Black-Scholes.  All three surfaces should agree within 0.5 vol points.
The resulting smile should show negative skew (lower strikes have
higher IV) consistent with rho=-0.68.

Construct methods: fft_pricing, monte_carlo
Comparison targets: heston_fft (fft_pricing), heston_cos (fft_pricing), heston_mc (monte_carlo)
Cross-validation harness:
  internal targets: heston_fft, heston_cos, heston_mc
  external targets: quantlib
New component: implied_vol_surface_extraction

Implementation target: heston_mc
Preferred method family: monte_carlo

Implementation target: heston_mc."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "'call'"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class EuropeanOptionMonteCarloPayoff:
    """Build a pricer for: Heston smile: FFT vs COS vs MC implied vol surface

Extract the implied volatility smile surface under the Heston
stochastic volatility model for SPX.
Use the market snapshot (as_of 2024-11-15) which provides:
  - SPX spot (S=5890)
  - Heston parameters: kappa=2.0, theta=0.04, xi=0.48, rho=-0.68, v0=0.04
  - USD OIS discount curve
  - A reference implied vol smile surface (6 expiries x 7 strikes)
Price European calls across strikes K=[5400, 5600, 5800, 5900, 6000,
6200, 6400] and maturities T=[0.25, 0.5, 1.0, 2.0] years.
Method 1: Heston FFT (Carr-Madan with the Heston characteristic function).
Method 2: Heston COS method.
Method 3: Heston MC (QE scheme, 200k paths).
From each set of prices, extract the implied vol surface by inverting
Black-Scholes.  All three surfaces should agree within 0.5 vol points.
The resulting smile should show negative skew (lower strikes have
higher IV) consistent with rho=-0.68.

Construct methods: fft_pricing, monte_carlo
Comparison targets: heston_fft (fft_pricing), heston_cos (fft_pricing), heston_mc (monte_carlo)
Cross-validation harness:
  internal targets: heston_fft, heston_cos, heston_mc
  external targets: quantlib
New component: implied_vol_surface_extraction

Implementation target: heston_mc
Preferred method family: monte_carlo

Implementation target: heston_mc."""

    def __init__(self, spec: EuropeanOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> EuropeanOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        return float(price_vanilla_equity_option_monte_carlo(market_state, spec))