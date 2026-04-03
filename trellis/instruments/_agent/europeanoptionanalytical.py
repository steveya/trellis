"""Agent-generated payoff: Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)

Construct methods: rate_tree, pde_solver, monte_carlo, fft_pricing
Comparison targets: crr_tree (rate_tree), bs_pde (pde_solver), mc_exact (monte_carlo), fft (fft_pricing), cos (fft_pricing), black_scholes (analytical)
Cross-validation harness:
  internal targets: crr_tree, bs_pde, mc_exact, fft, cos
  analytical benchmark: black_scholes

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class EuropeanOptionSpec:
    """Specification for Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)

Construct methods: rate_tree, pde_solver, monte_carlo, fft_pricing
Comparison targets: crr_tree (rate_tree), bs_pde (pde_solver), mc_exact (monte_carlo), fft (fft_pricing), cos (fft_pricing), black_scholes (analytical)
Cross-validation harness:
  internal targets: crr_tree, bs_pde, mc_exact, fft, cos
  analytical benchmark: black_scholes

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "'call'"
    day_count: DayCountConvention = DayCountConvention.ACT_365


class EuropeanOptionAnalyticalPayoff:
    """Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)

Construct methods: rate_tree, pde_solver, monte_carlo, fft_pricing
Comparison targets: crr_tree (rate_tree), bs_pde (pde_solver), mc_exact (monte_carlo), fft (fft_pricing), cos (fft_pricing), black_scholes (analytical)
Cross-validation harness:
  internal targets: crr_tree, bs_pde, mc_exact, fft, cos
  analytical benchmark: black_scholes

Implementation target: black_scholes
Preferred method family: analytical

Implementation target: black_scholes."""

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
        t = year_fraction(market_state.as_of, spec.expiry_date, spec.day_count)
        if t <= 0.0:
            intrinsic = max(spec.spot - spec.strike, 0.0) if str(spec.option_type).strip("'\"").lower() == "call" else max(spec.strike - spec.spot, 0.0)
            return float(spec.notional * intrinsic)

        df = market_state.discount.discount(t)
        sigma = market_state.vol_surface.black_vol(t, spec.strike)
        forward = spec.spot / max(df, 1e-12)

        opt_type = str(spec.option_type).strip("'\"").lower()
        if opt_type == "call":
            pv = df * black76_call(forward, spec.strike, sigma, t)
        elif opt_type == "put":
            pv = df * black76_put(forward, spec.strike, sigma, t)
        else:
            raise ValueError(f"Unsupported option_type: {spec.option_type!r}")

        return float(spec.notional * pv)
