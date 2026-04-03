"""Agent-generated payoff: Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)

Construct methods: rate_tree, pde_solver, monte_carlo, fft_pricing
Comparison targets: crr_tree (rate_tree), bs_pde (pde_solver), mc_exact (monte_carlo), fft (fft_pricing), cos (fft_pricing), black_scholes (analytical)
Cross-validation harness:
  internal targets: crr_tree, bs_pde, mc_exact, fft, cos
  analytical benchmark: black_scholes

Implementation target: mc_exact
Preferred method family: monte_carlo

Implementation target: mc_exact."""

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

Implementation target: mc_exact
Preferred method family: monte_carlo

Implementation target: mc_exact."""
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "'call'"
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class EuropeanOptionMonteCarloPayoff:
    """Build a pricer for: European equity call: 5-way (tree, PDE, MC, FFT, COS)

Construct methods: rate_tree, pde_solver, monte_carlo, fft_pricing
Comparison targets: crr_tree (rate_tree), bs_pde (pde_solver), mc_exact (monte_carlo), fft (fft_pricing), cos (fft_pricing), black_scholes (analytical)
Cross-validation harness:
  internal targets: crr_tree, bs_pde, mc_exact, fft, cos
  analytical benchmark: black_scholes

Implementation target: mc_exact
Preferred method family: monte_carlo

Implementation target: mc_exact."""

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
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            intrinsic = max(spec.spot - spec.strike, 0.0) if spec.option_type.lower().strip("'\"") == "call" else max(spec.strike - spec.spot, 0.0)
            return float(spec.notional * intrinsic)

        r = float(market_state.discount.zero_rate(T))
        sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

        from trellis.models.processes.gbm import GBM
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.core.differentiable import get_numpy

        np = get_numpy()

        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(
            process,
            n_paths=max(10000, int(spec.n_paths)),
            n_steps=max(1, int(spec.n_steps)),
            seed=42,
            method="exact",
        )

        def payoff_fn(paths):
            terminal = paths[:, -1]
            if spec.option_type.lower().strip("'\"") == "put":
                return np.maximum(spec.strike - terminal, 0.0) * spec.notional
            return np.maximum(terminal - spec.strike, 0.0) * spec.notional

        price = engine.price(spec.spot, T, payoff_fn, discount_rate=r)
        return float(price["price"])
