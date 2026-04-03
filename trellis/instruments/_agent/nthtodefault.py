"""Agent-generated payoff: Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: analytical_cds
Preferred method family: analytical

Implementation target: analytical_cds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put



@dataclass(frozen=True)
class NthToDefaultSpec:
    """Specification for Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: analytical_cds
Preferred method family: analytical

Implementation target: analytical_cds."""
    notional: float
    n_names: int
    n_th: int
    end_date: date
    correlation: float = 0.3
    recovery: float = 0.4
    day_count: DayCountConvention = DayCountConvention.ACT_360


class NthToDefaultPayoff:
    """Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: analytical_cds
Preferred method family: analytical

Implementation target: analytical_cds."""

    def __init__(self, spec: NthToDefaultSpec):
        self._spec = spec

    @property
    def spec(self) -> NthToDefaultSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"credit_curve", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
        if T <= 0.0:
            return 0.0

        if market_state.credit_curve is None:
            raise ValueError("NthToDefaultPayoff requires a credit curve")
        if market_state.discount is None:
            raise ValueError("NthToDefaultPayoff requires a discount curve")

        p_def = max(0.0, 1.0 - float(market_state.credit_curve.survival_probability(T)))
        n = int(spec.n_names)
        n_th = int(spec.n_th)
        rho = float(spec.correlation)

        if rho <= 1e-8:
            from math import comb

            p_nth = 1.0 - sum(
                comb(n, j) * (p_def ** j) * ((1.0 - p_def) ** (n - j))
                for j in range(n_th)
            )
        else:
            from math import comb
            from scipy import integrate
            from scipy.stats import norm

            p_thr = norm.ppf(max(1e-9, min(1.0 - 1e-9, p_def)))
            sq_rho = rho ** 0.5
            sq_1mr = (1.0 - rho) ** 0.5

            def integrand(z):
                pc = norm.cdf((p_thr - sq_rho * z) / sq_1mr)
                pk = 1.0 - sum(
                    comb(n, j) * (pc ** j) * ((1.0 - pc) ** (n - j))
                    for j in range(n_th)
                )
                return pk * norm.pdf(z)

            p_nth, _ = integrate.quad(integrand, -8.0, 8.0)
            p_nth = max(0.0, min(1.0, float(p_nth)))

        df = float(market_state.discount.discount(T))
        return float(p_nth * (1.0 - spec.recovery) * spec.notional * df)
