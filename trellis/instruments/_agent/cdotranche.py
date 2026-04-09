"""Agent-generated payoff: Build a pricer for: CDO tranche: Gaussian vs Student-t copula

Price a synthetic CDO mezzanine tranche on a 100-name investment-grade
portfolio.  Attachment point: 3%.  Detachment point: 7%.
Maturity: 5Y.  Notional per name: $1,000,000 (portfolio notional $100M).
Recovery rate: 40% flat across all names.
Use the IG credit curve from the market snapshot (as_of 2024-11-15)
as the representative single-name hazard curve for all 100 names.
Flat pairwise default correlation: 0.3.
Method 1: Gaussian copula (one-factor, semi-analytical via Vasicek
large-pool or recursive).
Method 2: Student-t copula (degrees of freedom = 5) for comparison
to see heavier-tail effects on the mezzanine tranche.
Report tranche fair spread (bp) and expected loss for each copula.

Construct methods: copula
Comparison targets: gaussian_copula (copula), student_t_copula (copula)
Cross-validation harness:
  internal targets: gaussian_copula, student_t_copula
  external targets: quantlib

Implementation target: student_t_copula
Preferred method family: copula

Implementation target: student_t_copula."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.credit_basket_copula import price_credit_basket_tranche



@dataclass(frozen=True)
class CDOTrancheSpec:
    """Specification for Build a pricer for: CDO tranche: Gaussian vs Student-t copula

Price a synthetic CDO mezzanine tranche on a 100-name investment-grade
portfolio.  Attachment point: 3%.  Detachment point: 7%.
Maturity: 5Y.  Notional per name: $1,000,000 (portfolio notional $100M).
Recovery rate: 40% flat across all names.
Use the IG credit curve from the market snapshot (as_of 2024-11-15)
as the representative single-name hazard curve for all 100 names.
Flat pairwise default correlation: 0.3.
Method 1: Gaussian copula (one-factor, semi-analytical via Vasicek
large-pool or recursive).
Method 2: Student-t copula (degrees of freedom = 5) for comparison
to see heavier-tail effects on the mezzanine tranche.
Report tranche fair spread (bp) and expected loss for each copula.

Construct methods: copula
Comparison targets: gaussian_copula (copula), student_t_copula (copula)
Cross-validation harness:
  internal targets: gaussian_copula, student_t_copula
  external targets: quantlib

Implementation target: student_t_copula
Preferred method family: copula

Implementation target: student_t_copula."""
    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    correlation: float = 0.3
    recovery: float = 0.4
    day_count: DayCountConvention = DayCountConvention.ACT_360


class CDOTranchePayoff:
    """Build a pricer for: CDO tranche: Gaussian vs Student-t copula

Price a synthetic CDO mezzanine tranche on a 100-name investment-grade
portfolio.  Attachment point: 3%.  Detachment point: 7%.
Maturity: 5Y.  Notional per name: $1,000,000 (portfolio notional $100M).
Recovery rate: 40% flat across all names.
Use the IG credit curve from the market snapshot (as_of 2024-11-15)
as the representative single-name hazard curve for all 100 names.
Flat pairwise default correlation: 0.3.
Method 1: Gaussian copula (one-factor, semi-analytical via Vasicek
large-pool or recursive).
Method 2: Student-t copula (degrees of freedom = 5) for comparison
to see heavier-tail effects on the mezzanine tranche.
Report tranche fair spread (bp) and expected loss for each copula.

Construct methods: copula
Comparison targets: gaussian_copula (copula), student_t_copula (copula)
Cross-validation harness:
  internal targets: gaussian_copula, student_t_copula
  external targets: quantlib

Implementation target: student_t_copula
Preferred method family: copula

Implementation target: student_t_copula."""

    def __init__(self, spec: CDOTrancheSpec):
        self._spec = spec

    @property
    def spec(self) -> CDOTrancheSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"credit_curve", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        return float(price_credit_basket_tranche(market_state, spec, copula_family="student_t", degrees_of_freedom=5.0, n_paths=40000, seed=42))