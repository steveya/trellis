"""CDO tranche adapter composed from public copula and loss-layer primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.copulas.correlation import equicorrelation_matrix
from trellis.models.copulas.student_t import StudentTCopula
from trellis.models.credit_basket_copula import resolve_credit_basket_inputs
from trellis.models.loss_layers import (
    bounded_layer_loss_fraction,
    homogeneous_pool_loss_fraction,
)


@dataclass(frozen=True)
class CDOTrancheSpec:
    """Homogeneous credit-pool tranche inputs for the bounded adapter route."""

    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    correlation: float = 0.3
    recovery: float = 0.4
    degrees_of_freedom: float = 5.0
    n_paths: int = 40_000
    seed: int | None = 42
    day_count: DayCountConvention = DayCountConvention.ACT_360


class CDOTranchePayoff:
    """Seeded Student-t tranche-loss composition used by the checked adapter."""

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
        resolved = resolve_credit_basket_inputs(market_state, spec)
        np = get_numpy()
        n_paths = int(spec.n_paths)
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if resolved.horizon <= 0.0 or resolved.default_probability <= 0.0:
            expected_loss_fraction = 0.0
        else:
            hazard_rates = np.full(resolved.n_names, resolved.hazard_rate)
            correlation = equicorrelation_matrix(
                resolved.n_names,
                resolved.correlation,
            )
            copula = StudentTCopula(
                correlation_matrix=correlation,
                df=float(spec.degrees_of_freedom),
            )
            default_times = copula.sample_default_times(
                hazard_rates,
                n_paths=n_paths,
                rng=np.random.default_rng(spec.seed),
            )
            default_counts = np.sum(default_times <= resolved.horizon, axis=1)
            pool_loss = homogeneous_pool_loss_fraction(
                default_counts,
                pool_size=resolved.n_names,
                recovery=resolved.recovery,
            )
            layer_loss = bounded_layer_loss_fraction(
                pool_loss,
                attachment=float(spec.attachment),
                detachment=float(spec.detachment),
            )
            expected_loss_fraction = float(np.mean(layer_loss))

        return float(
            resolved.notional
            * resolved.discount_factor
            * expected_loss_fraction
        )

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        """Return price plus bounded expected-loss and spread diagnostics."""
        spec = self._spec
        resolved = resolve_credit_basket_inputs(market_state, spec)
        price = float(self.evaluate(market_state))
        denominator = resolved.notional * resolved.discount_factor
        expected_loss_fraction = price / denominator if denominator > 0.0 else 0.0

        np = get_numpy()
        payment_count = max(int(np.ceil(resolved.horizon * 4.0)), 1)
        payment_times = (
            np.linspace(
                resolved.horizon / payment_count,
                resolved.horizon,
                payment_count,
            )
            if resolved.horizon > 0.0
            else np.asarray(())
        )
        annuity = float(
            np.sum(
                np.asarray(
                    [
                        float(market_state.discount.discount(float(payment_time)))
                        for payment_time in payment_times
                    ]
                )
            )
            / 4.0
        )
        tranche_width = float(spec.detachment) - float(spec.attachment)
        fair_spread_bp = (
            price / (resolved.notional * tranche_width * annuity) * 10_000.0
            if tranche_width > 0.0 and annuity > 0.0
            else 0.0
        )
        return {
            "price": price,
            "expected_loss_fraction": float(expected_loss_fraction),
            "fair_spread_bp": float(fair_spread_bp),
        }
