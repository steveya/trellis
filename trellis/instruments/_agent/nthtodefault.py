"""Generated adapter for name-weighted terminal nth-to-default pricing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.contingent_cashflows import (
    TriggerSettlement,
    ranked_event_expected_weight,
    trigger_settlement_pv,
)
from trellis.models.copulas.gaussian import GaussianCopula
from trellis.models.credit_basket_copula import resolve_credit_basket_inputs


@dataclass(frozen=True)
class NthToDefaultSpec:
    """Terminal basket terms and reproducible sampled-evidence controls."""

    notional: float
    n_names: int
    n_th: int
    end_date: date
    basket_names: tuple[str, ...] = ()
    basket_weights: tuple[float, ...] = ()
    correlation: float = 0.3
    recovery: float = 0.4
    spread: float | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_360
    n_paths: int = 250_000
    seed: int = 42


class NthToDefaultPayoff:
    """Compose sampled ranked exposure with an explicit terminal settlement."""

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
        n_names = int(spec.n_names)
        n_th = int(spec.n_th)
        if n_names < 2:
            raise ValueError("n_names must be at least two")
        if n_th <= 0 or n_th > n_names:
            raise ValueError("n_th must lie in [1, n_names]")
        n_paths = int(spec.n_paths)
        seed = int(spec.seed)
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        resolved = resolve_credit_basket_inputs(market_state, spec)
        if resolved.horizon <= 0.0 or resolved.default_probability <= 0.0:
            expected_weight = 0.0
        else:
            np = get_numpy()
            hazard_rates = np.full(resolved.n_names, resolved.hazard_rate)
            copula = GaussianCopula(
                correlation=resolved.correlation,
                n_names=resolved.n_names,
            )
            default_times = copula.sample_default_times(
                hazard_rates,
                n_paths=n_paths,
                rng=np.random.default_rng(seed),
            )
            expected_weight = ranked_event_expected_weight(
                default_times,
                event_weights=resolved.exposure_weights,
                rank=n_th,
                horizon=resolved.horizon,
            )

        settlement = TriggerSettlement(
            amount=resolved.notional * (1.0 - resolved.recovery),
            discount_factor=resolved.discount_factor,
            trigger_weight=expected_weight,
        )
        return trigger_settlement_pv(settlement)

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        """Return price and parallel one-basis-point spread CS01."""
        price = float(self.evaluate(market_state))
        spread = self._spec.spread
        if spread is None:
            return {"price": price}
        bumped_spec = replace(self._spec, spread=float(spread) + 1.0e-4)
        bumped_price = float(type(self)(bumped_spec).evaluate(market_state))
        return {
            "price": price,
            "spread_cs01": bumped_price - price,
        }
