"""Agent-generated payoff: Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: mc_cds
Preferred method family: monte_carlo

Implementation target: mc_cds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.conventions.calendar import BusinessDayAdjustment, WEEKEND_ONLY
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.date_utils import build_period_schedule
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.contingent_cashflows import (
    CouponAccrual,
    ProtectionPayment,
    build_default_event_grid,
    conditional_event_probabilities_from_curve,
    coupon_cashflow_pv,
    expected_first_event_weights,
    protection_payment_pv,
    sample_first_event_weights,
)



@dataclass(frozen=True)
class CDSSpec:
    """Specification for Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: mc_cds
Preferred method family: monte_carlo

Implementation target: mc_cds."""
    notional: float
    spread: float
    start_date: date
    end_date: date
    recovery: float = 0.4
    frequency: Frequency = Frequency.QUARTERLY
    day_count: DayCountConvention = DayCountConvention.ACT_360
    valuation_date: date | None = None
    pricing_method: str = "analytical"
    n_paths: int | None = None


class CDSPayoff:
    """Build a pricer for: CDS pricing: hazard rate MC vs survival prob analytical

Construct methods: monte_carlo
Comparison targets: mc_cds (monte_carlo), analytical_cds (analytical)
Cross-validation harness:
  internal targets: mc_cds, analytical_cds
  external targets: quantlib, financepy
New component: cds_pricing

Implementation target: mc_cds
Preferred method family: monte_carlo

Implementation target: mc_cds."""

    def __init__(self, spec: CDSSpec):
        self._spec = spec

    @property
    def spec(self) -> CDSSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"credit_curve", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec

        if market_state.credit_curve is None:
            raise ValueError("market_state.credit_curve is required for CDS pricing")
        if market_state.discount is None:
            raise ValueError("market_state.discount is required for CDS pricing")

        spread = float(spec.spread)
        if spread > 1.0:
            spread *= 1e-4

        schedule = build_period_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            day_count=spec.day_count,
            time_origin=spec.valuation_date or spec.start_date,
            calendar=WEEKEND_ONLY,
            bda=BusinessDayAdjustment.FOLLOWING,
            roll_convention=RollConvention.NONE,
            stub=StubType.SHORT_LAST,
            payment_lag_days=0,
        )

        credit_curve = market_state.credit_curve
        discount_curve = market_state.discount
        event_grid = build_default_event_grid(schedule)
        conditional_probabilities = conditional_event_probabilities_from_curve(
            credit_curve,
            event_grid.intervals,
        )
        pricing_method = str(
            getattr(spec, "pricing_method", "analytical") or "analytical"
        ).strip().lower()
        n_paths = getattr(spec, "n_paths", None)

        if pricing_method == "monte_carlo" or (
            pricing_method not in {"", "analytical"} and n_paths is not None
        ):
            path_count = int(n_paths) if n_paths is not None else 250000
            if path_count < 10000:
                path_count = 10000
            weights = sample_first_event_weights(
                conditional_probabilities,
                n_paths=path_count,
                seed=42,
            )
        else:
            weights = expected_first_event_weights(conditional_probabilities)

        premium_leg = 0.0
        protection_leg = 0.0
        accrued_on_event = 0.0
        accrued_to_valuation = 0.0
        interval_start = 0
        for period_index, period in enumerate(event_grid.periods):
            interval_stop = event_grid.period_interval_stops[period_index]
            if interval_stop <= interval_start:
                interval_start = interval_stop
                continue

            accrual = float(period.accrual_fraction)
            premium_leg += coupon_cashflow_pv(
                CouponAccrual(
                    notional=spec.notional,
                    rate=spread,
                    accrual=accrual,
                    discount_factor=float(
                        discount_curve.discount(
                            event_grid.period_payment_times[period_index]
                        )
                    ),
                    weight=weights.survival_weights[interval_stop - 1],
                )
            )
            accrued_to_valuation += (
                float(spec.notional)
                * spread
                * accrual
                * event_grid.elapsed_period_fractions[period_index]
            )

            for interval_index in range(interval_start, interval_stop):
                interval = event_grid.intervals[interval_index]
                event_weight = weights.event_weights[interval_index]
                if event_weight <= 0.0:
                    continue
                discount_factor = float(
                    discount_curve.discount(interval.settlement_time)
                )
                protection_leg += protection_payment_pv(
                    ProtectionPayment(
                        notional=spec.notional,
                        recovery=spec.recovery,
                        default_probability=event_weight,
                        discount_factor=discount_factor,
                    )
                )
                accrued_on_event += coupon_cashflow_pv(
                    CouponAccrual(
                        notional=spec.notional,
                        rate=spread,
                        accrual=accrual * interval.period_fraction_elapsed,
                        discount_factor=discount_factor,
                        weight=event_weight,
                    )
                )
            interval_start = interval_stop

        return float(
            protection_leg
            - premium_leg
            - accrued_on_event
            + accrued_to_valuation
        )
