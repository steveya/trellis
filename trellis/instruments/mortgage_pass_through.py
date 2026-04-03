"""Thin mortgage pass-through wrapper over shared prepayment/amortization kernels."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.core.market_state import MarketState
from trellis.models.cashflow_engine.amortization import level_pay
from trellis.models.cashflow_engine.prepayment import PSA
from trellis.models.contingent_cashflows import (
    CouponAccrual,
    PrincipalPayment,
    coupon_cashflow_pv,
    principal_payment_pv,
    project_prepayment_step,
)


@dataclass(frozen=True)
class MortgagePassThroughSpec:
    """Contract terms for a simple monthly-pay mortgage pass-through."""

    balance: float
    mortgage_rate: float
    pass_through_rate: float
    term_months: int
    psa_speed: float = 1.0


class MortgagePassThroughPayoff:
    """Price a pass-through by discounting scheduled and prepaid cashflows."""

    def __init__(self, spec: MortgagePassThroughSpec):
        self._spec = spec

    @property
    def spec(self) -> MortgagePassThroughSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        if spec.term_months <= 0 or spec.balance <= 0.0:
            return 0.0

        mortgage_rate = float(spec.mortgage_rate) / 12.0
        pass_through_rate = float(spec.pass_through_rate) / 12.0
        schedule = level_pay(float(spec.balance), mortgage_rate, int(spec.term_months))
        prepayment_model = PSA(speed=float(spec.psa_speed))

        pv = 0.0
        remaining_balance = float(spec.balance)

        for month, (scheduled_interest, scheduled_principal) in enumerate(schedule, start=1):
            if remaining_balance <= 0.0:
                break
            step = project_prepayment_step(
                beginning_balance=remaining_balance,
                scheduled_interest=scheduled_interest,
                scheduled_principal=scheduled_principal,
                smm=prepayment_model.smm(month),
            )
            discount_factor = float(market_state.discount.discount(month / 12.0))
            pv += coupon_cashflow_pv(
                CouponAccrual(
                    notional=step.beginning_balance,
                    rate=pass_through_rate,
                    accrual=1.0,
                    discount_factor=discount_factor,
                )
            )
            pv += principal_payment_pv(
                PrincipalPayment(
                    scheduled_principal=step.scheduled_principal,
                    prepaid_principal=step.prepaid_principal,
                    discount_factor=discount_factor,
                )
            )
            remaining_balance = step.remaining_balance

        return float(pv)
