"""Reusable event primitives and contingent cashflow kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Protocol

from scipy import integrate
from scipy.stats import norm


class CreditCurveLike(Protocol):
    """Curve interface required by the default-probability helpers."""

    def survival_probability(self, t: float) -> float:
        """Return survival probability to time ``t``."""
        ...


@dataclass(frozen=True)
class CouponAccrual:
    """One coupon or premium accrual emission."""

    notional: float
    rate: float
    accrual: float
    discount_factor: float
    weight: float = 1.0
    sign: float = 1.0


@dataclass(frozen=True)
class ProtectionPayment:
    """One protection payment driven by default probability."""

    notional: float
    recovery: float
    default_probability: float
    discount_factor: float
    sign: float = 1.0


@dataclass(frozen=True)
class PrincipalPayment:
    """One principal or amortization payment."""

    scheduled_principal: float
    prepaid_principal: float = 0.0
    discount_factor: float = 1.0
    sign: float = 1.0


@dataclass(frozen=True)
class TriggerSettlement:
    """One simple trigger/rebate settlement."""

    amount: float
    discount_factor: float = 1.0
    trigger_weight: float = 1.0
    sign: float = 1.0


@dataclass(frozen=True)
class PrepaymentStep:
    """One prepayment update step."""

    beginning_balance: float
    scheduled_interest: float
    scheduled_principal: float
    prepaid_principal: float
    total_principal: float
    remaining_balance: float
    smm: float


def interval_default_probability_from_survival(
    survival_start: float,
    survival_end: float,
) -> float:
    """Return conditional default probability from survival ratios."""
    survival_start = max(float(survival_start), 0.0)
    survival_end = max(float(survival_end), 0.0)
    if survival_start <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - survival_end / survival_start))


def terminal_default_probability(
    credit_curve: CreditCurveLike,
    horizon: float,
) -> float:
    """Return the default probability over ``[0, horizon]``."""
    horizon = max(float(horizon), 0.0)
    survival = float(credit_curve.survival_probability(horizon))
    return max(0.0, min(1.0, 1.0 - survival))


def coupon_cashflow_pv(coupon: CouponAccrual) -> float:
    """Return discounted coupon/premium PV with explicit sign and weight."""
    return (
        float(coupon.sign)
        * float(coupon.notional)
        * float(coupon.rate)
        * float(coupon.accrual)
        * float(coupon.discount_factor)
        * float(coupon.weight)
    )


def protection_payment_pv(payment: ProtectionPayment) -> float:
    """Return discounted protection-payment PV."""
    loss_given_default = max(0.0, 1.0 - float(payment.recovery))
    return (
        float(payment.sign)
        * float(payment.notional)
        * loss_given_default
        * float(payment.default_probability)
        * float(payment.discount_factor)
    )


def principal_payment_pv(payment: PrincipalPayment) -> float:
    """Return discounted principal/amortization PV."""
    total_principal = float(payment.scheduled_principal) + float(payment.prepaid_principal)
    return float(payment.sign) * total_principal * float(payment.discount_factor)


def trigger_settlement_pv(settlement: TriggerSettlement) -> float:
    """Return discounted trigger/rebate settlement PV."""
    return (
        float(settlement.sign)
        * float(settlement.amount)
        * float(settlement.discount_factor)
        * float(settlement.trigger_weight)
    )


def project_prepayment_step(
    *,
    beginning_balance: float,
    scheduled_interest: float,
    scheduled_principal: float,
    smm: float,
) -> PrepaymentStep:
    """Advance one scheduled-principal plus prepayment step."""
    balance = max(float(beginning_balance), 0.0)
    scheduled_interest = max(float(scheduled_interest), 0.0)
    scheduled_principal = max(0.0, min(float(scheduled_principal), balance))
    smm = max(0.0, min(float(smm), 1.0))

    balance_after_schedule = max(balance - scheduled_principal, 0.0)
    prepaid_principal = min(balance_after_schedule * smm, balance_after_schedule)
    total_principal = scheduled_principal + prepaid_principal
    remaining_balance = max(balance - total_principal, 0.0)

    return PrepaymentStep(
        beginning_balance=balance,
        scheduled_interest=scheduled_interest,
        scheduled_principal=scheduled_principal,
        prepaid_principal=prepaid_principal,
        total_principal=total_principal,
        remaining_balance=remaining_balance,
        smm=smm,
    )


def nth_to_default_probability(
    n_names: int,
    n_th: int,
    marginal_default_prob: float,
    correlation: float,
) -> float:
    """Return the probability that at least ``n_th`` names default."""
    n_names = int(n_names)
    n_th = int(n_th)
    if n_names <= 0:
        raise ValueError("n_names must be positive")
    if n_th <= 0 or n_th > n_names:
        raise ValueError("n_th must lie in [1, n_names]")

    p_def = max(0.0, min(1.0, float(marginal_default_prob)))
    rho = max(0.0, min(float(correlation), 0.999999))

    if rho <= 1e-8:
        return max(
            0.0,
            min(
                1.0,
                1.0
                - sum(
                    comb(n_names, j) * (p_def ** j) * ((1.0 - p_def) ** (n_names - j))
                    for j in range(n_th)
                ),
            ),
        )

    p_thr = norm.ppf(max(1e-9, min(1.0 - 1e-9, p_def)))
    sq_rho = rho ** 0.5
    sq_1mr = (1.0 - rho) ** 0.5

    def integrand(z: float) -> float:
        conditional_prob = norm.cdf((p_thr - sq_rho * z) / sq_1mr)
        triggered = 1.0 - sum(
            comb(n_names, j)
            * (conditional_prob ** j)
            * ((1.0 - conditional_prob) ** (n_names - j))
            for j in range(n_th)
        )
        return float(triggered) * float(norm.pdf(z))

    result, _ = integrate.quad(integrand, -8.0, 8.0)
    return max(0.0, min(1.0, float(result)))


__all__ = [
    "CouponAccrual",
    "PrepaymentStep",
    "PrincipalPayment",
    "ProtectionPayment",
    "TriggerSettlement",
    "coupon_cashflow_pv",
    "interval_default_probability_from_survival",
    "nth_to_default_probability",
    "principal_payment_pv",
    "project_prepayment_step",
    "protection_payment_pv",
    "terminal_default_probability",
    "trigger_settlement_pv",
]
