from __future__ import annotations

from datetime import date

import pytest

from trellis.agent.static_leg_contract import (
    CouponLeg,
    CouponPeriod,
    FixedCouponFormula,
    FloatingCouponFormula,
    KnownCashflow,
    KnownCashflowLeg,
    NotionalSchedule,
    NotionalStep,
    OvernightRateIndex,
    SettlementRule,
    SignedLeg,
    StaticLegContractIR,
    StaticLegIRWellFormednessError,
)


def _notional(amount: float) -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(
                start_date=date(2025, 1, 15),
                end_date=date(2030, 1, 15),
                amount=amount,
            ),
        )
    )


def _fixed_coupon_periods() -> tuple[CouponPeriod, ...]:
    return (
        CouponPeriod(
            accrual_start=date(2025, 1, 15),
            accrual_end=date(2025, 7, 15),
            payment_date=date(2025, 7, 15),
        ),
        CouponPeriod(
            accrual_start=date(2025, 7, 15),
            accrual_end=date(2026, 1, 15),
            payment_date=date(2026, 1, 15),
        ),
    )


class TestStaticLegContractIR:
    def test_static_leg_contract_collects_leg_and_settlement_currencies(self):
        fixed_leg = CouponLeg(
            currency="usd",
            notional_schedule=_notional(1_000_000.0),
            coupon_periods=_fixed_coupon_periods(),
            coupon_formula=FixedCouponFormula(0.05),
            day_count="30/360",
            payment_frequency="semiannual",
        )
        floating_leg = CouponLeg(
            currency="USD",
            notional_schedule=_notional(1_000_000.0),
            coupon_periods=_fixed_coupon_periods(),
            coupon_formula=FloatingCouponFormula(OvernightRateIndex("SOFR")),
            day_count="ACT/360",
            payment_frequency="quarterly",
        )

        contract = StaticLegContractIR(
            legs=(
                SignedLeg(direction="receive", leg=floating_leg),
                SignedLeg(direction="pay", leg=fixed_leg),
            ),
            settlement=SettlementRule(payout_currency="usd", settlement_lag_days=2),
        )

        assert contract.currencies == ("USD",)

    def test_known_cashflow_leg_requires_ordered_dates(self):
        with pytest.raises(StaticLegIRWellFormednessError):
            KnownCashflowLeg(
                currency="USD",
                cashflows=(
                    KnownCashflow(
                        payment_date=date(2030, 1, 15),
                        amount=1_000_000.0,
                        currency="USD",
                    ),
                    KnownCashflow(
                        payment_date=date(2029, 1, 15),
                        amount=50_000.0,
                        currency="USD",
                    ),
                ),
            )

    def test_coupon_leg_rejects_overlapping_coupon_periods(self):
        with pytest.raises(StaticLegIRWellFormednessError):
            CouponLeg(
                currency="USD",
                notional_schedule=_notional(1_000_000.0),
                coupon_periods=(
                    CouponPeriod(
                        accrual_start=date(2025, 1, 15),
                        accrual_end=date(2025, 7, 15),
                        payment_date=date(2025, 7, 15),
                    ),
                    CouponPeriod(
                        accrual_start=date(2025, 7, 1),
                        accrual_end=date(2026, 1, 15),
                        payment_date=date(2026, 1, 15),
                    ),
                ),
                coupon_formula=FixedCouponFormula(0.05),
                day_count="30/360",
                payment_frequency="semiannual",
            )
