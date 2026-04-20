from __future__ import annotations

from datetime import date

import pytest

from trellis.agent.static_leg_admission import (
    StaticLegLoweringNoMatchError,
    materialize_static_leg_lowering,
    select_static_leg_lowering,
)
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
    SignedLeg,
    StaticLegContractIR,
)
from trellis.instruments.swap import SwapSpec


def _notional(start_day: str, end_day: str, amount: float) -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(
                start_date=date.fromisoformat(start_day),
                end_date=date.fromisoformat(end_day),
                amount=amount,
            ),
        )
    )


def _periods(*bounds: tuple[str, str]) -> tuple[CouponPeriod, ...]:
    return tuple(
        CouponPeriod(
            accrual_start=date.fromisoformat(start_day),
            accrual_end=date.fromisoformat(end_day),
            payment_date=date.fromisoformat(end_day),
            fixing_date=date.fromisoformat(start_day),
        )
        for start_day, end_day in bounds
    )


def _fixed_float_swap() -> StaticLegContractIR:
    start = "2025-06-30"
    end = "2030-06-30"
    fixed_periods = _periods(
        ("2025-06-30", "2025-12-30"),
        ("2025-12-30", "2026-06-30"),
    )
    float_periods = _periods(
        ("2025-06-30", "2025-09-30"),
        ("2025-09-30", "2025-12-30"),
        ("2025-12-30", "2026-03-30"),
        ("2026-03-30", "2026-06-30"),
    )
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="pay",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    coupon_periods=fixed_periods,
                    coupon_formula=FixedCouponFormula(0.04),
                    day_count="30/360",
                    payment_frequency="semiannual",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    coupon_periods=float_periods,
                    coupon_formula=FloatingCouponFormula(OvernightRateIndex("SOFR")),
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                ),
            ),
        )
    )


def _basis_swap() -> StaticLegContractIR:
    start = "2025-06-30"
    end = "2030-06-30"
    periods = _periods(
        ("2025-06-30", "2025-09-30"),
        ("2025-09-30", "2025-12-30"),
    )
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="pay",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    coupon_periods=periods,
                    coupon_formula=FloatingCouponFormula(OvernightRateIndex("SOFR")),
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    coupon_periods=periods,
                    coupon_formula=FloatingCouponFormula(
                        OvernightRateIndex("FF"),
                        spread=0.0025,
                    ),
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                ),
            ),
        )
    )


def _fixed_coupon_bond() -> StaticLegContractIR:
    periods = _periods(
        ("2025-01-15", "2025-07-15"),
        ("2025-07-15", "2026-01-15"),
        ("2026-01-15", "2026-07-15"),
        ("2026-07-15", "2027-01-15"),
    )
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional("2025-01-15", "2027-01-15", 1_000_000.0),
                    coupon_periods=periods,
                    coupon_formula=FixedCouponFormula(0.05),
                    day_count="ACT/ACT",
                    payment_frequency="semiannual",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=KnownCashflowLeg(
                    currency="USD",
                    cashflows=(
                        KnownCashflow(
                            payment_date=date(2027, 1, 15),
                            amount=1_000_000.0,
                            currency="USD",
                            label="principal_redemption",
                        ),
                    ),
                ),
            ),
        )
    )


class TestStaticLegAdmission:
    def test_fixed_float_swap_selection_materializes_checked_swap_payoff(self):
        contract = _fixed_float_swap()

        selection = select_static_leg_lowering(contract)
        materialized = materialize_static_leg_lowering(contract, selection=selection)

        assert selection.declaration_id == "static_leg_fixed_float_swap"
        assert materialized["callable_ref"] == "trellis.instruments.swap.SwapPayoff"
        assert isinstance(materialized["call_kwargs"]["spec"], SwapSpec)
        assert materialized["call_kwargs"]["spec"].is_payer is True
        assert materialized["call_kwargs"]["spec"].fixed_rate == pytest.approx(0.04)

    def test_basis_swap_selection_is_explicit_but_remains_non_executable(self):
        selection = select_static_leg_lowering(_basis_swap())

        assert selection.declaration_id == "static_leg_basis_swap"
        with pytest.raises(NotImplementedError):
            materialize_static_leg_lowering(_basis_swap(), selection=selection)

    def test_fixed_coupon_bond_selection_materializes_bond_kwargs(self):
        contract = _fixed_coupon_bond()

        selection = select_static_leg_lowering(contract)
        materialized = materialize_static_leg_lowering(contract, selection=selection)

        assert selection.declaration_id == "static_leg_fixed_coupon_bond"
        assert materialized["callable_ref"] == "trellis.instruments.bond.Bond"
        assert materialized["call_kwargs"]["coupon"] == pytest.approx(0.05)
        assert materialized["call_kwargs"]["maturity_date"] == date(2027, 1, 15)

    def test_no_match_for_quote_linked_coupon_leg_until_richer_formula_lane_exists(self):
        contract = StaticLegContractIR(
            legs=(
                SignedLeg(
                    direction="receive",
                    leg=CouponLeg(
                        currency="USD",
                        notional_schedule=_notional("2025-01-15", "2026-01-15", 1_000_000.0),
                        coupon_periods=_periods(("2025-01-15", "2025-07-15")),
                        coupon_formula=FixedCouponFormula(0.0),
                        day_count="ACT/360",
                        payment_frequency="semiannual",
                        label="placeholder",
                    ),
                ),
            )
        )

        with pytest.raises(StaticLegLoweringNoMatchError):
            select_static_leg_lowering(contract)
