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
    PeriodRateOptionPeriod,
    PeriodRateOptionStripLeg,
    SignedLeg,
    StaticLegContractIR,
    TermRateIndex,
)
from trellis.instruments.swap import SwapSpec
from trellis.models.rate_basis_swap import RateBasisSwapSpec
from trellis.models.rate_cap_floor import CapFloorPeriod
from trellis.core.types import Frequency
from trellis.conventions.day_count import DayCountConvention


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


def _period_rate_option_strip() -> StaticLegContractIR:
    start = "2025-02-15"
    end = "2026-02-15"
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=PeriodRateOptionStripLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    option_periods=(
                        PeriodRateOptionPeriod(
                            accrual_start=date(2025, 2, 15),
                            accrual_end=date(2025, 5, 15),
                            fixing_date=date(2025, 2, 15),
                            payment_date=date(2025, 5, 15),
                        ),
                        PeriodRateOptionPeriod(
                            accrual_start=date(2025, 5, 15),
                            accrual_end=date(2025, 8, 15),
                            fixing_date=date(2025, 5, 15),
                            payment_date=date(2025, 8, 15),
                        ),
                    ),
                    rate_index=TermRateIndex("USD-SOFR", "3M"),
                    strike=0.04,
                    option_side="call",
                    day_count="ACT/360",
                    payment_frequency="quarterly",
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

    def test_basis_swap_selection_materializes_checked_basis_swap_spec(self):
        contract = _basis_swap()

        selection = select_static_leg_lowering(contract)
        materialized = materialize_static_leg_lowering(contract, selection=selection)

        assert selection.declaration_id == "static_leg_basis_swap"
        assert materialized["callable_ref"] == "trellis.models.rate_basis_swap.price_rate_basis_swap"
        assert isinstance(materialized["call_kwargs"]["spec"], RateBasisSwapSpec)
        assert materialized["call_kwargs"]["spec"].pay_leg.rate_index == "SOFR"
        assert materialized["call_kwargs"]["spec"].receive_leg.rate_index == "FF"
        assert materialized["call_kwargs"]["spec"].receive_leg.spread == pytest.approx(0.0025)

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

    def test_period_rate_option_strip_defaults_to_checked_analytical_lowering(self):
        contract = _period_rate_option_strip()

        selection = select_static_leg_lowering(contract)
        materialized = materialize_static_leg_lowering(contract, selection=selection)

        assert selection.declaration_id == "static_leg_period_rate_option_strip_analytical"
        assert (
            materialized["callable_ref"]
            == "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical"
        )
        assert materialized["call_kwargs"] == {
            "instrument_class": "cap",
            "periods": (
                CapFloorPeriod(
                    start_date=date(2025, 2, 15),
                    end_date=date(2025, 5, 15),
                    payment_date=date(2025, 5, 15),
                    fixing_date=date(2025, 2, 15),
                ),
                CapFloorPeriod(
                    start_date=date(2025, 5, 15),
                    end_date=date(2025, 8, 15),
                    payment_date=date(2025, 8, 15),
                    fixing_date=date(2025, 5, 15),
                ),
            ),
            "notional": pytest.approx(1_000_000.0),
            "strike": pytest.approx(0.04),
            "start_date": date(2025, 2, 15),
            "end_date": date(2025, 8, 15),
            "frequency": Frequency.QUARTERLY,
            "day_count": DayCountConvention.ACT_360,
            "rate_index": "USD-SOFR-3M",
        }

    def test_period_rate_option_strip_materializes_model_terms_for_analytical_helper(self):
        contract = _period_rate_option_strip()

        selection = select_static_leg_lowering(contract, requested_method="analytical")
        materialized = materialize_static_leg_lowering(
            contract,
            selection=selection,
            normalized_terms={
                "calendar_name": "weekend_only",
                "business_day_adjustment": "following",
                "model": "shifted_black",
                "shift": 0.01,
                "sabr": {
                    "alpha": 0.025,
                    "beta": 0.5,
                    "rho": -0.2,
                    "nu": 0.35,
                },
            },
        )

        assert selection.declaration_id == "static_leg_period_rate_option_strip_analytical"
        assert materialized["call_kwargs"]["calendar_name"] == "weekend_only"
        assert materialized["call_kwargs"]["business_day_adjustment"] == "following"
        assert materialized["call_kwargs"]["model"] == "shifted_black"
        assert materialized["call_kwargs"]["shift"] == pytest.approx(0.01)
        assert materialized["call_kwargs"]["sabr"] == {
            "alpha": 0.025,
            "beta": 0.5,
            "rho": -0.2,
            "nu": 0.35,
        }

    def test_period_rate_option_strip_can_select_checked_monte_carlo_lowering(self):
        contract = _period_rate_option_strip()

        selection = select_static_leg_lowering(contract, requested_method="monte_carlo")
        materialized = materialize_static_leg_lowering(
            contract,
            selection=selection,
            normalized_terms={"n_paths": 2048, "seed": 17},
        )

        assert selection.declaration_id == "static_leg_period_rate_option_strip_monte_carlo"
        assert (
            materialized["callable_ref"]
            == "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo"
        )
        assert materialized["call_kwargs"]["instrument_class"] == "cap"
        assert materialized["call_kwargs"]["n_paths"] == 2048
        assert materialized["call_kwargs"]["seed"] == 17
        assert "model" not in materialized["call_kwargs"]
        assert "shift" not in materialized["call_kwargs"]
        assert "sabr" not in materialized["call_kwargs"]

    def test_period_rate_option_strip_with_step_notional_fails_closed(self):
        contract = StaticLegContractIR(
            legs=(
                SignedLeg(
                    direction="receive",
                    leg=PeriodRateOptionStripLeg(
                        currency="USD",
                        notional_schedule=NotionalSchedule(
                            (
                                NotionalStep(
                                    start_date=date(2025, 2, 15),
                                    end_date=date(2025, 5, 15),
                                    amount=1_000_000.0,
                                ),
                                NotionalStep(
                                    start_date=date(2025, 5, 15),
                                    end_date=date(2025, 8, 15),
                                    amount=900_000.0,
                                ),
                            )
                        ),
                        option_periods=(
                            PeriodRateOptionPeriod(
                                accrual_start=date(2025, 2, 15),
                                accrual_end=date(2025, 5, 15),
                                fixing_date=date(2025, 2, 15),
                                payment_date=date(2025, 5, 15),
                            ),
                            PeriodRateOptionPeriod(
                                accrual_start=date(2025, 5, 15),
                                accrual_end=date(2025, 8, 15),
                                fixing_date=date(2025, 5, 15),
                                payment_date=date(2025, 8, 15),
                            ),
                        ),
                        rate_index=TermRateIndex("USD-SOFR", "3M"),
                        strike=0.04,
                        option_side="call",
                        day_count="ACT/360",
                        payment_frequency="quarterly",
                    ),
                ),
            )
        )

        with pytest.raises(StaticLegLoweringNoMatchError):
            select_static_leg_lowering(contract)

    def test_single_period_caplet_style_strip_fails_closed(self):
        contract = StaticLegContractIR(
            legs=(
                SignedLeg(
                    direction="receive",
                    leg=PeriodRateOptionStripLeg(
                        currency="USD",
                        notional_schedule=_notional("2025-02-15", "2025-05-15", 1_000_000.0),
                        option_periods=(
                            PeriodRateOptionPeriod(
                                accrual_start=date(2025, 2, 15),
                                accrual_end=date(2025, 5, 15),
                                fixing_date=date(2025, 2, 15),
                                payment_date=date(2025, 5, 15),
                            ),
                        ),
                        rate_index=TermRateIndex("USD-SOFR", "3M"),
                        strike=0.04,
                        option_side="call",
                        day_count="ACT/360",
                        payment_frequency="quarterly",
                    ),
                ),
            )
        )

        with pytest.raises(StaticLegLoweringNoMatchError):
            select_static_leg_lowering(contract)
