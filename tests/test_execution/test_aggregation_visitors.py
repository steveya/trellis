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
    SignedLeg,
    StaticLegContractIR,
)
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.execution import (
    build_future_value_cube_from_execution_ir,
    compile_static_leg_execution_ir,
    price_static_leg_execution_ir,
)
from trellis.execution.visitors.aggregation import (
    summarize_discounted_execution_ir,
    summarize_future_value_execution_ir,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.042, max_tenor=10.0),
        forecast_curves={"SOFR": YieldCurve.flat(0.046, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
    )


def _notional(amount: float) -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(
                start_date=SETTLE,
                end_date=date(2027, 11, 15),
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
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="pay",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(1_000_000.0),
                    coupon_periods=_periods(
                        ("2024-11-15", "2025-05-15"),
                        ("2025-05-15", "2025-11-15"),
                        ("2025-11-15", "2026-05-15"),
                        ("2026-05-15", "2026-11-15"),
                        ("2026-11-15", "2027-05-15"),
                        ("2027-05-15", "2027-11-15"),
                    ),
                    coupon_formula=FixedCouponFormula(0.045),
                    day_count="30/360",
                    payment_frequency="semiannual",
                    label="fixed",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(1_000_000.0),
                    coupon_periods=_periods(
                        ("2024-11-15", "2025-02-15"),
                        ("2025-02-15", "2025-05-15"),
                        ("2025-05-15", "2025-08-15"),
                        ("2025-08-15", "2025-11-15"),
                        ("2025-11-15", "2026-02-15"),
                        ("2026-02-15", "2026-05-15"),
                        ("2026-05-15", "2026-08-15"),
                        ("2026-08-15", "2026-11-15"),
                        ("2026-11-15", "2027-02-15"),
                        ("2027-02-15", "2027-05-15"),
                        ("2027-05-15", "2027-08-15"),
                        ("2027-08-15", "2027-11-15"),
                    ),
                    coupon_formula=FloatingCouponFormula(OvernightRateIndex("SOFR")),
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                    label="float",
                ),
            ),
        ),
        metadata={"semantic_family": "fixed_float_swap"},
    )


def _fixed_coupon_bond() -> StaticLegContractIR:
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(1_000_000.0),
                    coupon_periods=_periods(
                        ("2024-11-15", "2025-05-15"),
                        ("2025-05-15", "2025-11-15"),
                    ),
                    coupon_formula=FixedCouponFormula(0.05),
                    day_count="ACT/ACT",
                    payment_frequency="semiannual",
                    label="coupon",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=KnownCashflowLeg(
                    currency="USD",
                    cashflows=(
                        KnownCashflow(
                            payment_date=date(2025, 11, 15),
                            amount=1_000_000.0,
                            currency="USD",
                            label="principal_redemption",
                        ),
                    ),
                    label="principal",
                ),
            ),
        ),
        metadata={"semantic_family": "fixed_coupon_bond"},
    )


def test_discounted_execution_summary_matches_runtime_price():
    market_state = _market_state()
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    summary = summarize_discounted_execution_ir(ir, market_state)

    assert summary.source_kind == "static_leg_contract_ir"
    assert summary.product_family == "fixed_float_swap"
    assert summary.currency == "USD"
    assert summary.market_inputs == ("discount_curve:USD", "forward_curve:SOFR")
    assert summary.timeline_roles == ("fixing_dates", "payment_dates")
    assert summary.obligation_kinds == ("coupon_leg",)
    assert summary.present_value == pytest.approx(
        price_static_leg_execution_ir(ir, market_state),
        rel=1e-12,
        abs=1e-10,
    )
    assert summary.payment_dates[0] == date(2025, 2, 15)
    assert summary.payment_dates[-1] == date(2027, 11, 15)
    assert summary.compute_plan["aggregation_family"] == "discounted_execution_summary"


def test_future_value_execution_summary_matches_execution_backed_cube():
    market_state = _market_state()
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    summary = summarize_future_value_execution_ir(
        ir,
        market_state,
        position_name="payer_swap",
        n_paths=512,
        n_steps=96,
        seed=19,
        pfe_levels=(0.95,),
    )
    cube = build_future_value_cube_from_execution_ir(
        ir,
        market_state,
        position_name="payer_swap",
        n_paths=512,
        n_steps=96,
        seed=19,
    )

    assert summary.position_name == "payer_swap"
    assert summary.currency == "USD"
    assert summary.observation_dates == cube.observation_dates
    assert summary.expected_positive_exposure == pytest.approx(
        tuple(cube.expected_positive_exposure()),
        rel=0.0,
        abs=1e-12,
    )
    assert summary.expected_portfolio_value == pytest.approx(
        tuple(cube.portfolio_values().mean(axis=1)),
        rel=0.0,
        abs=1e-12,
    )
    assert summary.current_value == pytest.approx(
        price_static_leg_execution_ir(ir, market_state),
        rel=1e-12,
        abs=1e-10,
    )
    assert summary.terminal_value == pytest.approx(0.0, abs=1e-10)
    assert summary.potential_future_exposure[0][0] == pytest.approx(0.95)
    assert summary.potential_future_exposure[0][1] == pytest.approx(
        tuple(cube.potential_future_exposure(0.95)),
        rel=0.0,
        abs=1e-12,
    )
    assert summary.compute_plan["aggregation_family"] == "future_value_execution_summary"
    assert summary.position_provenance["bridge_family"] == "execution_ir"


def test_future_value_execution_summary_fails_closed_for_unsupported_bridge_family():
    market_state = _market_state()
    ir = compile_static_leg_execution_ir(_fixed_coupon_bond())

    with pytest.raises(ValueError, match="fixed_float_swap"):
        summarize_future_value_execution_ir(ir, market_state)
