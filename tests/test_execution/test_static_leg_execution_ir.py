from __future__ import annotations

from dataclasses import replace
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
    PeriodRateOptionPeriod,
    PeriodRateOptionStripLeg,
    SignedLeg,
    StaticLegContractIR,
    TermRateIndex,
)
from trellis.conventions.day_count import DayCountConvention
from trellis.core.market_state import MarketState
from trellis.core.types import Frequency
from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.agent.static_leg_admission import materialize_static_leg_lowering
from trellis.execution import (
    ContractExecutionIR,
    compile_static_leg_execution_ir,
    contract_execution_summary,
)
from trellis.execution.compiler import UnsupportedExecutionSemantics
from trellis.execution.runtime import price_static_leg_execution_ir
from trellis.execution.visitors.cashflow_expand import known_cashflow_obligations
from trellis.execution.visitors.normalize import normalize_execution_ir
from trellis.execution.visitors.requirements import derive_requirement_hints
from trellis.execution.visitors.schedule import execution_event_schedule
from trellis.instruments.bond import Bond
from trellis.instruments.swap import SwapPayoff
from trellis.models.rate_basis_swap import price_rate_basis_swap
from trellis.models.rate_cap_floor import price_rate_cap_floor_strip_analytical
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state(rate: float = 0.05, vol: float = 0.20) -> MarketState:
    curve = YieldCurve.flat(rate)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        forecast_curves={
            "SOFR": curve,
            "FF": YieldCurve.flat(rate + 0.002),
            "USD-SOFR-3M": curve,
        },
        vol_surface=FlatVol(vol),
    )


def _date_aware_market_state() -> MarketState:
    discount_curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0325,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    sofr_curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0410,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    ff_curve = DateAwareFlatYieldCurve(
        value_date=SETTLE,
        flat_rate=0.0365,
        curve_day_count=DayCountConvention.ACT_ACT_ISDA,
    )
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=discount_curve,
        forecast_curves={
            "SOFR": sofr_curve,
            "FF": ff_curve,
        },
    )


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
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="pay",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    coupon_periods=_periods(
                        ("2025-06-30", "2025-12-30"),
                        ("2025-12-30", "2026-06-30"),
                    ),
                    coupon_formula=FixedCouponFormula(0.04),
                    day_count="30/360",
                    payment_frequency="semiannual",
                    label="fixed",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(start, end, 1_000_000.0),
                    coupon_periods=_periods(
                        ("2025-06-30", "2025-09-30"),
                        ("2025-09-30", "2025-12-30"),
                        ("2025-12-30", "2026-03-30"),
                        ("2026-03-30", "2026-06-30"),
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
                    label="pay_sofr",
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
                    label="receive_ff",
                ),
            ),
        ),
        metadata={"semantic_family": "basis_swap"},
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
                    label="coupon",
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
        ),
        metadata={"semantic_family": "fixed_coupon_bond"},
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
                    label="cap_strip",
                ),
            ),
        ),
        metadata={"semantic_family": "period_rate_option_strip"},
    )


def test_static_leg_compile_emits_route_free_execution_ir_for_fixed_float_swap():
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    assert isinstance(ir, ContractExecutionIR)
    assert ir.source_track.source_kind == "static_leg_contract_ir"
    assert ir.source_track.product_family == "fixed_float_swap"
    assert ir.requirement_hints.market_inputs == frozenset(
        {"discount_curve:USD", "forward_curve:SOFR"}
    )
    assert {obligation.obligation_kind for obligation in ir.obligations} == {
        "coupon_leg"
    }
    assert {event.event_kind for event in ir.event_plan.events} >= {
        "fixing",
        "payment",
    }

    summary = contract_execution_summary(ir)
    assert summary["unsupported_reasons"] == ()
    assert summary["route_ids"] == ()
    assert summary["model_families"] == ()
    assert summary["obligation_kinds"] == ("coupon_leg",)


def test_static_leg_compile_fails_closed_for_unadmitted_shape():
    unsupported = StaticLegContractIR(
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
                ),
            ),
        )
    )

    ir = compile_static_leg_execution_ir(unsupported)
    assert ir.obligations == ()
    assert ir.execution_metadata.unsupported_reasons

    with pytest.raises(UnsupportedExecutionSemantics):
        compile_static_leg_execution_ir(unsupported, fail_on_unsupported=True)


def test_static_leg_compile_fails_closed_for_admitted_but_unsupported_terms():
    contract = _fixed_float_swap()
    floating_leg = contract.legs[1]
    unsupported = replace(
        contract,
        legs=(
            contract.legs[0],
            replace(
                floating_leg,
                leg=replace(
                    floating_leg.leg,
                    notional_schedule=_notional("2025-06-30", "2030-06-30", 2_000_000.0),
                ),
            ),
        ),
    )

    ir = compile_static_leg_execution_ir(unsupported)
    assert ir.obligations == ()
    assert ir.execution_metadata.unsupported_reasons

    with pytest.raises(UnsupportedExecutionSemantics):
        compile_static_leg_execution_ir(unsupported, fail_on_unsupported=True)


def test_static_leg_visitors_are_deterministic_for_schedule_requirements_and_cashflows():
    ir = compile_static_leg_execution_ir(_fixed_coupon_bond())

    schedule = execution_event_schedule(ir)
    assert schedule == execution_event_schedule(ir)
    assert tuple(item.event_kind for item in schedule).count("payment") == 5

    hints = derive_requirement_hints(ir)
    assert hints.market_inputs == frozenset({"discount_curve:USD"})
    assert hints.timeline_roles == frozenset({"payment_dates"})

    known = known_cashflow_obligations(ir)
    assert tuple(item.obligation_id for item in known) == (
        "known_cashflow:principal_redemption:0",
    )
    assert known[0].amount == pytest.approx(1_000_000.0)

    unsorted_ir = replace(
        ir,
        obligations=tuple(reversed(ir.obligations)),
        observables=tuple(reversed(ir.observables)),
        event_plan=replace(
            ir.event_plan,
            events=tuple(reversed(ir.event_plan.events)),
        ),
    )
    normalized = normalize_execution_ir(unsorted_ir)
    assert normalized == normalize_execution_ir(normalized)
    assert execution_event_schedule(normalized) == schedule


@pytest.mark.parametrize(
    ("contract_factory", "legacy_price"),
    (
        (
            _fixed_float_swap,
            lambda contract, market: price_payoff(
                SwapPayoff(
                    materialize_static_leg_lowering(contract)["call_kwargs"]["spec"]
                ),
                market,
            ),
        ),
        (
            _basis_swap,
            lambda contract, market: price_rate_basis_swap(
                market,
                materialize_static_leg_lowering(contract)["call_kwargs"]["spec"],
            ),
        ),
        (
            _fixed_coupon_bond,
            lambda contract, market: Bond(
                **materialize_static_leg_lowering(contract)["call_kwargs"]
            ).price(market.discount, settlement=market.settlement),
        ),
        (
            _period_rate_option_strip,
            lambda contract, market: price_rate_cap_floor_strip_analytical(
                market,
                **materialize_static_leg_lowering(contract)["call_kwargs"],
            ),
        ),
    ),
)
def test_static_leg_runtime_matches_existing_checked_helpers(contract_factory, legacy_price):
    contract = contract_factory()
    market = _market_state()
    ir = compile_static_leg_execution_ir(contract)

    assert price_static_leg_execution_ir(ir, market) == pytest.approx(
        legacy_price(contract, market),
        rel=1e-12,
        abs=1e-8,
    )


def test_static_leg_runtime_matches_basis_swap_helper_with_date_aware_curves():
    contract = _basis_swap()
    market = _date_aware_market_state()
    ir = compile_static_leg_execution_ir(contract)

    assert price_static_leg_execution_ir(ir, market) == pytest.approx(
        price_rate_basis_swap(
            market,
            materialize_static_leg_lowering(contract)["call_kwargs"]["spec"],
        ),
        rel=1e-12,
        abs=1e-8,
    )


def test_execution_artifact_can_be_reused_for_curve_bump_repricing():
    market = _market_state(rate=0.05)
    bumped = replace(market, discount=YieldCurve.flat(0.051), forecast_curves={"SOFR": YieldCurve.flat(0.051)})
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    base_price = price_static_leg_execution_ir(ir, market)
    bumped_price = price_static_leg_execution_ir(ir, bumped)

    assert base_price != pytest.approx(bumped_price)
