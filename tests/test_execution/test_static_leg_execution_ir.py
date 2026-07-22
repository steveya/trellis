from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import pytest

from trellis.agent.contract_ir import CurveQuote, ParRateTenor
from trellis.agent.static_leg_contract import (
    ConditionalAccrualLeg,
    ConditionalAccrualPeriod,
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
    QuotedCouponFormula,
    SettlementRule,
    SignedLeg,
    StaticLegContractIR,
    TermRateIndex,
)
from trellis.agent.semantic_observables import (
    BetweenPredicate,
    ObservationMetadata,
    RateIndexObservable,
)
from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import ExecutionBackedPayoff
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
from trellis.instruments.swap import SwapPayoff, SwapSpec
from trellis.models.rate_basis_swap import price_rate_basis_swap
from trellis.models.rate_cap_floor import price_rate_cap_floor_strip_analytical
from trellis.models.range_accrual import price_range_accrual
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


def _irregular_fixed_float_swap() -> StaticLegContractIR:
    contract = _fixed_float_swap()
    periods = _periods(
        ("2025-06-30", "2025-08-15"),
        ("2025-08-15", "2026-02-15"),
    )
    notional = _notional("2025-06-30", "2026-02-15", 1_000_000.0)
    fixed_signed_leg, floating_signed_leg = contract.legs
    return replace(
        contract,
        legs=(
            replace(
                fixed_signed_leg,
                leg=replace(
                    fixed_signed_leg.leg,
                    notional_schedule=notional,
                    coupon_periods=periods,
                ),
            ),
            replace(
                floating_signed_leg,
                leg=replace(
                    floating_signed_leg.leg,
                    notional_schedule=notional,
                    coupon_periods=periods,
                ),
            ),
        ),
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


def _range_accrual_contract() -> StaticLegContractIR:
    observation_dates = (
        date(2025, 1, 15),
        date(2025, 4, 15),
        date(2025, 7, 15),
        date(2025, 10, 15),
    )
    accrual_starts = (
        date(2024, 10, 15),
        date(2025, 1, 15),
        date(2025, 4, 15),
        date(2025, 7, 15),
    )
    condition = BetweenPredicate(
        observable=RateIndexObservable(
            observable_id="reference_rate_fixing",
            index_name="SOFR",
            observation=ObservationMetadata(
                schedule_role="observation_dates",
                fixing_date_role="fixing_dates",
                missing_fixing_policy="project_forward_for_future_only",
            ),
        ),
        lower_bound=0.01,
        upper_bound=0.08,
    )
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=ConditionalAccrualLeg(
                    currency="USD",
                    notional_schedule=_notional("2024-10-15", "2025-10-15", 1_000_000.0),
                    accrual_periods=tuple(
                        ConditionalAccrualPeriod(
                            accrual_start=start,
                            accrual_end=observation,
                            observation_date=observation,
                            fixing_date=observation,
                            payment_date=observation,
                        )
                        for start, observation in zip(accrual_starts, observation_dates)
                    ),
                    coupon_formula=FixedCouponFormula(0.0525),
                    day_count="ACT/365",
                    payment_frequency="quarterly",
                    accrual_condition_ref="reference_rate_fixing_in_range",
                    accrual_counter_ref="in_range_coupon_count",
                    settlement_rule="coupon_period_cash_settlement",
                    label="range_accrual_coupon",
                    metadata={"semantic_family": "range_accrual"},
                    accrual_condition=condition,
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=KnownCashflowLeg(
                    currency="USD",
                    cashflows=(
                        KnownCashflow(
                            payment_date=observation_dates[-1],
                            amount=1_000_000.0,
                            currency="USD",
                            label="principal_redemption",
                        ),
                    ),
                    label="range_accrual_principal",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
        metadata={"semantic_family": "range_accrual"},
    )


def test_static_leg_compile_emits_route_free_execution_ir_for_fixed_float_swap():
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    assert isinstance(ir, ContractExecutionIR)
    assert ir.source_track.source_kind == "static_leg_contract_ir"
    assert ir.source_track.product_family == "fixed_float_swap"
    assert ir.requirement_hints.market_inputs == frozenset(
        {
            "discount_curve:USD",
            "forward_curve:SOFR",
            "fixing_history:SOFR",
        }
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


def test_irregular_coupon_execution_preserves_exact_periods_and_avoids_regular_spec():
    contract = _irregular_fixed_float_swap()
    market = _market_state()

    ir = compile_static_leg_execution_ir(contract, fail_on_unsupported=True)
    price = price_static_leg_execution_ir(ir, market)

    assert (
        dict(ir.source_track.source_metadata)["static_leg_lowering_declaration_id"]
        == "static_leg_coupon_obligations"
    )
    assert ir.requirement_hints.market_inputs == frozenset(
        {
            "discount_curve:USD",
            "forward_curve:SOFR",
            "fixing_history:SOFR",
        }
    )
    assert (
        tuple(
            period
            for obligation in ir.obligations
            for period in dict(obligation.metadata)["periods"]
        )
        == (
            (
                date(2025, 6, 30),
                date(2025, 8, 15),
                date(2025, 8, 15),
                date(2025, 6, 30),
            ),
            (
                date(2025, 8, 15),
                date(2026, 2, 15),
                date(2026, 2, 15),
                date(2025, 8, 15),
            ),
        )
        * 2
    )

    expected = 0.0
    for signed_leg in contract.legs:
        leg = signed_leg.leg
        day_count = (
            DayCountConvention.THIRTY_360
            if leg.day_count == "30/360"
            else DayCountConvention.ACT_360
        )
        sign = 1.0 if signed_leg.direction == "receive" else -1.0
        for period in leg.coupon_periods:
            if isinstance(leg.coupon_formula, FixedCouponFormula):
                rate = leg.coupon_formula.rate
            else:
                curve = market.forecast_forward_curve("SOFR")
                start = year_fraction(
                    market.settlement, period.accrual_start, day_count
                )
                end = year_fraction(market.settlement, period.accrual_end, day_count)
                rate = curve.forward_rate(max(start, 0.0), max(end, start + 1e-6))
            accrual = year_fraction(period.accrual_start, period.accrual_end, day_count)
            discount_time = year_fraction(
                market.settlement, period.payment_date, day_count
            )
            expected += (
                sign
                * 1_000_000.0
                * float(rate)
                * accrual
                * market.discount.discount(discount_time)
            )

    regular_price = price_payoff(
        SwapPayoff(
            SwapSpec(
                notional=1_000_000.0,
                fixed_rate=0.04,
                start_date=date(2025, 6, 30),
                end_date=date(2026, 2, 15),
                fixed_frequency=Frequency.SEMI_ANNUAL,
                float_frequency=Frequency.QUARTERLY,
                fixed_day_count=DayCountConvention.THIRTY_360,
                float_day_count=DayCountConvention.ACT_360,
                rate_index="SOFR",
                is_payer=True,
            )
        ),
        market,
    )
    assert price == pytest.approx(expected, rel=1e-12, abs=1e-8)
    assert price != pytest.approx(regular_price, rel=1e-8, abs=1e-4)


def test_irregular_floating_coupon_uses_required_historical_fixing():
    contract = _irregular_fixed_float_swap()
    ir = compile_static_leg_execution_ir(contract, fail_on_unsupported=True)
    payoff = ExecutionBackedPayoff(ir)
    fixing_date = date(2025, 6, 30)
    same_day_market = replace(
        _market_state(),
        as_of=fixing_date,
        settlement=fixing_date,
    )
    assert math.isfinite(price_static_leg_execution_ir(ir, same_day_market))

    settlement = date(2025, 7, 15)
    missing_fixing_market = replace(
        _market_state(),
        as_of=settlement,
        settlement=settlement,
    )

    assert payoff.requirements == {
        "discount_curve",
        "fixing_history",
        "forward_curve",
    }
    with pytest.raises(MissingCapabilityError, match="fixing_history"):
        price_payoff(payoff, missing_fixing_market)
    with pytest.raises(ValueError, match="historical fixing"):
        price_static_leg_execution_ir(ir, missing_fixing_market)

    low_fixing_market = replace(
        missing_fixing_market,
        fixing_histories={"SOFR": {fixing_date: 0.01}},
    )
    high_fixing_market = replace(
        missing_fixing_market,
        fixing_histories={"SOFR": {fixing_date: 0.09}},
    )
    low_price = price_static_leg_execution_ir(ir, low_fixing_market)
    high_price = price_static_leg_execution_ir(ir, high_fixing_market)
    first_period = contract.legs[1].leg.coupon_periods[0]
    accrual = year_fraction(
        first_period.accrual_start,
        first_period.accrual_end,
        DayCountConvention.ACT_360,
    )
    discount_time = year_fraction(
        settlement,
        first_period.payment_date,
        DayCountConvention.ACT_360,
    )

    assert high_price - low_price == pytest.approx(
        1_000_000.0
        * (0.09 - 0.01)
        * accrual
        * low_fixing_market.discount.discount(discount_time),
        rel=1e-12,
        abs=1e-8,
    )


def test_irregular_fixed_bond_uses_exact_stub_accruals():
    contract = _fixed_coupon_bond()
    coupon_leg, redemption_leg = contract.legs
    irregular_periods = _periods(
        ("2025-01-15", "2025-08-15"),
        ("2025-08-15", "2026-01-15"),
        ("2026-01-15", "2026-07-15"),
        ("2026-07-15", "2027-01-15"),
    )
    contract = replace(
        contract,
        legs=(
            replace(
                coupon_leg,
                leg=replace(coupon_leg.leg, coupon_periods=irregular_periods),
            ),
            redemption_leg,
        ),
    )
    market = _market_state()

    ir = compile_static_leg_execution_ir(contract, fail_on_unsupported=True)
    price = price_static_leg_execution_ir(ir, market)

    coupon_pv = sum(
        1_000_000.0
        * 0.05
        * year_fraction(
            period.accrual_start,
            period.accrual_end,
            DayCountConvention.ACT_ACT,
        )
        * market.discount.discount(
            year_fraction(
                market.settlement,
                period.payment_date,
                DayCountConvention.ACT_ACT,
            )
        )
        for period in irregular_periods
    )
    principal_pv = 1_000_000.0 * market.discount.discount(
        year_fraction(
            market.settlement,
            date(2027, 1, 15),
            DayCountConvention.ACT_ACT,
        )
    )
    regular_frequency_shortcut = sum(
        1_000_000.0
        * 0.05
        * 0.5
        * market.discount.discount(
            year_fraction(
                market.settlement,
                period.payment_date,
                DayCountConvention.ACT_ACT,
            )
        )
        for period in irregular_periods
    ) + principal_pv

    assert dict(ir.source_track.source_metadata)[
        "static_leg_lowering_declaration_id"
    ] == "static_leg_coupon_obligations"
    assert price == pytest.approx(coupon_pv + principal_pv, rel=1e-12, abs=1e-8)
    assert price != pytest.approx(regular_frequency_shortcut, rel=1e-10, abs=1e-6)


def test_static_leg_compile_emits_conditional_accrual_execution_ir_for_range_accrual():
    ir = compile_static_leg_execution_ir(_range_accrual_contract())

    assert ir.source_track.product_family == "range_accrual"
    assert {obligation.obligation_kind for obligation in ir.obligations} == {
        "conditional_accrual_leg"
    }
    assert ir.requirement_hints.market_inputs == frozenset(
        {"discount_curve:USD", "fixing_history:SOFR", "forward_curve:SOFR"}
    )
    summary = contract_execution_summary(ir)
    assert summary["unsupported_reasons"] == ()
    assert summary["event_kinds"] == ("observation", "payment")
    assert summary["obligation_kinds"] == ("conditional_accrual_leg",)
    assert summary["settlement_kinds"] == ("conditional_accrual_present_value",)


def test_static_leg_compile_fails_closed_for_unadmitted_shape():
    unsupported = StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional("2025-01-15", "2026-01-15", 1_000_000.0),
                    coupon_periods=_periods(("2025-01-15", "2025-07-15")),
                    coupon_formula=QuotedCouponFormula(
                        CurveQuote(
                            curve_id="USD-SWAP",
                            coordinate=ParRateTenor("5Y"),
                            convention="par_rate",
                        )
                    ),
                    day_count="ACT/360",
                    payment_frequency="semiannual",
                ),
            ),
        )
    )

    ir = compile_static_leg_execution_ir(unsupported)
    assert ir.obligations == ()
    assert ir.execution_metadata.unsupported_reasons
    assert "static_coupon_obligation_formula_unsupported" in str(
        ir.execution_metadata.unsupported_reasons
    )

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
        (
            _range_accrual_contract,
            lambda contract, market: price_range_accrual(
                materialize_static_leg_lowering(contract)["call_kwargs"]["spec"],
                as_of=market.settlement,
                discount_curve=market.discount,
                forecast_curve=market.forecast_curves["SOFR"],
                fixing_history=market.fixing_histories["SOFR"]
                if market.fixing_histories
                else {},
                scenario_shifts_bps=(),
            ).price,
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


def test_execution_backed_payoff_prices_static_leg_ir_through_public_payoff_boundary():
    contract = _fixed_float_swap()
    market = replace(_market_state(), fixing_histories={"SOFR": {}})
    ir = compile_static_leg_execution_ir(contract)
    payoff = ExecutionBackedPayoff(ir)

    assert payoff.execution_ir is ir
    assert payoff.requirements == {
        "discount_curve",
        "fixing_history",
        "forward_curve",
    }
    assert price_payoff(payoff, market) == pytest.approx(
        price_static_leg_execution_ir(ir, market),
        rel=1e-12,
        abs=1e-8,
    )


def test_regular_fixed_float_payoff_preflights_fixing_history():
    ir = compile_static_leg_execution_ir(_fixed_float_swap())
    payoff = ExecutionBackedPayoff(ir)
    settlement = date(2025, 7, 15)
    market = replace(
        _market_state(),
        as_of=settlement,
        settlement=settlement,
        fixing_histories={},
    )

    with pytest.raises(MissingCapabilityError, match="fixing_history"):
        price_payoff(payoff, market)


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
