"""Bounded FpML European physical swaption normalization tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "fpml"
    / "confirmation_5_13_european_swaption.xml"
)


def _native_underlying_swap():
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        CouponPeriod,
        FixedCouponFormula,
        FloatingCouponFormula,
        NotionalSchedule,
        NotionalStep,
        SettlementRule,
        SignedLeg,
        StaticLegContractIR,
        TermRateIndex,
    )
    from trellis.conventions.schedule import generate_schedule
    from trellis.core.types import Frequency

    start = date(2025, 6, 30)
    end = date(2027, 6, 30)
    notional = NotionalSchedule((NotionalStep(start, end, 1_000_000.0),))

    def periods(frequency, *, floating: bool):
        ends = tuple(generate_schedule(start, end, frequency))
        starts = (start, *ends[:-1])
        return tuple(
            CouponPeriod(
                accrual_start=left,
                accrual_end=right,
                payment_date=right,
                fixing_date=left if floating else None,
            )
            for left, right in zip(starts, ends)
        )

    return StaticLegContractIR(
        legs=(
            SignedLeg(
                "pay",
                CouponLeg(
                    currency="USD",
                    notional_schedule=notional,
                    coupon_periods=periods(Frequency.SEMI_ANNUAL, floating=False),
                    coupon_formula=FixedCouponFormula(0.04),
                    day_count="30/360",
                    payment_frequency="semiannual",
                    label="fixed_leg",
                ),
            ),
            SignedLeg(
                "receive",
                CouponLeg(
                    currency="USD",
                    notional_schedule=notional,
                    coupon_periods=periods(Frequency.QUARTERLY, floating=True),
                    coupon_formula=FloatingCouponFormula(
                        TermRateIndex("USD-SOFR", "3M")
                    ),
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                    label="floating_leg",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
        metadata={"semantic_family": "fixed_float_swap"},
    )


def _native_contract(*, position: str = "long"):
    from trellis.agent.contract_ir import (
        Annuity,
        Constant,
        ContractIR,
        Exercise,
        FiniteSchedule,
        ForwardRate,
        Max,
        Observation,
        Scaled,
        Singleton,
        Strike,
        Sub,
        SwapRate,
        Underlying,
    )
    from trellis.agent.static_leg_contract import SettlementRule

    underlier_id = "USD-IRS-20250630-20270630"
    expiry = Singleton(date(2025, 6, 27))
    fixed_dates = FiniteSchedule(
        (
            date(2025, 12, 30),
            date(2026, 6, 30),
            date(2026, 12, 30),
            date(2027, 6, 30),
        )
    )
    return ContractIR(
        payoff=Scaled(
            Annuity(underlier_id, fixed_dates),
            Max(
                (
                    Sub(SwapRate(underlier_id, fixed_dates), Strike(0.04)),
                    Constant(0.0),
                )
            ),
        ),
        exercise=Exercise("european", expiry),
        observation=Observation("terminal", expiry),
        underlying=Underlying(ForwardRate(underlier_id, "lognormal_forward")),
        position=position,
        settlement=SettlementRule(
            settlement_kind="physical",
            payout_currency="USD",
        ),
        underlying_contract=_native_underlying_swap(),
    )


def _normalize(
    xml: bytes | None = None,
    *,
    valuation_party_id: str | None = "PARTY-A",
    valuation_date: date | None = date(2025, 1, 15),
):
    from trellis.io.fpml import normalize_fpml_document

    return normalize_fpml_document(
        xml if xml is not None else FIXTURE.read_bytes(),
        declared_view="confirmation",
        declared_version="5-13",
        valuation_party_id=valuation_party_id,
        valuation_date=valuation_date,
        require_valuation_date=True,
    )


def _blocker_ids(report) -> tuple[str, ...]:
    return tuple(blocker.id for blocker in report.blockers)


def test_normalizes_physical_european_payer_swaption_to_native_identity():
    from trellis.agent.contract_ir import ContractIR, contract_ir_economic_identity

    report = _normalize()
    native = _native_contract()

    assert report.status == "normalized"
    assert report.blockers == ()
    assert isinstance(report.normalized_contract, ContractIR)
    assert report.normalized_contract == native
    assert report.economic_identity == contract_ir_economic_identity(native)
    assert report.economic_identity.startswith("contract_ir:v1:")
    assert {item.semantic_field for item in report.mapping_provenance} >= {
        "position",
        "settlement.settlement_kind",
        "exercise.schedule.t",
        "underlying_contract",
    }


def test_seller_valuation_changes_position_without_reversing_underlying_swap():
    buyer = _normalize(valuation_party_id="PARTY-A")
    seller = _normalize(valuation_party_id="PARTY-B")

    assert buyer.normalized_contract.position == "long"
    assert seller.normalized_contract.position == "short"
    assert (
        buyer.normalized_contract.underlying_contract
        == seller.normalized_contract.underlying_contract
    )
    assert buyer.normalized_contract.payoff == seller.normalized_contract.payoff
    assert buyer.economic_identity != seller.economic_identity
    assert [
        item.normalized_value
        for item in seller.mapping_provenance
        if item.semantic_field == "valuation_party_id"
    ] == ["PARTY-B"]
    assert any(
        item.semantic_field.startswith("underlying_contract.legs[")
        for item in seller.mapping_provenance
    )


def test_absent_optional_swaption_straddle_normalizes_as_non_straddle():
    xml = FIXTURE.read_text().replace(
        "      <swaptionStraddle>false</swaptionStraddle>\n",
        "",
    ).encode()

    assert _normalize(xml).normalized_contract == _native_contract()


def test_imported_and_native_swaption_select_the_same_existing_route():
    from trellis.agent.contract_ir_solver_compiler import select_contract_ir_solver

    imported = select_contract_ir_solver(_normalize().normalized_contract)
    native = select_contract_ir_solver(_native_contract())

    assert imported == native
    assert imported.declaration_id == "swaption_payer_black76_resolved_kernel"


def test_imported_and_native_swaption_price_identically():
    from trellis.agent.contract_ir_solver_compiler import ContractIRPricingPayoff
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol

    market = MarketState(
        as_of=date(2025, 1, 15),
        settlement=date(2025, 1, 15),
        discount=YieldCurve.flat(0.035),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.038)},
        vol_surface=FlatVol(0.20),
    )
    imported = ContractIRPricingPayoff(_normalize().normalized_contract)
    native = ContractIRPricingPayoff(_native_contract())
    seller = ContractIRPricingPayoff(
        _normalize(valuation_party_id="PARTY-B").normalized_contract
    )

    imported_price = imported.evaluate(market)
    native_price = native.evaluate(market)
    seller_price = seller.evaluate(market)

    assert imported_price == pytest.approx(native_price, rel=1e-12, abs=1e-8)
    assert seller_price == pytest.approx(-imported_price, rel=1e-12, abs=1e-8)


def test_normalized_swaption_report_summary_is_body_free():
    from trellis.io.fpml import fpml_import_report_summary

    xml = FIXTURE.read_bytes()
    summary = fpml_import_report_summary(_normalize(xml))

    assert summary["normalized_contract"]["contract_type"] == "ContractIR"
    assert summary["economic_identity"].startswith("contract_ir:v1:")
    assert xml.decode("utf-8") not in repr(summary)


def _with_premium(*, payment_date: str) -> bytes:
    premium = f"""      <premium>
        <payerPartyReference href="PARTY-A" />
        <receiverPartyReference href="PARTY-B" />
        <paymentAmount>
          <currency>USD</currency>
          <amount>12500</amount>
        </paymentAmount>
        <paymentDate>
          <unadjustedDate>{payment_date}</unadjustedDate>
          <dateAdjustments>
            <businessDayConvention>NONE</businessDayConvention>
          </dateAdjustments>
        </paymentDate>
      </premium>
"""
    return FIXTURE.read_text().replace(
        "      <europeanExercise>",
        premium + "      <europeanExercise>",
    ).encode()


def test_historical_premium_is_reported_without_changing_contract_identity():
    without_premium = _normalize()
    with_premium = _normalize(_with_premium(payment_date="2025-01-14"))

    assert with_premium.economic_identity == without_premium.economic_identity
    assert with_premium.normalized_contract == without_premium.normalized_contract
    assert len(with_premium.premium_metadata) == 1
    assert with_premium.premium_metadata[0].amount == 12_500.0
    assert with_premium.premium_metadata[0].payment_date == date(2025, 1, 14)


def test_unsettled_premium_blocks_before_pricing():
    report = _normalize(_with_premium(payment_date="2025-01-16"))

    assert _blocker_ids(report) == (
        "external_import:fpml_swaption_unsettled_premium_unsupported",
    )


@pytest.mark.parametrize(
    ("replacements", "expected_id"),
    (
        (
            (
                ("<europeanExercise>", "<bermudaExercise>"),
                ("</europeanExercise>", "</bermudaExercise>"),
            ),
            "external_import:fpml_swaption_exercise_style_unsupported",
        ),
        (
            (("<physicalSettlement />", "<cashSettlement />"),),
            "external_import:fpml_swaption_cash_settlement_unsupported",
        ),
        (
            (
                (
                    "<swaptionStraddle>false</swaptionStraddle>",
                    "<swaptionStraddle>true</swaptionStraddle>",
                ),
            ),
            "external_import:fpml_swaption_straddle_unsupported",
        ),
    ),
)
def test_swaption_normalization_blocks_unadmitted_contract_shapes(
    replacements,
    expected_id,
):
    xml = FIXTURE.read_text()
    for old, new in replacements:
        xml = xml.replace(old, new)

    assert _blocker_ids(_normalize(xml.encode())) == (expected_id,)


def test_swaption_normalization_blocks_partial_exercise_terms():
    xml = FIXTURE.read_text().replace(
        "<expirationDate>",
        "<partialExercise><notionalReference href=\"FIXED-LEG\" /></partialExercise>\n"
        "        <expirationDate>",
    ).encode()

    assert _blocker_ids(_normalize(xml)) == (
        "external_import:fpml_european_exercise_feature_unsupported",
    )


def test_swaption_normalization_requires_buyer_and_seller_counterparties():
    xml = FIXTURE.read_text().replace(
        '      <buyerPartyReference href="PARTY-A" />\n',
        "",
    ).encode()

    report = _normalize(xml)

    assert _blocker_ids(report) == (
        "missing_contract_field:fpml_buyer_party_reference",
    )
    assert report.clarification.missing_fields == ("buyer_party_reference",)
