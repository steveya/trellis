"""Bounded FpML cap/floor strip normalization tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "fpml"
    / "confirmation_5_13_cap_floor.xml"
)


def _native_contract(
    *,
    option_side: str = "call",
    direction: str = "receive",
):
    from trellis.agent.static_leg_contract import (
        NotionalSchedule,
        NotionalStep,
        PeriodRateOptionPeriod,
        PeriodRateOptionStripLeg,
        SettlementRule,
        SignedLeg,
        StaticLegContractIR,
        TermRateIndex,
    )
    from trellis.conventions.schedule import generate_schedule
    from trellis.core.types import Frequency

    start = date(2025, 6, 30)
    end = date(2027, 6, 30)
    ends = tuple(generate_schedule(start, end, Frequency.QUARTERLY))
    starts = (start, *ends[:-1])
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction,
                PeriodRateOptionStripLeg(
                    currency="USD",
                    notional_schedule=NotionalSchedule(
                        (NotionalStep(start, end, 1_000_000.0),)
                    ),
                    option_periods=tuple(
                        PeriodRateOptionPeriod(
                            accrual_start=left,
                            accrual_end=right,
                            fixing_date=left,
                            payment_date=right,
                        )
                        for left, right in zip(starts, ends)
                    ),
                    rate_index=TermRateIndex("USD-SOFR", "3M"),
                    strike=0.04,
                    option_side=option_side,
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                    label="cap_strip" if option_side == "call" else "floor_strip",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
        metadata={"semantic_family": "period_rate_option_strip"},
    )


def _normalize(
    xml: bytes | None = None,
    *,
    valuation_party_id: str = "PARTY-A",
    valuation_date: date = date(2025, 1, 15),
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


def _floor_xml() -> bytes:
    return (
        FIXTURE.read_text()
        .replace("<capRateSchedule>", "<floorRateSchedule>")
        .replace("</capRateSchedule>", "</floorRateSchedule>")
        .encode()
    )


def test_normalizes_cap_floor_stream_to_native_strip_identity():
    from trellis.agent.static_leg_contract import static_leg_economic_identity

    report = _normalize()
    native = _native_contract()

    assert report.status == "normalized"
    assert report.blockers == ()
    assert report.normalized_contract == native
    assert report.economic_identity == static_leg_economic_identity(native)
    assert {item.semantic_field for item in report.mapping_provenance} >= {
        "legs[0].direction",
        "legs[0].currency",
        "legs[0].day_count",
        "legs[0].notional_schedule.steps[0].amount",
        "legs[0].notional_schedule.steps[0].start_date",
        "legs[0].notional_schedule.steps[0].end_date",
        "legs[0].option_periods",
        "legs[0].payment_frequency",
        "legs[0].rate_index",
        "legs[0].strike",
        "legs[0].option_side",
    }


def test_normalizes_floor_schedule_to_put_strip():
    report = _normalize(_floor_xml())

    assert report.normalized_contract == _native_contract(option_side="put")


def test_cap_floor_seller_valuation_reverses_strip_sign_only():
    buyer = _normalize()
    seller = _normalize(valuation_party_id="PARTY-B")

    assert buyer.normalized_contract == _native_contract(direction="receive")
    assert seller.normalized_contract == _native_contract(direction="pay")
    assert buyer.normalized_contract.legs[0].leg == seller.normalized_contract.legs[0].leg
    assert buyer.economic_identity != seller.economic_identity


def test_cap_floor_normalization_reports_cap_floor_counterparty_conflict():
    xml = FIXTURE.read_text().replace(
        '<receiverPartyReference href="PARTY-B" />',
        '<receiverPartyReference href="PARTY-A" />',
        1,
    )

    assert _blocker_ids(_normalize(xml.encode())) == (
        "contract_conflict:fpml_cap_floor_stream_parties",
    )


def test_cap_floor_identity_ignores_source_party_and_product_identifiers():
    xml = (
        FIXTURE.read_text()
        .replace("PARTY-A", "BANK-X")
        .replace("PARTY-B", "BANK-Y")
        .replace("CAP-001", "EXTERNAL-CAP-999")
        .encode()
    )

    baseline = _normalize()
    relabeled = _normalize(xml, valuation_party_id="BANK-X")

    assert relabeled.normalized_contract == baseline.normalized_contract
    assert relabeled.economic_identity == baseline.economic_identity


def test_cap_floor_normalization_rejects_collar():
    floor_schedule = """              <floorRateSchedule>
                <initialValue>0.02</initialValue>
                <buyer>Receiver</buyer>
                <seller>Payer</seller>
              </floorRateSchedule>
"""
    xml = FIXTURE.read_text().replace(
        "            </floatingRateCalculation>",
        floor_schedule + "            </floatingRateCalculation>",
    ).encode()

    assert _blocker_ids(_normalize(xml)) == (
        "external_import:fpml_cap_floor_collar_unsupported",
    )


def test_cap_floor_normalization_requires_one_strike_schedule():
    xml = FIXTURE.read_text()
    start = xml.index("              <capRateSchedule>")
    end = xml.index("              </capRateSchedule>") + len(
        "              </capRateSchedule>\n"
    )
    report = _normalize((xml[:start] + xml[end:]).encode())

    assert _blocker_ids(report) == (
        "missing_contract_field:fpml_cap_floor_strike_schedule",
    )
    assert report.clarification.missing_fields == ("strike_schedule",)


def test_cap_floor_normalization_rejects_duplicate_strike_schedule():
    schedule = """              <capRateSchedule>
                <initialValue>0.05</initialValue>
                <buyer>Payer</buyer>
                <seller>Receiver</seller>
              </capRateSchedule>
"""
    xml = FIXTURE.read_text().replace(
        "            </floatingRateCalculation>",
        schedule + "            </floatingRateCalculation>",
    ).encode()

    assert _blocker_ids(_normalize(xml)) == (
        "contract_ambiguity:fpml_cap_floor_strike_schedule",
    )


def test_cap_floor_normalization_requires_explicit_strike_buyer_and_seller():
    xml = FIXTURE.read_text().replace("                <buyer>Payer</buyer>\n", "")
    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "missing_contract_field:fpml_strike_buyer",
    )
    assert report.clarification.missing_fields == ("strike_buyer",)


def test_imported_and_native_cap_select_the_same_existing_static_route():
    from trellis.agent.static_leg_admission import select_static_leg_lowering

    imported = select_static_leg_lowering(_normalize().normalized_contract)
    native = select_static_leg_lowering(_native_contract())

    assert imported == native
    assert imported.declaration_id == "static_leg_period_rate_option_strip_analytical"


def test_imported_and_native_cap_price_identically():
    from trellis.core.market_state import MarketState
    from trellis.core.payoff import ExecutionBackedPayoff
    from trellis.curves.yield_curve import YieldCurve
    from trellis.execution import compile_static_leg_execution_ir
    from trellis.models.vol_surface import FlatVol

    market = MarketState(
        as_of=date(2025, 1, 15),
        settlement=date(2025, 1, 15),
        discount=YieldCurve.flat(0.035),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.038)},
        vol_surface=FlatVol(0.20),
    )

    def price(contract):
        payoff = ExecutionBackedPayoff(
            compile_static_leg_execution_ir(contract, fail_on_unsupported=True)
        )
        return payoff.evaluate(market)

    imported_price = price(_normalize().normalized_contract)
    native_price = price(_native_contract())
    seller_price = price(
        _normalize(valuation_party_id="PARTY-B").normalized_contract
    )

    assert imported_price == pytest.approx(native_price, rel=1e-12, abs=1e-8)
    assert seller_price == pytest.approx(-imported_price, rel=1e-12, abs=1e-8)


def _with_premium(*, payment_date: str) -> bytes:
    premium = f"""      <premium>
        <payerPartyReference href="PARTY-A" />
        <receiverPartyReference href="PARTY-B" />
        <paymentAmount>
          <currency>USD</currency>
          <amount>9000</amount>
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
        "    </capFloor>",
        premium + "    </capFloor>",
    ).encode()


def test_historical_cap_premium_is_metadata_outside_contract_identity():
    without_premium = _normalize()
    with_premium = _normalize(_with_premium(payment_date="2025-01-14"))

    assert with_premium.economic_identity == without_premium.economic_identity
    assert with_premium.normalized_contract == without_premium.normalized_contract
    assert len(with_premium.premium_metadata) == 1
    assert with_premium.premium_metadata[0].amount == 9_000.0


def test_unsettled_cap_premium_blocks_before_pricing():
    report = _normalize(_with_premium(payment_date="2025-01-16"))

    assert _blocker_ids(report) == (
        "external_import:fpml_cap_floor_unsettled_premium_unsupported",
    )


def test_cap_floor_with_already_fixed_unpaid_period_blocks():
    report = _normalize(valuation_date=date(2025, 6, 30))

    assert _blocker_ids(report) == (
        "external_import:fpml_historical_fixing_runtime_unsupported",
    )


@pytest.mark.parametrize(
    ("old", "new", "expected_id"),
    (
        (
            "<initialValue>0.04</initialValue>",
            "<initialValue>0.04</initialValue>"
            "<step><stepDate>2026-06-30</stepDate><stepValue>0.05</stepValue></step>",
            "external_import:fpml_step_strike_schedule_unsupported",
        ),
        (
            "<buyer>Payer</buyer>",
            "<buyer>Receiver</buyer>",
            "contract_conflict:fpml_cap_floor_option_parties",
        ),
        (
            "<indexTenor>",
            "<spreadSchedule><initialValue>0.001</initialValue></spreadSchedule>"
            "<indexTenor>",
            "external_import:fpml_cap_floor_coupon_transform_unsupported",
        ),
        (
            "<indexTenor>",
            "<floatingRateMultiplierSchedule><initialValue>0.5</initialValue>"
            "</floatingRateMultiplierSchedule><indexTenor>",
            "external_import:fpml_cap_floor_coupon_transform_unsupported",
        ),
        (
            "<indexTenor>",
            "<averagingMethod>Weighted</averagingMethod><indexTenor>",
            "external_import:fpml_cap_floor_averaging_unsupported",
        ),
        (
            "<calculation>",
            "<calculation><compoundingMethod>Flat</compoundingMethod>",
            "external_import:fpml_compounding_unsupported",
        ),
        (
            "<floatingRateCalculation>",
            "<fixedRateSchedule><initialValue>0.01</initialValue>"
            "</fixedRateSchedule><floatingRateCalculation>",
            "external_import:fpml_cap_floor_fixed_coupon_unsupported",
        ),
        (
            "<currency>USD</currency>",
            "<currency>USD</currency><step><stepDate>2026-06-30</stepDate>"
            "<stepValue>900000</stepValue></step>",
            "external_import:fpml_amortizing_notional_unsupported",
        ),
    ),
)
def test_cap_floor_normalization_rejects_unrepresented_rate_terms(
    old,
    new,
    expected_id,
):
    xml = FIXTURE.read_text().replace(old, new, 1).encode()

    assert _blocker_ids(_normalize(xml)) == (expected_id,)


@pytest.mark.parametrize(
    ("element", "expected_id"),
    (
        (
            "<additionalPayment />",
            "external_import:fpml_cap_floor_additional_payment_unsupported",
        ),
        (
            "<earlyTerminationProvision />",
            "external_import:fpml_cap_floor_early_termination_unsupported",
        ),
    ),
)
def test_cap_floor_normalization_rejects_unconsumed_product_economics(
    element,
    expected_id,
):
    xml = FIXTURE.read_text().replace(
        "    </capFloor>",
        f"      {element}\n    </capFloor>",
    ).encode()

    assert _blocker_ids(_normalize(xml)) == (expected_id,)
