"""Bounded FpML fixed-float swap normalization tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest


FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "fpml"
    / "confirmation_5_13_fixed_float_swap.xml"
)


def _native_contract():
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

    start = date(2025, 6, 30)
    end = date(2027, 6, 30)

    def periods(months: int) -> tuple[CouponPeriod, ...]:
        from trellis.conventions.schedule import generate_schedule
        from trellis.core.types import Frequency

        frequency = {
            6: Frequency.SEMI_ANNUAL,
            3: Frequency.QUARTERLY,
        }[months]
        ends = tuple(generate_schedule(start, end, frequency))
        starts = (start, *ends[:-1])
        return tuple(
            CouponPeriod(
                accrual_start=period_start,
                accrual_end=period_end,
                payment_date=period_end,
                fixing_date=period_start if months == 3 else None,
            )
            for period_start, period_end in zip(starts, ends)
        )

    notional = NotionalSchedule((NotionalStep(start, end, 1_000_000.0),))
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                "pay",
                CouponLeg(
                    currency="USD",
                    notional_schedule=notional,
                    coupon_periods=periods(6),
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
                    coupon_periods=periods(3),
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


def _normalize(
    xml: bytes | None = None,
    *,
    valuation_party_id: str | None = "PARTY-A",
    valuation_date: date | None = None,
    require_valuation_date: bool = False,
):
    from trellis.io.fpml import normalize_fpml_document

    return normalize_fpml_document(
        xml if xml is not None else FIXTURE.read_bytes(),
        declared_view="confirmation",
        declared_version="5-13",
        valuation_party_id=valuation_party_id,
        valuation_date=valuation_date,
        require_valuation_date=require_valuation_date,
    )


def _blocker_ids(report) -> tuple[str, ...]:
    return tuple(blocker.id for blocker in report.blockers)


def test_normalizes_admitted_swap_to_native_static_leg_identity():
    from trellis.agent.static_leg_contract import static_leg_economic_identity

    report = _normalize()
    native = _native_contract()

    assert report.status == "normalized"
    assert report.blockers == ()
    assert report.normalized_contract == native
    assert report.economic_identity == static_leg_economic_identity(native)
    assert report.economic_identity.startswith("static_leg:v1:")
    assert report.mapping_provenance
    assert {item.semantic_field for item in report.mapping_provenance} >= {
        "legs[0].direction",
        "legs[0].coupon_formula.rate",
        "legs[1].coupon_formula.rate_index",
        "valuation_party_id",
    }
    assert all(item.xml_path.startswith("/dataDocument/trade/swap") for item in report.mapping_provenance if item.xml_path)


def test_normalized_and_native_contracts_select_identical_structural_artifacts():
    from trellis.agent.static_leg_admission import select_static_leg_lowering
    from trellis.execution import compile_static_leg_execution_ir

    imported = _normalize().normalized_contract
    native = _native_contract()

    assert select_static_leg_lowering(imported) == select_static_leg_lowering(native)
    imported_ir = compile_static_leg_execution_ir(imported, fail_on_unsupported=True)
    native_ir = compile_static_leg_execution_ir(native, fail_on_unsupported=True)
    assert imported_ir == native_ir
    assert dict(imported_ir.source_track.source_metadata) == {
        "static_leg_lowering_declaration_id": "static_leg_fixed_float_swap",
        "validation_bundle_id": "static_leg_fixed_float_swap_contract",
        "requested_method": "",
        "callable_ref": "trellis.instruments.swap.SwapPayoff",
    }


def test_normalized_and_native_contracts_price_identically():
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.execution import compile_static_leg_execution_ir
    from trellis.execution.runtime import price_static_leg_execution_ir

    market = MarketState(
        as_of=date(2025, 1, 15),
        settlement=date(2025, 1, 15),
        discount=YieldCurve.flat(0.035),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.038)},
    )

    imported_price = price_static_leg_execution_ir(
        compile_static_leg_execution_ir(
            _normalize().normalized_contract,
            fail_on_unsupported=True,
        ),
        market,
    )
    native_price = price_static_leg_execution_ir(
        compile_static_leg_execution_ir(_native_contract(), fail_on_unsupported=True),
        market,
    )

    assert imported_price == pytest.approx(native_price, rel=1e-12, abs=1e-8)


def test_opposite_valuation_party_reverses_signed_position():
    party_a = _normalize(valuation_party_id="PARTY-A")
    party_b = _normalize(valuation_party_id="PARTY-B")

    assert tuple(leg.direction for leg in party_a.normalized_contract.legs) == (
        "pay",
        "receive",
    )
    assert tuple(leg.direction for leg in party_b.normalized_contract.legs) == (
        "receive",
        "pay",
    )
    assert party_a.economic_identity != party_b.economic_identity


def test_normalization_blocks_seasoned_floating_coupon_before_runtime_pricing():
    report = _normalize(
        valuation_date=date(2025, 7, 1),
        require_valuation_date=True,
    )

    assert _blocker_ids(report) == (
        "external_import:fpml_historical_fixing_runtime_unsupported",
    )


def test_normalization_requires_valuation_date_when_requested_for_pricing():
    report = _normalize(require_valuation_date=True)

    assert _blocker_ids(report) == (
        "missing_contract_field:fpml_valuation_date",
    )
    assert report.clarification.missing_fields == ("valuation_date",)


def test_normalized_report_summary_is_body_free_and_machine_readable():
    from trellis.io.fpml import fpml_import_report_summary

    xml = FIXTURE.read_bytes()
    summary = fpml_import_report_summary(_normalize(xml))

    assert summary["status"] == "normalized"
    assert summary["economic_identity"].startswith("static_leg:v1:")
    assert summary["normalized_contract"]["contract_type"] == "StaticLegContractIR"
    assert summary["mapping_provenance"]
    assert xml.decode("utf-8") not in repr(summary)


def test_normalized_report_artifacts_cannot_be_relabelled_as_inspection_only():
    report = _normalize()

    with pytest.raises(ValueError, match="normalized artifacts"):
        replace(report, status="inspected")


def test_normalization_requires_an_explicit_valuation_party():
    report = _normalize(valuation_party_id=None)

    assert report.status == "blocked"
    assert _blocker_ids(report) == (
        "missing_contract_field:fpml_valuation_party_id",
    )
    assert report.clarification.missing_fields == ("valuation_party_id",)
    assert report.normalized_contract is None


def test_normalization_rejects_valuation_party_outside_trade():
    report = _normalize(valuation_party_id="PARTY-Z")

    assert _blocker_ids(report) == (
        "contract_conflict:fpml_valuation_party_id",
    )


@pytest.mark.parametrize(
    ("old", "new", "expected_id"),
    (
        (
            "<currency>USD</currency>\n              </notionalStepSchedule>",
            "<currency>USD</currency>\n                <step><stepDate>2026-06-30</stepDate><stepValue>900000</stepValue></step>\n              </notionalStepSchedule>",
            "external_import:fpml_amortizing_notional_unsupported",
        ),
        (
            "<dayCountFraction>ACT/360</dayCountFraction>",
            "<dayCountFraction>ACT/360</dayCountFraction>\n            <compoundingMethod>Flat</compoundingMethod>",
            "external_import:fpml_compounding_unsupported",
        ),
        (
            "<currency>USD</currency>\n              </notionalStepSchedule>\n            </notionalSchedule>\n            <floatingRateCalculation>",
            "<currency>EUR</currency>\n              </notionalStepSchedule>\n            </notionalSchedule>\n            <floatingRateCalculation>",
            "external_import:fpml_cross_currency_swap_unsupported",
        ),
        (
            "<calculationPeriodFrequency>",
            "<firstRegularPeriodStartDate>2025-09-30</firstRegularPeriodStartDate>\n          <calculationPeriodFrequency>",
            "external_import:fpml_stub_period_unsupported",
        ),
    ),
)
def test_normalization_rejects_unsupported_swap_economics(old, new, expected_id):
    xml = FIXTURE.read_text().replace(old, new, 1).encode()
    report = _normalize(xml)

    assert report.status == "blocked"
    assert expected_id in _blocker_ids(report)
    assert report.normalized_contract is None


def test_normalization_reports_missing_payment_schedule_as_clarification():
    xml = FIXTURE.read_text()
    start = xml.index("        <paymentDates>")
    end = xml.index("        </paymentDates>", start) + len("        </paymentDates>\n")
    report = _normalize((xml[:start] + xml[end:]).encode())

    assert _blocker_ids(report) == (
        "missing_contract_field:fpml_payment_dates",
    )
    assert report.clarification.missing_fields == ("payment_dates",)


def test_normalization_rejects_broken_schedule_reference():
    xml = FIXTURE.read_text().replace(
        'calculationPeriodDatesReference href="FIXED-CALC-DATES"',
        'calculationPeriodDatesReference href="OTHER-DATES"',
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "contract_conflict:fpml_schedule_reference",
    )


def test_normalization_rejects_implicit_stub_schedule():
    xml = FIXTURE.read_text().replace("2027-06-30", "2027-05-30")

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_stub_period_unsupported",
    )


def test_normalization_rejects_unimplemented_end_of_month_roll():
    xml = (
        FIXTURE.read_text()
        .replace("2025-06-30", "2025-02-28")
        .replace("2027-06-30", "2025-08-31")
        .replace(
            "<rollConvention>30</rollConvention>",
            "<rollConvention>NONE</rollConvention>",
        )
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_stub_period_unsupported",
    )


def test_normalization_rejects_clamped_high_day_schedule():
    xml = (
        FIXTURE.read_text()
        .replace("2025-06-30", "2025-01-31")
        .replace("2027-06-30", "2026-01-31")
        .replace(
            "<rollConvention>30</rollConvention>",
            "<rollConvention>31</rollConvention>",
        )
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_stub_period_unsupported",
    )


def test_normalization_rejects_duplicate_roll_conventions():
    xml = FIXTURE.read_text().replace(
        "<rollConvention>30</rollConvention>",
        "<rollConvention>30</rollConvention>\n"
        "            <rollConvention>31</rollConvention>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "contract_ambiguity:fpml_roll_convention",
    )


def test_normalization_rejects_initial_fixing_override_leaf():
    xml = FIXTURE.read_text().replace(
        "<resetRelativeTo>CalculationPeriodStartDate</resetRelativeTo>",
        "<initialFixingDate>2025-06-27</initialFixingDate>\n"
        "          <resetRelativeTo>CalculationPeriodStartDate</resetRelativeTo>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_initial_fixing_override_unsupported",
    )


def test_normalization_rejects_conflicting_supplied_adjusted_date():
    xml = FIXTURE.read_text().replace(
        "            </dateAdjustments>\n          </effectiveDate>",
        "            </dateAdjustments>\n"
        "            <adjustedDate>2025-07-01</adjustedDate>\n"
        "          </effectiveDate>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "contract_conflict:fpml_effective_date_adjusted_date",
    )


def test_normalization_accepts_matching_supplied_adjusted_date():
    xml = FIXTURE.read_text().replace(
        "            </dateAdjustments>\n          </effectiveDate>",
        "            </dateAdjustments>\n"
        "            <adjustedDate>2025-06-30</adjustedDate>\n"
        "          </effectiveDate>",
        1,
    )

    report = _normalize(xml.encode())

    assert report.status == "normalized"


def test_payment_dates_are_anchored_to_adjusted_calculation_period_ends():
    xml = (
        FIXTURE.read_text()
        .replace("2025-06-30", "2025-06-29")
        .replace("2027-06-30", "2027-06-29")
        .replace(
            "<rollConvention>30</rollConvention>",
            "<rollConvention>29</rollConvention>",
        )
        .replace(
            "<dateAdjustments>\n"
            "              <businessDayConvention>NONE</businessDayConvention>\n"
            "            </dateAdjustments>",
            "<dateAdjustments>\n"
            "              <businessDayConvention>FOLLOWING</businessDayConvention>\n"
            "              <businessCenters><businessCenter>USNY</businessCenter>"
            "</businessCenters>\n"
            "            </dateAdjustments>",
        )
        .replace(
            "<calculationPeriodDatesAdjustments>\n"
            "            <businessDayConvention>NONE</businessDayConvention>\n"
            "          </calculationPeriodDatesAdjustments>",
            "<calculationPeriodDatesAdjustments>\n"
            "            <businessDayConvention>FOLLOWING</businessDayConvention>\n"
            "            <businessCenters><businessCenter>USNY</businessCenter>"
            "</businessCenters>\n"
            "          </calculationPeriodDatesAdjustments>",
        )
    )

    report = _normalize(xml.encode())

    assert report.status == "normalized"
    periods = tuple(
        period
        for signed_leg in report.normalized_contract.legs
        for period in signed_leg.leg.coupon_periods
    )
    assert all(period.payment_date == period.accrual_end for period in periods)
    assert any(period.accrual_end == date(2026, 3, 30) for period in periods)


def test_normalization_rejects_unknown_stream_party_reference():
    xml = FIXTURE.read_text().replace('href="PARTY-B"', 'href="PARTY-Z"', 1)

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "contract_conflict:fpml_swap_stream_party_reference",
    )


def test_normalization_rejects_different_counterparty_pairs_across_streams():
    xml = FIXTURE.read_text().replace(
        '<swapStream id="FLOAT-LEG">\n'
        '        <payerPartyReference href="PARTY-B" />',
        '<swapStream id="FLOAT-LEG">\n'
        '        <payerPartyReference href="PARTY-C" />',
        1,
    )
    xml = xml.replace(
        "</dataDocument>",
        '  <party id="PARTY-C"><partyId>PARTY-C-ID</partyId></party>\n'
        "</dataDocument>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "contract_conflict:fpml_swap_stream_counterparties",
    )


def test_normalization_rejects_reset_schedule_on_fixed_stream():
    xml = FIXTURE.read_text().replace(
        "        <calculationPeriodAmount>",
        '        <resetDates id="FIXED-RESET-DATES" />\n'
        "        <calculationPeriodAmount>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_fixed_stream_reset_dates_unsupported",
    )


def test_normalization_is_invariant_to_fpml_stream_order():
    from xml.etree import ElementTree

    namespace = "http://www.fpml.org/FpML-5/confirmation"
    root = ElementTree.fromstring(FIXTURE.read_bytes())
    swap = root.find(f".//{{{namespace}}}swap")
    streams = swap.findall(f"{{{namespace}}}swapStream")
    swap.remove(streams[0])
    swap.remove(streams[1])
    swap.insert(0, streams[1])
    swap.insert(1, streams[0])

    reordered = _normalize(ElementTree.tostring(root, encoding="utf-8"))
    original = _normalize()

    assert reordered.normalized_contract == original.normalized_contract
    assert reordered.economic_identity == original.economic_identity
    fixed_rate_mapping = next(
        item
        for item in reordered.mapping_provenance
        if item.semantic_field == "legs[0].coupon_formula.rate"
    )
    assert "/swapStream[2]/" in fixed_rate_mapping.xml_path


def test_normalization_rejects_unconsumed_swap_economics():
    xml = FIXTURE.read_text().replace(
        "    </swap>",
        "      <additionalPayment />\n    </swap>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_swap_feature_unsupported",
    )


@pytest.mark.parametrize(
    ("old", "new", "expected_id"),
    (
        (
            "</floatingRateCalculation>",
            "<capRateSchedule><initialValue>0.05</initialValue></capRateSchedule>\n"
            "            </floatingRateCalculation>",
            "external_import:fpml_floating_rate_calculation_feature_unsupported",
        ),
        (
            "<paymentDatesAdjustments>",
            "<paymentDaysOffset><periodMultiplier>2</periodMultiplier>"
            "<period>D</period><dayType>Business</dayType></paymentDaysOffset>\n"
            "          <paymentDatesAdjustments>",
            "external_import:fpml_payment_dates_feature_unsupported",
        ),
        (
            "</calculationPeriodFrequency>",
            "<periodRule>unsupported</periodRule></calculationPeriodFrequency>",
            "external_import:fpml_calculation_period_frequency_feature_unsupported",
        ),
        (
            "</paymentFrequency>",
            "<periodRule>unsupported</periodRule></paymentFrequency>",
            "external_import:fpml_payment_frequency_feature_unsupported",
        ),
        (
            "</resetFrequency>",
            "<periodRule>unsupported</periodRule></resetFrequency>",
            "external_import:fpml_reset_frequency_feature_unsupported",
        ),
        (
            "</indexTenor>",
            "<periodRule>unsupported</periodRule></indexTenor>",
            "external_import:fpml_index_tenor_feature_unsupported",
        ),
        (
            "</dateAdjustments>",
            "<periodRule>unsupported</periodRule></dateAdjustments>",
            "external_import:fpml_effective_date_adjustments_feature_unsupported",
        ),
        (
            "</calculationPeriodDatesAdjustments>",
            "<periodRule>unsupported</periodRule></calculationPeriodDatesAdjustments>",
            "external_import:fpml_calculation_period_dates_adjustments_feature_unsupported",
        ),
        (
            "</paymentDatesAdjustments>",
            "<periodRule>unsupported</periodRule></paymentDatesAdjustments>",
            "external_import:fpml_payment_dates_adjustments_feature_unsupported",
        ),
        (
            "</resetDatesAdjustments>",
            "<periodRule>unsupported</periodRule></resetDatesAdjustments>",
            "external_import:fpml_reset_dates_adjustments_feature_unsupported",
        ),
        (
            "</initialValue>",
            '<vendor:override xmlns:vendor="urn:vendor" /></initialValue>',
            "external_import:fpml_notional_feature_unsupported",
        ),
        (
            '<payerPartyReference href="PARTY-A" />',
            '<payerPartyReference href="PARTY-A"><vendor:override '
            'xmlns:vendor="urn:vendor" /></payerPartyReference>',
            "external_import:fpml_payer_party_reference_feature_unsupported",
        ),
        (
            '<calculationPeriodDatesReference href="FIXED-CALC-DATES" />',
            '<calculationPeriodDatesReference href="FIXED-CALC-DATES">'
            '<vendor:override xmlns:vendor="urn:vendor" />'
            "</calculationPeriodDatesReference>",
            "external_import:fpml_calculation_period_dates_reference_feature_unsupported",
        ),
        (
            "<unadjustedDate>2025-06-30</unadjustedDate>",
            "<unadjustedDate>2025-06-30</unadjustedDate>"
            '<adjustedDate>2025-06-30<vendor:override xmlns:vendor="urn:vendor" />'
            "</adjustedDate>",
            "external_import:fpml_effective_date_adjusted_date_feature_unsupported",
        ),
        (
            "<rollConvention>30</rollConvention>",
            '<rollConvention>30<vendor:override xmlns:vendor="urn:vendor" />'
            "</rollConvention>",
            "external_import:fpml_roll_convention_feature_unsupported",
        ),
    ),
)
def test_normalization_rejects_unconsumed_nested_economics(old, new, expected_id):
    report = _normalize(FIXTURE.read_text().replace(old, new, 1).encode())

    assert _blocker_ids(report) == (expected_id,)


def test_normalization_rejects_unconsumed_business_center_children():
    xml = FIXTURE.read_text().replace(
        "<calculationPeriodDatesAdjustments>\n"
        "            <businessDayConvention>NONE</businessDayConvention>",
        "<calculationPeriodDatesAdjustments>\n"
        "            <businessDayConvention>NONE</businessDayConvention>\n"
        "            <businessCenters><calendarRule>unsupported</calendarRule>"
        "</businessCenters>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_business_centers_feature_unsupported",
    )


@pytest.mark.parametrize(
    ("business_centers", "expected_id"),
    (
        (
            "<businessCenters><businessCenter><calendarRule>unsupported</calendarRule>"
            "</businessCenter></businessCenters>",
            "external_import:fpml_business_center_feature_unsupported",
        ),
        (
            "<businessCenters><businessCenter> </businessCenter></businessCenters>",
            "missing_contract_field:fpml_business_center",
        ),
        (
            "<businessCenters><businessCenter>USNY</businessCenter></businessCenters>"
            "<businessCenters><businessCenter>GBLO</businessCenter></businessCenters>",
            "contract_ambiguity:fpml_business_centers",
        ),
    ),
)
def test_normalization_rejects_ambiguous_or_malformed_business_centers(
    business_centers,
    expected_id,
):
    xml = FIXTURE.read_text().replace(
        "<calculationPeriodDatesAdjustments>\n"
        "            <businessDayConvention>NONE</businessDayConvention>",
        "<calculationPeriodDatesAdjustments>\n"
        "            <businessDayConvention>NONE</businessDayConvention>\n"
        f"            {business_centers}",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (expected_id,)


@pytest.mark.parametrize(
    ("old", "new", "expected_id"),
    (
        (
            "<initialValue>1000000</initialValue>",
            "<initialValue>1e6</initialValue>",
            "external_import:fpml_malformed_notional",
        ),
        (
            "<initialValue>0.04</initialValue>",
            "<initialValue>4e-2</initialValue>",
            "external_import:fpml_malformed_fixed_rate",
        ),
        (
            "<periodMultiplier>6</periodMultiplier>",
            "<periodMultiplier>6_0</periodMultiplier>",
            "external_import:fpml_malformed_period_multiplier",
        ),
    ),
)
def test_normalization_rejects_non_xml_numeric_lexical_forms(old, new, expected_id):
    report = _normalize(FIXTURE.read_text().replace(old, new, 1).encode())

    assert _blocker_ids(report) == (expected_id,)


@pytest.mark.parametrize("fixed_rate", ("+0.04", ".04", "4."))
def test_normalization_accepts_xml_decimal_lexical_forms(fixed_rate):
    xml = FIXTURE.read_text().replace(
        "<initialValue>0.04</initialValue>",
        f"<initialValue>{fixed_rate}</initialValue>",
        1,
    )

    report = _normalize(xml.encode())

    assert report.status == "normalized"


def test_normalization_rejects_fixed_and_floating_economics_on_one_stream():
    xml = FIXTURE.read_text()
    fixed_rate = """<fixedRateSchedule>
              <initialValue>0.04</initialValue>
            </fixedRateSchedule>"""
    floating_rate = """<floatingRateCalculation>
              <floatingRateIndex>USD-SOFR</floatingRateIndex>
              <indexTenor>
                <periodMultiplier>3</periodMultiplier>
                <period>M</period>
              </indexTenor>
            </floatingRateCalculation>"""
    xml = xml.replace(floating_rate, "", 1)
    xml = xml.replace(fixed_rate, fixed_rate + "\n            " + floating_rate, 1)

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_fixed_float_leg_shape_unsupported",
    )


def test_normalization_rejects_index_tenor_mismatched_to_coupon_frequency():
    xml = FIXTURE.read_text().replace(
        "<indexTenor>\n"
        "                <periodMultiplier>3</periodMultiplier>\n"
        "                <period>M</period>\n"
        "              </indexTenor>",
        "<indexTenor>\n"
        "                <periodMultiplier>6</periodMultiplier>\n"
        "                <period>M</period>\n"
        "              </indexTenor>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_index_tenor_frequency_mismatch",
    )


def test_normalization_records_spread_and_gearing_provenance_when_supplied():
    xml = FIXTURE.read_text().replace(
        "            </floatingRateCalculation>",
        "              <spreadSchedule><initialValue>0.0015</initialValue>"
        "</spreadSchedule>\n"
        "              <floatingRateMultiplierSchedule><initialValue>1.25"
        "</initialValue></floatingRateMultiplierSchedule>\n"
        "            </floatingRateCalculation>",
        1,
    )

    report = _normalize(xml.encode())

    assert report.status == "normalized"
    provenance = {item.semantic_field: item for item in report.mapping_provenance}
    assert provenance["legs[1].coupon_formula.spread"].normalized_value == "0.0015"
    assert provenance["legs[1].coupon_formula.spread"].xml_path.endswith(
        "/spreadSchedule/initialValue"
    )
    assert provenance["legs[1].coupon_formula.gearing"].normalized_value == "1.25"
    assert provenance["legs[1].coupon_formula.gearing"].xml_path.endswith(
        "/floatingRateMultiplierSchedule/initialValue"
    )


def test_normalization_rejects_duplicate_floating_spread_schedules():
    spread = "<spreadSchedule><initialValue>0.001</initialValue></spreadSchedule>"
    xml = FIXTURE.read_text().replace(
        "            </floatingRateCalculation>",
        f"              {spread}\n              {spread}\n"
        "            </floatingRateCalculation>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "contract_ambiguity:fpml_floating_spread_schedule",
    )
    assert report.clarification.ambiguous_fields == ("floating_spread_schedule",)


def test_normalization_rejects_foreign_namespaced_economics():
    xml = FIXTURE.read_text().replace(
        "            </floatingRateCalculation>",
        "              <vendor:couponOverride xmlns:vendor=\"urn:vendor\">"
        "0.02</vendor:couponOverride>\n"
        "            </floatingRateCalculation>",
        1,
    )

    report = _normalize(xml.encode())

    assert _blocker_ids(report) == (
        "external_import:fpml_floating_rate_calculation_feature_unsupported",
    )


def test_economic_identity_ignores_labels_and_source_provenance():
    from trellis.agent.static_leg_contract import static_leg_economic_identity

    native = _native_contract()
    relabeled = replace(
        native,
        legs=tuple(
            replace(signed, leg=replace(signed.leg, label=f"other-{index}"))
            for index, signed in enumerate(native.legs)
        ),
        metadata={"source": "native", "arbitrary": 17},
    )

    assert static_leg_economic_identity(native) == static_leg_economic_identity(relabeled)
