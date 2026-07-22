"""Shared, product-neutral primitives for bounded FpML normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import re

from trellis.agent.static_leg_contract import (
    CouponLeg,
    CouponPeriod,
    FloatingCouponFormula,
    PeriodRateOptionStripLeg,
    StaticLegContractIR,
    TermRateIndex,
)
from trellis.conventions.calendar import (
    BRAZIL,
    SYDNEY,
    TARGET,
    TOKYO,
    TORONTO,
    UK_SETTLEMENT,
    US_SETTLEMENT,
    WEEKEND_ONLY,
    ZURICH,
    BusinessDayAdjustment,
    JointCalendar,
)
from trellis.conventions.schedule import generate_schedule
from trellis.core.types import Frequency
from trellis.io.fpml.contracts import (
    FpMLClarification,
    FpMLFieldProvenance,
    FpMLImportBlocker,
    FpMLImportReport,
    FpMLPremiumMetadata,
)
from trellis.io.fpml.importer import (
    _direct_children,
    _first_direct_child,
    _optional_text,
    _parse_xsd_date,
    _split_tag,
)


_CALENDARS = {
    "AUSY": SYDNEY,
    "BRSP": BRAZIL,
    "CATO": TORONTO,
    "CHZU": ZURICH,
    "EUTA": TARGET,
    "GBLO": UK_SETTLEMENT,
    "JPTO": TOKYO,
    "USNY": US_SETTLEMENT,
}
_BUSINESS_DAY_ADJUSTMENTS = {
    "NONE": BusinessDayAdjustment.UNADJUSTED,
    "FOLLOWING": BusinessDayAdjustment.FOLLOWING,
    "MODFOLLOWING": BusinessDayAdjustment.MODIFIED_FOLLOWING,
    "PRECEDING": BusinessDayAdjustment.PRECEDING,
    "MODPRECEDING": BusinessDayAdjustment.MODIFIED_PRECEDING,
}
_FREQUENCIES = {
    (1, "Y"): (Frequency.ANNUAL, "annual"),
    (12, "M"): (Frequency.ANNUAL, "annual"),
    (6, "M"): (Frequency.SEMI_ANNUAL, "semiannual"),
    (3, "M"): (Frequency.QUARTERLY, "quarterly"),
    (1, "M"): (Frequency.MONTHLY, "monthly"),
}
_INDEX_TENORS_BY_FREQUENCY = {
    Frequency.ANNUAL: frozenset({"1Y", "12M"}),
    Frequency.SEMI_ANNUAL: frozenset({"6M"}),
    Frequency.QUARTERLY: frozenset({"3M"}),
    Frequency.MONTHLY: frozenset({"1M"}),
}
_XML_DECIMAL_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
_XML_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+")
_DAY_COUNTS = {
    "ACT/360": "ACT/360",
    "ACT/365.FIXED": "ACT/365",
    "ACT/365": "ACT/365",
    "ACT/ACT": "ACT/ACT",
    "30/360": "30/360",
}
_STUB_FIELDS = {
    "firstPeriodStartDate",
    "firstRegularPeriodStartDate",
    "lastRegularPeriodEndDate",
    "firstPaymentDate",
    "lastRegularPaymentDate",
}
_ALLOWED_PARTY_CHILDREN = {"partyId"}
_ALLOWED_TRADE_HEADER_CHILDREN = {"partyTradeIdentifier", "tradeDate"}
_ALLOWED_PARTY_TRADE_IDENTIFIER_CHILDREN = {"partyReference", "tradeId"}
_ALLOWED_STREAM_CHILDREN = {
    "calculationPeriodAmount",
    "calculationPeriodDates",
    "payerAccountReference",
    "payerPartyReference",
    "paymentDates",
    "receiverAccountReference",
    "receiverPartyReference",
    "resetDates",
}
_STREAM_METADATA_REFERENCES = {"payerAccountReference", "receiverAccountReference"}
_ALLOWED_CALCULATION_DATES_CHILDREN = {
    "calculationPeriodDatesAdjustments",
    "calculationPeriodFrequency",
    "effectiveDate",
    "terminationDate",
}
_ALLOWED_ADJUSTABLE_DATE_CHILDREN = {
    "adjustedDate",
    "dateAdjustments",
    "unadjustedDate",
}
_ALLOWED_DATE_ADJUSTMENT_CHILDREN = {
    "businessCenters",
    "businessDayConvention",
}
_ALLOWED_BUSINESS_CENTERS_CHILDREN = {"businessCenter"}
_ALLOWED_LEAF_CHILDREN: set[str] = set()
_ALLOWED_PAYMENT_DATES_CHILDREN = {
    "calculationPeriodDatesReference",
    "payRelativeTo",
    "paymentDatesAdjustments",
    "paymentFrequency",
}
_ALLOWED_RESET_DATES_CHILDREN = {
    "calculationPeriodDatesReference",
    "fixingDates",
    "initialFixingDate",
    "resetFrequency",
    "resetRelativeTo",
    "resetDatesAdjustments",
}
_ALLOWED_FIXING_DATES_CHILDREN = {
    "businessCenters",
    "businessDayConvention",
    "dateRelativeTo",
    "dayType",
    "period",
    "periodMultiplier",
}
_ALLOWED_CALCULATION_CHILDREN = {
    "dayCountFraction",
    "fixedRateSchedule",
    "floatingRateCalculation",
    "notionalSchedule",
}
_ALLOWED_CALCULATION_PERIOD_AMOUNT_CHILDREN = {"calculation"}
_ALLOWED_NOTIONAL_SCHEDULE_CHILDREN = {"notionalStepSchedule"}
_ALLOWED_NOTIONAL_STEP_SCHEDULE_CHILDREN = {"currency", "initialValue", "step"}
_ALLOWED_RATE_SCHEDULE_CHILDREN = {"initialValue", "step"}
_ALLOWED_FLOATING_RATE_CALCULATION_CHILDREN = {
    "floatingRateIndex",
    "floatingRateMultiplierSchedule",
    "indexTenor",
    "spreadSchedule",
}
_RATE_OPTION_STRIKE_CHILDREN = {"capRateSchedule", "floorRateSchedule"}
_ALLOWED_PREMIUM_CHILDREN = {
    "payerPartyReference",
    "paymentAmount",
    "paymentDate",
    "receiverPartyReference",
}
_ALLOWED_PAYMENT_AMOUNT_CHILDREN = {"amount", "currency"}


@dataclass(frozen=True)
class _LegTerms:
    kind: str
    direction: str
    payer: str
    receiver: str
    currency: str
    notional: float
    start: date
    end: date
    frequency: Frequency
    frequency_name: str
    day_count: str
    periods: tuple[CouponPeriod, ...]
    fixed_rate: float | None = None
    rate_index: TermRateIndex | None = None
    spread: float = 0.0
    gearing: float = 1.0
    spread_supplied: bool = False
    gearing_supplied: bool = False


class _NormalizationBlocked(Exception):
    def __init__(self, blocker: FpMLImportBlocker):
        self.blocker = blocker
        super().__init__(blocker.summary)


def _validate_document_metadata(root, trade, *, namespace: str | None) -> None:
    for party in _direct_children(root, "party", namespace=namespace):
        _reject_unadmitted_direct_children(
            party,
            allowed=_ALLOWED_PARTY_CHILDREN,
            scope="party",
            namespace=namespace,
        )
        _reject_nested_metadata_children(
            party,
            names=_ALLOWED_PARTY_CHILDREN,
            namespace=namespace,
        )

    trade_header = _required_child(
        trade,
        "tradeHeader",
        namespace=namespace,
        missing_field="trade_header",
    )
    _reject_unadmitted_direct_children(
        trade_header,
        allowed=_ALLOWED_TRADE_HEADER_CHILDREN,
        scope="trade_header",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        trade_header,
        names={"tradeDate"},
        namespace=namespace,
    )
    for identifier in _direct_children(
        trade_header,
        "partyTradeIdentifier",
        namespace=namespace,
    ):
        _reject_unadmitted_direct_children(
            identifier,
            allowed=_ALLOWED_PARTY_TRADE_IDENTIFIER_CHILDREN,
            scope="party_trade_identifier",
            namespace=namespace,
        )
        _reject_nested_metadata_children(
            identifier,
            names=_ALLOWED_PARTY_TRADE_IDENTIFIER_CHILDREN,
            namespace=namespace,
        )


def _normalize_stream(
    stream,
    *,
    kind: str,
    namespace: str | None,
    valuation_party_id: str,
    allow_rate_option_strikes: bool = False,
    stream_scope: str = "swap_stream",
) -> _LegTerms:
    payer = _required_href(
        stream,
        "payerPartyReference",
        namespace=namespace,
        missing_field="payer_party_reference",
    )
    receiver = _required_href(
        stream,
        "receiverPartyReference",
        namespace=namespace,
        missing_field="receiver_party_reference",
    )
    if payer == receiver:
        _fail(
            f"contract_conflict:fpml_{stream_scope}_parties",
            "contract_conflict",
            "The stream payer and receiver must differ.",
        )
    if valuation_party_id == payer:
        direction = "pay"
    elif valuation_party_id == receiver:
        direction = "receive"
    else:
        _fail(
            "contract_conflict:fpml_valuation_party_id",
            "contract_conflict",
            "The valuation party is not payer or receiver on the admitted stream.",
        )

    calculation_dates = _required_child(
        stream,
        "calculationPeriodDates",
        namespace=namespace,
        missing_field="calculation_period_dates",
    )
    _reject_unadmitted_direct_children(
        calculation_dates,
        allowed=_ALLOWED_CALCULATION_DATES_CHILDREN,
        scope="calculation_period_dates",
        namespace=namespace,
    )
    calculation_dates_id = _required_attribute(
        calculation_dates,
        "id",
        missing_field="calculation_period_dates_id",
    )
    payment_dates = _required_child(
        stream,
        "paymentDates",
        namespace=namespace,
        missing_field="payment_dates",
    )
    _reject_unadmitted_direct_children(
        payment_dates,
        allowed=_ALLOWED_PAYMENT_DATES_CHILDREN,
        scope="payment_dates",
        namespace=namespace,
    )
    _require_reference(
        payment_dates,
        "calculationPeriodDatesReference",
        calculation_dates_id,
        namespace=namespace,
    )
    calculation_period_amount = _required_child(
        stream,
        "calculationPeriodAmount",
        namespace=namespace,
        missing_field="calculation_period_amount",
    )
    _reject_unadmitted_direct_children(
        calculation_period_amount,
        allowed=_ALLOWED_CALCULATION_PERIOD_AMOUNT_CHILDREN,
        scope="calculation_period_amount",
        namespace=namespace,
    )
    calculation = _required_child(
        calculation_period_amount,
        "calculation",
        namespace=namespace,
        missing_field="calculation",
    )
    _reject_unadmitted_direct_children(
        calculation,
        allowed=_ALLOWED_CALCULATION_CHILDREN,
        scope="calculation",
        namespace=namespace,
    )

    start_unadjusted, start = _adjustable_date(
        calculation_dates,
        "effectiveDate",
        namespace=namespace,
    )
    end_unadjusted, end = _adjustable_date(
        calculation_dates,
        "terminationDate",
        namespace=namespace,
    )
    if start >= end:
        _fail(
            "contract_conflict:fpml_swap_dates",
            "contract_conflict",
            "Swap effective date must precede termination date.",
        )

    calculation_frequency_element = _required_child(
        calculation_dates,
        "calculationPeriodFrequency",
        namespace=namespace,
        missing_field="calculation_period_frequency",
    )
    frequency, frequency_name = _frequency(
        calculation_frequency_element,
        namespace=namespace,
        scope="calculation_period_frequency",
        allow_roll_convention=True,
    )
    payment_frequency_element = _required_child(
        payment_dates,
        "paymentFrequency",
        namespace=namespace,
        missing_field="payment_frequency",
    )
    payment_frequency, _ = _frequency(
        payment_frequency_element,
        namespace=namespace,
        scope="payment_frequency",
    )
    if payment_frequency != frequency:
        _fail(
            "external_import:fpml_multiple_calculation_periods_per_payment_unsupported",
            "unsupported_contract",
            "Calculation and payment frequencies must match in the non-compounding cohort.",
        )
    pay_relative_to = _required_text(
        payment_dates,
        "payRelativeTo",
        namespace=namespace,
        missing_field="pay_relative_to",
    )
    if pay_relative_to != "CalculationPeriodEndDate":
        _fail(
            "external_import:fpml_payment_relative_date_unsupported",
            "unsupported_contract",
            "Payments must be relative to calculation-period end dates.",
        )

    calc_adjustments = _required_child(
        calculation_dates,
        "calculationPeriodDatesAdjustments",
        namespace=namespace,
        missing_field="calculation_period_dates_adjustments",
    )
    calc_bda, calc_calendar = _date_adjustment(
        calc_adjustments,
        namespace=namespace,
        scope="calculation_period_dates_adjustments",
        allowed=_ALLOWED_DATE_ADJUSTMENT_CHILDREN,
    )
    payment_adjustments = _required_child(
        payment_dates,
        "paymentDatesAdjustments",
        namespace=namespace,
        missing_field="payment_dates_adjustments",
    )
    payment_bda, payment_calendar = _date_adjustment(
        payment_adjustments,
        namespace=namespace,
        scope="payment_dates_adjustments",
        allowed=_ALLOWED_DATE_ADJUSTMENT_CHILDREN,
    )
    _validate_roll_convention(
        calculation_frequency_element,
        start_unadjusted,
        namespace=namespace,
    )
    _validate_regular_schedule(
        start_unadjusted,
        end_unadjusted,
        frequency,
    )
    unadjusted_ends = tuple(
        generate_schedule(start_unadjusted, end_unadjusted, frequency)
    )
    adjusted_ends = tuple(
        calc_calendar.adjust(item, calc_bda)
        if calc_bda != BusinessDayAdjustment.UNADJUSTED
        else item
        for item in unadjusted_ends
    )
    if not adjusted_ends or adjusted_ends[-1] != end:
        _fail(
            "contract_conflict:fpml_termination_date_adjustments",
            "contract_conflict",
            "Termination-date and calculation-period adjustments produce different dates.",
        )
    adjusted_starts = (start, *adjusted_ends[:-1])
    payment_end_dates = tuple(
        payment_calendar.adjust(item, payment_bda)
        if payment_bda != BusinessDayAdjustment.UNADJUSTED
        else item
        for item in adjusted_ends
    )
    if any(
        payment_date < accrual_end
        for payment_date, accrual_end in zip(payment_end_dates, adjusted_ends)
    ):
        _fail(
            "external_import:fpml_payment_before_accrual_end_unsupported",
            "unsupported_contract",
            "The normalized payment date cannot precede its accrual end date.",
        )

    fixing_dates: tuple[date | None, ...]
    if kind == "floating":
        fixing_dates = _floating_fixing_dates(
            stream,
            adjusted_starts=adjusted_starts,
            frequency=frequency,
            calculation_dates_id=calculation_dates_id,
            namespace=namespace,
        )
    else:
        fixing_dates = tuple(None for _ in adjusted_starts)
    periods = tuple(
        CouponPeriod(
            accrual_start=period_start,
            accrual_end=period_end,
            payment_date=payment_date,
            fixing_date=fixing_date,
        )
        for period_start, period_end, payment_date, fixing_date in zip(
            adjusted_starts,
            adjusted_ends,
            payment_end_dates,
            fixing_dates,
        )
    )

    notional_container = _required_child(
        calculation,
        "notionalSchedule",
        namespace=namespace,
        missing_field="notional_schedule",
    )
    _reject_unadmitted_direct_children(
        notional_container,
        allowed=_ALLOWED_NOTIONAL_SCHEDULE_CHILDREN,
        scope="notional_schedule",
        namespace=namespace,
    )
    notional_schedule = _required_child(
        notional_container,
        "notionalStepSchedule",
        namespace=namespace,
        missing_field="notional_step_schedule",
    )
    _reject_unadmitted_direct_children(
        notional_schedule,
        allowed=_ALLOWED_NOTIONAL_STEP_SCHEDULE_CHILDREN,
        scope="notional_step_schedule",
        namespace=namespace,
    )
    notional = _finite_float(
        _required_text(
            notional_schedule,
            "initialValue",
            namespace=namespace,
            missing_field="notional",
        ),
        field_name="notional",
        positive=True,
    )
    currency = _required_text(
        notional_schedule,
        "currency",
        namespace=namespace,
        missing_field="currency",
    ).upper()
    day_count_token = _required_text(
        calculation,
        "dayCountFraction",
        namespace=namespace,
        missing_field="day_count_fraction",
    ).upper()
    try:
        day_count = _DAY_COUNTS[day_count_token]
    except KeyError:
        _fail(
            "external_import:fpml_day_count_unsupported",
            "unsupported_contract",
            f"Day-count convention {day_count_token!r} is outside the admitted closure.",
        )

    fixed_rate = None
    rate_index = None
    spread = 0.0
    gearing = 1.0
    spread_supplied = False
    gearing_supplied = False
    if kind == "fixed":
        schedule = _required_child(
            calculation,
            "fixedRateSchedule",
            namespace=namespace,
            missing_field="fixed_rate_schedule",
        )
        if _direct_children(schedule, "step", namespace=namespace):
            _fail(
                "external_import:fpml_step_rate_schedule_unsupported",
                "unsupported_contract",
                "Stepped fixed rates are outside the admitted swap cohort.",
            )
        _reject_unadmitted_direct_children(
            schedule,
            allowed=_ALLOWED_RATE_SCHEDULE_CHILDREN,
            scope="fixed_rate_schedule",
            namespace=namespace,
        )
        fixed_rate = _finite_float(
            _required_text(
                schedule,
                "initialValue",
                namespace=namespace,
                missing_field="fixed_rate",
            ),
            field_name="fixed_rate",
        )
    else:
        floating = _required_child(
            calculation,
            "floatingRateCalculation",
            namespace=namespace,
            missing_field="floating_rate_calculation",
        )
        _reject_unadmitted_direct_children(
            floating,
            allowed=(
                _ALLOWED_FLOATING_RATE_CALCULATION_CHILDREN
                | _RATE_OPTION_STRIKE_CHILDREN
                if allow_rate_option_strikes
                else _ALLOWED_FLOATING_RATE_CALCULATION_CHILDREN
            ),
            scope="floating_rate_calculation",
            namespace=namespace,
        )
        index_name = _required_text(
            floating,
            "floatingRateIndex",
            namespace=namespace,
            missing_field="floating_rate_index",
        )
        tenors = _direct_children(floating, "indexTenor", namespace=namespace)
        if len(tenors) != 1:
            _fail(
                "external_import:fpml_index_tenor_count_unsupported",
                "unsupported_contract",
                "The admitted term-rate cohort requires exactly one index tenor.",
            )
        tenor = _period_label(tenors[0], namespace=namespace)
        if tenor not in _INDEX_TENORS_BY_FREQUENCY[frequency]:
            _fail(
                "external_import:fpml_index_tenor_frequency_mismatch",
                "unsupported_contract",
                "The admitted term-rate cohort requires index tenor to match "
                "calculation and reset frequency.",
            )
        rate_index = TermRateIndex(index_name, tenor)
        spread_schedules = _direct_children(
            floating,
            "spreadSchedule",
            namespace=namespace,
        )
        if len(spread_schedules) > 1:
            _fail(
                "contract_ambiguity:fpml_floating_spread_schedule",
                "contract_ambiguity",
                "FpML normalization found multiple floating spread schedules.",
                ambiguous_fields=("floating_spread_schedule",),
            )
        spread_schedule = spread_schedules[0] if spread_schedules else None
        if spread_schedule is not None:
            spread_supplied = True
            if _direct_children(spread_schedule, "step", namespace=namespace):
                _fail(
                    "external_import:fpml_step_spread_schedule_unsupported",
                    "unsupported_contract",
                    "Stepped floating spreads are outside the admitted swap cohort.",
                )
            _reject_unadmitted_direct_children(
                spread_schedule,
                allowed=_ALLOWED_RATE_SCHEDULE_CHILDREN,
                scope="spread_schedule",
                namespace=namespace,
            )
            spread = _finite_float(
                _required_text(
                    spread_schedule,
                    "initialValue",
                    namespace=namespace,
                    missing_field="floating_spread",
                ),
                field_name="floating_spread",
            )
        multiplier_schedules = _direct_children(
            floating,
            "floatingRateMultiplierSchedule",
            namespace=namespace,
        )
        if len(multiplier_schedules) > 1:
            _fail(
                "contract_ambiguity:fpml_floating_rate_multiplier_schedule",
                "contract_ambiguity",
                "FpML normalization found multiple floating rate multiplier schedules.",
                ambiguous_fields=("floating_rate_multiplier_schedule",),
            )
        multiplier_schedule = multiplier_schedules[0] if multiplier_schedules else None
        if multiplier_schedule is not None:
            gearing_supplied = True
            if _direct_children(multiplier_schedule, "step", namespace=namespace):
                _fail(
                    "external_import:fpml_step_multiplier_schedule_unsupported",
                    "unsupported_contract",
                    "Stepped floating multipliers are outside the admitted swap cohort.",
                )
            _reject_unadmitted_direct_children(
                multiplier_schedule,
                allowed=_ALLOWED_RATE_SCHEDULE_CHILDREN,
                scope="floating_rate_multiplier_schedule",
                namespace=namespace,
            )
            gearing = _finite_float(
                _required_text(
                    multiplier_schedule,
                    "initialValue",
                    namespace=namespace,
                    missing_field="floating_rate_multiplier",
                ),
                field_name="floating_rate_multiplier",
            )

    return _LegTerms(
        kind=kind,
        direction=direction,
        payer=payer,
        receiver=receiver,
        currency=currency,
        notional=notional,
        start=start,
        end=end,
        frequency=frequency,
        frequency_name=frequency_name,
        day_count=day_count,
        periods=periods,
        fixed_rate=fixed_rate,
        rate_index=rate_index,
        spread=spread,
        gearing=gearing,
        spread_supplied=spread_supplied,
        gearing_supplied=gearing_supplied,
    )


def _has_unresolved_historical_fixing(
    contract: StaticLegContractIR,
    valuation_date: date,
) -> bool:
    """Return whether an unpaid floating coupon already requires a known fixing."""

    for signed_leg in contract.legs:
        leg = signed_leg.leg
        if isinstance(leg, PeriodRateOptionStripLeg):
            if any(
                period.fixing_date <= valuation_date < period.payment_date
                for period in leg.option_periods
            ):
                return True
            continue
        if not isinstance(leg, CouponLeg):
            continue
        if not isinstance(leg.coupon_formula, FloatingCouponFormula):
            continue
        for period in leg.coupon_periods:
            if (
                period.fixing_date is not None
                and period.fixing_date <= valuation_date < period.payment_date
            ):
                return True
    return False


def _floating_fixing_dates(
    stream,
    *,
    adjusted_starts: tuple[date, ...],
    frequency: Frequency,
    calculation_dates_id: str,
    namespace: str | None,
) -> tuple[date, ...]:
    reset_dates = _required_child(
        stream,
        "resetDates",
        namespace=namespace,
        missing_field="reset_dates",
    )
    _reject_unadmitted_direct_children(
        reset_dates,
        allowed=_ALLOWED_RESET_DATES_CHILDREN,
        scope="reset_dates",
        namespace=namespace,
    )
    _require_reference(
        reset_dates,
        "calculationPeriodDatesReference",
        calculation_dates_id,
        namespace=namespace,
    )
    reset_dates_id = _required_attribute(
        reset_dates,
        "id",
        missing_field="reset_dates_id",
    )
    if (
        _first_direct_child(reset_dates, "initialFixingDate", namespace=namespace)
        is not None
    ):
        _fail(
            "external_import:fpml_initial_fixing_override_unsupported",
            "unsupported_contract",
            "Initial fixing overrides are outside the admitted swap cohort.",
        )
    relative_to = _required_text(
        reset_dates,
        "resetRelativeTo",
        namespace=namespace,
        missing_field="reset_relative_to",
    )
    if relative_to != "CalculationPeriodStartDate":
        _fail(
            "external_import:fpml_reset_relative_date_unsupported",
            "unsupported_contract",
            "Floating resets must be relative to calculation-period start dates.",
        )
    reset_frequency = _required_child(
        reset_dates,
        "resetFrequency",
        namespace=namespace,
        missing_field="reset_frequency",
    )
    parsed_reset_frequency, _ = _frequency(
        reset_frequency,
        namespace=namespace,
        scope="reset_frequency",
    )
    if parsed_reset_frequency != frequency:
        _fail(
            "external_import:fpml_reset_frequency_unsupported",
            "unsupported_contract",
            "Floating reset frequency must match calculation frequency.",
        )
    reset_adjustments = _required_child(
        reset_dates,
        "resetDatesAdjustments",
        namespace=namespace,
        missing_field="reset_dates_adjustments",
    )
    reset_bda, _ = _date_adjustment(
        reset_adjustments,
        namespace=namespace,
        scope="reset_dates_adjustments",
        allowed=_ALLOWED_DATE_ADJUSTMENT_CHILDREN,
    )
    if reset_bda != BusinessDayAdjustment.UNADJUSTED:
        _fail(
            "external_import:fpml_reset_date_adjustment_unsupported",
            "unsupported_contract",
            "Adjusted reset-date schedules are outside the admitted swap cohort.",
        )
    fixing = _required_child(
        reset_dates,
        "fixingDates",
        namespace=namespace,
        missing_field="fixing_dates",
    )
    _reject_unadmitted_direct_children(
        fixing,
        allowed=_ALLOWED_FIXING_DATES_CHILDREN,
        scope="fixing_dates",
        namespace=namespace,
    )
    _require_reference(
        fixing,
        "dateRelativeTo",
        reset_dates_id,
        namespace=namespace,
    )
    multiplier = _integer_text(
        fixing,
        "periodMultiplier",
        namespace=namespace,
        missing_field="fixing_date_offset",
    )
    period = _required_text(
        fixing,
        "period",
        namespace=namespace,
        missing_field="fixing_date_offset_period",
    )
    if period != "D":
        _fail(
            "external_import:fpml_fixing_offset_period_unsupported",
            "unsupported_contract",
            "Fixing-date offsets must be expressed in days.",
        )
    day_type = _required_text(
        fixing,
        "dayType",
        namespace=namespace,
        missing_field="fixing_date_day_type",
    )
    if day_type == "Business" and not _descendants(
        fixing,
        "businessCenter",
        namespace=namespace,
    ):
        _fail(
            "missing_contract_field:fpml_business_centers",
            "contract_gap",
            "Business-day fixing offsets require explicit admitted business centers.",
            missing_fields=("business_centers",),
        )
    bda, calendar = _date_adjustment(
        fixing,
        namespace=namespace,
        scope="fixing_dates",
        allowed=_ALLOWED_FIXING_DATES_CHILDREN,
    )
    values = []
    for period_start in adjusted_starts:
        if day_type == "Business":
            value = calendar.add_business_days(period_start, multiplier)
        elif day_type == "Calendar":
            value = period_start + timedelta(days=multiplier)
        else:
            _fail(
                "external_import:fpml_fixing_day_type_unsupported",
                "unsupported_contract",
                f"Fixing day type {day_type!r} is outside the admitted closure.",
            )
        if bda != BusinessDayAdjustment.UNADJUSTED:
            value = calendar.adjust(value, bda)
        values.append(value)
    return tuple(values)


def _adjustable_date(parent, name: str, *, namespace: str | None) -> tuple[date, date]:
    blocker_field = {
        "effectiveDate": "effective_date",
        "terminationDate": "termination_date",
    }.get(name, name)
    element = _required_child(
        parent,
        name,
        namespace=namespace,
        missing_field=name,
    )
    _reject_unadmitted_direct_children(
        element,
        allowed=_ALLOWED_ADJUSTABLE_DATE_CHILDREN,
        scope=f"{name}_adjustable_date",
        namespace=namespace,
    )
    value = _required_text(
        element,
        "unadjustedDate",
        namespace=namespace,
        missing_field=f"{name}.unadjusted_date",
    )
    try:
        unadjusted = _parse_xsd_date(value)
    except ValueError:
        _fail(
            "external_import:fpml_malformed_economic_date",
            "malformed_document",
            f"FpML {name} must be a valid XML Schema date.",
        )
    adjustments = _required_child(
        element,
        "dateAdjustments",
        namespace=namespace,
        missing_field=f"{name}.date_adjustments",
    )
    bda, calendar = _date_adjustment(
        adjustments,
        namespace=namespace,
        scope=f"{blocker_field}_adjustments",
        allowed=_ALLOWED_DATE_ADJUSTMENT_CHILDREN,
    )
    adjusted = calendar.adjust(unadjusted, bda)
    supplied_adjusted = _direct_children(element, "adjustedDate", namespace=namespace)
    if len(supplied_adjusted) > 1:
        _fail(
            f"contract_ambiguity:fpml_{blocker_field}_adjusted_date",
            "contract_ambiguity",
            f"FpML {name} contains multiple supplied adjusted dates.",
            ambiguous_fields=(f"{name}.adjusted_date",),
        )
    if supplied_adjusted:
        _reject_unadmitted_direct_children(
            supplied_adjusted[0],
            allowed=_ALLOWED_LEAF_CHILDREN,
            scope=f"{blocker_field}_adjusted_date",
            namespace=namespace,
        )
        supplied_value = _optional_text(supplied_adjusted[0].text)
        if supplied_value is None:
            _fail(
                f"missing_contract_field:fpml_{blocker_field}_adjusted_date",
                "contract_gap",
                f"FpML {name} adjusted date cannot be empty.",
                missing_fields=(f"{name}.adjusted_date",),
            )
        try:
            declared_adjusted = _parse_xsd_date(supplied_value)
        except ValueError:
            _fail(
                "external_import:fpml_malformed_economic_date",
                "malformed_document",
                f"FpML {name} adjusted date must be a valid XML Schema date.",
            )
        if declared_adjusted != adjusted:
            _fail(
                f"contract_conflict:fpml_{blocker_field}_adjusted_date",
                "contract_conflict",
                f"FpML {name} supplied adjusted date conflicts with its date adjustments.",
            )
    return unadjusted, adjusted


def _date_adjustment(
    element,
    *,
    namespace: str | None,
    scope: str,
    allowed: set[str],
):
    _reject_unadmitted_direct_children(
        element,
        allowed=allowed,
        scope=scope,
        namespace=namespace,
    )
    token = _required_text(
        element,
        "businessDayConvention",
        namespace=namespace,
        missing_field="business_day_convention",
    ).upper()
    try:
        bda = _BUSINESS_DAY_ADJUSTMENTS[token]
    except KeyError:
        _fail(
            "external_import:fpml_business_day_convention_unsupported",
            "unsupported_contract",
            f"Business-day convention {token!r} is outside the admitted closure.",
        )
    business_centers = _direct_children(
        element,
        "businessCenters",
        namespace=namespace,
    )
    if len(business_centers) > 1:
        _fail(
            "contract_ambiguity:fpml_business_centers",
            "contract_ambiguity",
            "FpML date adjustments contain multiple business-center containers.",
            ambiguous_fields=("business_centers",),
        )
    if business_centers:
        _reject_unadmitted_direct_children(
            business_centers[0],
            allowed=_ALLOWED_BUSINESS_CENTERS_CHILDREN,
            scope="business_centers",
            namespace=namespace,
        )
    center_elements = (
        _direct_children(
            business_centers[0],
            "businessCenter",
            namespace=namespace,
        )
        if business_centers
        else ()
    )
    center_values = []
    for center in center_elements:
        _reject_unadmitted_direct_children(
            center,
            allowed=_ALLOWED_LEAF_CHILDREN,
            scope="business_center",
            namespace=namespace,
        )
        text = _optional_text(center.text)
        if text is None:
            _fail(
                "missing_contract_field:fpml_business_center",
                "contract_gap",
                "FpML business-center values cannot be empty.",
                missing_fields=("business_center",),
            )
        center_values.append(text)
    if business_centers and not center_values:
        _fail(
            "missing_contract_field:fpml_business_center",
            "contract_gap",
            "FpML business-center containers require at least one center.",
            missing_fields=("business_center",),
        )
    if not center_values:
        if bda != BusinessDayAdjustment.UNADJUSTED:
            _fail(
                "missing_contract_field:fpml_business_centers",
                "contract_gap",
                "Adjusted FpML dates require explicit admitted business centers.",
                missing_fields=("business_centers",),
            )
        return bda, WEEKEND_ONLY
    calendars = []
    for center in center_values:
        try:
            calendars.append(_CALENDARS[center])
        except KeyError:
            _fail(
                "external_import:fpml_business_center_unsupported",
                "unsupported_contract",
                f"Business center {center!r} is outside the Trellis calendar closure.",
            )
    calendar = calendars[0] if len(calendars) == 1 else JointCalendar(*calendars)
    return bda, calendar


def _frequency(
    element,
    *,
    namespace: str | None,
    scope: str,
    allow_roll_convention: bool = False,
):
    allowed = {"periodMultiplier", "period"}
    if allow_roll_convention:
        allowed.add("rollConvention")
    _reject_unadmitted_direct_children(
        element,
        allowed=allowed,
        scope=scope,
        namespace=namespace,
    )
    multiplier = _integer_text(
        element,
        "periodMultiplier",
        namespace=namespace,
        missing_field="period_multiplier",
    )
    period = _required_text(
        element,
        "period",
        namespace=namespace,
        missing_field="period",
    ).upper()
    try:
        return _FREQUENCIES[(multiplier, period)]
    except KeyError:
        _fail(
            "external_import:fpml_frequency_unsupported",
            "unsupported_contract",
            f"Frequency {multiplier}{period} is outside the admitted closure.",
        )


def _period_label(element, *, namespace: str | None) -> str:
    _reject_unadmitted_direct_children(
        element,
        allowed={"periodMultiplier", "period"},
        scope="index_tenor",
        namespace=namespace,
    )
    multiplier = _integer_text(
        element,
        "periodMultiplier",
        namespace=namespace,
        missing_field="index_tenor_multiplier",
    )
    period = _required_text(
        element,
        "period",
        namespace=namespace,
        missing_field="index_tenor_period",
    ).upper()
    if multiplier <= 0 or period not in {"D", "W", "M", "Y"}:
        _fail(
            "external_import:fpml_index_tenor_unsupported",
            "unsupported_contract",
            f"Index tenor {multiplier}{period} is outside the admitted closure.",
        )
    return f"{multiplier}{period}"


def _validate_roll_convention(element, start: date, *, namespace: str | None) -> None:
    roll_elements = _direct_children(element, "rollConvention", namespace=namespace)
    if len(roll_elements) > 1:
        _fail(
            "contract_ambiguity:fpml_roll_convention",
            "contract_ambiguity",
            "FpML normalization found multiple roll conventions for one schedule.",
            ambiguous_fields=("roll_convention",),
        )
    if roll_elements:
        _reject_unadmitted_direct_children(
            roll_elements[0],
            allowed=_ALLOWED_LEAF_CHILDREN,
            scope="roll_convention",
            namespace=namespace,
        )
    roll = _optional_text(roll_elements[0].text if roll_elements else None)
    if roll is None or roll.upper() == "NONE":
        return
    if roll.isdigit() and int(roll) == start.day:
        return
    _fail(
        "external_import:fpml_roll_convention_unsupported",
        "unsupported_contract",
        f"Roll convention {roll!r} is outside the regular schedule closure.",
    )


def _validate_regular_schedule(start: date, end: date, frequency: Frequency) -> None:
    months_per_period = 12 // frequency.value
    month_span = (end.year - start.year) * 12 + end.month - start.month
    generated_dates = tuple(generate_schedule(start, end, frequency))
    expected_periods = month_span // months_per_period if month_span > 0 else 0
    if (
        month_span <= 0
        or month_span % months_per_period
        or start.day != end.day
        or len(generated_dates) != expected_periods
        or any(item.day != start.day for item in generated_dates)
    ):
        _fail(
            "external_import:fpml_stub_period_unsupported",
            "unsupported_contract",
            "The effective and termination dates imply an unsupported stub or "
            "end-of-month roll.",
        )


def _required_child(parent, name: str, *, namespace: str | None, missing_field: str):
    matches = _direct_children(parent, name, namespace=namespace)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _fail(
            f"missing_contract_field:fpml_{missing_field}",
            "contract_gap",
            f"FpML normalization requires {missing_field.replace('_', ' ')}.",
            missing_fields=(missing_field,),
        )
    _fail(
        f"contract_ambiguity:fpml_{missing_field}",
        "contract_ambiguity",
        f"FpML normalization found multiple {missing_field.replace('_', ' ')} values.",
        ambiguous_fields=(missing_field,),
    )


def _required_text(
    parent, name: str, *, namespace: str | None, missing_field: str
) -> str:
    element = _required_child(
        parent,
        name,
        namespace=namespace,
        missing_field=missing_field,
    )
    _reject_unadmitted_direct_children(
        element,
        allowed=_ALLOWED_LEAF_CHILDREN,
        scope=missing_field,
        namespace=namespace,
    )
    text = _optional_text(element.text)
    if text is None:
        _fail(
            f"missing_contract_field:fpml_{missing_field}",
            "contract_gap",
            f"FpML normalization requires {missing_field.replace('_', ' ')}.",
            missing_fields=(missing_field,),
        )
    return text


def _required_href(
    parent, name: str, *, namespace: str | None, missing_field: str
) -> str:
    element = _required_child(
        parent,
        name,
        namespace=namespace,
        missing_field=missing_field,
    )
    _reject_unadmitted_direct_children(
        element,
        allowed=_ALLOWED_LEAF_CHILDREN,
        scope=missing_field,
        namespace=namespace,
    )
    href = _optional_text(element.attrib.get("href"))
    if href is None:
        _fail(
            f"missing_contract_field:fpml_{missing_field}",
            "contract_gap",
            f"FpML normalization requires {missing_field.replace('_', ' ')} href.",
            missing_fields=(missing_field,),
        )
    return href


def _required_attribute(element, name: str, *, missing_field: str) -> str:
    value = _optional_text(element.attrib.get(name))
    if value is None:
        _fail(
            f"missing_contract_field:fpml_{missing_field}",
            "contract_gap",
            f"FpML normalization requires {missing_field.replace('_', ' ')}.",
            missing_fields=(missing_field,),
        )
    return value


def _require_reference(
    parent,
    name: str,
    expected_id: str,
    *,
    namespace: str | None,
) -> None:
    reference = _required_child(
        parent,
        name,
        namespace=namespace,
        missing_field=_snake_case(name),
    )
    _reject_unadmitted_direct_children(
        reference,
        allowed=_ALLOWED_LEAF_CHILDREN,
        scope=_snake_case(name),
        namespace=namespace,
    )
    href = _optional_text(reference.attrib.get("href"))
    if href is None:
        _fail(
            f"missing_contract_field:fpml_{_snake_case(name)}",
            "contract_gap",
            f"FpML normalization requires {_snake_case(name).replace('_', ' ')} href.",
            missing_fields=(_snake_case(name),),
        )
    if href != expected_id:
        _fail(
            "contract_conflict:fpml_schedule_reference",
            "contract_conflict",
            f"FpML schedule reference {href!r} does not identify {expected_id!r}.",
        )


def _snake_case(value: str) -> str:
    pieces: list[str] = []
    for character in value:
        if character.isupper() and pieces:
            pieces.append("_")
        pieces.append(character.lower())
    return "".join(pieces)


def _integer_text(
    parent, name: str, *, namespace: str | None, missing_field: str
) -> int:
    text = _required_text(
        parent,
        name,
        namespace=namespace,
        missing_field=missing_field,
    )
    if _XML_INTEGER_PATTERN.fullmatch(text) is None:
        _fail(
            f"external_import:fpml_malformed_{missing_field}",
            "malformed_document",
            f"FpML {missing_field.replace('_', ' ')} must use XML integer syntax.",
        )
    return int(text)


def _finite_float(value: str, *, field_name: str, positive: bool = False) -> float:
    if _XML_DECIMAL_PATTERN.fullmatch(value) is None:
        parsed = math.nan
    else:
        parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        _fail(
            f"external_import:fpml_malformed_{field_name}",
            "malformed_document",
            f"FpML {field_name.replace('_', ' ')} must be a finite"
            + (" positive" if positive else "")
            + " XML decimal.",
        )
    return parsed


def _descendants(parent, name: str, *, namespace: str | None):
    return tuple(
        element
        for element in parent.iter()
        if _split_tag(element.tag) == (namespace, name)
    )


def _descendant(parent, name: str, *, namespace: str | None):
    matches = _descendants(parent, name, namespace=namespace)
    return matches[0] if matches else None


def _contains_any(parent, names: set[str], *, namespace: str | None) -> bool:
    return any(
        element_namespace == namespace and local_name in names
        for element in parent.iter()
        for element_namespace, local_name in (_split_tag(element.tag),)
    )


def _reject_nested_metadata_children(
    parent,
    *,
    names: set[str],
    namespace: str | None,
) -> None:
    for name in names:
        for element in _direct_children(parent, name, namespace=namespace):
            _reject_unadmitted_direct_children(
                element,
                allowed=_ALLOWED_LEAF_CHILDREN,
                scope=_snake_case(name),
                namespace=namespace,
            )


def _reject_unadmitted_direct_children(
    parent,
    *,
    allowed: set[str],
    scope: str,
    namespace: str | None,
) -> None:
    unsupported = tuple(
        sorted(
            {
                (
                    local_name
                    if child_namespace == namespace
                    else f"foreign_namespace:{local_name}"
                )
                for child in tuple(parent)
                for child_namespace, local_name in (_split_tag(child.tag),)
                if child_namespace != namespace or local_name not in allowed
            }
        )
    )
    if unsupported:
        _fail(
            f"external_import:fpml_{scope}_feature_unsupported",
            "unsupported_contract",
            "The bounded FpML normalizer does not consume "
            + ", ".join(unsupported)
            + f" in {scope.replace('_', ' ')}.",
        )


def _normalize_option_premiums(
    product,
    *,
    namespace: str | None,
    valuation_date: date | None,
    known_party_ids: tuple[str, ...],
    option_party_ids: tuple[str, str],
    product_scope: str,
    product_label: str,
) -> tuple[FpMLPremiumMetadata, ...]:
    result = []
    for premium in _direct_children(product, "premium", namespace=namespace):
        _reject_unadmitted_direct_children(
            premium,
            allowed=_ALLOWED_PREMIUM_CHILDREN,
            scope=f"{product_scope}_premium",
            namespace=namespace,
        )
        payer = _required_href(
            premium,
            "payerPartyReference",
            namespace=namespace,
            missing_field="premium_payer_party_reference",
        )
        receiver = _required_href(
            premium,
            "receiverPartyReference",
            namespace=namespace,
            missing_field="premium_receiver_party_reference",
        )
        if (
            payer == receiver
            or not {payer, receiver}.issubset(set(known_party_ids))
            or {payer, receiver} != set(option_party_ids)
        ):
            _fail(
                f"contract_conflict:fpml_{product_scope}_premium_parties",
                "contract_conflict",
                f"{product_label} premium parties must be distinct identified trade parties.",
            )
        amount_element = _required_child(
            premium,
            "paymentAmount",
            namespace=namespace,
            missing_field="premium_payment_amount",
        )
        _reject_unadmitted_direct_children(
            amount_element,
            allowed=_ALLOWED_PAYMENT_AMOUNT_CHILDREN,
            scope="premium_payment_amount",
            namespace=namespace,
        )
        currency = _required_text(
            amount_element,
            "currency",
            namespace=namespace,
            missing_field="premium_currency",
        ).upper()
        amount = _finite_float(
            _required_text(
                amount_element,
                "amount",
                namespace=namespace,
                missing_field="premium_amount",
            ),
            field_name="premium_amount",
            positive=True,
        )
        _, payment_date = _adjustable_date(
            premium,
            "paymentDate",
            namespace=namespace,
        )
        if valuation_date is not None and payment_date >= valuation_date:
            _fail(
                f"external_import:fpml_{product_scope}_unsettled_premium_unsupported",
                "unsupported_contract",
                "Only premiums settled before the valuation date are admitted as "
                "separate source metadata.",
            )
        result.append(
            FpMLPremiumMetadata(
                payer_party_id=payer,
                receiver_party_id=receiver,
                payment_date=payment_date,
                currency=currency,
                amount=amount,
            )
        )
    return tuple(result)


def _provenance(
    xml_path: str, semantic_field: str, value: object
) -> FpMLFieldProvenance:
    return FpMLFieldProvenance(
        xml_path=xml_path,
        semantic_field=semantic_field,
        normalized_value=value.isoformat() if isinstance(value, date) else str(value),
    )


def _blocker(
    blocker_id: str,
    category: str,
    summary: str,
    *,
    missing_fields: tuple[str, ...] = (),
    ambiguous_fields: tuple[str, ...] = (),
) -> FpMLImportBlocker:
    return FpMLImportBlocker(
        id=blocker_id,
        category=category,
        summary=summary,
        missing_fields=missing_fields,
        ambiguous_fields=ambiguous_fields,
    )


def _fail(
    blocker_id: str,
    category: str,
    summary: str,
    *,
    missing_fields: tuple[str, ...] = (),
    ambiguous_fields: tuple[str, ...] = (),
) -> None:
    raise _NormalizationBlocked(
        _blocker(
            blocker_id,
            category,
            summary,
            missing_fields=missing_fields,
            ambiguous_fields=ambiguous_fields,
        )
    )


def _clarification(blocker: FpMLImportBlocker) -> FpMLClarification:
    messages = ()
    if blocker.missing_fields:
        messages += (
            "Provide unambiguous FpML values for: "
            + ", ".join(blocker.missing_fields)
            + ".",
        )
    if blocker.ambiguous_fields:
        messages += (
            "Disambiguate the FpML fields: "
            + ", ".join(blocker.ambiguous_fields)
            + ".",
        )
    return FpMLClarification(
        requires_clarification=bool(blocker.missing_fields or blocker.ambiguous_fields),
        missing_fields=blocker.missing_fields,
        ambiguous_fields=blocker.ambiguous_fields,
        messages=messages,
    )


def _blocked_from(
    inspected: FpMLImportReport,
    blocker: FpMLImportBlocker,
) -> FpMLImportReport:
    return FpMLImportReport(
        status="blocked",
        profile=inspected.profile,
        document=inspected.document,
        trade=inspected.trade,
        trade_envelope=inspected.trade_envelope,
        blockers=(blocker,),
        clarification=_clarification(blocker),
    )
